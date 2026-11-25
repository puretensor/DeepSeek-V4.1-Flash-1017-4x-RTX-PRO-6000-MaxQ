#!/usr/bin/env python3
"""Set or show NVML clock offsets (and optionally fan speed) on every NVIDIA GPU in the machine.

usage (root): oc.py show | oc.py set CORE_MHZ MEM_MHZ | oc.py fan PERCENT | oc.py fan auto | oc.py reset
Offsets are the driver's own controls (on RTX PRO 6000 Blackwell Max-Q the driver allows core -1000..1000 and
memory -2000..6000 MHz) and are volatile:
`reset`, a driver reload or a reboot returns them to 0. Hard caps here keep a typo from asking for more than was tested.
"""
import ctypes
import sys

import pynvml as n

CORE_CAP, MEM_CAP = 500, 6000
n.nvmlInit()
H = [n.nvmlDeviceGetHandleByIndex(i) for i in range(n.nvmlDeviceGetCount())]


def get(h, t):
    info = n.c_nvmlClockOffset_t(); info.version = n.nvmlClockOffset_v1; info.type = t; info.pstate = 0
    n.nvmlDeviceGetClockOffsets(h, ctypes.byref(info))
    return info.clockOffsetMHz


def put(h, t, mhz):
    info = n.c_nvmlClockOffset_t(); info.version = n.nvmlClockOffset_v1; info.type = t; info.pstate = 0
    info.clockOffsetMHz = mhz
    n.nvmlDeviceSetClockOffsets(h, ctypes.byref(info))


def show():
    for i, h in enumerate(H):
        fans = []
        for f in range(n.nvmlDeviceGetNumFans(h)):
            fans.append(n.nvmlDeviceGetFanSpeed_v2(h, f))
        print(f"gpu{i} core {get(h, 0):+d} mem {get(h, 2):+d} MHz  fan {fans} %  temp {n.nvmlDeviceGetTemperature(h, 0)} C  "
              f"clock {n.nvmlDeviceGetClockInfo(h, 0)} MHz  {n.nvmlDeviceGetPowerUsage(h) / 1000:.0f} W")


cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
if cmd == "set":
    core, mem = int(sys.argv[2]), int(sys.argv[3])
    if not (0 <= core <= CORE_CAP and 0 <= mem <= MEM_CAP):
        sys.exit(f"refused: core 0..{CORE_CAP}, mem 0..{MEM_CAP}")
    for h in H:
        put(h, 0, core); put(h, 2, mem)
elif cmd == "reset":
    for h in H:
        put(h, 0, 0); put(h, 2, 0)
elif cmd == "fan":
    for h in H:
        for f in range(n.nvmlDeviceGetNumFans(h)):
            if sys.argv[2] == "auto":
                n.nvmlDeviceSetDefaultFanSpeed_v2(h, f)
            else:
                n.nvmlDeviceSetFanSpeed_v2(h, f, max(30, min(100, int(sys.argv[2]))))
show()
