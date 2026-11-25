#!/usr/bin/env bash
# Step the GPU clock offsets against a running `./serve.sh bench` server and keep the fastest stable setting.
# Core first, then memory, two passes of five runs per step. It stops climbing on an axis at the first GPU error (Xid),
# server failure or slower decode step, and ALWAYS returns the clocks to stock and the fans to automatic on exit.
# usage: sudo -v && ./sweep.sh        (env: PORT, CORE_STEPS, MEM_STEPS, OUT)
# Needs: python3 with nvidia-ml-py for root (oc.py), a cold room and an open chassis help.
set -u
PORT="${PORT:-8000}"; OUT="${OUT:-push.jsonl}"
CORE_STEPS="${CORE_STEPS:-75 150 225 300 375 450}"
MEM_STEPS="${MEM_STEPS:-1500 3000 4500 5250 6000}"
HERE=$(cd "$(dirname "$0")" && pwd)
OC="sudo python3 $HERE/oc.py"
B="python3 $HERE/bench.py --url http://127.0.0.1:$PORT --model dsv41 --no-think --prompts $HERE/prompts/codeonly.json --only fraser-webhook --fraser-reps 1 --out $OUT"
log() { echo "$(date -u +%T) $*"; }
xid() { sudo dmesg | grep -c "NVRM: Xid"; }
alive() { curl -sf -m 10 "localhost:$PORT/v1/models" >/dev/null; }
stat() { python3 "$HERE/stat.py" "$OUT" "$1"; }          # median tok/s, median step ms, runs
trap '$OC reset >/dev/null; $OC fan auto >/dev/null; log "clocks back at stock, fans automatic"' EXIT
alive || { echo "no server on :$PORT; run ./serve.sh bench first"; exit 1; }
$OC fan 100 >/dev/null; X0=$(xid)
passes() { for _ in $(seq 1 "$2"); do $B --label "$1" >/dev/null 2>&1 || return 1; done; }
step() {   # label core mem
  $OC set "$2" "$3" >/dev/null || return 1
  sleep 5
  passes "$1" 2 && alive || { log "$1: bench failed or the server died"; return 1; }
  [ "$(xid)" = "$X0" ] || { log "$1: new Xid: $(sudo dmesg | grep 'NVRM: Xid' | tail -1)"; return 1; }
  log "$1 (core +$2, mem +$3): $(stat "$1")  [tok/s, step ms, runs]"
}
faster() { python3 -c "import sys; sys.exit(0 if float('$1') < float('$2') else 1)"; }
passes STOCK 3; bs=$(stat STOCK | awk '{print $2}'); bc=0; bm=0
log "STOCK: $(stat STOCK)"
for c in $CORE_STEPS; do
  step "C$c" "$c" 0 || { $OC set $bc $bm >/dev/null; break; }
  s=$(stat "C$c" | awk '{print $2}'); faster "$s" "$bs" || break; bs=$s; bc=$c
done
alive && for m in $MEM_STEPS; do
  step "C${bc}M$m" "$bc" "$m" || { $OC set $bc $bm >/dev/null; break; }
  s=$(stat "C${bc}M$m" | awk '{print $2}'); faster "$s" "$bs" || break; bs=$s; bm=$m
done
if alive; then
  $OC set $bc $bm >/dev/null; sleep 5
  passes BEST 4
  log "BEST core +$bc, mem +$bm: $(stat BEST)  [tok/s, step ms, runs]"
fi
