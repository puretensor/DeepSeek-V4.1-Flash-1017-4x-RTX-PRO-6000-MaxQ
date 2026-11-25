# DeepSeek V4.1 Flash 1017 on four RTX PRO 6000 Blackwell Max-Q: 760 tok/s, one stream

Single-stream decode reached 760.1 tok/s (batch 1) on four RTX PRO 6000 Blackwell Max-Q cards at their stock 325 W power cap, overclocked with the driver's own clock offsets: core +450 MHz and memory +5250 MHz. On 2026-11-25, this was the fastest run for `deepseek-ai/DeepSeek-V4.1-Flash-1017` on LocalMaxxing. The previous best was 713.3 by Fraser Price (`fraserprice`) on the same card model. The same session set the fastest Verified run on LocalMaxxing's canonical `code-v1` prompt for this model at 623.0 tok/s, ahead of the previous best of 622.3 on 600 W Workstation Edition cards.

The engine is not ours. It is Fraser Price's BRRR recipe (fused decode kernels, Apache-2.0) on Local Inference Lab's `b12x` vLLM, built unmodified. Warm thanks to both for building and sharing it. This repository holds the host settings, the overclock procedure, the benchmark and the results. Every number in the results tables links to its LocalMaxxing run page.

## Quick start

```bash
./build.sh                                  # builds the BRRR image at the pinned revision
MODEL=/path/to/DeepSeek-V4.1-Flash-1017 ./serve.sh bench
python3 bench.py --url http://127.0.0.1:8000 --model dsv41 --no-think --prompts prompts/canonical_prompts.json --only code-v1,reasoning-v1 --label stock --out results.jsonl
./sweep.sh                                  # steps the clock offsets, keeps the fastest stable setting, resets on exit
```

## Results

Single-slot bench arm (32K context, 1 slot), batch 1, temperature 0:

| Clock offsets | Prompt | tok/s | Verified | Run |
| - | - | - | - | - |
| core +450, memory +5250 MHz | code-only, 1.9K tokens | 760.1 | no (not a canonical prompt) | https://www.localmaxxing.com/runs/cmwwkkwv70016qd0181keoetm |
| core +225, memory +4500 MHz | code-only, 1.9K tokens | 753.5 | no (not a canonical prompt) | https://www.localmaxxing.com/runs/cmwwk60ed000xqd014ewk5juy |
| core +225, memory +4500 MHz | code-v1 | 623.0 | yes | https://www.localmaxxing.com/runs/cmwwkkxku0019qd01a3x65w2c |
| core +450, memory +5250 MHz | code-v1 | 622.8 | yes | https://www.localmaxxing.com/runs/cmwwkky8c001cqd01d03mtqmf |
| core +450, memory +5250 MHz | webhook x10, 2.7K tokens | 672.1 | no (not a canonical prompt) | https://www.localmaxxing.com/runs/cmwwkkyhs001fqd01jhe8fn7n |

Stock clocks on the 16-slot, 1M-context serving config from the same day:

| Prompt | tok/s | Verified | Run |
| - | - | - | - |
| code-only, 1.9K tokens | 695.1 | no | https://www.localmaxxing.com/runs/cmwwj6qwq000tqd0125zp24b4 |
| code-v1 | 561.4 | yes | https://www.localmaxxing.com/runs/cmwwj6lds000eqd01pacq7o52 |
| reasoning-v1 | 319.1 | yes | https://www.localmaxxing.com/runs/cmwwj6m3j000hqd01qqm1omks |
| webhook x10, 2.7K tokens | 589.6 | no | https://www.localmaxxing.com/runs/cmwwj6mcy000kqd01j7211wl9 |
| code-v1, thinking high | 439.1 | yes | https://www.localmaxxing.com/runs/cmwwj6n0j000nqd01nwpjazo6 |
| reasoning-v1, thinking high | 355.8 | yes | https://www.localmaxxing.com/runs/cmwwj6nq1000qqd01q8h1qy0r |

## What each lever bought

Medians on the `code-only` prompt, single-slot arm:

| Setting | tok/s | Decode step |
| - | - | - |
| Stock clocks | 691 | 8.79 ms |
| Core +225 MHz | 714 | 8.60 ms |
| Core +225, memory +4500 MHz | 754 | 8.08 ms |
| Core +450, memory +5250 MHz | 761 | 7.98 ms |

Decode speed is tokens per step divided by step time. With DSpark drafting 7 tokens, this prompt emits about 6.07 tokens per step; 6.07 / 7.99 ms = 760.

Memory clock was the big lever. Core offset alone was worth about 3%, and adding the memory offset gave about 6% more. Decode on this model is memory-bound. Memory +6000 MHz was no faster than +5250, while core offsets above +225 added about 1%. The cards stay at the 325 W cap the whole time; under a power cap, a positive core offset gives more clock for the same power.

With GPU fans at 100%, the chassis open and the room cold, cards ran at 56 to 71 C under load. Earlier the same day, with the chassis closed in a warm room, the hottest card had reached 92 C. Neither session logged any Xid errors, and the same request-shape checks pass as on the stock configuration.

The single-slot arm alone at stock clocks measured 691 against 695 on the 16-slot serving config, so on this day it was not worth anything by itself. The offsets are volatile: they clear on a driver reload or reboot, and `sweep.sh` resets them on exit. We serve at stock clocks.

## Host settings

The workstation uses a Threadripper PRO 9975WX, 512 GB DDR5 and four cards on PCIe 5.0 x16, with no NVLink. It runs Ubuntu 26.04 and NVIDIA driver 620.70.

Set the kernel command line to include `amd_iommu=off iommu=off`. `host/nvidia-p2p.conf` sets the NVIDIA peer-to-peer registry keys (the same ones the BRRR recipe's `p2p.sh` sets), and `host/nvidia-ro.conf` enables PCIe relaxed ordering (`NVreg_EnablePCIERelaxedOrderingMode=1`). On this engine we measured relaxed ordering as neutral to slightly positive, with a decode step of 8.8 ms against 8.8 to 9.0 ms without it on a quiet host, so we keep it on. Both files need `update-initramfs -u` and a reboot.

During benchmarks, CPU boost is on, the governor is set to performance and the deepest C-state is off. Engram tables are read from NVMe through `io_uring`, so the model directory must sit on local `ext4` or `xfs`.

## Method and caveats

The weights are the official `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint, revision `1bae0e10`, exactly as released (510 GB): FP8 with 4-bit (`fp4`) expert weights, as its own `config.json` declares. Nothing was re-quantized. The KV cache is FP8.

Runs are batch 1, temperature 0 and streamed, using cold prompts with a cache-bust nonce. Each pass is 2 warm-ups and 5 runs. The 760 and 753 results are the run closest to the median of 20 runs (four passes).

The `code-only` prompt is Fraser Price's own BRRR bench prompt, a repeated webhook coding task with an instruction to output code only. LocalMaxxing marks it unverified because it is not one of its canonical prompts; the `code-v1` runs are Verified. Code-only output is what lifts acceptance to about 6 tokens per step. On prose, acceptance and speed are much lower: `reasoning-v1` does 319 tok/s at stock.

This is a benchmark configuration. One slot and 32K context is not how we serve. Give a freshly started server a few minutes of traffic before timing it, because the first runs after a start took 12 to 27 ms per step before settling. Overclocking can hang a GPU or corrupt results. `sweep.sh` climbs in small steps, stops at the first GPU error, failure or slower step, and always resets. Your cards and cooling will have a different ceiling.

## Files

- `build.sh`
- `serve.sh`
- `sweep.sh`
- `oc.py`: clock offsets and fan control through NVML
- `bench.py`: the benchmark, LocalMaxxing rules
- `stat.py`
- `prompts/`
- `host/`
- `results.json`: every run with its page

## Licence

Our files are MIT; the engine recipe is Apache-2.0 and the model MIT, see `NOTICE`.
