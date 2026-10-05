# RTX 5060 Ti 16 GB: Bonsai 2 27B on Windows

This branch builds NInfer for the RTX 5060 Ti through Ubuntu WSL2 and serves the
bundled chat app at `http://localhost:8080`. Double-click `Start-Bonsai.cmd` after
building. Keep its terminal open; Ctrl+C stops the server and releases its GPU allocations.
Startup and request logs are under `logs/launch/`.

The launcher selects one request at a time, `rk4v4` KV, FP16 GDN state, a 65,536-token
context, and DFlash2 with seven drafts and the optimized proposal head. Temperature
defaults to zero and thinking is disabled for quicker replies. Explicit request
settings override these response defaults; `-Temperature` and `-Thinking` change
them for the session. These choices reduce sampling variety and reasoning effort.
`rk4v4` stores
rotated Lloyd-Max 4-bit keys with FP16 scales per 64 values and signed INT4 values
with FP16 scales per 32 values. The model's ternary weights stay in their stored
representation.

## Source and model

The requested upstream is [Neroued/ninfer](https://github.com/Neroued/ninfer).
The original checkout was `d44ab584`. Bonsai needs the ternary codec, activation
Hadamard transforms, and embedding restoration supplied by the compatible
[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) fork. This port uses
that fork's `809d4c3f` as its baseline. Its existing commits retain their authorship;
the 5060 Ti build, benchmark controls, numerical qualification, route profile and
launcher are documented in separate commits on `codex/rtx-5060-ti-bonsai`.
The baseline diverges from current upstream: it does not incorporate the later
preemptive scheduler or cache changes. Integration with current upstream requires
resolving that divergence and qualifying the combined engine.

Use the explicit artifact below; a GGUF file cannot be passed to the C++ Engine.

| Field | Value |
|---|---|
| Published model | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) |
| Revision | `b85b33627b27b9757a5a094fc74785a53e99f8ee` |
| Local file | `models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer` |
| File size | 9,520,051,456 bytes (8.87 GiB) |
| SHA256 | `cdc4810b0ff17c40d0f62cf214b6e0bcd08346e9eb05ca53371507037793c14a` |

The artifact includes optional MTP, DFlash2, proposal-head and Vision components.
Only the components selected for a run are loaded. The Windows launcher selects text
generation. The downloaded `NOTICE`, `SHA256SUMS` and conversion report are beside the model.

## Build in Ubuntu WSL2

The verified toolchain is CUDA 13.1.115, GCC 15.2.0, CMake 4.2.3, Ninja and Python 3.11.16
on Ubuntu 26.04.1. The Windows NVIDIA driver is 591.86. CUDA uses the Windows driver
through WSL2.

Run these commands in Ubuntu from the repository root:

```bash
sudo apt-get update
sudo apt-get install --yes build-essential cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev

/usr/local/cuda/bin/nvcc --version
/usr/lib/wsl/lib/nvidia-smi

# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
# Select Python 3.11 in a repository-relative environment.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  pytest numpy jinja2 requests huggingface_hub

bash scripts/fetch-webui.sh b11165
cmake --preset rtx5060-dev \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j 4
```

`rtx5060` builds the applications; `rtx5060-dev` also builds tests and benchmarks.
Both target `sm_120a` with the `mma.sync` compatibility implementations used by Bonsai.
The preset embeds the pinned llama.cpp `b11165` chat UI. The build does not download it.
CUDA compilation retains quoted host options so paths containing spaces work under `/mnt/c`.
On a 16 GiB WSL instance, unrestricted build parallelism started swapping. Four workers
kept the rebuild in RAM; this affects build time rather than the runtime configuration.

To restore the model in Ubuntu:

```bash
.venv/bin/hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 \
  Ternary-Bonsai-2-27B-ninfer-v3.ninfer SHA256SUMS NOTICE \
  Ternary-Bonsai-2-27B-ninfer-v3.ninfer.conversion.json \
  --revision b85b33627b27b9757a5a094fc74785a53e99f8ee --local-dir models/bonsai2
(cd models/bonsai2 && sha256sum --check SHA256SUMS --ignore-missing)
```

## Launch settings

Double-click `Start-Bonsai.cmd`, or run it from PowerShell:

```powershell
.\Start-Bonsai.cmd
.\Start-Bonsai.cmd -Spec none -Context 65536
.\Start-Bonsai.cmd -Spec dflash2 -DraftTokens 5 -Context 65536
.\Start-Bonsai.cmd -Spec mtp -DraftTokens 3
.\Start-Bonsai.cmd -Temperature 0.7 -Thinking
.\Start-Bonsai.cmd -Port 8081 -NoBrowser
.\Start-Bonsai.cmd -FastPrefill
```

`-Distribution` selects another Ubuntu WSL distribution. `-Model` and `-Server` accept
explicit Windows paths. An occupied port, missing model, missing binary, startup
failure or readiness timeout produces an error with the relevant log location.
The app opens after `/health` reports ready. The server listens on `127.0.0.1`.

The same Engine is available directly from Ubuntu:

```bash
build/apps/ninfer-serve models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --host 127.0.0.1 --port 8080 --model-id bonsai2-27b \
  --max-context 65536 --max-concurrency 1 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --temperature 0 --no-thinking
```

The [CLI](cli.md), [serving guide](serving.md) and executable `--help` describe the
other supported options. [Device profiles](device-profiles.md) explains how to refresh
the GPU's measured schedules.

## Measured settings

Measurements use this exact artifact, one RTX 5060 Ti with 36 SMs and 15.93 GiB
usable VRAM, CUDA 13.1, driver 591.86 and the WSL2 build above. Under load, GPU
clocks were about 2737 MHz with 13801 MHz memory; observed power and temperatures
spanned 168–178 W and 68–71 °C. The desktop remained active. Power and clocks were
not overridden. The final matrices ran serially without a concurrent build.

The comparison calls the public Engine with `bench/fixtures/bench_corpus.ids`,
greedy generation, disabled model stop defaults, one lane, 65,536-token capacity,
`rk4v4`, FP16 GDN state and CUDA Graphs. Each workload generates 128 measured
decode tokens after prefill; one warmup is discarded and three repetitions are
reported as arithmetic mean ± sample standard deviation. Prefix caching and
n-gram drafting are disabled. Rates depend on prompt content, sampling and draft
acceptance; arbitrary chat requests can be substantially slower.

The profile-off baseline uses the inherited GDN route. The profile pilot predates
the precision correction below. MTP and DFlash rows use the final qualified
SM120 arithmetic and built-in schedules. Earlier matrices remain pilot evidence;
their results were not overwritten when the correction changed acceptance.

| Configuration | Prompt tokens | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|---:|
| No speculation, profile off | 512 | 2243.29 ± 0.25 | 51.97 ± 0.031 | 2.6927 ± 0.0018 |
| No speculation, profile off | 8192 | 2077.33 ± 2.31 | 50.87 ± 0.002 | 6.4608 ± 0.0045 |
| No speculation, profile off | 32768 | 1707.13 ± 0.29 | 47.79 ± 0.028 | 21.8759 ± 0.0037 |
| No speculation, measured profile (pilot) | 512 | 2244.34 ± 13.17 | 52.12 ± 0.132 | 2.6853 ± 0.0069 |
| No speculation, measured profile (pilot) | 8192 | 2085.96 ± 2.62 | 50.78 ± 0.319 | 6.4500 ± 0.0206 |
| No speculation, measured profile (pilot) | 32768 | 1702.74 ± 3.37 | 48.01 ± 0.100 | 21.9131 ± 0.0369 |
| MTP3, optimized head | 512 | 2223.10 ± 0.60 | 79.12 ± 0.004 | 1.8488 ± 0.0001 |
| MTP3, optimized head | 8192 | 2093.81 ± 1.81 | 95.91 ± 0.019 | 5.2478 ± 0.0036 |
| MTP3, optimized head | 32768 | 1726.97 ± 0.22 | 89.50 ± 0.007 | 20.4056 ± 0.0025 |
| DFlash5, optimized head | 512 | 2238.16 ± 3.13 | 86.51 ± 0.059 | 1.7094 ± 0.0008 |
| DFlash5, optimized head | 8192 | 2071.33 ± 1.76 | 149.49 ± 0.044 | 4.8119 ± 0.0036 |
| DFlash5, optimized head | 32768 | 1709.48 ± 0.18 | 203.49 ± 0.086 | 19.7983 ± 0.0023 |
| DFlash7, optimized head (default) | 512 | 2252.77 ± 2.60 | 85.96 ± 0.012 | 1.7171 ± 0.0004 |
| DFlash7, optimized head (default) | 8192 | 2083.60 ± 0.72 | 154.85 ± 0.025 | 4.7588 ± 0.0014 |
| DFlash7, optimized head (default) | 32768 | 1709.40 ± 0.35 | 241.12 ± 0.155 | 19.7011 ± 0.0035 |
| DFlash7 plus fast prefill | 512 | 2237.96 ± 1.85 | 84.47 ± 0.008 | 1.7449 ± 0.0001 |
| DFlash7 plus fast prefill | 8192 | 2118.62 ± 0.71 | 154.83 ± 0.020 | 4.6939 ± 0.0012 |
| DFlash7 plus fast prefill | 32768 | 1841.42 ± 0.09 | 240.90 ± 0.205 | 18.3274 ± 0.0011 |

DFlash7 raises decode throughput by 65.41%, 204.38%, 404.56% against the
profile-off baseline at 512/8192/32768 prompt tokens. Total request time falls by
36.23%, 26.34%, 9.94%; prefill changes by +0.42%, +0.30%, +0.13%.
Against corrected MTP3, decode improves by 8.64%, 61.45%, 169.42% and total time
falls by 7.13%, 9.32%, 3.45%, with prefill changes +1.33%, −0.49%, −1.02%.
This selects speculative acceleration of an existing optimized ternary engine;
the inherited Bonsai fork already has a prefill tile tuned for this GPU.

Seven drafts costs 0.64% short decode and 0.45% short total time compared with
five. At 8K/32K, seven improves decode by 3.59%/18.49% and total time by
1.10%/0.49%; prefill changes +0.59%/−0.005%. The default selects seven for
the supported context range; `-DraftTokens 5` selects the faster short case.
Across three repetitions, DFlash7 accepted 204/288/327 drafts over 177/96/57
rounds at the three prompt lengths, with n-gram rounds zero. The very high
long-context acceptance belongs to this corpus and cannot be assumed for chat.

Larger draft windows were also measured on the same complete matrix after the
live app checks, with GPU work serial and no build overlap:

| Configuration | Prompt tokens | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|---:|
| DFlash9, optimized head | 512 | 2216.58 ± 5.43 | 72.13 ± 0.037 | 2.0064 ± 0.0005 |
| DFlash9, optimized head | 8192 | 2062.36 ± 1.38 | 143.30 ± 0.089 | 4.8662 ± 0.0026 |
| DFlash9, optimized head | 32768 | 1689.59 ± 0.30 | 284.01 ± 0.581 | 19.8457 ± 0.0043 |
| DFlash15, optimized head | 512 | 2216.23 ± 3.66 | 68.48 ± 0.044 | 2.1010 ± 0.0013 |
| DFlash15, optimized head | 8192 | 2062.87 ± 1.22 | 130.79 ± 0.176 | 4.9505 ± 0.0036 |
| DFlash15, optimized head | 32768 | 1688.46 ± 1.09 | 229.88 ± 0.092 | 19.9647 ± 0.0124 |

Against seven, nine changes decode by −16.08%, −7.46%, +17.79%, prefill
−1.61%, −1.02%, −1.16%, and total time +16.85%, +2.26%, +0.73%. Its higher
32K decode rate did not offset slower prefill over this 128-token completion.
Fifteen changes decode by −20.33%, −15.54%, −4.66%, prefill −1.62%, −0.99%,
−1.23%, and total time +22.36%, +4.03%, +1.34%. Both defaults were rejected.
A three-draft short screen reached 82.45 ± 0.042 decode tokens/s and
2206.19 ± 10.63 prefill tokens/s, with 1.7858 ± 0.0002 s total time:
−4.08% decode and +4.00% total against seven. These findings retain the
seven-draft default and five-draft short alternative.

`-FastPrefill` selects the inherited quantized prompt kernel. Against ordinary
DFlash7, prefill changes −0.66%, +1.68%, +7.72%, decode −1.73%, −0.012%,
−0.091%, and total time +1.62%, −1.36%, −6.97%. Keep it off for the default
short-response behavior; it reduces long-prompt latency in the measured cases.
Short accepted drafts changed from 204/177 rounds to 201/183. Its FP16 PV
partials and INT8 private query arithmetic can change the numerical trajectory.
Corrected fast MTP3 reached 80.40/92.48/86.19 decode tokens/s, improving total
time 1.52%/1.19%/6.50% against ordinary MTP3 while slowing long decode
3.58%/3.69%. At 32K, ordinary DFlash7 total time is 3.26% slower than fast
MTP3 (19.701 versus 19.079 s); fast DFlash7 reaches 18.327 s with the short
cost shown above. Each alternative remains available explicitly.

The measured profile alone, in its pilot, changed decode by +0.30%, −0.19%,
+0.46%, prefill by +0.05%, +0.42%, −0.26%, and total time by −0.28%, −0.17%,
+0.17%. Calibration did not establish a material whole-inference gain by itself.
The 16 measured route keys for this card and SM count are now built in. Their
catalog is `ac2d7b3e311831b0`, measured on 2026-10-04. The tables summarize the
measurement results; the commands below reproduce the workloads. Temporary
experimental schedules were removed.

To reproduce the final default comparison from Ubuntu:

```bash
build/bench/ninfer_bench \
  --weights models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --kv-dtype rk4v4 --gdn-state-fp16 --max-ctx 65536 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 0 \
  -pg "512,128;8192,128;32768,128" --warmup 1 -r 3 -o json
```

## Rejected settings and kernel experiments

The requested prefill chunk remains 1024; alignment to this card's attention
waves resolves it to 1152. Pilot requests of 2048/4096 reached only 2045/2013
prefill tokens/s at 8K versus 2054, with total times 5.389/5.471 s versus 5.387 s.
Workspace rose from 175 to 289/632 MiB. Larger chunks were rejected.

Before the precision correction, MTP3 neural decode reached 89.15/91.51/85.08.
MTP2/4/5/7 short screens reached 80.67/67.41/80.07/58.26. Native DFlash5/7
reached 70.70/77.05 and optimized DFlash5/7 reached 76.62/82.29. These screens
overlapped CPU compilation/linking while GPU work remained serial. Full matrices
for the final backend decision ran after the correction without a build.

An MTP3 draft attention window of 1024 reduced long acceptance and reached only
45.95/44.42 decode tokens/s at 8K/32K, losing 52.09%/50.36% against the final
full-history MTP3. It was rejected. MTP1 reached 79.68/78.85/74.99, gaining
0.70% short decode but losing 17.79%/16.21% long decode. The native MTP3
proposal head reached 75.66 versus 79.12 with identical short acceptance; it
saves 170.5 MiB of weights but was rejected for the speed default.

The pre-correction fast MTP3 pilot improved prefill by 1.32%/2.65%/8.42%,
but changed decode by −12.20%/+2.01%/+6.65% and total time by
+11.73%/−2.42%/−7.66%. That default was rejected. The final corrected
comparisons above supersede its backend choice, preserving the adverse pilot.

With default n-gram drafting enabled, pilot MTP3 reached 449/440 tokens/s at
8K/32K and DFlash2 reached 499–527. Almost all long rounds copied matching
prompt continuations. The launcher retains this facility, with 15 draft tokens
and minimum match 12; these rates cannot be attributed to neural generation.

Nsight Systems attributed about 63% of GPU time in one measured 512/128 MTP3
neural request to narrow ternary projections, and about 0.9% to activation
quantization. CUDA node tracing adds overhead; traces establish attribution.

The wider 64-row/1024-K tile used 108 registers and 37,888 bytes shared memory
and won no calibration key. A 64-row/512-K tile used 61 registers and 20,480
bytes, passed four independent T2 oracle suites, and won isolated widths. Its
broad profile changed decode by −16.47%, +1.79%, +3.97% and total time by
+16.78%, −0.75%, −0.46%, also reducing short acceptance. Retaining the prior
reduction order restored acceptance but reached 89.10 tokens/s with no
established gain. Matched traces showed gate projection −0.27% and GDN input
+0.63%, nearly cancelling. Both tiles and registrations were removed.

## Numerical qualification and maintenance

An independent FP64 recurrence check at the actual 1152-token prefill width
found output relative L2 error 0.00417553/0.00417139, exceeding the frozen
0.0041 limit. Explicit rounding of private FP32 state/control operands into
TF32 fixed this on the SM120 two-stage route. The unchanged checks now measure
0.00408413/0.00407843 output error and 0.00301536/0.00301778 state error,
below the state limit 0.00318828. The worst output has only 0.39% margin;
these represented fixtures establish qualification rather than a universal
error guarantee. The final matrices were remeasured after this correction.
Against the earlier MTP3 pilot, short decode fell 11.25% and total time rose
10.48%; long decode improved 4.81%/5.19% and total time fell 2.59%/2.50%.
Numerical limits were not relaxed to retain the faster pilot.

The [CUDA PTX conversion contract](https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html)
requires SM90 or newer for RN conversion. The fork's SM86 recurrence retains its
existing arithmetic. A compile check first caught the unsupported instruction
and passed after the architecture branch was added. This is a production-file
compile check with CUDA 13.1, not RTX 3090 numerical/performance qualification
or a validation of the other machine's MSVC/CUDA 12.8 environment. The local
5060 preset is independent of that host's existing `build-ninja` tree.

Independent FP32/FP64 Op oracles cover packed T2 A8 Linear, residual LinearAdd,
fused attention/GDN inputs, the actual vocabulary head, recurrence and replay;
exact transform/codec and KV append tests cover stored scales and packed data.
FP16 state is decoded independently and reference carry rounds only between
public calls. Permanent short/ragged, distinct-state, changed graph and
1152+carry tests protect the correction. Affected FP32 two-stage tests also pass.
Large T2 matrices and the full vocabulary head include independently checked
sampled output rows; the smaller/full comparison cases check every output.

The rk4v4 attention suite covers full-output short cases and 37 sampled long
cases through 262,144 keys, graphs, tails, cache preservation and buffer guards.
Two of these cases qualify 10/16-column speculative verification at 65,536
visible keys, including graph replay; the final suite passed in 99.35 s.
Fast rk4v4 adds 13 cases, including actual 512/1152 prompt widths and 65,536
visible keys with large value scales. Long cases compare selected query rows
and heads, every output dimension and every visible key. Other outputs receive
finite/buffer checks. They do not compare every long-context output element.
The first fast test crashed because its host fixture allocated INT8 instead of
packed Lloyd4 keys; the fixture was repaired and the original criteria passed.

Builds and affected host option/profile/report/schema tests passed. Launcher
tests pass in PowerShell 5.1 and 7, including paths with spaces, default and
explicit flags, invalid temperatures, readiness failures, exit codes and parent
death cleanup. Fractional temperature bounds initially slipped through integer
validation limits; floating-point limits and NaN rejection fixed that regression.
The public Engine, artifact representation and ownership remain explicit; this
port adds a measured device row, localized math correction, build preset and
launcher integration. It introduces no runtime weight repacking or new model
architecture. Full model task accuracy/perplexity, Vision and RTX 3090 runtime
performance have not been assessed in this optimization task.

## Resources and live application

At 65,536 capacity, the default neural benchmark reserves 8.158 GiB of weights,
1.291 GiB of sequence storage, 174.64 MiB of workspace and a 480 MiB graph
allowance. Main KV payload is 1.094 GiB. Compared with MTP3, weights increase
0.875 GiB and planned runtime reservation increases 0.419 GiB. Reported free
device memory after startup is 5.162 GiB under the benchmark's cache policy.

The full boundary run populated 65,500 prompt tokens and generated 33 outputs
within 65,536 capacity. It completed in 48.218 s, including 48.063 s prefill;
one repetition and no warmup establishes boundary execution, not a statistically
supported throughput claim.

The serving defaults enable prefix caching, unlike the inference benchmark.
They allocate one device checkpoint, eight host state checkpoints and an eager
8192 MiB pinned host KV arena. Keep the model resident while chatting to amortize
artifact loading and graph startup.

The actual Windows launcher started the selected server with the built-in
16-key profile, DFlash7, temperature zero and thinking off. Engine startup took
39.5 s, including 27.7 s weight upload, 0.8 s pinning 918.6 MiB of host states,
8.4 s pinning host KV and 1.5 s graph preparation; HTTP warmup added 0.64 s.
This startup was measured from the artifact under `/mnt/c`, not optimized by
moving or duplicating it. Runtime reservation with serving caches is 2.53 GiB.

Six live API requests passed: repeated greedy generation, SSE completion and
usage framing, omitted settings using launcher defaults, explicit temperature
0.7, and an explicit thinking override. Repeated replies matched in this sample;
prefix reuse covered 16 and then all 23 prompt tokens. Request times ranged
0.404–0.840 s. The thinking request spent its 48-token limit entirely on
reasoning and had no final answer; this verifies the override, not arithmetic
task accuracy. Larger output budgets are needed for that reasoning response.
Health, model discovery, props, embedded HTML and scheduler/cache stats passed.

The real bundled chat UI showed `Default: 0` for temperature, submitted a streamed
request with thinking off, and displayed a coherent GPU explanation. Its
353-token prompt and 37-token reply completed in 0.660 s, with server-reported
106.0 decode tokens/s. This single interaction checks integration and is not
the three-workload performance comparison.

After API and UI checks, total GPU usage including the desktop was 11,681 MiB,
with 4,370 MiB free. WSL used about 11.5 GiB RAM, had about 4.1 GiB available,
and used no swap. Ending only the owned Windows launcher disconnected health
immediately. The server was still exiting at the first sub-second process probe;
bounded follow-up confirmed its Linux process group was gone and total GPU usage
returned near the original 1,473 MiB desktop baseline. Parent death is tested on the
real server; physical Ctrl+C and console-close gestures were not driven through
Windows UI automation.

The final affected server/test build and three recurrence oracle suites passed
after preserving the SM86 path.
