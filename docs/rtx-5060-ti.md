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

Bonsai needs the ternary codec, activation Hadamard transforms, and embedding
restoration of [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all), a
descendant of [Neroued/ninfer](https://github.com/Neroued/ninfer). This port lives in a
fork of ninfer-all and uses its `809d4c3f` as the baseline. Its existing commits retain
their authorship; the 5060 Ti build, benchmark controls, numerical qualification,
route profile, launcher and optimizations are separate commits on
`codex/rtx-5060-ti-bonsai`. ninfer-all's `master` has advanced past the baseline,
and Neroued's upstream has diverged further (its later preemptive scheduler and
cache changes are not included). Those newer commits have not been qualified on
this card; integrating them requires resolving the divergence and qualifying the
combined engine.

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
Both target `sm_120a` with the `mma.sync` compatibility implementations used by Bonsai,
and set `NINFER_PDL=ON` so decode-graph kernels launch as programmatic dependents.
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
.\Start-Bonsai.cmd -Concurrency 1
```

`-Concurrency` (1 to 8, default 2) sets how many requests run at the same time; see
[concurrent requests](#concurrent-requests).

`-Distribution` selects another Ubuntu WSL distribution. `-Model` and `-Server` accept
explicit Windows drive paths or local `\\wsl.localhost\<distribution>\...` and
`\\wsl$\<distribution>\...` paths. The share's distribution must match
`-Distribution`. An occupied port, missing model, missing binary, startup
failure or readiness timeout produces an error with the relevant log location.
The app opens after `/health` reports ready. The server listens on `127.0.0.1`.
The console shows the server log as it is written (loading progress, each request, and
throughput every five seconds while it generates), and each launch also saves it under `logs\launch\`.

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

The built-in 5060 Ti profile selects prompt attention automatically for Bonsai's
24-query-head `rk4v4` geometry. Query tails shorter than 256 tokens keep standard
arithmetic. Every wider chunk, at any visible history through the model's
262144-token limit, uses the warp-specialized prompt kernel: INT8 QK and FP16 PV
partials over an FP16 value tile that a producer warpgroup decodes while the
consumer warps compute. The visible history is each chunk's own, including its
current tokens, rather than reserved capacity. Decode at width 1 and DFlash
verification at widths 6 to 8 form the small-T attention's P·V from e4m3 operands
([decode attention](#decode-attention-e4m3-pv)). `-FastPrefill` forces fast attention
on narrow chunks too; `NINFER_PROMPT_FAST=0` disables automatic fast selection when
the explicit fast option is absent.

### Faster startup from WSL storage

Keeping a copy of the model on the Linux filesystem substantially reduced loading
time on this machine. The launcher can select that copy explicitly. From Ubuntu,
with the repository as the working directory:

```bash
model_cache_dir="$HOME/.cache/ninfer-rtx5060-startup"
mkdir -p "$model_cache_dir"
cp --no-clobber -- models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  "$model_cache_dir/Ternary-Bonsai-2-27B-ninfer-v3.ninfer"
sha256sum models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  "$model_cache_dir/Ternary-Bonsai-2-27B-ninfer-v3.ninfer"
```

Check that both hashes equal the pinned artifact hash above. The copy consumes
9,520,051,456 additional bytes (8.87 GiB); copying and verifying both complete
files took 112.28 seconds in this preparation. From PowerShell in the repository:

```powershell
$bonsaiLinuxHome = (& wsl.exe --distribution Ubuntu --exec printenv HOME).Trim()
$bonsaiNativeModel = '\\wsl.localhost\Ubuntu' + $bonsaiLinuxHome.Replace('/', '\') +
  '\.cache\ninfer-rtx5060-startup\Ternary-Bonsai-2-27B-ninfer-v3.ninfer'
.\Start-Bonsai.cmd -Model $bonsaiNativeModel
```

Use the same distribution in both commands when selecting another Ubuntu instance.
Model selection remains explicit, and launch performs no copying or conversion.

The measured comparison used the current production binary, the same 17-key device
profile, `rk4v4`, FP16 GDN, DFlash2 with seven draft tokens and the draft head,
default n-gram/presence settings, one lane and 262144-token capacity. Each run
started a fresh server, waited for health readiness, requested a 19-token prompt
with 32 output tokens at temperature zero with thinking off, and stopped its
owned server. Sources ran in Windows/Linux/Linux/Windows order. Both files were
fully read during hash verification before timing; OS caches were not dropped.
These are two observations per source, without an estimated variance or a claim
about first use after reboot.

| Entry point / block | Model filesystem | Health readiness, seconds | Request, seconds |
|---|---|---:|---:|
| Direct Linux server / 1 | Windows drive (`/mnt/c`) | 43.2693 | 0.728924 |
| Direct Linux server / 2 | Linux | 15.8725 | 0.725247 |
| Direct Linux server / 3 | Linux | 15.8599 | 0.733202 |
| Direct Linux server / 4 | Windows drive (`/mnt/c`) | 43.8980 | 0.725813 |
| Windows PowerShell launcher / 1 | Windows drive | 49.6300 | 0.720 |
| Windows PowerShell launcher / 2 | Linux through WSL share | 18.7514 | 0.717 |
| Windows PowerShell launcher / 3 | Linux through WSL share | 18.2809 | 0.725 |
| Windows PowerShell launcher / 4 | Windows drive | 44.3219 | 0.718 |

Direct-server request times include the client exchange; Windows-launcher request
times are the server's millisecond-resolution request log. Compare within each
entry point. Matched direct-server startup improved 63.32% and 63.87%; matched
Windows-launcher startup improved 62.22% and 58.75%. The first Windows launch
took 5.31 seconds longer than the last; the source comparison does not identify
the cause of that variation. The `.cmd` wrapper and browser opening were outside
the Windows PowerShell timing.

The complete replies matched across all eight runs, with the same 32 output tokens
and draft acceptance. Direct-server request time changed -0.50% and +1.02%, while
server-reported Windows-launcher request time changed -0.42% and +0.97%; neither
comparison establishes an inference gain. Linux-source long generation and task
accuracy were not measured as part of this storage comparison. Host KV pinning
still took 8.3 seconds in both native-source Windows launches, with 918.6 MiB of
host states taking 0.77–0.90 seconds. Startup gains do not remove that resource
cost. PowerShell 5.1 and 7 launcher suites passed, including both WSL share
prefixes, spaces, distribution selection and mismatch rejection. Real native-copy
launches returned the matching API reply and released their server group when
the Windows parent stopped. A real nondefault-distribution launch is unverified.
Review found no issues in the bounded launcher change. A private timing helper's
missing quote was corrected and parsed before execution; an incorrect diagnostic
log filename was corrected without rerunning the measurements.

### Host KV pinning investigation

A private allocator experiment compared portable `cudaHostAlloc` with anonymous
memory populated and then registered using `cudaHostRegisterPortable`, either
with ordinary pages or local `MADV_HUGEPAGE` advice. CUDA's
[portable registration contract](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__MEMORY.html)
keeps the registered buffer pinned across device contexts. The WSL kernel was
already configured for huge pages on advice; no system setting changed. These
measurements used the same toolchain and GPU with the model unloaded. Each fresh
process prepared exactly 8 GiB, checked every byte through H2D/D2H transfers
against an independent integer pattern, and replayed a transfer graph twice with
different data. Preparation includes population for both registration methods;
qualification time is separate. There were two observations per method.

| Block / allocation | Preparation, seconds | Release, seconds | Qualification, seconds | Anonymous huge pages after pinning, MiB |
|---|---:|---:|---:|---:|
| 1 / portable allocation | 8.27209 | 1.74450 | 2.34671 | 0 |
| 2 / ordinary-page registration | 8.63568 | 0.79230 | 2.55224 | 0 |
| 3 / huge-page registration | 2.15092 | 0.25309 | 2.54035 | 8192 |
| 4 / huge-page registration | 1.02369 | 0.25685 | 2.50422 | 8192 |
| 5 / ordinary-page registration | 5.37557 | 0.74709 | 3.81103 | 0 |
| 6 / portable allocation | 7.04439 | 1.86657 | 2.51450 | 0 |

All six exact checks passed: 8,724,152,320 checked bytes per run, including graph
replays, totaling 52,344,913,920 bytes. Huge-page preparation was 74.00% and
85.47% shorter than the matched outer allocation controls. Ordinary-page
registration was 4.40% slower and 23.69% faster, so it did not provide a consistent
preparation improvement. Qualification was 8.25% slower in the first huge-page
run and 0.41% faster in the second than the matched allocation controls. These
whole-transfer checks mix CPU pattern work and copies and do not isolate
transfer throughput.
No separate bandwidth comparison was made. Preparation varied substantially
between repeated methods; its cause and variance remain unresolved. A 64 MiB
smoke test also passed exact ordinary/graph transfers for all three methods.

The same current public server then ran a fresh-server allocation/huge/huge/
allocation comparison using the verified native model copy, 262144-token
capacity and the settings in the storage comparison. A private interposer
replaced only the 8 GiB allocation; the binary and cache capacities were identical.
The probe asserted activation before accepting a timing and checked the entire
reply against the earlier native-copy control.

| Block / allocation | Health readiness, seconds | 19-prompt / 32-output request, seconds |
|---|---:|---:|
| 1 / current portable allocation | 16.8684 | 0.735432 |
| 2 / private huge-page allocation | 11.5666 | 0.737284 |
| 3 / private huge-page allocation | 10.7960 | 0.742673 |
| 4 / current portable allocation | 16.3160 | 0.783966 |

Matched readiness improved 31.43% and 33.83%; request time changed +0.25% and
-5.27%, which does not establish an inference gain. All four complete replies
matched, and all owned servers stopped successfully. The first interposer matrix
was invalid: the loader split its library path at a space and ignored it. Those
four timings are not allocation comparisons. The corrected library path had no
spaces and required the activation marker. An initial private probe compile
failed because `std::min` received different integer types; the explicit size
type fixed it before qualification.

The Core implementation now has an explicit `PreferHugePages` preference for
Host KV arenas. It preserves the requested pinned capacity, owns its anonymous
mapping and unregisters before unmapping. The existing portable allocator serves
small buffers and platforms without this preference. If unregistering fails,
cleanup reports the error and retains the mapping until process exit; that error
can retain the whole host arena. `PinnedHostBuffer` grows from 16 to 32 bytes and
its internal constructor changes, requiring its consumers to rebuild.

The five affected build targets passed with the pinned toolchain. Arena, arena
rank, physical packed KV and Qwen context-store tests passed. The new arena case
checks a 64 MiB + 4096 + 17 byte buffer against an independent exact byte pattern
through two graph replays after moves, then verifies ordinary-buffer replacement
and release of the old registration. A real Bonsai DFlash2 Responses workload
with 512-token device capacity and a 256 MiB host arena confirmed seven main KV
pages spilled to Host, three returned to Device, one state image restored and
321 cached prompt tokens reused. The pre-change and production full replies,
transfer counts and capacities agreed. The first probe incorrectly required an
MTP backend pool: this artifact's DFlash2 configuration has no persistent backend
KV pages. That failed assertion is preserved, and the corrected checks use the
existing recorded baseline. These functional runs overlapped linking and supply
no performance evidence.

Production then ran the same native-storage, 262144-capacity startup comparison,
with compilation stopped, the saved pre-change server and no interposer. Both
versions retained 8 GiB host KV and eight host state slots. Each process remained
resident for six seconds after its identical reply so adapter counters could be
sampled; that hold is excluded from readiness and request timings.

| Block / production allocation | Health readiness, seconds | 19-prompt / 32-output request, seconds | Peak dedicated, MiB | Peak shared, MiB |
|---|---:|---:|---:|---:|
| 1 / pre-change portable allocation | 16.8723 | 0.737550 | 15191.00 | 9251.06 |
| 2 / preferred huge-page allocation | 11.5471 | 0.729535 | 15214.95 | 9251.08 |
| 3 / preferred huge-page allocation | 10.7998 | 0.779731 | 15190.88 | 9251.08 |
| 4 / pre-change portable allocation | 15.8118 | 0.728156 | 15192.65 | 9235.02 |

Matched readiness improved 31.56% and 31.70%. Short request time changed -1.09%
and +7.08%; the second slowdown remains unexplained and these timings establish
no inference improvement. All four full replies, token counts and cache
capacities agreed, and owned servers stopped successfully. There were six to
eight adapter samples per process at roughly three-second cadence. Observed
dedicated headroom was 1096–1120 MiB; one production run peaked 24 MiB above its
matched baseline, while the other was 2 MiB below. These whole-adapter counters
include desktop activity and cannot attribute those differences to an allocation
or identify driver paging. The intentional host allocations still account for
most shared memory.

A subsequent serial baseline/production/production/baseline comparison used the
same native artifact and current TMA route, the 30018-token Navier–Stokes prompt
and 512 output tokens. It retained 262144-token capacity, `rk4v4`, FP16 GDN,
DFlash2 window seven with the optimized head, n-gram defaults, temperature zero,
thinking disabled and the explicit 17-key profile. Each fresh server had no
prefix reuse; startup is excluded from these inference timings.

| Block / host allocation | Prefill, tokens/s | Decode, tokens/s | Request total, seconds | Peak dedicated, MiB | Peak shared, MiB |
|---|---:|---:|---:|---:|---:|
| 1 / pre-change portable allocation | 1884.454 | 108.857 | 20.682544 | 15191.15 | 9264.63 |
| 2 / preferred huge-page allocation | 1895.321 | 109.048 | 20.580766 | 15200.19 | 9265.66 |
| 3 / preferred huge-page allocation | 1891.558 | 109.278 | 20.598248 | 15208.94 | 9235.87 |
| 4 / pre-change portable allocation | 1898.159 | 109.110 | 20.554912 | 15188.71 | 9238.50 |

Matched prefill changed +0.577% and -0.348%, decode +0.175% and +0.154%, and total
time -0.492% and +0.211%. The adverse prefill and total changes remain part of
the result; this small comparison establishes no general inference gain or
cause for the separate +7.08% short-request slowdown. All four complete
2095-character answers matched each other and the earlier production TMA control,
with 347/1149 accepted drafts, 163 speculative rounds, one fallback and zero
accepted n-gram tokens. Adapter sampling produced 11–13 observations per process;
production dedicated peaks were 9 and 20 MiB above the matched controls. This
allocation change is retained for the measured startup improvement and verified
cache behavior. Earlier startup runs followed complete artifact verification but
did not explicitly warm files immediately before each matrix. WSL can restart
after idle periods, so cache retention across those periods was not controlled;
these are observed fresh-process comparisons. Other Linux configurations, multi-device execution, native
Windows execution, 224.8K inference after this allocation change and a completely
filled 262K request remain unverified.

### Host state pinning investigation

With the production host KV preference active, a second private interposer
replaced only the portable 963198976-byte host state allocation: eight complete
120399872-byte images, about 918.6 MiB. It used the same Core allocation method
tested above, the same native artifact, 262K capacity and short request settings.
The interposer's activation marker was required in both candidate runs.

| Block / state allocation | Health readiness, seconds | 19-prompt / 32-output request, seconds | Host state pinning, seconds | Host KV pinning, seconds |
|---|---:|---:|---:|---:|
| 1 / current portable allocation | 12.6023 | 0.722711 | 0.901 | 3.9 |
| 2 / private preferred registration | 10.7199 | 0.720927 | 0.539079 | 3.1 |
| 3 / private preferred registration | 9.5406 | 0.730558 | 0.181123 | 2.4 |
| 4 / current portable allocation | 11.1450 | 0.724040 | 0.526 | 3.0 |

Matched readiness changed -14.94% and -14.40%; request time changed -0.247% and
+0.900%. All four full replies agreed, and owned servers stopped successfully.
Host state pinning shortened in both pairs, but host KV pinning varied as well;
the comparison cannot attribute the whole readiness difference to state pinning
alone. The `HostStatePool` constructor now requests the existing Core preference.
Its five-target incremental build, StateImage byte-copy checks, StateImage layout
and ContextStore tests passed. The real four-slot DFlash2 restore workload also
passed against the saved KV-only record, preserving complete replies, 321 reused
tokens, main KV/state transfers and capacities. The owned server's `smaps`
recorded a 470312 KiB state mapping with the huge page advice flag and 468992 KiB
(458 MiB) of anonymous huge pages, confirming the large backing for the real
459.3 MiB state pool. The bounded caller review found no source findings.

Production used the saved KV-only binary against the new state preference with
the same native artifact, cache capacities and settings. Both binaries and the
complete native artifact were read in the same WSL session immediately before
the four fresh-server runs. The six-second post-request residency window remained
outside readiness and request timings.

| Block / state allocation | Health readiness, seconds | 19-prompt / 32-output request, seconds | Host state pinning, seconds | Host KV pinning, seconds | Peak dedicated, MiB | Peak shared, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / KV-only portable state | 12.8565 | 0.712038 | 0.831 | 4.2 | 15205.30 | 9265.02 |
| 2 / preferred state backing | 10.5441 | 0.763641 | 0.164 | 2.6 | 15202.61 | 9263.83 |
| 3 / preferred state backing | 9.0382 | 0.785891 | 0.175 | 1.8 | 15205.42 | 9263.60 |
| 4 / KV-only portable state | 9.5390 | 0.772591 | 0.529 | 1.7 | 15205.24 | 9265.02 |

Matched readiness improved 17.99% and 5.25%. Both short requests were slower,
by 7.25% and 1.72%; those costs and the substantial variation in host KV pinning
remain unresolved. All four complete replies and capacities matched, with all
owned servers stopping successfully. Five to seven adapter samples per process
showed 1106–1108 MiB of dedicated headroom. Whole-adapter peaks do not establish
physical paging or attribute memory differences to the state pool.

The production state preference then ran the same matched 30018-prompt,
512-output paper comparison used for host KV, with the same explicit file reads
before the four-run matrix. Startup remained outside inference timings.

| Block / state allocation | Prefill, tokens/s | Decode, tokens/s | Request total, seconds | Peak dedicated, MiB | Peak shared, MiB |
|---|---:|---:|---:|---:|---:|
| 1 / KV-only portable state | 1884.869 | 109.165 | 20.664644 | 15205.29 | 9265.38 |
| 2 / preferred state backing | 1893.274 | 109.263 | 20.593572 | 15301.28 | 9265.33 |
| 3 / preferred state backing | 1901.725 | 109.410 | 20.517455 | 15279.35 | 9235.59 |
| 4 / KV-only portable state | 1903.213 | 109.172 | 20.513684 | 15206.79 | 9236.76 |

Matched prefill changed +0.446% and -0.078%, decode +0.090% and +0.217%, and total
time -0.344% and +0.018%. All four complete 2095-character answers matched each
other and the earlier TMA control, with the same 347/1149 accepted drafts, 163
speculative rounds, one fallback and zero accepted n-gram tokens. This establishes
no general inference improvement and does not explain the short-request costs.
There were 10–12 adapter samples per process. Candidate dedicated peaks were 96
and 73 MiB above the controls, leaving 1010–1032 MiB of observed headroom; their
cause is unverified. Shared peaks stayed in the same approximate 9.2 GiB range.
The state preference is retained for the observed startup gains and qualified
transfer behavior, with those timing and resource uncertainties disclosed.
Long inference after both allocation changes, completely filled 262K context,
physical paging, other platforms/GPUs, task quality and full-suite coverage remain
unverified. The optimization goal is paused for a documented handoff.

## Measured settings

Measurements use this exact artifact, one RTX 5060 Ti with 36 SMs and 15.93 GiB
usable VRAM, CUDA 13.1, driver 591.86 and the WSL2 build above. Under load, GPU
clocks were about 2737 MHz with 13801 MHz memory; observed power and temperatures
spanned 168–178 W and 68–71 °C. The desktop remained active. Power and clocks were
not overridden. The final matrices ran serially without a concurrent build.

The earlier comparison below calls the public Engine with `bench/fixtures/bench_corpus.ids`,
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
| DFlash7, optimized head, ordinary prefill | 512 | 2252.77 ± 2.60 | 85.96 ± 0.012 | 1.7171 ± 0.0004 |
| DFlash7, optimized head, ordinary prefill | 8192 | 2083.60 ± 0.72 | 154.85 ± 0.025 | 4.7588 ± 0.0014 |
| DFlash7, optimized head, ordinary prefill | 32768 | 1709.40 ± 0.35 | 241.12 ± 0.155 | 19.7011 ± 0.0035 |
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

In that comparison, `-FastPrefill` selects the inherited quantized prompt kernel. Against ordinary
DFlash7, prefill changes −0.66%, +1.68%, +7.72%, decode −1.73%, −0.012%,
−0.091%, and total time +1.62%, −1.36%, −6.97%. These results kept explicit
fast selection off by default. The automatic profile above now applies it only
to the qualified longer chunks.
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
The profile contains 18 route keys for this card and SM count. The 16 earlier
keys were measured on 2026-10-04; the prompt policy was added on 2026-10-06 and
remeasured with its e4m3 PV candidate on 2026-10-08, when the proposal head's
`t2_i8_small/131072x5120` key was added. The current catalog is
`eaf84b6eea2436ac`. Updating the catalog does not imply that every earlier key
was remeasured. Temporary experimental switches were removed.

To reproduce the earlier ordinary comparison from Ubuntu, disable automatic
fast selection for this command:

```bash
NINFER_PROMPT_FAST=0 build/bench/ninfer_bench \
  --weights models/bonsai2/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --kv-dtype rk4v4 --gdn-state-fp16 --max-ctx 65536 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 0 \
  -pg "512,128;8192,128;32768,128" --warmup 1 -r 3 -o json
```

The final automatic policy was compared with a fresh standard-prefill baseline
under the same conditions and command above. The automatic run omits
`NINFER_PROMPT_FAST=0`; both use the 17-key profile. Mean ± sample standard
deviation across three measured repetitions was:

| Route | Prompt tokens | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|---:|
| Standard | 512 | 2255.58 ± 0.84 | 85.925 ± 0.029 | 1.71734 ± 0.00052 |
| Automatic | 512 | 2261.06 ± 1.32 | 85.955 ± 0.014 | 1.71630 ± 0.00032 |
| Standard | 8192 | 2093.35 ± 1.81 | 154.826 ± 0.045 | 4.74067 ± 0.00357 |
| Automatic | 8192 | 2103.32 ± 1.38 | 154.908 ± 0.036 | 4.72168 ± 0.00262 |
| Standard | 32768 | 1719.98 ± 0.42 | 241.116 ± 0.098 | 19.58316 ± 0.00482 |
| Automatic | 32768 | 1874.46 ± 0.90 | 240.993 ± 0.169 | 18.01362 ± 0.00920 |

At 32K, prefill throughput rose 8.98%, prefill time fell 8.24% and total time
fell 8.01%; decode was 0.051% lower. All three workloads retained identical
draft acceptance and round counts. Both runs use standard attention at 512/8K,
where measured rate differences were below 0.5%.

An Nsight Systems trace of the same automatic 32768/128 Engine workload
(one warmup, one captured repetition, DFlash7, n-gram drafting disabled) found
29 prefill chunks. Runtime-correlated GPU work inside these chunks was active
for 17.418 of the 17.521 seconds from its first to last activity: 0.589% idle,
including gaps between chunks. Of 3.673 seconds in the chunk submission ranges,
3.649 overlapped GPU work. The four main ternary projection families accounted
for 11.733 seconds, about 67% of this GPU work. This distinguishes host submission
time from an exposed CPU bottleneck at this workload; it does not establish the
same attribution for every context length. Trace instrumentation times are
excluded from the throughput comparisons above.

## Paper explanation requests

The serving comparison used the same artifact, one lane, 262144 capacity,
`rk4v4`, FP16 GDN state, DFlash7 and the optimized proposal head. Each request
started a fresh server and asked `Explain this paper.` (`Explain these papers.`
for the combined long input). Temperature was zero, thinking disabled and
presence penalty the server's default 1.5. Prefix caching and default n-gram
drafting were enabled; every request had zero prefix-cache hits. Normal model
stops were enabled. All eight replies reached the 512-token output limit.
Decode rates count the 511 tokens after the prefill token. Each row is one cold
request, without a warmup or an estimate of run-to-run variance.

| Input | Prompt tokens | Route | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---|---:|---:|---:|
| Beale–Kato–Majda | 4177 | Standard | 1763.48 | 144.02 | 5.954 |
| Beale–Kato–Majda | 4177 | Automatic | 1770.86 | 143.79 | 5.950 |
| Deutsch | 17936 | Standard | 1849.62 | 134.96 | 13.530 |
| Deutsch | 17936 | Automatic | 1923.74 | 126.94 | 13.396 |
| Navier–Stokes excerpt | 30018 | Standard | 1713.31 | 108.02 | 22.305 |
| Navier–Stokes excerpt | 30018 | Automatic | 1847.10 | 112.57 | 20.844 |
| Navier–Stokes + Euler | 224759 | Standard | 681.50 | 67.79 | 337.620 |
| Navier–Stokes + Euler | 224759 | Automatic | 929.86 | 61.92 | 250.233 |

Automatic selection reduced total time by 0.059%, 0.991%, 6.55% and 25.88%
at the four respective lengths. Prefill throughput changed +0.418%, +4.01%,
+7.81% and +36.44%; decode changed −0.163%, −5.94%, +4.22% and −8.66%.
The 4K pair uses the same attention route and retained identical draft
acceptance. The 18K and 225K automatic runs needed more DFlash verification
rounds (151 versus 142, and 183 versus 167); their approximate prompt arithmetic
changed the greedy continuation. The lower decode rates remain part of the
result even though total time improved. N-gram drafting accepted three tokens
in each 4K run and none in the other pairs; the 225K runs attempted no n-gram
rounds. These results do not reproduce the much higher n-gram rates of the
earlier synthetic pilots.

Two subsequent automatic runs of the same 224759-token request compared smaller
draft windows under the same settings:

| Drafts | Prefill tokens/s | Decode tokens/s | Total seconds | Accepted drafts | Draft rounds |
|---|---:|---:|---:|---:|---:|
| 7 | 929.86 | 61.92 | 250.233 | 327 | 183 |
| 5 | 929.61 | 69.04 | 249.447 | 346 | 165 |
| 3 | 928.68 | 66.46 | 249.978 | 311 | 199 |

Against seven drafts, five improved decode 11.48% and reduced total time 0.314%;
prefill throughput was 0.027% lower. Three was 3.73% slower at decode than five
and took 0.213% longer overall. Its cheaper attention verification tile did not
compensate for the additional rounds. The seven-draft run accepted only 14
tokens at draft positions six and seven. These are single cold runs with
changed continuations; they do not establish a universal draft-window optimum.
Seven remains the general default because the earlier 8K/32K matrix favored
it. For this long-context workload, launch with
`Start-Bonsai.cmd -Context 262144 -DraftTokens 5`.

The five-draft long request also tested `--ngram-min-match 8` against the default
12, retaining 15 copy drafts. It attempted one copy round and accepted none of
its 15 tokens, reaching 68.91 decode tokens/s and 249.877 s total. Decode was
0.18% lower and total time 0.17% higher than the five-draft default run; prefill
throughput was 0.17% lower. This single cold trial did not justify changing the
n-gram default; the greedy continuation also changed.

In the last 10% of the long prefill, fully prefilling five-second log intervals
had median throughput 460.80 tokens/s with standard attention and 691.20 with
automatic selection. Standard's 11 intervals ranged 230.40–460.80; automatic's
seven ranged 460.80–691.20. Counts advance in completed 1152-token chunks,
so these interval rates also reflect the logging cadence. The slowdown as
history grows persists with the faster implementation.

Inputs came from the [author's typeset Deutsch paper](https://www.daviddeutsch.org.uk/wp-content/deutsch85.pdf),
the [original Beale–Kato–Majda paper](https://projecteuclid.org/download/pdf_1/euclid.cmp/1103941230),
and OpenAI's [Navier–Stokes](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf)
and [Euler](https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf)
PDFs. Plain extraction of Navier–Stokes produced about 156K tokens; its first
30000 text tokens formed the excerpt. Layout extraction plus the distinct Euler
paper formed the 225K stress request. Layout extraction warned about two
rotated sections. The original user-reported 230K request, its UI extraction
and sampling settings are unavailable; these requests establish comparable
local workloads with the simple question. Downloaded papers, extracted text,
generated replies and raw logs remain outside the tracked source tree.

## Prompt attention and evaluated settings

The existing `NINFER_T2_A8_RASTER=rows|cols` override was compared with the public
Linear Op on eight text-layer shapes at T1152, `a8int`, graph replay, a 256 MiB
L2 eviction buffer, three warmups and 15 repetitions. Median microseconds were:

| Output × input rows | Columns first | Rows first | Rows time change |
|---|---:|---:|---:|
| 1024 × 5120 | 129.824 | 129.856 | +0.025% |
| 4096 × 5120 | 394.048 | 396.064 | +0.512% |
| 6144 × 5120 | 572.224 | 572.192 | −0.006% |
| 7168 × 5120 | 660.256 | 664.384 | +0.625% |
| 12288 × 5120 | 1098.528 | 1145.632 | +4.29% |
| 34816 × 5120 | 2889.504 | 6374.432 | +120.61% |
| 5120 × 6144 | 580.416 | 574.240 | −1.06% |
| 5120 × 17408 | 1595.200 | 1566.528 | −1.80% |

Rows first was rejected as a global default. Four further passes on each output
projection alternated the order of the two settings and used 25 repetitions.
For input width 6144, rows first changed median time by −0.35% across the passes;
individual changes ranged −0.57% to −0.006%. For input width 17408, the median
change reversed to +0.73%, with three slower passes and individual changes from
−1.43% to +2.25%. These results did not justify a production route change.
This screen measures the complete public Linear call, including activation
quantization; it does not measure the fused input or residual Ops. No gain was
established in whole inference or at other widths.

A private copy of the existing T2 tile probe compared four warp partitions
against its original 2×2 partition, retaining a 128×64 tile, two stages and
per-G128 activation scales. At T1152, two warmups and nine cold-L2 repetitions
with a 128 MiB eviction buffer, median GEMM microseconds were:

| Shape | 2×2, three CTAs | 4×1, three CTAs | 1×4, three CTAs | 4×2, two CTAs | 2×4, two CTAs |
|---|---:|---:|---:|---:|---:|
| 34816 × 5120 | 3062.8 | 3958.8 | 3318.8 | 3347.5 | 3415.1 |
| 5120 × 17408 | 1474.6 | 1899.6 | 1593.3 | 1659.9 | 1672.2 |
| 12288 × 5120 | 1083.4 | 1370.1 | 1140.7 | 1165.3 | 1194.0 |

All four alternatives lost at every shape, with the worst GEMM slowdown 29.3%,
and were discarded. This probe has a runtime K and a separate quantizer, so its
kernel timings are distinct from public Op results. Its A16 output comparison
was diagnostic and does not qualify a production route against an independent
oracle. The initial standalone link recipe lacked the current NVFP4/GGML
archives and the CUDA driver stub directory; it linked after using those
dependencies. No projection implementation was changed by these experiments.

A later restructuring retained the 4×1 partition in the production tile. Each warp
owns 32 rows across all 64 columns, so each code word is decoded once per tile.
Under 2×2 every word was decoded by two warps. A warp walks the n-tiles in pairs,
interleaving the two partials' integer MMAs before the per-group FP32 scaling, and
each output keeps its group order. The probe above predates the pairing and had a
runtime K; the cause of its 4×1 loss was not isolated. With the same change, each group partial's first MMA reads a zero accumulator
(`mma_s8_zero`) and converts the exact sum with one I2FP; the kernel had started
from the bits of 1.5 × 2^23 and subtracted that as a float. A private probe with
compile-time K found outputs bit-identical to the previous tile and −4 to −13%
GEMM time on the five text-layer shapes. The kernel keeps 168 registers, and its
stack frame fell from 8 to 0 bytes.
The T2 Linear, LinearAdd, fused SwiGLU and attention-input tests pass. In the paper
requests, with fresh servers per run, prefill throughput rose 4.1% at 18K
(2130→2216 tokens/s), 3.5% at 30K (2062→2134) and 1.9% at 225K (1042→1063).
Decode changed by at most 0.14%, and every reply matched.

A further private probe changed the tile aspect ratio with the same two stages,
per-G128 activation scales, T1152 and nine cold-L2 samples. GEMM microseconds were:

| Shape | 128×64, current | 128×32 | 256×32 | 128×96 |
|---|---:|---:|---:|---:|
| 34816 × 5120 | 3066.9 | 3460.1 | 3204.1 | 2962.4 |
| 5120 × 17408 | 1472.5 | 1641.5 | 1599.5 | 1531.9 |
| 12288 × 5120 | 1083.4 | 1173.5 | 1130.5 | 1038.3 |

Both narrower variants lost at every shape. The 128×96 tile improved two shapes
by 3.41%/4.16% and slowed the down projection 4.03% in that probe. A second private
comparison copied the production header with compile-time K and its quantizer;
the wider tile then slowed GEMM by 0.60%, 7.29% and 3.56%, respectively, and was
discarded. It used 255 registers and 40320 bytes shared memory, versus 168
registers and 30976 bytes for the 128×64 kernel in this comparison. A16 output
comparisons remained diagnostic; these discarded tiles were not independently
qualified or installed as production routes.

The existing private cuBLAS probe was rebuilt against the current optimized T2
route and run at T1152 on the same three shapes. Two warmups and nine cold-L2
samples used a 128 MiB eviction buffer. Complete conversion, GEMM and epilogue
times, summed from each stage's median, were:

| Shape | Current integer Op | cuBLAS INT8 | cuBLAS FP8 | cuBLAS BF16 | cuBLAS FP16 |
|---|---:|---:|---:|---:|---:|
| 34816 × 5120 | 2942.8 | 3636.1 | 3822.6 | 9278.3 | 5899.1 |
| 5120 × 17408 | 1620.8 | 1710.9 | 1737.3 | 4635.9 | 3016.5 |
| 12288 × 5120 | 1088.3 | 1331.8 | 1858.8 | 3287.4 | 2125.4 |

Every alternative lost at every shape, so none was added. INT8 GEMM alone
reached 172.6–186.4 TOP/s, but materializing weights and rescaling outputs erased
that advantage. These variants require a dense temporary weight plane; INT8
also needs the full INT32 product before conversion. INT8 and FP8 fold group
weight scales into row scales and quantize activations per token; FP16 uses
FP16 accumulation. Their A16 comparisons are diagnostics rather than independent
qualification or model-quality measurements. The public T2 Linear route still
uses the groupwise integer implementation under `--prefill-cublas`; this private
comparison does not add a new engine option or production path.

A private T2 prefill scheduling trial interleaved two or four independent
integer MMA accumulators before applying the original per-group FP32 scaling
and FMA. It retained the production 128×64 tile, three-block launch bounds,
quantizer and epilogues. All three variants passed the public T2 A8 Linear,
LinearAdd, attention-input projection and GDN-input projection oracle suites.
The baseline and two-product variant also passed width-193/1152 graph checks
at all eight shapes below, with input preservation and an additional activation
case with one sign. The FP64 reference independently materializes packed weights;
the original A8 comparison criteria were retained. Full output is checked for
1024×5120 and both 5120-row output projections; the other shapes sample rows
and compare every query column.

The public complete Linear Op, including quantization, used CUDA Graph replay,
256 MiB L2 eviction, three warmups and 15 repetitions at T=1152. Initial medians
in microseconds were:

| Output × input rows | Baseline | Two products | Four products |
|---|---:|---:|---:|
| 1024 × 5120 | 129.856 | 129.888 | 129.856 |
| 4096 × 5120 | 396.096 | 394.016 | 398.080 |
| 6144 × 5120 | 574.272 | 572.192 | 577.120 |
| 7168 × 5120 | 664.320 | 660.224 | 664.384 |
| 12288 × 5120 | 1102.656 | 1094.432 | 1098.528 |
| 34816 × 5120 | 3053.376 | 2916.128 | 2916.128 |
| 5120 × 6144 | 584.448 | 578.272 | 582.432 |
| 5120 × 17408 | 1597.248 | 1602.624 | 1601.344 |

Two further passes used 31 repetitions, first reversing and then restoring
variant order. The two-product variant tied on 1024×5120 in both passes and
reduced medians on the other seven shapes. Paired changes were −0.53/−1.01%,
−1.04/−0.60%, −3.20/−1.85%, −2.21/−0.93%, −1.52/−0.55%,
−1.04/−0.73% and −1.15/−0.53%, in table order excluding the tied shape.
Its initial 5120×17408 median was 0.34% slower, and that shape's p95 was
1.95–5.82% slower across all three passes. Those costs remain part of the result.
The four-product variant had mixed gains and losses, including a 0.70% loss
on 5120×6144 in the last pass. It was discarded. The inspected two/four-product
store and residual kernels used 168 registers and 30976 shared bytes without
spills; the baseline store kernel had four bytes of spill loads/stores.
The public Engine comparison used the same Bonsai artifact and checked profile,
`rk4v4`, FP16 GDN state, DFlash5 with the optimized head, n-gram drafting disabled,
65536 capacity, the benchmark corpus, 32768 prompt tokens and 1024 measured
decode tokens. Four process blocks ran in baseline/candidate/candidate/baseline
order, each with one warmup and three measured repetitions:

| Block | Prefill tokens/s, mean ± SD | Decode tokens/s, mean ± SD | Total seconds, mean ± SD |
|---|---:|---:|---:|
| Baseline 1 | 1823.414 ± 3.198 | 186.934 ± 0.623 | 23.453142 ± 0.048790 |
| Two products 1 | 1836.458 ± 2.256 | 186.121 ± 1.598 | 23.349663 ± 0.069152 |
| Two products 2 | 1832.757 ± 5.364 | 187.111 ± 0.233 | 23.355926 ± 0.048787 |
| Baseline 2 | 1822.645 ± 0.466 | 187.221 ± 0.338 | 23.451943 ± 0.007787 |

The paired prefill gains were 0.715% and 0.555%, and total times fell 0.441%
and 0.409%. Decode was 0.435% and 0.059% lower; no decode improvement is claimed.
Every repetition accepted 830 of 968 drafted tokens over 194 rounds, without
fallback or n-gram drafts. Baseline total times spanned 23.421913–23.509364 s;
candidate times spanned 23.299342–23.428516 s. The slowest first-block candidate
was slightly slower than two baseline repetitions. Workspace peak remained
183124992 bytes in this comparison. The two-product schedule was retained for
its repeated prefill and total-time gains, with the decode and Op p95 costs
above disclosed. The production engine benchmark, server, calibrator, Op
benchmarks and test executable rebuilt successfully. All four affected production
oracle suites passed in 26.26 s with the checked device profile. An initial private
wrapper used CTest projection aliases as executable program names; those
invocations exited without running tests and were corrected to the dispatcher
names with `--t2-only`. Their failed logs were preserved.

The rebuilt production server completed the normal 30018-token Navier–Stokes
excerpt request, `Explain this paper.`, with DFlash7, the optimized head,
262144 capacity, default prefix caching and 15-token n-gram drafting. It returned
512 tokens at the output limit, with no prefix reuse, in 21.489152 s: 1791.936
prefill and 109.289 decode tokens/s. The earlier single automatic app run had
1847.101/112.573 tokens/s and 20.843825 s total. The new run was 2.987% slower
in prefill, 2.917% slower in decode and 3.096% slower overall. Its complete
response text matched the earlier response exactly, including the same draft
acceptance counts. This adverse result prompted a fresh controlled server
comparison with both schedules in one binary; the older isolated run is retained
as an observation rather than treated as a controlled baseline.

The fresh app comparison used one private server binary with a schedule selector,
the same artifact and automatic profile, and the same request and serving settings
as the production check. Every block started its own server and sent one cold
request, in baseline/candidate/candidate/baseline order:

| Block | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|
| Baseline 1 | 1786.477 | 109.238 | 21.545582 |
| Two products 1 | 1809.679 | 108.961 | 21.342038 |
| Two products 2 | 1806.734 | 109.206 | 21.352642 |
| Baseline 2 | 1792.035 | 107.699 | 21.557183 |

Paired prefill throughput rose 1.299% and 0.820%, and total times fell 0.945%
and 0.949%. Decode changed −0.253% and +1.399%; this screen does not establish
a decode gain. All four complete response texts matched each other and the earlier
paper response. Each computed 30018 prompt tokens without prefix reuse and
returned 512 tokens at the output limit, accepting 347 of 1149 DFlash proposals
over 163 rounds with one fallback and no n-gram acceptance. Every owned server
exited with code zero. The current controlled baseline was also slower than the
older isolated run; the cause of that difference has not been established.
This is two cold observations per schedule on one paper excerpt, with other
papers and long-context app performance unmeasured for this scheduling change.
Task accuracy and performance on other GPUs have not been assessed.

A subsequent T2 prefill trial replaced the weight and activation `cp.async`
transactions with two byte-addressed tensor copies per 128-wide group. Packed
ternary weights and permuted s8 activations retain their representation; scales
keep their existing copy pipeline and each output keeps its FP32 group order.
Both shared activation stages start on 1024-byte boundaries for the 128-byte
swizzle. The sm_120 implementation uses stack descriptors passed by value to the
kernel, with the existing copy tile serving other operand views. It introduces
another physical staging implementation beside that tile, sharing the quantizer
and decoder. Descriptor encoding happens per projection and allocates no device
storage.

The independent byte-copy oracle checked K5120/6144/17408, both shared buffers,
eager execution and two graph replays: all 33,030,144 byte comparisons, guards
and input-preservation checks passed. Current and tensor-copy variants passed
the four public Linear, LinearAdd, attention-input and GDN-input T2 oracle
suites. Both also passed the independent FP64 width-193/1152 fixture described
above, including biased activations and graph guards; all 36 numerical error
metric rows matched. These qualify the represented operations, not paper-answer
accuracy or perplexity.

At T1152, graph replay, cold L2, three warmups and 15 measured calls, complete
public Linear median times in microseconds were:

| Output × input rows | Current two products | Tensor copies | Change |
|---|---:|---:|---:|
| 1024 × 5120 | 129.856 | 123.648 | −4.78% |
| 4096 × 5120 | 394.016 | 373.536 | −5.20% |
| 6144 × 5120 | 572.224 | 537.408 | −6.08% |
| 7168 × 5120 | 658.240 | 617.280 | −6.22% |
| 12288 × 5120 | 1100.544 | 1020.736 | −7.25% |
| 34816 × 5120 | 2916.128 | 2729.792 | −6.39% |
| 5120 × 6144 | 578.368 | 545.600 | −5.67% |
| 5120 × 17408 | 1582.880 | 1492.800 | −5.69% |

Two 31-call repeats reversed and restored the variant order. Every shape's
median improved in both repeats, by 4.05–5.92% and 4.66–6.96%. Tail costs
remained visible: MLP-down p95 rose 2.22% in the initial pass and 3.01% in the
first repeat; 7168×5120 p95 rose 24.19% in the second repeat. The other p95
comparisons improved. The compiler reported 168 registers, an 8-byte stack frame
and four-byte spill loads/stores for the tensor-copy kernel, versus no spills
for the current kernel. Shared storage was 30736 versus 30976 bytes.

Eager cold-L2 checks included per-call descriptor encoding at T193/257/1152 on
1024×5120, 34816×5120 and 5120×17408, with three warmups and 31 calls. Eight of
nine medians improved 2.92–7.25%; 1024×5120 at T257 slowed 0.11%, with equal
p95. The improvements are confined to these measured shapes and widths.

The same public Engine workload as the preceding scheduling comparison then
ran current/tensor-copy/tensor-copy/current, one warmup and three repetitions
per block. This includes descriptor encoding, complete model execution and
decoding. Mean ± standard deviation was:

| Block | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|
| Current 1 | 1829.104 ± 4.391 | 187.170 ± 0.252 | 23.390044 ± 0.044422 |
| Tensor copies 1 | 1917.703 ± 1.827 | 187.218 ± 0.214 | 22.561031 ± 0.022596 |
| Tensor copies 2 | 1916.143 ± 4.599 | 186.967 ± 0.233 | 22.582202 ± 0.040650 |
| Current 2 | 1831.387 ± 5.325 | 187.446 ± 0.267 | 23.359480 ± 0.048217 |

The two pairs improved prefill 4.84%/4.63% and reduced total time 3.54%/3.33%.
Decode changed +0.03%/−0.26%; this does not establish a decode improvement.
Each repetition accepted 830 of 968 proposals in 194 rounds without fallback,
and workspace peak remained 183124992 bytes. All six tensor-copy totals
(22.535437–22.624006 s) were below all six current totals
(23.320395–23.441316 s). The native staging implementation was selected on that
whole-request evidence while retaining the Op tail regressions above.

The same matched fresh-server API method as the preceding scheduling trial
then compared current/tensor-copy/tensor-copy/current on the 30018-token
paper excerpt, asking `Explain this paper.`. Capacity was 262144, with DFlash7,
the optimized proposal head, n-gram15/min12, temperature zero, thinking off and
512 output tokens. The four observations were:

| Block | Prefill tokens/s | Decode tokens/s | Request seconds |
|---|---:|---:|---:|
| Current 1 | 1790.271 | 109.225 | 21.509604 |
| Tensor copies 1 | 1892.334 | 109.177 | 20.598542 |
| Tensor copies 2 | 1889.268 | 109.255 | 20.622633 |
| Current 2 | 1801.160 | 108.776 | 21.419617 |

The two pairs improved prefill 5.70%/4.89% and reduced request time 4.24%/3.72%.
Decode changed −0.044%/+0.440%, so no decode improvement is established here.
All four complete response texts matched. Each request computed the full
30018-token prompt without a prefix hit, stopped at the output limit and
accepted 347 of 1149 proposals in 163 rounds with one fallback; the n-gram
proposal accepted zero of 15 tokens in one round. Startup took 43.0–44.4 s
and is excluded from request time. This remains two cold observations per
schedule on one excerpt. That frozen prototype was not measured on the long
paper request; production measurements follow below. Paper-answer accuracy
remains unmeasured.

The production engine benchmark, server, calibrator, Op benchmarks and test
executable rebuilt successfully after tensor-copy integration. The four public
T2 oracle suites passed with the measured profile and again without a profile.
Production also passed the independent FP64 width-193/1152 graph fixture and
guarded 16-byte-offset weight views, which exercise the existing copy tile for
operands that lack 128-byte alignment. A separate sm_86 compile check passed;
runtime behavior and performance on that GPU remain untested. The comparisons
above use the qualified frozen prototype object.

Review found that admitting the by-value tensor maps on native Windows would
bypass the existing MSVC descriptor ABI workaround. The T2 tensor-copy include,
encoder and launch now exclude `NINFER_TMA_STAGED_DESCRIPTORS`; those builds
retain the copy tile. Compiling the sm_120a translation unit with that staged
descriptor flag passed on WSL, and review confirmed the source guard. Native
MSVC compilation and Windows TMA throughput remain unverified; the new T2 TMA
path is selected only in builds that support its by-value descriptors.

A fresh baseline/production/production/baseline comparison then used the
rebuilt public Engine benchmark, with the same 32768-token prompt, 1024 measured
decode tokens, capacity 65536, r4 KV, FP16 GDN, DFlash5, optimized proposal head,
n-gram drafting off, measured profile, one warmup and three repetitions per
block. The baseline used the preserved two-product copy implementation. Mean
± standard deviation was:

| Block | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|
| Baseline 1 | 1832.816 ± 4.050 | 187.401 ± 0.330 | 23.346777 ± 0.047809 |
| Production 1 | 1915.054 ± 3.471 | 187.329 ± 0.257 | 22.581252 ± 0.025026 |
| Production 2 | 1914.523 ± 8.145 | 186.127 ± 1.440 | 22.621693 ± 0.104639 |
| Baseline 2 | 1833.718 ± 3.280 | 187.212 ± 0.343 | 23.343924 ± 0.041882 |

The two pairs improved prefill 4.49%/4.41% and reduced total time 3.28%/3.09%.
Decode fell 0.038%/0.580%; the second production block had greater variation,
and the cause of that decode slowdown is unresolved. Every production total
(22.536901–22.738635 s) was below every baseline total
(23.300236–23.395761 s). All twelve repetitions accepted 830 of 968 proposals
in 194 rounds without fallback, with the same prefill signature and
183124992-byte workspace peak. This confirms the production improvement for
the measured workload while retaining the kernel's spill and Op tail costs.
That Engine comparison does not establish a decode improvement, performance
outside the measured workload, paper-answer accuracy or Windows device residency.

The rebuilt production server also completed the same matched fresh-server
paper comparison, with the 30018-token prompt, `Explain this paper.`, 512
outputs, capacity 262144 and the DFlash7/n-gram15 settings above:

| Block | Prefill tokens/s | Decode tokens/s | Request seconds |
|---|---:|---:|---:|
| Baseline 1 | 1805.122 | 107.613 | 21.434896 |
| Production 1 | 1890.156 | 108.823 | 20.635102 |
| Production 2 | 1885.541 | 107.850 | 20.715312 |
| Baseline 2 | 1812.367 | 108.864 | 21.316717 |

The two pairs improved prefill 4.71%/4.04% and reduced request time 3.73%/2.82%.
Decode changed +1.124%/−0.932%, including a slowdown in the reverse pair;
these observations do not establish a decode improvement. All four full
response texts matched, with no prefix reuse, 347/1149 accepted proposals,
163 rounds and one fallback. The decode rate counts 511 tokens because the
first output belongs to prefill. Startup was 43.04–43.09 s and is excluded
from request time. Both production request times were below both baseline
times. All owned servers exited successfully. This is two cold observations
per implementation on one paper excerpt; other papers and task accuracy remain
unverified.

The production server then completed a fresh baseline/production/production/
baseline comparison on the 224759-token Navier–Stokes + Euler request,
`Explain these papers.`, 512 outputs and capacity 262144. Both implementations
used the same automatic staged-V profile, `rk4v4`, FP16 GDN, DFlash7, optimized
proposal head, default n-gram15/minimum12, temperature zero, thinking off and
default presence penalty 1.5. Windows adapter counters were sampled throughout
each owned run, approximately every three seconds. Compilation, indexing and
other inference measurements did not overlap these requests.

| Block | Prefill tokens/s | Decode tokens/s | Request seconds |
|---|---:|---:|---:|
| Baseline 1 | 905.433 | 60.074 | 257.023517 |
| Production 1 | 922.911 | 59.950 | 252.335238 |
| Production 2 | 922.146 | 59.973 | 252.534309 |
| Baseline 2 | 904.767 | 59.935 | 257.217556 |

The two pairs improved prefill 1.930%/1.921% and reduced request time
1.824%/1.821%. Both production totals were below both baseline totals. Decode
changed −0.205%/+0.064%; these observations do not establish a decode gain.
All four complete 2301-character response texts matched exactly. Each computed
the entire prompt with zero prefix reuse, accepted 327/1270 drafts in 183 rounds,
used one fallback and attempted no n-gram rounds. The decode rate counts 511
tokens. Fresh-server startup was 44.04–44.07 seconds and is excluded from request
time. Every owned server and the telemetry helper exited successfully.

All four runs had seven fully prefilling log intervals in the last 10% of the
prompt, each with median 691.2 tokens/s and range 460.8–691.2. These intervals
count completed chunks and reflect logging cadence; the long-context slowdown
persists, and this logging resolution did not distinguish a late-prefill gain.

| Block | Peak dedicated MiB | Peak shared MiB | Dedicated headroom MiB |
|---|---:|---:|---:|
| Baseline 1 | 15138.9 | 9264.0 | 1172.1 |
| Production 1 | 15132.3 | 9249.2 | 1178.7 |
| Production 2 | 15130.4 | 9249.3 | 1180.6 |
| Baseline 2 | 15132.1 | 9248.6 | 1178.9 |

Headroom uses the reported 16311 MiB physical card capacity. Startup explicitly
pinned 8 GiB of host KV and 918.6 MiB of host states, accounting for most shared
usage, and reported all 4096 KV pages reserved for the 262144-token capacity.
That reservation completed this 224759-token request plus its response with
roughly 1170–1180 MiB of measured dedicated headroom. Adapter counters do not
identify Windows driver paging events or prove physical residency of every
allocation. A request filling the entire 262144-token capacity, other papers,
PPL and task accuracy remain unverified. These are two cold observations per
implementation on one long input, without a variance estimate; they extend
the production performance evidence at this scope.

A private follow-up reduced tensor-copy completion polling to the producer
thread or one lane per warp, retaining the CTA barrier before buffer reuse.
Both variants passed exact byte-copy checks, the four public T2 suites and the
independent width-193/1152 graph oracle. Each candidate's 36 extended numerical
error-stat rows matched production; that does not establish exact full-output
equivalence. Warp-leader polling removed the stack spills in all nine compiler
specializations, while producer-only polling retained them. Both still used
168 registers and 30736 shared bytes.

Complete Linear graph replay at T1152, cold L2, three warmups and 31 measured
calls gave these changes against the current tensor-copy implementation:

| Output × input rows | Producer median | Producer p95 | Warp leader median | Warp leader p95 |
|---|---:|---:|---:|---:|
| 1024 × 5120 | +1.50% | +1.54% | +3.15% | +0.92% |
| 4096 × 5120 | +0.55% | +0.89% | +1.10% | +1.42% |
| 6144 × 5120 | −0.01% | +1.00% | +1.10% | +1.75% |
| 7168 × 5120 | 0.00% | +0.08% | +1.12% | +0.85% |
| 12288 × 5120 | −2.99% | −0.32% | −1.79% | +18.86% |
| 34816 × 5120 | −0.18% | +0.34% | +1.09% | −1.52% |
| 5120 × 6144 | +0.75% | +0.90% | +2.28% | +3.02% |
| 5120 × 17408 | +0.15% | −3.90% | +0.98% | −5.03% |

The 12288×5120 exception was repeated with reversed and restored order.
Producer-only median changed −0.003% and −2.09%; warp-leader median improved
1.40% and 1.08%, with p95 costs of 0.99% and 0.71%. Most shapes regress in
the general replacement, the producer-only exception depends on the observed
order, and the warp-leader exception retains a tail cost. Both variants remain
private; no shape-specific route or whole-inference improvement was established.
Removing compiler spills alone did not justify selecting either implementation.

A second private follow-up used three tensor-copy buffers and a two-block launch
bound, against the current two-buffer, three-block bound. It preserved the
quantizer, decoder, per-group arithmetic, scale rings and epilogues. Both buffer
counts passed 33,030,144 independent byte comparisons, including phased buffer
reuse and two graph replays. The candidate passed all four public T2 suites and
the independent width-193/1152 graph oracle; its 36 numerical error-stat rows
matched production. Compiler resources increased from 168 to 186 registers and
30736 to 44064 shared bytes, while the eight-byte stack spills disappeared.
Actual resident-block occupancy was not measured.

Complete Linear graph replay at T1152, cold L2, three warmups and 31 measured
calls produced the following changes against two buffers:

| Output × input rows | Three-buffer median | Three-buffer p95 |
|---|---:|---:|
| 1024 × 5120 | −1.60% | 0.00% |
| 4096 × 5120 | +3.82% | +3.49% |
| 6144 × 5120 | +3.43% | +3.14% |
| 7168 × 5120 | +3.96% | +3.60% |
| 12288 × 5120 | +5.12% | +4.73% |
| 34816 × 5120 | +7.29% | +8.10% |
| 5120 × 6144 | +2.66% | +2.25% |
| 5120 × 17408 | +2.50% | +2.26% |

The 1024×5120 exception was checked at three token widths with reversed and
restored implementation order, using the same graph/cold-L2 method:

| Tokens | First median / p95 | Second median / p95 |
|---|---:|---:|
| 193 | +5.26% / +0.08% | +5.35% / +0.16% |
| 257 | −3.71% / −3.52% | −3.65% / −3.58% |
| 1152 | −1.66% / −1.55% | −0.96% / −0.59% |

The general replacement regressed on seven shapes, and the smaller projection's
benefit depends on width. Three buffers remain private; fused projection timing
and whole-inference performance for a scoped route are unverified. Two local
shell helpers reported failure after their compile/link and numerical commands
had completed: a trailing carriage return and editing an executing helper caused
the respective wrapper errors. Their completed logs and numerical results were
checked independently, and the helpers were repaired; the wrapper exit codes
are not reported as successful checks.

A separate CUDA resource query for the width-5120 Store epilogue calculated a
maximum of three resident blocks/twelve warps for two buffers and two blocks/eight
warps for three buffers. The card reports 102400 shared bytes per SM and 1024
reserved shared bytes per block. These are calculated resource limits, not
observations of active occupancy during a workload. The diagnostic's first
compile used an incorrect namespace for the existing epilogue; correcting that
local source resolved the compile failure.

A private 64×64 tile then halved the row extent and thread count, retaining the
two buffers, activation quantizer, scale rings, decoder and arithmetic order.
Its four-block launch bound did not result in four calculated resident blocks:
24592 static shared bytes plus the 1024-byte reserve limited it to three blocks,
or six warps, versus twelve warps for the current 128×64 tile. It used 184
registers without spills. All 55,050,240 independent byte comparisons, the four
public T2 suites and the width-193/1152 FP64 graph oracle passed. All 36 extended
error-stat rows matched production after removing diagnostic stderr markers
that interrupted two stdout lines; the initial log-comparison assertion failed
on that interleaving and the repaired comparison passed.

Complete Linear graph replay at T1152, cold L2, three warmups and 31 measured
calls gave these costs against the current tile:

| Output × input rows | 64-row median | 64-row p95 |
|---|---:|---:|
| 1024 × 5120 | +13.25% | +13.27% |
| 4096 × 5120 | +15.91% | +15.82% |
| 6144 × 5120 | +16.37% | +15.97% |
| 7168 × 5120 | +16.86% | +16.63% |
| 12288 × 5120 | +16.73% | +17.62% |
| 34816 × 5120 | +24.97% | +19.89% |
| 5120 × 6144 | +16.02% | +15.92% |
| 5120 × 17408 | +16.74% | +12.74% |

Every shape regressed. The 64-row tile was rejected without a whole-inference
trial; its finer work distribution did not produce an Op improvement.

A private 128×32 tile next halved the column extent while retaining four warps
per block, two buffers and the same quantizer, decoder, per-group arithmetic and
epilogues. It used 106 registers, 22544 shared bytes and no stack spills. CUDA's
resource calculation allowed four blocks/sixteen warps per SM, versus three
blocks/twelve warps for the current tile. All 44,040,192 independent byte
comparisons, four public T2 suites and the width-193/1152 FP64 graph oracle passed;
all 36 extended error-stat rows matched production.

Complete Linear graph replay at T1152, cold L2, three warmups and 31 measured
calls produced these changes against the current tile:

| Output × input rows | 32-column median | 32-column p95 |
|---|---:|---:|
| 1024 × 5120 | +4.97% | +6.58% |
| 4096 × 5120 | +10.62% | +10.56% |
| 6144 × 5120 | +11.39% | +12.49% |
| 7168 × 5120 | +11.89% | +11.26% |
| 12288 × 5120 | +9.65% | −3.88% |
| 34816 × 5120 | +11.81% | +10.83% |
| 5120 × 6144 | +11.28% | +10.51% |
| 5120 × 17408 | +11.93% | +11.26% |

Every median regressed, despite the lower 12288×5120 p95. The narrower tile was
rejected without a whole-inference trial. Higher calculated occupancy and
removing spills did not establish a speed improvement for either smaller tile.

The long-context prompt experiment compared the public attention Op at physical
width 1152, Q24/KV4/D256, `rk4v4`, batch one, append entry, CUDA Graph replay,
cold L2, three warmups and 15 measured repetitions. GPU work ran serially after
the build. Median time in microseconds was:

| Prior KV tokens | Standard | Legacy fast | Staged V | Staged change against legacy |
|---|---:|---:|---:|---:|
| 32768 | 21637.920 | 15032.128 | 13891.360 | −7.59% |
| 131072 | 88735.519 | 64579.391 | 55020.321 | −14.80% |
| 229376 | 155244.934 | 112637.756 | 99048.256 | −12.07% |

These are Op results. A 32-key, four-warp alternative passed its numerical
checks but took 17099.552/74140.419/128703.262 microseconds at those histories,
13.4–15.0% slower than its earlier legacy baseline
(15083.296/64462.624/112493.858 microseconds), and was removed. Standard attention with
FP16 PV reached 17973.057/73029.411/128663.803; packing GQA yielded
21545.759/88735.550/155481.369. Both lost to staged V. Staged V costs 99328
dynamic shared bytes and 1056 static bytes per CTA, with 255 registers and a
16-byte stack frame in the measured build; legacy packed fast uses 84992
dynamic and 1024 static bytes.

A later private staged-V trial used 32 keys and four warps, with offsets inside
the physical 64-token KV pages. The runtime occupancy query reported two
resident blocks per SM, 255 registers per thread, 49664 dynamic shared bytes
and 16 static shared bytes. Both this route and its 32-key legacy control passed
ten append/cached attention cases against the independent FP64 oracle, including
fragmented pages, graph replay, 65/129-token tails, sampled queries through 8193
visible keys and V scale 512. The performance workload repeated the width-1152,
graph, cold-cache, three-warmup, 15-repetition comparison above:

| Prior KV tokens | Current staged 64 | Private staged 32 | Private legacy 32 |
|---|---:|---:|---:|
| 32768 | 14475.968 | 19097.376 | 17620.800 |
| 131072 | 56592.159 | 74549.026 | 74974.335 |
| 229376 | 101128.990 | 131077.179 | 131588.669 |

Times are microseconds. Staged 32 lost 29.61–31.92% against the concurrently
measured current route, and legacy 32 lost 21.72–32.48%. Both were discarded;
neither was integrated or tested through the whole Engine. The initial numerical
screen did not qualify width 1152 or the long benchmark histories. An incorrect
benchmark flag caused the first timing invocation to exit before measuring;
the corrected invocation completed all nine cases.

An Engine pilot used the extracted OpenAI Navier–Stokes paper with a
detailed explanation question: 156550 rendered prompt tokens, 512 measured
decode tokens, 262144 capacity, DFlash7, optimized head, graphs and n-gram
drafting disabled. Each row is one cold repetition without warmup, with model
stop defaults disabled by the benchmark:

| Prompt implementation | Prefill tokens/s | Decode tokens/s | Total seconds |
|---|---:|---:|---:|
| Standard | 871.14 | 108.807 | 184.417 |
| Legacy fast | 1080.41 | 110.399 | 149.542 |
| Staged V prototype | 1131.79 | 110.332 | 142.966 |

Against legacy fast, staged V reduced prefill time 4.54% and total time 4.40%,
with decode 0.0603% lower. Against standard, prefill time fell 23.03% and total
time fell 22.48%. This pilot with one repetition predates the final automatic
policy and does not reproduce a 230K prompt or the app's normal stopping behavior.

At width 129, staged V lost to standard at all measured histories: +65.21% at
an empty prefix, +14.51% at 4096, +12.53% at 8192 and +8.31% at 229376.
Width 256 also lost +11.98% at an empty prefix, while 256/512/1152 won at the
long histories. The automatic policy therefore preserves short prompts and
narrow tails. An initial staged prototype used dimension indices where packed
byte/group indices were required; the independent oracle rejected it. Corrected
indices were qualified before any reported timings.

For narrow long-context verification, ten existing schedules were compared at
width 8 and the same three histories, cold L2 and graph replay, with three
warmups and 25 repetitions. The existing `6x2x32q` remained fastest at
197.408/713.536/1258.300 microseconds. The 229K alternatives spanned
1399.650–2354.980 microseconds, so the decode schedule was retained.

Private cached-entry prototypes then targeted width 6, which DFlash5 verifies,
with the same Q24/KV4/D256 geometry, `rk4v4`, batch one, FP32 PV accumulation,
graph replay and cold 256 MiB L2 flush. Each entry is the public Op's median of
25 repetitions after three warmups. The first screen doubled the key tile
to 64 with twelve warps and one resident CTA; early fetching was also tested:

| Visible keys | Current 6×2×32q | 12×1×64q | 12×1×64qe |
| ---: | ---: | ---: | ---: |
| 32768 | 197.664 µs | 205.632 µs | 211.712 µs |
| 131072 | 740.160 µs | 801.568 µs | 817.952 µs |
| 229376 | 1301.312 µs | 1393.472 µs | 1435.456 µs |

Every alternative lost, by approximately 4–11%. Another screen kept six warps
and raised the register allowance by requiring only one resident CTA. The
32-key unmasked kernel eliminated its 76-byte spill stores/loads and used
242 registers, but both one-CTA alternatives were 34–42% slower:

| Visible keys | Current 6×2×32q | 6×1×32q | 6×1×64q |
| ---: | ---: | ---: | ---: |
| 32768 | 199.296 µs | 268.896 µs | 267.040 µs |
| 131072 | 744.224 µs | 1059.648 µs | 1051.456 µs |
| 229376 | 1303.296 µs | 1851.168 µs | 1832.736 µs |

The 64-key six-warp kernel still had 16-byte spill stores/loads at 255 registers.
Workspace query and execution agreed in both screens. These rejected candidates
were not qualified against an independent mathematical oracle or integrated into
the Engine; the timings establish their Op costs, with no claim of model quality
or whole-inference performance. The measured production schedule remains selected.

Shared Q-scale and packed-K staging candidates preserved the six-warp,
two-CTA, 32-key schedule and FP32 PV accumulation. Q scales used 768 additional
shared bytes per CTA; packed K reused retired V codes after dequantization, with
additional synchronization. Six independent FP64 public cached-Op cases passed
for each of four variants, covering 32/65/129/2049/8192 visible keys, fragmented
or offset pages, graph replay, a 512× V scale, cache preservation and guards.
For the unmasked kernel, spill stores/loads fell from 76 bytes to 12 with shared
Q, 64 with shared K and 24 with both, at the same 168-register limit. All were
slower in the same public-Op timing conditions:

| Visible keys | Current | Shared Q scales | Shared packed K | Both |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 199.424 µs | 201.504 µs | 215.872 µs | 228.160 µs |
| 131072 | 744.224 µs | 753.696 µs | 811.776 µs | 860.960 µs |
| 229376 | 1303.328 µs | 1321.056 µs | 1431.296 µs | 1518.656 µs |

The costs were 1.04–1.36%, 8.25–9.82% and 14.41–16.52%, respectively.
Fewer spills did not establish a gain, so these candidates were discarded.
The private checks qualify their represented short outputs; they do not qualify
masked batches, every long output, or a model task. No production memory or
execution change resulted from these screens.

A twelve-warp screen kept 32 keys per tile and six QK producer warps, allowing
six V-loader warps to run alongside them. Two-CTA launch bounds doubled the
scheduled warps but constrained registers to 80 per thread: ordinary fetching
spilled 196/192 bytes of stores/loads, and early fetching spilled 268/252 bytes.
The one-CTA control used 168 registers without spills. All four schedules passed
the same six full-output FP64 cached-Op cases before timing:

| Visible keys | Current 6×2×32q | 12×1×32q | 12×2×32q | 12×2×32qe |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 199.424 µs | 226.080 µs | 244.512 µs | 258.112 µs |
| 131072 | 744.224 µs | 889.600 µs | 916.288 µs | 963.328 µs |
| 229376 | 1303.360 µs | 1556.288 µs | 1603.328 µs | 1668.928 µs |

Every alternative lost, by 13–29%. These schedules were discarded; increased
occupancy and dequantization overlap did not establish a performance gain.

A separate exact Lloyd4 decoder candidate replaced the negative-table byte
permutation with bytewise complement and addition. All 65,536 packed four-nibble
inputs matched the independent signed-code oracle exactly. The same six FP64
cached-Op cases passed, with unchanged register and shared-memory counts in the
private width-6 kernel. Four alternating-order timing passes, with three warmups
and 31 repetitions per entry, changed median Op time by −1.03% to 0.00% at 32K,
−0.83% to −0.46% at 131K, and −0.41% to −0.16% at 229K. The initial 32K gain
mostly disappeared on repetition. Both decoder variants subsequently passed
16 public append/cached cases, including sampled 229K outputs, fragmented pages,
graphs, large V scales, ragged batches, gates and cache/guard checks. Long cases
compare first/last queries, one head per KV group, every output dimension and
every visible key; remaining outputs receive finite and guard checks.
Fused append at width 6 measured 199.456/746.272/1307.424 µs for the existing
decoder and 199.456/742.176/1297.984 µs for the candidate at 32K/131K/229K,
with three warmups, 31 repetitions, graph replay and cold L2: a tie, −0.55%
and −0.72%, respectively. Register/shared counts and public workspace usage
agreed in these measured specializations.

The first whole-Engine comparison used Bonsai2, this checked profile, `rk4v4`,
FP16 GDN state, DFlash5 with its optimized head, no n-gram proposals, 65536
capacity, the benchmark corpus, and 32768 prompt plus 1024 measured generated
tokens. One warmup and three repetitions produced decode 186.254±0.831 versus
186.896±4.298 tokens/s, a 0.34% mean increase. Prefill changed by +1.06% and
total time by −0.87%, although the private override affects only width-6 append
attention. Every repetition had 830 accepted of 968 drafted tokens in 194
rounds. The candidate's slowest decode took 5.617803 s, slower than all three
baseline repetitions (5.480574–5.526053 s). This comparison does not establish
a whole-inference gain. Reversing the block order produced candidate decode
187.041±0.167 versus baseline 187.427±0.203 tokens/s: the candidate lost
0.21%. Prefill changed −0.005%, and total time rose 0.052% (23.439010 versus
23.426768 s). Speculative counts remained identical. Across both blocks the
mean decode difference was only +0.069%, with a larger spread for the candidate;
the adverse repetition and reverse-order loss remain part of the result.
The Op improvement did not establish an application improvement, so this
candidate was discarded. No decoder change was integrated into production.
Nsight Compute could not access GPU counters (`ERR_NVGPUCTRPERM`); its
instrumented timing was excluded from the comparisons.

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

### Warp-specialized prompt attention

Microbenchmarks on this card set the ceilings used below: `mma.sync` sustained
203 TOP/s for s8 m16n8k32, 103 for f16 with FP16 accumulation, 51 for f16 with
FP32 accumulation and 205 for e4m3 with FP16 accumulation; streaming DRAM reads
reached 385–413 GB/s and L2-resident reads about 1.2–1.4 TB/s. `setmaxnreg`
compiles and runs for an explicit `compute_120a`/`sm_120a` target; a generic
`compute_120` PTX target rejects it.

Timing-only ablations of the staged-V kernel at width 1152 (removing one phase,
results invalid by design) showed it bound neither by DRAM nor by Tensor Cores:
removing next-tile loads changed 32K/131K time by −0.6%/+0.4%, QK MMA by
−7.6%/−7.1%, PV MMA by −11.4%/−10.1%, V decode by −11.7%/−10.2%, key expansion by
−11.9%/−9.1%, and both MMAs by −24.2%/−23.0%. Its loop issues about 2,500
instructions per warp and 64-key tile at the 255-register limit with two warps
per SM sub-partition, and every warp passes through the same serial phases.

The production `staged` launch now runs `prompt_i8_ws.cuh`: twelve warps per CTA.
Two consumer warpgroups (eight warps, `setmaxnreg` 224) keep the staged kernel's
per-row arithmetic. One producer warpgroup (56 registers) loads packed keys and
values, expands keys two tiles ahead into the two key stages and decodes the next
value tile while the consumers run QK and softmax. Named barriers replace the CTA
barriers. Shared memory was 99,328 dynamic bytes (100,352 with the FP32 key scales
below). Each output performs the same
operations in the same order, so a private comparison found zero differing BF16
values out of 7,077,888 at 1152, 8192 and 32768 visible keys; the staged-V kernel
was then removed. Its eight-warp fast-kernel fallback remains for builds without
`NINFER_SM120_FP8`.

Public cached Op, width 1152, graph replay, cold L2, two warmups and seven
repetitions, three alternating passes (medians in microseconds):

| Visible keys | Staged V | Warp-specialized | Change |
|---:|---:|---:|---:|
| 8192 | 3944 | 3548 | −10.0% |
| 32768 | 14951 | 13261 | −11.3% |
| 131072 | 58650 | 56673 | −3.4% |
| 229376 | 103489 | 99656 | −3.7% |

Against standard attention, the warp-specialized kernel was 39–47% faster at width
1152 from an empty prefix to 8192 keys, 28–47% faster at width 512, tied at width
256 with an empty prefix and 22–24% faster at 4K/8K, and slower at width 129 with
an empty prefix (88.1 versus 54.0 µs), tying at 4K/8K. The profile therefore routed
every chunk of at least 256 tokens to it at all visible histories
(`attn_prompt/h24/rk4v4`: `[[262144, "staged"]]`); narrower tails keep standard.
[Prompt attention: e4m3 PV](#prompt-attention-e4m3-pv) describes the current bands.

Whole-app paper requests used the harness of
[Paper explanation requests](#paper-explanation-requests) with fresh servers,
alternating the previous production server and the warp-specialized one under the
previous 8K profile boundary. Every reply was identical, with identical acceptance:

| Input | Prefill time change | Total time change |
|---|---:|---:|
| Deutsch, 17936 tokens | −1.55% | −1.03% |
| Navier–Stokes, 30018 tokens (two pairs) | −2.84%, −1.99% | −2.15%, −1.54% |
| Navier–Stokes + Euler, 224759 tokens | −7.86% (256.96→236.78 s) | −7.56% (266.23→246.10 s) |

Decode changed −0.07% to +0.06% at 18K/30K and −0.74% at 225K. Extending the
route below 8K changes the arithmetic of short histories, so its replies differ.
With the same server, Deutsch prefill rose 2.5% (1879→1926 tokens/s, two pairs)
and total time fell 1.6%; the 30K prefill rose 0.4–1.0%, while its changed
continuation (1895 versus 2095 characters) raised decode about 10%, which is not a
kernel speedup. The Engine benchmark (DFlash7, no n-gram, three repetitions,
alternating) measured prefill +1.1% at a 2048-token prompt and +3.7% at 8192;
8K decode rose 3.3% with changed acceptance. At 224759 tokens (four fresh servers,
order all-history, 8K, 8K, all-history) prefill took 226.23/236.62 s against
227.63/236.11 s and total time 234.60/245.56 s against 236.79/245.27 s: −0.2% and
−0.4% on the means. Both pairs ran about 4% slower than the first pair, which bounds
the comparison's resolution. Decode measured 63.13/59.27 against 57.66/57.62
tokens/s with a changed continuation (331 versus 327 accepted drafts, 180 versus
183 rounds). Changed replies remained coherent explanations of the same papers;
task quality was not measured.

Rejected in the same investigation, each timed against a concurrent control:

| Candidate | Result |
|---|---|
| Interleaving key expansion into the staged PV loop | +5–7% slower at 32K–229K |
| L2 prefetch of later tiles from the producer (distances 3/5/8) | no measurable change |
| Conflict-free value swizzle (exact) | within noise; more spills |
| Value tile split into dimension halves with four barriers | +1–9% slower than warp-specialized |
| Hoisted P fragments with x4 V loads in the small-T verify kernel | W4 −3–4%, W8 +5.6–7.3%; ptxas already deduplicated P |
| FP16 PV accumulation in verify (`NINFER_SMALLT_PV_F16=1`) | W8 −3.3–4.0%, W6 −0.4–0.7%; changes numerics, not adopted |
| Warp-specialized T2 prefill GEMM, 256×64, producer warpgroup | +4–15% slower on four shapes, tie on 1024×5120; two consumer warps per sub-partition |
| One 12-warp T2 CTA, 256×96, three- or four-stage TMA ring | +3–23% slower than three 4-warp CTAs |
| Prefill chunk 2304 | 30K prefill −3.3%, total +2.0%, changed reply |

A T2 ablation explained these results: removing integer MMA saved 28–37%,
per-group FP32 scaling 8–16% and ternary decode 2–12%, while loads alone took
about half of the kernel time at roughly L2 bandwidth. Latency within each warp,
not L2 traffic, limited the alternatives with fewer resident warps. At 32K
decode, the T2 GEMV ran at about 400 GB/s in steady state; its mean was about 10%
slower because of external outliers (p99 3.5× the median). Verify attention at
229K (W8, 1.25 ms per layer) remains the main long-context decode cost; its
ablations showed DRAM loads −3.0%, QK MMA −4.6%, PV MMA −11.8%, V dequantization
−9.3% and key expansion −11.4%.

### Prompt exponentials, warpgroup barriers and the fused rotated FFN

Three later changes are retained on top of the warp-specialized kernel. The first two were
judged by the paper requests with a fresh server per run in B/F/U/U/F/B order;
replies were identical and decode changed by at most 0.1%.

- **FTZ exponentials and FP32 key scales.** The consumers evaluate softmax with
  `ex2.approx.ftz.f32` (`exp2_approx_ftz`), and the producer widens each key
  stage's G64 scales once into an FP32 `[group][key]` block (+1024 shared bytes),
  so a key pair's scales are one load. Only results below 2^-126 can differ, and
  a private comparison found zero differing BF16 outputs at 1152 and 32768 keys.
  The consumer loop fell from 2247 to 2074 SASS instructions per tile. The Op at
  width 1152 measured −2.9% at 229K keys (96.1→93.3 ms) and −2.7% at 32K
  (13.96→13.58 ms). The rk4v4 and fast-prompt FP64 oracle suites pass.
- **Fused rotated FFN.** `swiglu_hadamard_linear_add` runs the Dense FFN of a
  rotated checkpoint after the input rotation. Where both projections are T2 on
  the integer prefill tile, the gate/up GEMM stages each gate row block with its
  up partners and writes BF16 SwiGLU directly. This drops the BF16 `[2I, T]`
  plane, 80 MB at T1152. `t2_prefill_hadamard_quantize_kernel` then applies the
  1024-point transform and writes the down projection's INT8 codes and scales,
  without a BF16 transformed tensor. Narrower calls compose `linear`,
  `silu_mul_hadamard` and `linear_add`. The fused path rounds every intermediate
  to BF16 exactly as the composition does. Its test compares outputs word for
  word at T 1–1152 on the real FFN shapes. Where the fused path runs, it also
  compares the paired tile against `silu_mul` and the quantizer's codes and
  scales, padding included, byte for byte. The unpaired tile kernels kept 168
  registers and their shared memory; the paired one also uses 168.

Prefill seconds, mean of two runs per server:

| Input | Previous | FTZ exponentials | + first fused FFN |
|---|---:|---:|---:|
| Deutsch, 17936 tokens | 8.694 | 8.634 (−0.69%) | 8.451 (−2.80%) |
| Navier–Stokes, 30018 tokens | 15.073 | 14.983 (−0.60%) | 14.700 (−2.47%) |
| Navier–Stokes + Euler, 224759 tokens | 226.33 | 220.34 (−2.65%) | 218.19 (−3.60%) |

The first fused form still wrote the gate/up plane and fused only SwiGLU, the
transform and quantization. The current form replaced it. In a later session
(three runs per server at 18K/30K, two at 225K) it measured −0.78%, −0.83% and
−0.22% prefill against the first form. In that session the same first-form binary
ran 6.1–8.5% slower in prefill and 8.8–9.4% slower in decode than in the table
above, and host KV pinning took
4.0 instead of 2.2 s. GPU clocks, power limits and Windows load were unchanged;
the cause is unresolved, so compare only within a session.

**Per-warpgroup barriers.** Each consumer warpgroup now waits on its own key-stage
and value-tile full barriers. Previously all 384 threads met at every barrier, so
both warpgroups started each phase together. Each iteration runs the previous
tile's PV, then this tile's QK and softmax, and releases the key stage before the
softmax. The arithmetic is unchanged: a private comparison found zero differing
outputs at 0, 1000, 8192 and 32768 keys, and the rk4v4 and fast-prompt oracle
suites pass. Private Op timings at width 1152 (two alternating passes, medians):

| Variant | 32K keys, µs | 229K keys, µs |
|---|---:|---:|
| Previous (all threads per barrier) | 12626 / 12385 | 94679 / 94226 |
| Per-warpgroup barriers, rotated loop (retained) | 11907 / 12281 | 91351 / 90590 |
| Per-warpgroup barriers, previous order | 12290 / 12099 | 93675 / 92254 |
| Same, key stage released after the softmax | 12317 / 12315 | 92381 / 91581 |
| Enforced pingpong turns between warpgroups | 15998 / 15617 | 111620 / 110783 |
| Rotated loop with a 40/232 producer/consumer register split | 21686 / 21581 | 152490 / 150201 |

Enforced turns serialize each warpgroup's mixed MMA and scaling phase. The
40-register producers spill 572 bytes, and their delay stalls the consumers;
both were rejected. In the paper requests (fresh servers, three runs per server at
18K/30K and two at 225K, alternating), prefill fell 0.59% (8.482→8.432 s), 0.19%
(14.677→14.649 s) and 1.03% (218.40→216.16 s). Decode was within 0.3% and every
reply matched.

### Decode: GDN replay, exact-order gating and value decode

Three exact decode changes are retained; every paper reply below matched the baseline
byte for byte.

- **GDN record and fold.** Both replay kernels stage their window in shared memory
  and run the recurrence with four state rows per lane. The rows share every staged
  key and query load and every reduction step. Each row dot product sums its column
  chunks in the lane order of `apply_gdn_transition` and finishes with that
  recurrence's butterfly, so each result is bit-identical to it. The replay record,
  fold (including the 394 s wide case), state and two-stage tests pass.
- **Exact-order gating.** `gdn_norm_gating_proj` for the 27B now runs 256 physical
  threads. Through 14 tokens they keep the summation order of the previous 512-thread
  kernel; beyond that they keep the 256-thread order. Every width therefore produces
  the same bits as before this change. With the weights out of L2, as a decode graph
  meets them, a private harness measured T=8 at 13.0 µs instead of 15.4 µs. One-token
  tiles run through 4 tokens, two-token tiles through 28, and four-token tiles above
  that.
- **FP16 value decode.** `kv_cache_int4_dequant_f16x8_from` takes the FP16 scale and
  decodes without a conversion instruction. The biased nibble under exponent 1024 is
  FP16 1024 + n. Subtracting 1032 recovers the code, and one HMUL2 then rounds
  code × scale once, exactly as FP16-RN of the FP32 product. A private harness
  compared every finite FP16 scale with every byte value in every position against
  the previous FP32 path and found zero differing words. The small-T kernel also
  stores each even/odd P pair as one FP16x2 word.

The public attention benchmark does not install the device profile, so it times the
fallback small-T launches (W1 8×2×32, W8 6×2×32). The value decode was timed instead
with private copies of the production routes: cached `rk4v4`, graph replay, cold L2,
three warmups and 21 repetitions, medians of two alternating passes.

| Route | 32K keys, µs | 229K keys, µs |
|---|---:|---:|
| W8 `6x2x32q`: previous → retained | 195.4 → 183.1 | 1262.4 → 1175.0 |
| W4 `4x2x32qe` | 146.3 → 135.4 | 936.8 → 854.8 |
| W1 `8x2x32e` | 106.3 → 103.2 | 672.5 → 654.0 |
| W1 fallback `8x2x32` | 115.5 → 111.4 | 742.0 → 707.0 |

Paper requests ran four servers with fresh processes in X/Z/V/F order and its
reverse. They ran twice per server at 18K and 30K and once per block at 225K.
X is the baseline, Z adds the replay and gating changes, V adds the value decode,
and F is V with FP16 PV sums. The table gives median milliseconds per decode round;
at 225K it gives each block's run.

| Input | X | Z | V | V vs X |
|---|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 29.60 | 29.11 | 28.70 | −3.0% |
| Navier–Stokes, 30018 tokens | 29.80 | 29.16 | 28.94 | −2.9% |
| Navier–Stokes + Euler, 224759 tokens | 48.34 / 44.92 | 47.75 / 44.41 | 46.30 / 43.01 | −4.2% / −4.3% |

Decode throughput rose from 114.3 to 117.9 tokens/s at 18K, from 115.1 to 118.5 at
30K and from 60.96 to 63.66 at 225K. Prefill stayed within ±0.5%. The 225K blocks
differ by 7–8% for every server, so only same-block comparisons are meaningful.
This session's baseline decoded 8–9% slower than the same binary in earlier sessions.
F tied V at 18K and 30K (28.72 and 28.97 ms). At 225K it measured 42.73/42.70 ms, but
its reply changed, so the per-round comparison mixes in a different continuation.
FP16 PV sums remain off; the earlier verdict below stands.

Rejected in this investigation:

| Candidate | Result |
|---|---|
| Unmasked fast path for key tiles visible to every row | W8 −1.2% at 229K; W4/W1 tie; spills (8→24 bytes) cost the 128-register W1 fallback +13% |
| Pipelined W8 kernel: six producer warps one tile ahead of six PV warps, two stages, one CTA per SM | 1509 µs at 229K against 1172 for `6x2x32q` and 1400 for `12x1x32q` |
| L2 bulk prefetch 1–3 tiles ahead of that pipeline | 1493–1523 µs; no change |
| Skipping the accumulator rescale when a warp's alphas are all one | +0.7% at 229K |
| NVFP4 activations for the T2 prefill GEMMs (fake-quant screen) | held-out perplexity 5.4138 → 5.5408 (+2.3%); INT8 activations measured 5.4147 |

Ablations of the retained W8 route at 229K (1177 µs) showed where the time goes.
Skipping PV saved 297 µs, QK and its scaling 181 µs, the softmax 141 µs, the next
tile's loads 74 µs and the value dequantization 49 µs. With everything skipped,
171 µs remained. The phases are serial within a CTA and are overlapped only by the
second CTA on each SM. Neither tensor throughput nor bandwidth bounds the kernel:
FP16 MMA with FP32 accumulation runs at 51.6 TFLOPS here, against 100.8 with FP16
accumulation, which puts PV at about 0.44 ms of tensor time at peak. In the
pipelined variant PV and dequantization on six warps took 1283 µs alone. A single
warp per sub-partition does reach peak MMA rate, so warp count did not cause that
time; the variant only moved the serial chain onto fewer warps.

### Decode: proposal head and programmatic launch

Two further decode changes are retained.

- **Proposal head under its Linear policy.** `linear_topk` takes the head's
  `LinearPolicy`. The ternary head projects through T2 Linear under that policy, so with
  `prefill_a8` the DFlash2 proposal head (131072 × 5120) gets s8 activations, as the
  output head already did. Only the candidates change; the full head still verifies
  them. The top-k producer is now one warp per 128 rows: four keys per lane and a
  bitonic select-16 over the exact score/id order keys. It publishes an unordered set,
  which the merge already sorts. `ninfer_linear_topk_test` checks the full and indexed
  T2 heads against FP64 oracles under both policies, including the all-zero tie case.
  At 16 columns the whole Op went from 699 to 589 µs (642 → 464 µs at 8 columns,
  unchanged at 1). In nsys the producer took 33 µs instead of 121–134 µs, and the s8
  projection 531 µs against 642 µs for the A16 kernel.
- **Programmatic dependent launch.** The `rtx5060` presets set `NINFER_PDL=ON`, so each
  decode-graph kernel stages its weights while the previous one finishes.

Paper requests ran four fresh servers in V/T/P/F order and its reverse: twice per
server at 18K and 30K, once per block at 225K. V is the previous build, T adds the
proposal-head change, P is T with `NINFER_PDL=ON`, and F is V with FP16 PV sums. The
table gives median milliseconds per decode round. V, T and P replies were byte-identical,
with the same round counts (151, 149, 180). Prefill moved by at most 0.2%.

| Input | V | T | P | F | P vs V |
|---|---:|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 26.14 | 25.85 | 25.46 | 25.93 | −2.6% |
| Navier–Stokes, 30018 tokens | 26.88 | 26.59 | 26.22 | 26.82 | −2.4% |
| Navier–Stokes + Euler, 224759 tokens | 42.99 | 42.74 | 42.35 | 42.72 | −1.5% |

Decode throughput went from 129.5 to 132.9 tokens/s at 18K, from 127.6 to 130.8 at
30K and from 66.0 to 67.0 at 225K. This time F changed the reply on all three papers
(152, 154 and 172 rounds), and per round it gained only 0.8%, 0.2% and 0.6%. At 30K
its extra rounds made decode slower (123.7 tokens/s). FP16 PV sums stay off.

### Prompt attention: e4m3 PV

`staged_fp8` is the warp-specialized prompt kernel with its PV product in e4m3. This
section measures its first form. Each consumer rounded its probabilities to e4m3
against the row's tile maximum times 2^4. The producers rounded each decoded value tile
to e4m3 under one power of two per tile and 32-dimension group, chosen to keep |V| below
32. PV ran as m16n8k32 e4m3 MMA with FP16 accumulation (205 TOP/s here, against 103 for
f16), and its FP32 promotion applied the tile factor and both unscales. The current form
is the block-scaled PV described [below](#prompt-attention-block-scaled-e4m3-pv). The
FP32 row sums still add the unrounded probabilities. The two e4m3 value stages fit in
the FP16 value tile's space. With a
single stage, the producers' value decode sat on the consumers' critical path and the
kernel ran slower than FP16 PV.

The producers store each key as a 256-byte row at its probability's k position, in
16-byte chunks XOR-swizzled by row. The consumers load their B fragments with
`ldmatrix.m16n16.x2.trans.b8`, which only the explicit `sm_120a` target accepts. Each
lane then holds dimensions 32g + 8j + t and 32g + 8j + t + 4 of tile j and stores
them directly. The first form staged the tile transposed, `[dimension][key]`, which
took four byte permutes per four values in the producers. The natural tile removed
35% of the kernel's PRMT and 26% of its STS instructions; spill traffic rose from
122 to 186 STL/LDL instructions, mostly in the prologue. Outputs did not change: the
oracle statistics of both forms are identical and every paper reply matched.

Public cached Op at width 1152, graph replay and cold L2 (microseconds):

| Visible keys | `staged` (FP16 PV) | Transposed e4m3 tile | Change |
|---:|---:|---:|---:|
| 8192 | 2984 | 2566 | −14.0% |
| 32768 | 11068 | 9950 | −10.1% |
| 229376 | 83267 | 70520 | −15.3% |

These are medians of an alternating ABBA run. In a later session (minimum of nine
runs per binary over three alternating passes), the natural tile measured
2548–2574 µs against 2558–2559 at 8K, a tie. It measured 9943–10137 against
10549–10830 at 32K and 71.9–75.4 ms against 75.7–77.4 ms at 229K.

Calibration with route catalog `eaf84b6eea2436ac` timed every schedule (µs):

| Visible keys | `standard` | `fast` | `staged` | `staged_fp8` |
|---:|---:|---:|---:|---:|
| 512 | 148.2 | 144.3 | 137.2 | 148.4 |
| 8192 | 5401 | 3821 | 2650 | 2273 |
| 32768 | 21976 | 15650 | 11684 | 10100 |
| 131072 | 92886 | 64475 | 51342 | 42600 |
| 262144 | 190768 | 129373 | 101607 | 86012 |

The profile now routes `attn_prompt/h24/rk4v4` as `[[512, "staged"], [262144,
"staged_fp8"]]`. `NINFER_PROMPT_PV_FP8=0|1` forces FP16 or e4m3 PV wherever the
profile names either staged launch.

e4m3 PV is a numerical change. Each probability and each decoded value moves by up
to 2^-4 of itself, where FP16 PV moves them by at most 2^-11. Its oracle criterion
(`kAttentionRk4v4Fp8PvCriterion`) is derived in the test from those roundings:
relative L2 5.1e-2 against rk4v4's 3.25e-3, and elements within 2^-4 of the largest
reference output. In this first form its cases measured relative L2 2.2e-2 to 3.2e-2 through 262,144
keys and a largest element error of 4.3e-2 of the largest output. On model quality,
the quick perplexity check moved from 5.414747 to 5.414032. Six 65K-token windows
moved from 5.353920 to 5.356144 (+0.04%).

Paper requests, first round: P is the previous build and E the transposed e4m3
tile, fresh servers in P/E/E/P/P/E/E/P order (P/E/E/P at 225K), mean seconds:

| Input | Prefill P → E | Total P → E | ms/round P / E |
|---|---:|---:|---:|
| Deutsch, 17936 tokens | 8.08 → 7.97 (−1.4%) | 11.98 → 11.91 | 25.48 / 25.30 |
| Navier–Stokes, 30018 tokens | 14.08 → 13.79 (−2.1%) | 18.04 → 17.81 | 26.25 / 26.25 |
| Navier–Stokes + Euler, 224759 tokens | 211.6 → 193.7 (−8.5%) | 219.6 → 200.9 | 42.39 / 42.68 |

All three replies changed (151/154, 149/151 and 180/162 rounds). The decode kernels
are unchanged, so the per-round differences follow the changed continuations. Task
quality was not measured. In a second round, both builds ran with
`NINFER_PROMPT_PV_FP8=1`. The natural tile cut 225K prefill from 209.76/209.72 to
205.19/205.11 s (−2.2%), 30K from 15.01 to 14.97 s and Deutsch from 8.685 to 8.66 s.
Milliseconds per round were unchanged and every reply matched. That session ran the
transposed build about 8% slower at 225K than the first round did, so compare only
within a round.

Ablations of `staged_fp8` at width 1152 skipped one producer phase each, so their
results are invalid by design. They cut 8K/229K time by 10.4%/7.4% without key
expansion and 13.3%/14.9% without value decode. Without both, the cut was
14.6%/17.7%, and 16.0%/21.1% when the loop's global loads were also skipped.
Instrumented barrier waits took at most 0.6% of loop cycles in either warp group.
Neither group waits on the other; producer work most likely slows the consumers by
sharing sub-partition issue and load/store throughput.

That decode was also repeated: at width 1152, each KV head's tiles were decoded by
all 54 CTAs of its six query heads (nine row blocks each).
`causal_attention_prompt_ws_image_kernel` now decodes every (KV head, key tile)
once per launch into a workspace image. The image holds the key stage (INT8 codes
and FP32 scales, 17,408 bytes), then the e4m3 value tile and its eight group
unscales (16,416 bytes), and the producers only copy it with `cp.async`. A value
range now covers the tile's keys visible to the launch, not only to the CTA's rows.
This can change the power of two only on a CTA's diagonal tile, and only for keys
its rows mask. The FP64 oracle statistics of every rk4v4 fast-prompt and
`staged_fp8` case stayed identical at their printed precision. The FP8 kernel fell
from 28,916 to 22,212 SASS lines, and its spill instructions from 186 to 117. The
images take 4 × ⌈keys / 64⌉ × 33,824 bytes of workspace, reserved for the largest
envelope that can select `staged_fp8`: 138.5 MB at 65,536 keys and 554.2 MB at
262,144. A timing screen with probe-filled images copied the value images alone
(−1.3%, −3.2% and −12.5% at 8K, 32K and 229K) and both images (−8 to −17%, −7.6%
and −14.6%), without the decode pass.

Public cached Op at width 1152, graph replay, cold L2, decode pass included; minimum
of nine repetitions per run over three alternating pairs (µs):

| Visible keys | Natural tile | Tile images | Change |
|---:|---:|---:|---:|
| 8192 | 2546–2552 | 2335–2338 | −8.4% |
| 32768 | 9711–9737 | 9081–9227 | −6.0% |
| 229376 | 72330–74238 | 64615–65398 | −11.0% |

Paper requests with fresh servers in N/I/I/N order per input, N the natural tile
and I the tile images, same profile and no overrides (seconds and ms per round):

| Input | Prefill N | Prefill I | ms/round N | ms/round I |
|---|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 8.67 / 7.95 | 8.42 / 7.91 | 28.07 / 25.30 | 25.37 / 25.29 |
| Navier–Stokes, 30018 tokens | 14.98 / 15.06 | 14.77 / 14.88 | 28.98 / 29.21 | 28.87 / 29.20 |
| Navier–Stokes + Euler, 224759 tokens | 203.67 / 203.90 | 193.37 / 193.88 | 47.00 / 46.74 | 47.08 / 47.15 |

225K prefill fell 5.0% and total time 211.7 → 201.5 s. Every reply matched its
counterpart. At 225K, decode was 0.5% slower per round, and both image runs were
slower than both baseline runs. The decode kernels did not change and the cause is
unresolved. The first Deutsch request of each binary was slow in both. With a
262,144-token capacity, startup reported runtime 5.81 → 6.24 GiB and free VRAM
1.53 → 1.12 GiB, and every KV page remained on the device (4,096/4,096).

Rejected in this investigation:

| Candidate | Result |
|---|---|
| Single e4m3 value stage | +2.7% at 8K, +16% at 32K, +5% at 229K against FP16 PV |
| QK accumulators seeded with 0x4B400000 for an FADD conversion instead of I2FP | +3.4% |
| 72/216 producer/consumer register split | consumers spill 113 bytes |
| Softmax of tile k and PV of tile k − 1 in one barrier-free region | +5.5%; 86 spill stores, 100 loads |
| Lazy running maximum, rescaling only past a 2^8 growth | −0.7% at 8K, +0.2% at 229K |
| Quad transpose of the accumulators into the FP16 epilogue's layout | +1.3% at 8K; collective shuffle loops |
| 40/232 register split for the copying producers | +2.7–3.4% at 8K–229K, though spill instructions fell from 117 to 99 |

### Prompt attention: block-scaled e4m3 PV

On this card, `mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X` with e4m3 operands and
UE8M0 scales accumulates in FP32 at the full e4m3 rate (205 TOP/s). The unscaled FP32-accumulating
e4m3 form runs at half that rate. With thread and byte selector 0, lane 4g supplies the scale of
A row g, lane 4g + 1 that of A row g + 8, and lane 4c that of B column c. When the accumulator is
much larger than the products, the MMA's FP32 addition is off by 2–3.5 units in the last place
against an FP64 sum of the same e4m3 operands, so the running sum is not accumulated inside the
MMA.

`staged_fp8` now takes each row's probabilities against the running maximum, lowered by the
whole octaves that bring it within one of the row's tile maximum, times 2^8. The tile's largest
probability lies in (128, 256]. The row's octaves and the 2^-8 return through its UE8M0 A scale.
Each decoded value is rounded to e4m3 under a power of two per (key tile, 32-dimension group)
that maps the group's bound into [128, 256), and that power's inverse is the B scale. A tile's
two 32-key MMAs per n8 tile chain from a zero accumulator. One FFMA per element then rescales the
running sum and adds the FP32 partial. This replaces the FP16 partial's conversion, multiply and
fused multiply-add. Elements below about 2^-14 of their tile's largest probability or value
become subnormal; the FP16-sum form's threshold was 2^-10. The image kernel's value unscale no
longer carries the probability shift. Each consumer keeps 168 registers.

The public FP64 oracle suites pass, including the profile, route-coverage, rk4v4, fast-prompt,
INT8, NVFP4 and k8v4 suites. Across the 62 fast-prompt cases, the worst criterion use stayed at
0.692. Relative L2 rose by 1.0% on average and by 4–7% in the 1152-wide cases from 33K to 262K
keys, which now measure up to 3.27e-2. Held-out perplexity moved 5.416098 → 5.415032 on the quick
corpus and 5.351154 → 5.353397 (+0.042%) at 65,536 tokens.

A variant took probabilities against the tile's row maximum, so the largest was exactly 256. It
used a constant A scale and multiplied the partial by the tile factor, with one FFMA whenever no
row of the warp raised its maximum. It recovered the oracle's long-case L2 (mean ratio 1.004) but
not the 65K perplexity (5.353427). It was 0.8% slower at 8K and was rejected.

Public cached Op at width 1152, graph replay and cold L2; minimum of nine repetitions per run
over three alternating pairs (µs):

| Visible keys | FP16 sums | Block-scaled FP32 | Change |
|---:|---:|---:|---:|
| 8192 | 2320–2324 | 2246 | −3.2% |
| 32768 | 8520–8559 | 8243–8333 | −2.7% |
| 229376 | 58622–58686 | 56696–56873 | −3.2% |

Paper requests with fresh servers in X/P/P/X order per input. X is the per-token T2 build below
and P adds block-scaled PV:

| Input | Prefill tok/s X → P | ms/round X / P | Accepted drafts per round X → P | Total s X → P |
|---|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 2243.1 → 2256.2 (+0.6%) | 27.86 / 27.72 | 2.38 → 2.36 | 12.26 → 12.22 |
| Navier–Stokes, 30018 tokens | 2169.3 → 2183.1 (+0.6%) | 28.88 / 28.79 | 2.36 → 1.84 | 18.30 → 19.00 |
| Navier–Stokes + Euler, 224759 tokens | 1212.6 → 1230.0 (+1.4%) | 46.23 / 46.27 | 1.99 → 2.10 | 193.6 → 190.7 |

That session ran about 8% slower than the per-token T2 session (X at 2243 against 2437 prefill
tokens/s on Deutsch). Every reply changed. The 30K reply accepted 22% fewer drafts, so its
request took 3.8% longer overall. Acceptance was then measured on one server per build, eight
questions on each of the Deutsch and 30K documents. Across all 16
replies, the per-group T2 build accepted 2.016 drafts per round, the per-token build 2.039 and
block-scaled PV 2.002. Single replies moved by −23% to +18% between any two builds, so neither
change has a detectable systematic effect on acceptance at that resolution (about ±2.5%).

### Prefill: per-token T2 activation scales

The T2 prefill tile quantizes its activations to int8 with one FP16 scale per token, where it
previously used one FP32 scale per token and 128-wide K group. Each group's int32 sum is
converted and multiplied by its weight scale (`acc = fma(float(sum), w_scale, acc)`). The token
scale is applied once, in the store epilogue. Before the change, the K5120 mainloop spent 192 of
its 515 instructions per group and warp on scaling: 64 I2FP, 64 FMUL and 64 FFMA. The FMUL
applied the activation group scale. The Bonsai weight scales differ from group to group (about 7%
coefficient of variation, around 38 distinct values per 5120-wide row), so the int32 sum cannot
run over all of K. The plain and TMA-staged tiles lose the scale bytes from each stage. The
fused Hadamard quantizer of the rotated FFN down input runs one CTA per token column and holds
the BF16-rounded transform blocks in registers until the token maximum is known. At K17408 it
uses 159 registers. The plain quantizer uses 80 registers and an 8-byte stack frame there.

Every T2 GEMM input on this artifact is Hadamard-rotated (`rmsnorm_hadamard`,
`gated_rmsnorm_hadamard`, `sigmoid_mul_hadamard`, `gdn_norm_gating_proj_rotated` and the fused
FFN transform). The rotation spreads outlier channels, which is why a per-token absmax holds up
here. It does not hold for unrotated W4A8 activations
([quality trade experiments](maintainer/quality-trade-experiments.md)).

A probe of the production K5120 tile measured −5.7% to −6.8% GEMM time. Removing the scaling
entirely gave −6.5% to −6.8%. Integer group multipliers in place of FP32 weight scales gave
−5.7% to −6.5%, which does not justify an artifact change. A magic-number FADD conversion of the
int32 sums, exact but slower, measured +1.3% to +1.9%.

Held-out perplexity (rk4v4, FP16 GDN state):

| Corpus | Per-group scales | Per-token scales |
|---|---:|---:|
| Quick | 5.414032 | 5.416098 (+0.038%) |
| 65,536-token context, 32,768 stride | 5.356144 | 5.351154 (−0.093%) |

Paper requests with fresh servers in I/X/X/I order per input, I the per-group build and X the
per-token build, same profile and no overrides:

| Input | Prefill tok/s I | Prefill tok/s X | ms/round I | ms/round X | Decode tok/s I | Decode tok/s X |
|---|---:|---:|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 2276.7 | 2436.7 (+7.0%) | 25.33 / 25.29 | 25.30 / 25.29 | 131.10 | 133.77 |
| Navier–Stokes, 30018 tokens | 2211.6 | 2359.7 (+6.7%) | 26.21 / 26.21 | 26.22 / 26.21 | 129.12 | 128.25 |
| Navier–Stokes + Euler, 224759 tokens | 1260.7 | 1307.4 (+3.7%) | 42.62 / 42.69 | 42.38 / 42.34 | 73.95 | 70.54 |

The two runs of each binary agree to within 0.2%. Decode time per round did not change. The
change alters numerics, so every reply differs from its baseline. Decode tokens per second
follow how many drafts the new reply accepts. Every request ran to the same token budget. At
225K, X needed 171 rounds against 162 (−7.4% accepted drafts per round), so decode fell 4.6%.
At 30K acceptance fell by 0.9%, and on Deutsch it rose by 2.8%. With a single reply per
input, the 225K acceptance loss is unresolved. Total client time still fell at every input:
11.82 → 11.23 s, 17.59 → 16.76 s and 185.5 → 179.4 s.

The public T2 A8 Linear, residual LinearAdd, fused SwiGLU–Hadamard, attention and GDN input
projection (T2 routes) and top-k proposal-head oracle tests pass. `ninfer_gdn_input_proj_test`
still fails only in its FP8 A16 cases, as it did before the change.

### Prompt attention: one QK chain per key

After block-scaled PV, converting and scaling the QK partials was the largest remaining cost in
the `staged_fp8` consumers. Each 64-dimension key group's int32 sum was converted and multiplied
by its own key and query scales. A timing-only ablation with one int32 chain per score measured
−4.0%, −3.0% and −4.0% at 8K, 32K and 229K visible keys on the FP16-sum kernel, and −9.1%,
−8.8% and −9.4% with block-scaled PV.

`staged_fp8` now gives each key one INT8 scale. Per key and G64 group, the tile-image kernel
builds a shared-memory table of the 16 Lloyd-Max levels times the group scale over the key's
largest group scale, rounded to INT8, and expands the 4-bit codes through it. Each image keeps
one FP32 scale per key. The consumers quantize each query row under one scale and chain the
four groups' s8 MMAs into one int32 sum per score, multiplied once by the two scales.
Re-rounding moves a key element by at most half an INT8 step of its key's largest group. The
FP16 `staged` route keeps its per-group scales. The image kernel uses 72 registers and 5504
bytes of shared memory. The consumers keep 168 registers, with a 24-byte stack frame.

Across the fast-prompt FP64 oracle cases, relative L2 changed by a mean factor of 1.0002 against
block-scaled PV (largest 1.0035). The worst criterion use fell from 0.692 to 0.677. Held-out
perplexity moved 5.415032 → 5.415790 (+0.014%) on the quick corpus and 5.353397 → 5.352907
(−0.009%) at 65,536 tokens.

Public cached Op at width 1152, graph replay and cold L2; minimum of nine repetitions per run
over three alternating pairs (µs):

| Visible keys | Per-group QK | One chain per key | Change |
|---:|---:|---:|---:|
| 8192 | 2255–2257 | 1990–1997 | −11.6% |
| 32768 | 8743–8874 | 7742–7866 | −11.4% |
| 229376 | 61455–62535 | 53487–55259 | −11.6% |

Paper requests with fresh servers in P/K/K/P order per input. P is block-scaled PV and K adds
one chain per key:

| Input | Prefill tok/s P → K | ms/round P / K | Accepted drafts per round P → K | Total s P → K |
|---|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 2227.2 → 2251.2 (+1.1%) | 28.16 / 28.15 | 2.36 → 2.43 | 12.39 → 12.22 |
| Navier–Stokes, 30018 tokens | 2159.1 → 2203.2 (+2.0%) | 29.23 / 29.21 | 1.84 → 1.85 | 19.24 → 18.92 |
| Navier–Stokes + Euler, 224759 tokens | 1224.5 → 1305.5 (+6.6%) | 46.88 / 47.10 | 2.10 → 1.87 | 191.6 → 180.9 |

Both runs of each binary agree to within 1%. Every reply changed. At 225K, time per round rose
0.5% in both K runs although no decode kernel changed. The tile-image change also showed +0.5%
there; the cause is unresolved. That reply accepted 11% fewer drafts per round, so decode fell
from 66.1 to 60.9 tokens/s, and the request still finished 5.6% sooner. On the 16-reply
acceptance probe (eight questions on each of the Deutsch and 30K documents), K accepted 1.988
drafts per round against block-scaled PV's 2.002. K was higher on 7 of the 16 replies, −6.4% on
the 30K replies and +5.3% on the Deutsch replies, so there is no detectable systematic shift.

### Decode attention: e4m3 PV

Decode and speculative verification (query widths 1 to 8) run the small-T INT8-family attention.
Its 32-key tiers formed P·V on the FP16 MMA from values decoded to FP16. The `f` schedules of the
`rk4v4` tiers form it instead as one block-scaled e4m3 m16n8k32 MMA per n8 tile, which this card
issues at four times the FP16 rate. The producers scale each row's probabilities by whole octaves
and 2^8 so that the tile's largest lies in (128, 256], round them to e4m3, and return the scale
through the row's UE8M0 A scale. Every decoded value is rounded to e4m3 under its 32-dimension
group's power of two, which maps eight times the group's largest scale into [128, 256); its
inverse is the group's B scale. Each tile's MMA starts from zero and one FFMA per element adds it
to the rescaled FP32 running sum. The row sums keep the unrounded FP32 probabilities. The other
widths and KV formats keep FP16 PV.

The first build stored the output in the prompt kernel's interleaved column order, which belongs
to that kernel's nibble-interleaved value conversion; this kernel's value tile keeps dimensions in
order. The FP64 oracle rejected it at relative L2 1.19, and calibration rejected the routes for
output 0.28–0.73 away from the compiled route. With the natural order store, 103 public Op cases
that force each `f` schedule at the widths it serves pass the FP64 oracle at relative L2
0.0218–0.0314 against a 0.051 limit (largest criterion use 0.655). They cover a partial first
key block with the fused append, cached entries at 5000 and 262144 visible keys, graph replay at
65536 keys on the profile's tiers, value amplitudes 2^-6 and 2^9, a value-group ramp spanning
2^12, masked batch rows and a single visible key. The FP16 schedules in the same run stay at or
below 0.0018, so the decode attention output error rises about seventeenfold.

Calibration with the L2 flushed, median of eleven (µs). The previous routes took FP16 PV:

| Width | Visible keys | Previous route | Previous | `f` route | `f` | Change |
|---:|---:|---|---:|---|---:|---:|
| 6 | 8192 | `12x1x32q` | 88.9 | `12x1x32qef` | 82.1 | −7.6% |
| 6 | 65536 | `6x2x32q` | 404.5 | `6x2x32qef` | 361.0 | −10.8% |
| 6 | 262144 | `6x2x32q` | 1490.7 | `6x2x32qef` | 1338.1 | −10.2% |
| 7 | 8192 | `12x1x32qe` | 82.5 | `12x1x32qef` | 76.5 | −7.3% |
| 7 | 65536 | `6x2x32q` | 391.4 | `6x2x32qef` | 352.8 | −9.9% |
| 7 | 262144 | `6x2x32q` | 1438.7 | `6x2x32qef` | 1299.0 | −9.7% |
| 8 | 8192 | `12x1x32qe` | 82.8 | `12x1x32qef` | 76.6 | −7.5% |
| 8 | 65536 | `6x2x32q` | 393.3 | `6x2x32qef` | 351.4 | −10.7% |
| 8 | 262144 | `6x2x32q` | 1440.4 | `6x2x32qef` | 1284.9 | −10.8% |

Width 1 also takes `8x2x32ef`, within 0.2–1.2% of `8x2x32e`. Widths 3 to 5 keep FP16 PV, where no
`f` form was faster, and width 2 moved to `4x2x32e` past 8192 keys in the same calibration.

Paper requests with fresh servers in K/S/S/K order per input. K is the previous engine and S adds
e4m3 PV; both runs of each binary returned the same reply, and every reply changed between them:

| Input | Prefill tok/s K → S | ms/round K / S | Accepted drafts per round K → S | Decode tok/s K → S | Total s K → S |
|---|---:|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 2452.5 → 2444.4 (−0.3%) | 25.31 / 25.17 (−0.5%) | 2.43 → 2.43 | 135.5 → 136.2 | 11.13 → 11.14 |
| Navier–Stokes, 30018 tokens | 2389.9 → 2387.2 (−0.1%) | 26.38 / 26.01 (−1.4%) | 1.85 → 2.43 | 108.2 → 131.9 | 17.34 → 16.51 |
| Navier–Stokes + Euler, 224759 tokens | 1416.1 → 1415.7 (−0.0%) | 42.69 / 40.49 (−5.2%) | 1.87 → 2.04 | 67.2 → 75.1 | 166.6 → 165.8 |

The time per round is the kernel's effect; most of the decode rate at 30K and part of it at 225K
come from the changed replies accepting more drafts. On the 16-reply acceptance probe S accepted
2.003 drafts per round against K's 1.988, higher on 6 of the 16 replies (per-reply ratio
0.83–1.31), +5.6% on the 30K replies and −3.8% on the Deutsch replies, so there is no detectable
systematic shift. Held-out perplexity scores prompt chunks, which this kernel does not serve, so
it does not measure this change; the oracle error above is its numerical qualification, and the
effect on answer quality has not been measured.

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
visible keys, including graph replay; the current suite passed in 92.40 s.
Fast rk4v4 now covers 37 public Op cases, including actual 512/1152 prompt
widths through 262,144 visible keys, append and cached entry points, graph
replay, large value scales, and masked or empty rows. Automatic selection is
qualified at 255/256/257 query tokens and around the 8192/262144 key boundaries.
Short cases compare every output; long cases compare selected query rows and
heads, every output dimension and every visible key. Other outputs receive
finite/buffer checks. They do not compare every long-context output element.
The 37 fast cases passed in 99.54 s with the 17-key profile fixture selected.
With the warp-specialized kernel behind the `staged` launch, the rk4v4, fast-prompt,
INT8-prompt, device-profile and route-coverage suites passed under the previous
8K profile boundary (223.7 s), and the rk4v4 and fast-prompt suites passed again
with every depth routed to it (199.4 s), so the automatic cases at 255/256/257
query tokens and 8191–8193 keys then exercise it from short histories.
The first fast test crashed because its host fixture allocated INT8 instead of
packed Lloyd4 keys; the fixture was repaired and the original criteria passed.
The `staged_fp8` suite runs 19 more public Op cases on that route under its own
criterion: partial blocks, graph replay, 1152-token chunks through 262,144 keys,
value scales 2^-6 and 2^9, masked and empty batch rows, and one- and two-tile
sweeps. When the installed profile stages these launches, the FP16 fast suite pins
`staged`, so both kernels are qualified at every depth whichever bands the profile
names. With the 18-key profile fixture, the seven checks below passed in 494.93 s, and in
569.04 s once its decode tiers took e4m3 PV; the `rk4v4` suite then also runs the forced `f`
cases.

To qualify the measured attention routes on the 5060 Ti, select the checked
profile fixture explicitly:

```bash
unset NINFER_PROMPT_FAST
NINFER_TEST_DEVICE_PROFILE_PATH="$PWD/src/runtime/engine/device_profiles.json" \
NINFER_OP_REPORT_STATS=1 ctest --test-dir build --output-on-failure -j 1 \
  -R '^ninfer_(device_profile|route_coverage|softmax_attention_(rk4v4|rk4v4_fast_prompt|int8_prompt|nvfp4|k8v4))_test$'
```

The common attention checks exposed an existing NVFP4 workspace defect: eight
cases passed their independent math comparison but failed the exact workspace
high-water check. Six queried shapes reserved 44.4–66.9 MB beyond the selected
launch's requirement, including launches needing zero scratch. The query now
uses execution's split plan and the maximum actual allocation over the legal
width interval, including early prompt routes under `small_prefill`. This
changes NVFP4 scratch sizing; Bonsai's `rk4v4` route uses no NVFP4 split scratch.
The query scans the width interval on the CPU; its startup cost for an NVFP4
model has not been measured. Interval witnesses and append/cached graph cases
cover the sizing contract and the transition into prompt execution.
The corrected NVFP4 suite passed in 124.23 s, including 32 interval witnesses
and eight transition graph calls. No comparison criteria were relaxed.
All seven selected attention, profile and catalog checks passed in 392.80 s.

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

At 262144 capacity, Main KV alone reserves 4697620480 bytes (4.375 GiB):
16 full-attention layers, four KV heads, D256, 4-bit keys with FP16 G64 scales,
and 4-bit values with FP16 G32 scales. The cache-enabled app was also started
at this capacity with DFlash7 and default 15-token n-gram drafting. It reserved
8.158 GiB of selected weights, 4.702 GiB of sequence storage and 174.64 MiB of
workspace. The startup ledger reported 1.565 GiB free after initialization;
32 MiB of graphs were actually allocated against a 960 MiB allowance.

Earlier Windows adapter counters peaked at 15109 MiB dedicated and 9266 MiB
shared during the completed 224759-token automatic paper request at this capacity.
The matched production tensor-copy comparison above measured 15130–15139 MiB
dedicated and 9249–9264 MiB shared, leaving 1172–1181 MiB of dedicated headroom.
The app intentionally pinned 8 GiB of host KV and 918.6
MiB of host states, which accounts for approximately 9 GiB of shared memory.
Shared usage by itself does not establish that resident weights or active KV
have spilled. The earlier dedicated peak left about 1.18 GiB of this card's
physical memory free. These workloads completed within that capacity;
other GPU allocations can still make 262K capacity unsuitable on a busier
desktop. Internal Engine spill counters do not measure Windows driver residency,
so they cannot establish whether the original user run experienced paging.

The earlier Windows launcher check started the selected server with the built-in
16-key profile, DFlash7, temperature zero and thinking off. Engine startup took
39.5 s, including 27.7 s weight upload, 0.8 s pinning 918.6 MiB of host states,
8.4 s pinning host KV and 1.5 s graph preparation; HTTP warmup added 0.64 s.
This earlier startup used the artifact under `/mnt/c`; the
[storage comparison above](#faster-startup-from-wsl-storage) measures a native
copy. Runtime reservation with serving caches is 2.53 GiB.

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

## Result against the first port

The prompt and decode sections above each measured one change against the engine before it. A
direct paper comparison of the first port of this branch (O, with the profile it shipped with) against the
current engine (C), fresh servers in O/C/C/O order per input, one request at a time:

| Input | Prefill tok/s O → C | ms/round O → C | Accepted drafts per round O → C | Decode tok/s O → C | Total s O → C |
|---|---:|---:|---:|---:|---:|
| Deutsch, 17936 tokens | 2020.3 → 2441.6 (+20.9%) | 26.68 → 25.20 (−5.6%) | 2.38 → 2.43 | 126.8 → 136.1 | 12.97 → 11.15 (−14.0%) |
| Navier–Stokes, 30018 tokens | 1941.4 → 2392.7 (+23.2%) | 27.86 → 26.02 (−6.6%) | 2.13 → 2.43 | 112.5 → 131.8 | 20.06 → 16.48 (−17.8%) |
| Navier–Stokes + Euler, 224759 tokens | 947.3 → 1415.1 (+49.4%) | 45.23 → 40.54 (−10.4%) | 1.79 → 2.04 | 61.7 → 75.0 | 245.8 → 165.9 (−32.5%) |

Both runs of each engine returned the same reply, and every reply differs between the engines.
The time per round and the prefill rate measure the engines; the decode rate also includes each
reply's acceptance, which favoured the current engine's replies here (the 16-reply probes above
show no systematic shift from the individual changes).

## Concurrent requests

`Start-Bonsai.cmd` serves two requests at the same time by default. With `-Concurrency N` above 1
the launcher passes `--max-concurrency N`, `--kv-capacity` of N times `-Context` up to 262144
tokens, and `--concurrent-prefill`, so a request is admitted while another one prefills. For every
N it passes `--cuda-graph-allowance-mib` of 128 MiB per request: startup measured 32–41 MiB of CUDA
graphs for one request and 77–85 MiB for two, against a computed allowance of 960 MiB per request,
and the fixed allowance returns that reservation to the memory left beside the KV pool.

The requests share one KV pool. At the default 65536-token context each of two requests can still
use its whole context. At `-Context 262144` the pool stays at 262144 tokens, so two requests run
together only while both fit, and a request that does not fit waits in the queue. There, a request
without an output limit receives a budget within one lane's half of the pool, unless its prompt
alone overruns that half ([default output limit](serving.md#default-output-limit)).

The scheduler alternates one decode round over every decoding request with one prefill chunk. A
chunk serves the staged prompt with the fewest tokens left (the lowest lane on ties), so a short
prompt that arrives during a long one is ingested within its own chunks instead of behind the whole
long prompt. Prefill chunks of different requests and decode rounds are not batched together.

Measured with DFlash7, greedy decoding, thinking off and a fresh server per configuration, at the
default 65536-token context. "One at a time" sends the two requests serially to a one-request
server, "both at once" sends them together to the default two-request server. The decode workload
is two short prompts with 768-token answers, papers are the Deutsch paper and the first 30000
tokens of the Navier–Stokes paper with 512-token answers, and mixed is the Deutsch paper and one
short prompt:

| Workload | One at a time: wall s, tok/s | Both at once: wall s, tok/s | Aggregate | Per request, alone → together (s) |
|---|---:|---:|---:|---|
| Decode | 14.86, 103.4 | 9.13, 168.2 | +62.7% | 7.97 → 9.13, 6.89 → 8.04 |
| Papers | 27.27, 37.6 | 25.13, 40.7 | +8.5% | 11.14 → 23.96, 16.13 → 25.08 |
| Mixed | 18.73, 68.3 | 15.63, 81.9 | +19.8% | 11.14 → 12.18, 7.59 → 15.58 |

Serving two requests together raised the aggregate rate on every workload, and each request takes
longer than it would alone. A decode round over two requests costs more than one over a single
request, the papers each wait for the other's prefill chunks, and the mixed workload's short
request advances one decode round per chunk of the paper's prefill, which keeps prefill at full
speed.

A single request on the two-request configuration runs as fast as on the one-request
configuration. At 65536 tokens, the decode and paper workloads sent one at a time took 14.84 s and
27.25 s against 14.86 s and 27.27 s. At a 262144-token context (fresh servers in 1/2/2/1
order; the one-request runs kept the computed graph allowance):

| Input | One-request server: ms/round, prefill tok/s | Two-request server: ms/round, prefill tok/s |
|---|---:|---:|
| Deutsch, 17936 tokens | 25.19, 2435.7 | 25.19, 2440.1 |
| Navier–Stokes + Euler, 224759 tokens | 40.60, 1415.8 | 40.57, 1414.9 |

Both runs of each configuration returned the same replies, and the two configurations returned the
same reply for each input.

With the Deutsch paper sent first and a short prompt 1.5 s later to a two-request server, the
short prompt's first token arrived after 1.12 s, against 5.98 s for the previous engine, which
served the lowest lane first. The paper's own request took 12.12 s against 11.87 s. The two engines
also differ by the e4m3 decode PV, which changes the replies; the first-token difference comes from
the order of prefill chunks.

Device memory after startup, as the startup ledger reports it:

| Launch | KV pool, tokens | Runtime reservation | Free |
|---|---:|---:|---:|
| `-Concurrency 1` | 65,536 | 1.76 GiB | 4.98 GiB |
| default (two requests) | 131,072 | 3.23 GiB | 3.50 GiB |
| `-Concurrency 1 -Context 262144`, 960 MiB graph allowance | 262,144 | 6.24 GiB | 1.22 GiB |
| `-Context 262144` (two requests) | 262,144 | 5.81 GiB | 937 MiB |

At a 262144-token context two requests leave less free memory than one; other GPU applications
can make that too tight, and `-Concurrency 1` restores the one-request layout. With the display
on, the desktop compositor shares the GPU: in one session every measurement, with one request or
two, ran about 10% slower until the display turned off, so compare timings made under the same
conditions.

## Further work

These are the open candidates on this card, each with the evidence that bounds it. None of them
is in the engine.

- **Decode weight traffic.** In an nsys profile of the Deutsch paper (154 rounds), the T2 small-T
  projections took 69.5% of decode time and every T2 decode shape streamed weights at
  390–417 GB/s, close to this card's memory bandwidth. Decode speed per round is therefore bounded
  by the bytes of the stored ternary weights. A denser ternary packing (1.75 instead of 2.125 bits
  per weight, decoded exactly) measured about 3 ms less per round but 6–18% slower prefill, so the
  paper totals lost; it would need a decode-only copy of the weights, which 16 GB does not hold
  next to a long context.
- **T2 activation quantization.** The separate per-token quantization before each T2 decode
  projection ran 40244 launches for 118 ms in that profile (about 0.8 ms per round). Folding it
  into the producing kernels is estimated at 0.5 ms per round. In prefill the equivalent SiLU and
  quantize fusion is estimated at about 2% at 30K tokens and 1% at 225K.
- **DFlash drafter kernels.** The drafter's Q4/Q8 projections took about 2.8 ms per round
  (432 ms over 154 rounds), about a tenth of a 30K-context round. The 5060 Ti profile names no
  groupwise route, so they run their compiled schedules; whether the groupwise calibration finds
  faster ones for the drafter's widths on this card has not been measured.
- **GDN decode state work.** Recording, norm-gating and folding the GDN state took about 1.5 ms per
  round in the same profile.
- **T2 small-T routes after the newer kernels.** The T2 decode schedules were calibrated before the
  later prefill and attention changes; a fresh `ninfer-calibrate --only t2_i8_small` checks whether
  they still hold.
- **Long-context decode after the per-key QK scale.** Decode at 224759 tokens measured 0.5% more
  time per round with the per-key prompt QK scale although the decode kernels did not change. The
  cause has not been found.
- **e4m3 conversion in decode attention.** With e4m3 PV the small-T kernel converts each 4-bit value
  and probability to e4m3. `cvt.e4m3x2.f16x2` issues at 25.6 per clock per SM against 46 for a
  logic operation, which is most of the gap to the conversion-free bound of the W8 kernel
  (1079 µs at 229K keys in a timing-only probe). A KV format whose values are stored as FP4 (E2M1)
  with block scales would feed the block-scaled FP4 MMA without conversion; that is a new KV format
  and needs its own perplexity qualification.
- **Decode beside a prefill.** With two requests, the scheduler alternates one decode round with
  one prefill chunk, and prefill units of different requests and decode rounds are never batched
  together. A request that is decoding while another ingests a long prompt therefore advances one
  round per chunk: a 1152-token chunk takes about 0.4 s over the Deutsch paper and 0.8 s near
  225K tokens, against a 25–45 ms round.
  This keeps prefill at full speed, the chosen policy here. Smaller chunks while another lane
  decodes would trade prefill speed for decode latency, but each prefill unit is planned as one
  service-work quantum for the context-cache planner, so the chunk size cannot change per unit
  without changing that planning. A combined launch of a prefill chunk and the other lane's decode
  step would avoid the trade and is a larger scheduler and Program change.
- **Two-request decode rounds.** In an nsys profile of the decode workload with both requests at
  once, a round over both requests took a median 28.4 ms of kernel time against 24.5 ms for the
  rounds where one request decoded alone (285 and 61 rounds in the same profile). The T2
  projections cost the same (19.7 and 19.9 ms), as both requests share each weight read. Most of
  the 3.9 ms comes from the GDN state record, fold and norm-gating (+1.0 ms together), the
  drafter's Q4 projections (the row-split Q4 GEMM took 0.76 ms per two-request round and less than
  0.17 ms per single round, the k-split kernel +0.14 ms) and the T2 activation quantization
  (+0.2 ms). Calibrating the groupwise routes at the drafter's two-request widths and sharing the
  GDN state kernels' launches across lanes are the candidates; neither has been measured.
- **A short prompt's first token beside a long prefill.** In the staggered check above the short
  prompt still waited 1.12 s. That matches two of the paper's chunks running before its two
  prefill units: one while it was admitted and one while it offered its prompt-frontier capture,
  which yields the lane's unit to other prefill. This decomposition is inferred from the request
  timings and was not traced. Letting a short prompt finish without yielding to another lane's
  chunk changes the active-capture transaction of the context cache.
- **First request after startup.** The first request's time to first token is about 550 ms and
  later ones about 130 ms. The startup warmup runs a 4-token prompt without caching; a global
  `CUDA_MODULE_LOADING=EAGER` and a longer warmup were tried and rejected earlier. Which first-use
  cost remains has not been attributed.
