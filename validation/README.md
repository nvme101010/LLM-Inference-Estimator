# Calibrating and checking the speed model

The page's speed figures come from a roofline model: each board's published memory bandwidth and dense compute, times the share of each that real serving reaches, plus the engine's fixed costs. This folder fits those shares and costs to published serving benchmarks and measures the fit on benchmarks it did not see, so the error the page quotes is measured, not guessed. Everything runs the model exactly as `index.html` does.

```bash
node validation/fetch-benchmarks.mjs   # rebuild benchmarks.json from the sources (optional)
node validation/fit.mjs --apply        # fit, cross-validate, and write CAL and RANGE into index.html (about 2 minutes)
node validation/compare.mjs --write    # compare the page as it stands and rewrite results.md
```

Node 18 or later, no dependencies. `compare.mjs --index <file>` checks another copy of the page, such as an earlier release.

## Sources

| Source | Model | Boards | Engine | Points |
|---|---|---|---|---|
| [NVIDIA NIM benchmarks](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html), version 1.8.0 tables | Llama 3.1 8B, FP8 and BF16 | H200 (DGX H200, sized as HGX H200), L40S | NIM (TensorRT-LLM) | 88 |
| [SemiAnalysis InferenceX](https://inferencex.semianalysis.com) (formerly InferenceMAX), [public API](https://inferencex.semianalysis.com/api), latest runs | Llama 3.3 70B FP8; gpt-oss-120b as released (MXFP4) | HGX H200, HGX B200, 1 to 8 GPUs per copy | vLLM 0.10.2 (Llama), 0.15–0.22 (gpt-oss) | 227 |
| InferenceX, as above: a check only, never fitted | the same two | AMD MI355X, which the page doesn't size; its MI350X is the same chip at a lower power limit, and its MI350P half of one | vLLM on ROCm | 91 |

Every source holds a fixed number of requests in flight, each sending the next as soon as the last is answered, with synthetic prompts that never repeat. That is the page's own load model with prompt caching off, so each point maps onto it directly: same model, board, weight and KV-cache precision, GPUs per copy, prompt and reply lengths, prefill piece size and requests in flight, all taken from the source's own settings. Each point in `benchmarks.json` names the table or the run it came from.

## The calibration (estimator 1.2.0)

**Board figures** are the vendors' own: memory bandwidth, and dense BF16 tensor compute (half the "with sparsity" figure NVIDIA headlines). Until 1.1.0 the page used hand-derated "effective" compute, about two thirds of the dense figure, and then took 45% of that.

**Fitted** (`fit.mjs`), as written into `CAL` in `index.html`:

| Constant | Value | Meaning |
|---|---|---|
| `mbu` | 0.85 | share of memory bandwidth decode reaches |
| `mfuDecode` | 0.20 | share of dense compute decode reaches, when it is compute-bound |
| `mfuPrefill` | 0.48 | share of dense compute prompt reading reaches |
| `fp8p` | 2 | FP8 compute speed against BF16 (2 on the datasheets) |
| `fp8d` | 0.82 | share of FP8 weights' halved bytes that decode realizes |
| `linkLatency` | 23 µs | fixed time per all-reduce over NVLink, two per layer when a copy is split |
| `step` | 1.4 ms | vLLM's fixed time per step (scheduling, sampling, launch) |
| `request` | 37 ms | vLLM's time to take a request in (HTTP, tokenization, scheduling), part of every first token |
| `rocm` | 0.75 | share of the bandwidth and compute shares reached on AMD's stack, fitted on MI355X runs (since 1.5.0; 0.68 before, see AMD below) |

Not fitted, since no benchmark here exercises them: the PCIe all-reduce latency (150 µs) and the precisions other than FP8 and the gpt-oss checkpoint's own.

**How.** The loss, per group of points (one source, model and board), is the mean pseudo-Huber log error of speed per user over every point that fits in memory, plus that of the first token at each curve's lowest concurrency. Errors beyond ±50% count linearly, so no single benchmark's quirks set the constants, and every group counts equally. The engines' fixed costs differ several-fold (TensorRT-LLM takes 3 ms to take a request in, against vLLM's 37), so each engine gets its own step and request time. The page ships vLLM's, as it models vLLM. The search runs Nelder–Mead from three starting points and keeps the best.

**Held out.** Each of the six groups is predicted by a fit made without it, so every figure below is out of sample:

| Group left out | Points | Speed per user | Output, all users | First token, lowest concurrency |
|---|---|---|---|---|
| All | 295 | 1.02 (0.80–1.31), 77% | 1.06 (0.82–1.38), 74% | 1.02 (0.66–1.49), 50% |
| InferenceX gpt-oss-120b, HGX B200 | 56 | 0.80 (0.68–0.99), 46% | 0.81 (0.70–1.02), 59% | 1.58 (0.76–2.18), 25% |
| InferenceX gpt-oss-120b, HGX H200 | 51 | 1.14 (1.02–1.30), 86% | 1.19 (1.08–1.56), 75% | 0.84 (0.62–1.31), 33% |
| InferenceX Llama 3.3 70B, HGX B200 | 58 | 1.04 (0.93–1.24), 93% | 1.09 (0.96–1.31), 86% | 0.71 (0.50–0.85), 25% |
| InferenceX Llama 3.3 70B, HGX H200 | 58 | 1.05 (0.92–1.24), 91% | 1.06 (0.94–1.28), 88% | 1.05 (1.02–1.14), 100% |
| NIM Llama 3.1 8B, HGX H200 | 44 | 1.23 (0.89–1.57), 52% | 1.30 (0.91–1.59), 41% | 1.17 (0.81–1.36), 50% |
| NIM Llama 3.1 8B, L40S | 28 | 0.92 (0.88–0.99), 100% | 0.92 (0.89–1.05), 100% | 1.10 (0.98–1.25), 83% |

Each cell is predicted ÷ measured: the median, the middle 80% in brackets, and the share within ±25%. The fitted constants hardly move from fold to fold (bandwidth share 0.83–0.85, prompt-reading share 0.44–0.59, all-reduce latency 20–24 µs), so no one group is carrying the fit. Fitted on all six, in sample, speed per user comes to 1.00 (0.87–1.18) with 91% within ±25%.

**The likely range.** For the middle 80% of held-out points, measured speed per user came to between 0.77 and 1.3 times the estimate, and the first token with nothing queued between 0.67 and 1.5 times. `fit.mjs --apply` writes these into `RANGE`. The page shows them as the likely range beside each figure, and judges the speed goal at the cautious end: speed at the low end, first token at the slow end. Held out, the central estimate promises more than 25% too much speed on 37 of 295 points; the cautious end does so on 3 (and by more than 10% on 19).

**AMD.** The MI355X's 8 GPUs are wired to each other with no switch, one Infinity Fabric link per pair (76.8 GB/s each way), so a run split across 2 or 4 of them uses only 1 or 3 links. Since 1.5.0 the model counts that (`mesh` in `GPUS`); before, every split got all 7 links' 538 GB/s. The constants fitted on NVIDIA boards promise 14% too much speed on the MI355X (1.14, 1.00–1.43). With one factor for AMD's stack, the estimate is checked by fitting that factor on one model's MI355X runs and predicting the other's: 0.99 (0.85–1.25), 81% within ±25%, as good as the NVIDIA held-out figures. The factor comes to 0.73 fitted on Llama 3.3 70B (vLLM 0.10, October 2025) and 0.78 fitted on gpt-oss (vLLM 0.22, 2026). With every split given all 7 links, those were 0.64 and 0.77, read at the time as ROCm catching up; most of that gap was the 70B runs' 2- and 4-GPU splits, which the gpt-oss runs barely use. The page ships 0.75, fitted on both, for the MI350P and MI350X.

**Recommendations.** Over 48 scenarios (four models × four presets × 8, 32 and 128 users), 23 get the same answer from 1.0.3, 1.1.0 and 1.2.0. Of the other 25, 1.2.0 gives 1.1.0's answer in 15 and lands between the two in 6. The cautious end (× 0.77) is close to the 20% that 1.1.0 ran slow by accident, so the sizes mostly match 1.1.0's, but now the margin is measured and the figures shown are central. Four need more than either version asked for:

- Llama 3.1 8B coding agent, 8 users: 1 × H200 NVL instead of an RTX PRO 6000.
- Llama 3.1 8B coding agent, 128 users: an HGX H200 system instead of 6 MI350Ps, once the ROCm factor applies.
- Llama 3.3 70B chat, 128 users: an HGX H200 system instead of 4 H200 NVL.
- gpt-oss-120b long-document summary, 8 users: an HGX H200 system instead of 4 H200 NVL, from the first token's slow end on 64K-token prompts.

## What still doesn't fit

- **gpt-oss on B200 comes out slow** (0.80 held out, 0.88 in sample), and its first tokens long. Its runs use vLLM 0.22 with all-reduce fusion, against the 0.10 behind the Llama runs: newer engines have lower fixed costs than one set of constants can carry.
- **Short first tokens on NIM come out long** in `results.md`, which compares the page as it stands: the page charges vLLM's 37 ms to take a request in, and TensorRT-LLM takes about 3.
- **First token with many requests in flight is not used.** These benchmarks start every request at once, so the first wave queues behind the others' prompts. In InferenceX runs above about 32 in flight, that burst makes the mean first token 2–3 × the median, and the 99th percentile about 10 ×. The comparison uses InferenceX's medians, which reflect the steady state; NIM publishes only means, which include the burst. The page's wait assumes requests arrive at random, so neither tests it. A benchmark at a random request rate (`vllm bench serve --request-rate`) would.
- **Boards with no benchmark** (L4, RTX PRO 4500 and 6000, H200 NVL, B300, MI350P) rest on the fitted shares applied to their datasheets, and the MI350P additionally on the ROCm factor.
- **Coverage.** Three models (dense and one mixture-of-experts); nothing on hybrid or latent-attention models, or with prompt caching on.

**Memory, for completeness:** of the 20 points the page would not size onto that many GPUs, 18 show first tokens of 2.8 s to 18 minutes, most of them 10–300 × the estimate: the engine holding requests back for want of cache space. The other two, Llama 3.3 70B on one H200 or one B200 with 64 requests of 1K in and 8K out, ran normally, so the page was cautious there.

## Before calibration

Speed per user, the same 295 points:

| | Median, middle 80% | Within ±25% | Over-promises by more than 10% / 25% / 50% |
|---|---|---|---|
| 1.0.3 | 0.92 (0.72–1.34) | 62% | 75 / 43 / 20 points |
| 1.1.0 | 0.80 (0.65–0.97) | 51% | 5 / 0 / 0 points |
| 1.2.0, held out, central estimate | 1.02 (0.80–1.31) | 77% | 103 / 37 / 12 points |
| 1.2.0, held out, cautious end (judges the goal) | 0.79 (0.62–1.01) | — | 19 / 3 / 0 points |

1.0.3's median looked close because two errors cancelled: decode constants that ran slow, and prompt load left out, which ran fast; with 5K–8K-token prompts it promised up to 1.9 × the measured speed. 1.1.0 added the prompt load and erred safe, but most of its remaining error was prompt reading 1.9–2.5 × too slow, from the derated compute figures, which it also charged against everyone's streaming. 1.2.0 shows the central estimate, accurate to within the ranges above, and keeps 1.1.0's caution where it matters, in the goal check.
