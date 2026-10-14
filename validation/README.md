# Checking the speed model against published benchmarks

The page's speed figures come from a roofline model with efficiency constants that were set by judgement, not fitted to measurements. This folder checks that model, exactly as `index.html` runs it, against published serving benchmarks, so the error is known rather than guessed. Nothing here has been used to tune the constants, so every point is out of sample.

```bash
node validation/fetch-benchmarks.mjs   # rebuild benchmarks.json from the sources (optional)
node validation/compare.mjs --write    # rerun the comparison and rewrite results.md
```

Node 18 or later, no dependencies. `compare.mjs` runs the page's own script, so rerun it after any change to the model.

## Sources

| Source | Model | Boards | Engine | Points |
|---|---|---|---|---|
| [NVIDIA NIM benchmarks](https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html), version 1.8.0 tables | Llama 3.1 8B, FP8 and BF16 | H200 (DGX H200, sized as HGX H200), L40S | NIM (TensorRT-LLM) | 88 |
| [SemiAnalysis InferenceX](https://inferencex.semianalysis.com) (formerly InferenceMAX), [public API](https://inferencex.semianalysis.com/api), latest runs | Llama 3.3 70B FP8; gpt-oss-120b as released (MXFP4) | HGX H200, HGX B200, 1 to 8 GPUs per copy | vLLM 0.10.2 (Llama), 0.15–0.22 (gpt-oss) | 227 |

Both hold a fixed number of requests in flight, each sending the next as soon as the last is answered, with synthetic prompts that never repeat. That is the page's own load model with prompt caching off, so each point maps onto it directly: same model, board, weight and KV-cache precision, GPUs per copy, prompt and reply lengths, piece size (8,192 tokens) and requests in flight. Boards the page doesn't size (H100, MI300 and others) are left out. Each point in `benchmarks.json` names the table or the run it came from.

## Results

Predicted ÷ measured over the 295 points that fit in memory by the page's own sizing: the median, the middle 80% in brackets, and the share within ±25%.

| | Speed per user | Output, all users | First token, lowest concurrency |
|---|---|---|---|
| All | 0.80 (0.65–0.97), 51% | 0.83 (0.66–1.00), 58% | 0.90 (0.26–2.25), 15% |
| Prompts up to 1K | 0.83 (0.71–0.99), 60% | 0.85 (0.72–1.02), 69% | 0.62 (0.24–1.82), 18% |
| Prompts of 5K–8K | 0.73 (0.63–0.89), 31% | 0.76 (0.63–0.95), 33% | 1.15 (0.41–2.45), 11% |
| Prompts of 20K | 0.73 (0.62–0.87), 33% | 0.74 (0.64–0.84), 33% | 2.33 (2.26–2.40), 0% |

By source, model and board, and every point, in [results.md](results.md).

**Speed per user errs slow, by about 20%.** The page predicts 0.80 × the measured speed (80% of points between 0.65 and 0.97), so a setup the page says meets a speed goal usually does, with room to spare, and some it rejects would pass. It is closest when one GPU holds the model and few requests are in flight (Llama 3.1 8B on one H200 with prompts up to 1K: 0.99–1.03; gpt-oss on one B200: 0.98–1.01), and furthest when:

- **one copy is split across GPUs with few requests in flight**, where steps are short and the fixed costs dominate: gpt-oss over 2 to 8 B200s comes out at 0.61–0.76, Llama 3.3 70B over 2 H200s at about 0.73. The per-layer all-reduce latency (0.03 ms per crossing over NVLink, two per layer) and the 1.5 ms per step look too high.
- **prompts are long**, because reading them is too slow (below), and under load that time comes out of everyone's streaming: 0.73 at 5K–20K-token prompts, down to 0.54 for Llama 3.1 8B with 5K prompts on one H200.

**Reading a prompt is too slow, by up to 2.5×.** With one request in flight, the first token for Llama 3.1 8B in FP8 comes out 1.9–2.5 × the measured time on both H200 and L40S for prompts of 1K tokens or more, and 1.5 × for Llama 3.3 70B on H200. On B200 it is close (1.02). The compute figures behind it are conservative: the page takes the H200 at 720 TFLOPs BF16 and FP8 at 1.4 × BF16 speed, against 989 TFLOPs and 2 × in NVIDIA's dense specifications, then applies 45% utilization on top.

**Short first tokens leave out fixed costs, as the page says.** For gpt-oss, the first token comes out at 0.25–0.43 × measured: the page has its 1K-token prompts read in 5–9 ms, and the 15–40 ms more that the benchmark measures is mostly the HTTP request, tokenization and scheduling, which the page leaves out, and says so.

**First token with many requests in flight is not compared.** These benchmarks start every request at once, so the first wave queues behind the others' prompts. In InferenceX runs above about 32 in flight, that burst makes the mean first token 2–3 × the median and the 99th percentile about 10 × the median; the comparison uses its medians, which reflect the steady state. NIM publishes only means, which include that burst. The page's wait assumes requests arrive at random, so neither is a clean test of it, and results.md reports those ratios without weighing them here. A benchmark at a random request rate (`vllm bench serve --request-rate`) would test it.

**Memory: where the page says it won't fit, the engine mostly ran out of room.** Of the 20 points the page would not size onto that many GPUs, 18 show first tokens of 2.8 s to 18 minutes, most of them 10–300 × the page's estimate: the engine holding requests back for want of cache space. The other two, Llama 3.3 70B on one H200 or one B200 with 64 requests of 1K in and 8K out, ran normally, so the page was cautious there.

## Before and after 1.1.0

Version 1.1.0 added the work of reading prompts that keep arriving. The same comparison run on 1.0.3's model (speed per user, the same 295 points):

| | Median, middle 80% | Within ±25% | Over-promises by more than 10% / 25% / 50% | Typical error |
|---|---|---|---|---|
| 1.0.3 | 0.92 (0.72–1.34) | 62% | 75 / 43 / 20 points | × 1.18 |
| 1.1.0 | 0.80 (0.65–0.97) | 51% | 5 / 0 / 0 points | × 1.24 |

1.0.3's median looks closer because two errors cancelled: decode constants that run slow, and prompt load left out, which runs fast. With 5K–8K-token prompts it promised up to 1.9 × the measured speed (90th percentile), so a setup it passed could miss the goal badly. 1.1.0's error is about as large but one-sided: it almost never promises more speed than was measured. Most of what remains is prompt reading being too slow, which 1.1.0 also charges against everyone's streaming, so fixing the compute constants is the next step.

## What this doesn't cover

- **Engines.** NIM runs TensorRT-LLM; the page assumes vLLM's defaults. Some of the NIM error is the engine, not the model.
- **Coverage.** Three models, three boards; nothing on RTX PRO, MI350P or B300, on hybrid or latent-attention models, or with prompt caching on.
- **Fitting.** These numbers measure the constants as they stand. Fitting them to these points would make this table look better without saying how the fit does on anything else; a fit should be checked on points it wasn't fitted to.
