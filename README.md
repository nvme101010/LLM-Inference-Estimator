# LLM Inference Estimator

Single-node sizing estimate: how many GPUs one server needs for your model and users. Does not cover multi-node or NVL rack-scale systems.

A single-page tool for estimating what LLM inference needs. Pick a model, the number of concurrent users, a workload and a speed goal, and it recommends the node estimated to meet the goal with the least GPU memory in total:

- **Memory:** model weights, KV cache, activations and overhead, per GPU and in total, in GB or GiB.
- **Speed:** estimated tokens per second per user and time to first reply while new prompts keep arriving, including thinking time for models that reason before answering.
- **Capacity:** how many GPUs of each board one node takes, and how many users it can handle: what fits in memory, and what also reaches the speed goal on the estimate.

It covers 40+ popular open models (Qwen, Gemma, Llama, DeepSeek, gpt-oss, GLM, Kimi, Mistral and others) on NVIDIA L4, L40S, H200 NVL, HGX H200, HGX B200, HGX B300, RTX PRO 4500/6000 Blackwell, and AMD MI350P.

## Single node

Every result is one server of up to 8 GPUs. If one node of a board can't serve every user, its row says how many it does serve. For production, plan at least one more node (or replica) than this, so the service keeps running if one fails or is being updated.

GPU counts round up to what can be bought. H200 NVL servers take 1, 2, 4 or 8 cards, but an NVLink bridge joins at most 4 of them, so one copy of a model spans at most 4 and 8 cards run two copies. The HGX H200, HGX B200 and HGX B300 are sold only as 8-GPU systems, with all 8 GPUs joined through NVSwitch. RTX PRO 4500 servers hold up to 5 cards (the limit on HPE servers).

Separately, one copy of a model is split across 1, 2, 4 or 8 GPUs. Serving engines need the split to divide the model's attention heads evenly, so this is a software rule that applies to every board. On a 5-card RTX PRO 4500 server, one copy uses at most 4 cards and the fifth can run another copy.

**Speed under load:** each concurrent user is taken to send a new request as soon as the last is answered, so new prompts keep arriving while replies stream. Reading them takes a share of the engine's time, in steps shared with everyone's next token, so speed per user is the average over prompt reads and the time between them, and the first token includes the average wait for prompts queued ahead. Speed on this setup shows the request rate, how busy reading prompts keeps the engine, and the speed between prompt reads. Long new prompts (RAG, long documents) cost the most; chat barely notices.

**Prompt caching** is modelled the way vLLM and SGLang run by default: the repeated part of each prompt (a shared system prompt, earlier turns) is reused, so only the new part is read before the first reply, and the shared part is stored once per GPU. It can be turned off in Advanced settings.

**Several workloads:** one model often serves more than one kind of request at once, such as 16 chat users alongside 10 RAG users. **Add another workload** gives each its own number of concurrent users and its own prompt and reply lengths, up to four in all. They share the same copies of the model: memory adds up across them, and because one decode step serves the whole batch, every user streams at the same speed. Each workload has its own first-token time: the headline shows the slowest, and the speed goal is checked against it. Every workload's prompts join one queue, so each first token is the same average wait for prompts ahead plus the time to read that workload's own prompt. Workloads that need a different model are not covered.

**RAG pipelines:** the estimate covers the LLM only. RAG also needs an embedding model and usually a reranker, which aren't sized. Small ones (under ~1B parameters) add a few GB and can share a GPU or run on CPU at low traffic; a 4–8B reranker, or heavy traffic, needs its own GPU.

Not sure how many concurrent users to plan for? **Estimate from daily traffic** turns people, requests per day and hours of use into a concurrent-user count.

## Use it

Open `index.html` in a browser. It is one self-contained file with no build step and no server.

## Run it in a container

The image is nginx plus that one file, about 20 MB, running as a non-root user on port 8080.

```bash
docker compose up --build
```

Then open <http://localhost:8080>. The same image works anywhere a container runs:

```bash
docker build -t llm-inference-estimator .
docker run --rm -p 8080:8080 --read-only --tmpfs /tmp llm-inference-estimator
```

A prebuilt image for amd64 and arm64 is published to GHCR by the [Container workflow](.github/workflows/container.yml) on every merge to `main` (`:latest` and `:sha-<commit>`) and on every `v*` tag (`:1.2.3`, `:1.2`):

```bash
docker run --rm -p 8080:8080 --read-only --tmpfs /tmp ghcr.io/nvme101010/llm-inference-estimator:latest
```

`/healthz` answers `200 ok` for health checks. The page loads its two fonts from Google Fonts; without internet access it falls back to system fonts and everything else works offline.

### Kubernetes

[`deploy/k8s`](deploy/k8s) holds a Deployment (2 replicas, non-root, read-only root filesystem, no capabilities, readiness and liveness probes on `/healthz`, 10m CPU and 16 Mi memory requested), a Service on port 80, a PodDisruptionBudget, and an optional Ingress. Apply it from the repository or straight from GitHub:

```bash
kubectl apply -k "https://github.com/nvme101010/LLM-Inference-Estimator//deploy/k8s?ref=v1.0.3"
```

Then port-forward to try it (`kubectl port-forward svc/llm-inference-estimator 8080:80`) or set the host in `ingress.yaml` and add it to the kustomization. The image runs two nginx workers regardless of node size, so the 32 Mi limit holds everywhere.

## Share a sizing

- **Download summary** saves a one-page report of the recommendation, the inputs, every board compared and the assumptions. Open it and choose Print to save it as a PDF.
- **Copy link** appears when the page is served from the web, for example GitHub Pages. The address holds every setting, so the link opens the same scenario.

Both record the estimator's version (`v=` in the link). A link made with an older version opens with a note listing what has changed in the figures since, and the image tag to run to see them as they were (`ghcr.io/nvme101010/llm-inference-estimator:<version>`). Links made before versions were recorded are treated as 1.0.3.

When a release changes what the figures come to, bump `VERSION` in `index.html`, add a line to `CHANGES` saying what changed, and tag the release with the same version.

## Accuracy

Model and board data as of October 2026. These are planning estimates, not benchmarks. Memory figures follow each model's published architecture, and board memory is the size on the box, with 90% treated as usable (vLLM's default limit). Speed and first-token times are estimates from a simple roofline model of each board, with efficiency constants that have not been fitted to measurements, so a speed goal is met on the estimate only. They are averages over steady traffic at the server: they leave out the network round trip and tokenization, which can outweigh a short first token, and bursts of traffic make the wait for a first token longer. Real results vary with the serving engine and its settings. Before committing to hardware, confirm with a load test at your request rate (for example `vllm bench serve` against vLLM or SGLang).

**Measured error.** Against published benchmarks (295 points the page would size, of 315, from NVIDIA NIM and SemiAnalysis InferenceX: Llama 3.1 8B, Llama 3.3 70B and gpt-oss-120b on H200, B200 and L40S), the estimates err on the slow side:

- **Speed per user** comes out at 0.80 × measured (80% of points between 0.65 and 0.97). It is slowest with long prompts (0.73) and with one copy split across GPUs serving few users (about 0.6–0.75), so a setup the page says meets a speed goal usually does.
- **Prompt reading** is up to 2.5 × too slow, so first tokens for long prompts come out too long. Short first tokens come out too short, since the page leaves out the request's fixed costs.
- **Memory:** where the page says a load won't fit, the engine held requests back for want of cache space in 18 of 20 cases.

Method, sources, caveats and every point: [validation](validation/README.md). Rerun with `node validation/compare.mjs` after any change to the model.
