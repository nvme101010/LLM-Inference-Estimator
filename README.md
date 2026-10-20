# LLM Inference Estimator

Sizing estimate: how many GPUs your models and users need, with each model on one node at most. Does not cover models spread over several nodes, or NVL rack-scale systems.

A single-page tool for estimating what LLM inference needs. Pick a model, the number of concurrent users, a workload and a speed goal, and it recommends the node estimated to meet the goal with the least GPU memory in total. Add up to four models and it sizes them together on one kind of board:

- **Memory:** model weights, KV cache, activations and overhead, per GPU and in total, in GB or GiB.
- **Speed:** estimated tokens per second per user and time to first reply while new prompts keep arriving, each with a likely range measured against published benchmarks, including thinking time for models that reason before answering.
- **Capacity:** how many GPUs of each board one node takes, and how many users it can handle: what fits in memory, and what also reaches the speed goal on the estimate.
- **Server (optional, beta):** the CPUs, system memory and local storage around those GPUs, following NVIDIA's recommendations for certified inference servers.

It covers 40+ popular open models (Qwen, Gemma, Llama, DeepSeek, gpt-oss, GLM, Kimi, Mistral and others) on NVIDIA L4, L40S, H200 NVL, HGX H200, HGX B200, HGX B300, RTX PRO 4500/6000 Blackwell, and AMD MI350P and MI355X.

## Single node

Every result is one server of up to 8 GPUs. If one node of a board can't serve every user, its row says how many it does serve. For production, plan at least one more node (or replica) than this, so the service keeps running if one fails or is being updated.

GPU counts round up to what can be bought. H200 NVL servers take 1, 2, 4 or 8 cards, but an NVLink bridge joins at most 4 of them, so one copy of a model spans at most 4 and 8 cards run two copies. **Pipeline across bridge groups** in Advanced settings (off by default) lets one copy span both groups as two pipeline stages, tensor-parallel within each and handing activations between them over PCIe (vLLM's `--tensor-parallel-size 4 --pipeline-parallel-size 2`): that fits a model or a KV cache that one group can't hold, at about one group's speed per user, since each stage still reads all of its weights for every token; two copies of 4 are still preferred when they fit. Pipelined figures are modeled, not yet fitted to a benchmark. The HGX H200, HGX B200 and HGX B300 are sold only as 8-GPU systems, with all 8 GPUs joined through NVSwitch. So is the AMD MI355X, but with no switch: each GPU has one Infinity Fabric link to each of the other 7, so a copy split across 2 or 4 of them uses only the links among them and pays more per layer than on an HGX. RTX PRO 4500 servers hold up to 5 cards (the limit on HPE servers).

### Servers view

The answer opens by server. **Servers in scope** picks the HPE servers to consider (all by default; the customer's standard builds, say). The page sizes the model on every configuration of those servers — each server with each GPU it takes, at the count and wiring HPE supports — and shows:

- **the recommendation**: the configuration that meets the speed goal with the least GPU memory (then not split over PCIe, not pipelined, the most room, the fewest rack units), with its GPU bays, figures, `vllm serve` line and build checks;
- **step down / step up**: a smaller server that also works (or comes closest), the same server with its open bays filled, and a bigger server;
- **every server in scope**: the configurations that hold the model, and for the rest the most GPU memory they carry against what the model needs;
- **one configuration in detail**: how its cards are wired, every layout the wiring allows (TP within an NVLink group or NVSwitch; TP within groups × PP across them; on PCIe cards, TP over PCIe or PP over PCIe) at each number of copies, and its build.

Pipelined layouts are modeled, not yet fitted to a benchmark, and say so. The server catalog is the JSON block `id="catalog-hpe"` in `index.html`; edit it when HPE revises its QuickSpecs. URL keys: `vw=gpu` opens the GPU view, `hs` lists the servers in scope (dot-separated ids), `cf` the configuration in detail.

### HPE servers

Each board is named with the HPE server it is sized in, and **HPE server** in Advanced settings narrows the comparison to one server at its own limits (URL key `hs`). The list is every current HPE ProLiant server that takes one of these GPUs:

| HPE server | Boards (most cards per server) | Notes |
|---|---|---|
| ProLiant Compute DL380a Gen12 · 4U · 2 × Xeon 6 | L4, L40S, RTX PRO 6000 (8), H200 NVL | H200 NVL in two 4-way NVLink groups (S4A91C, supported only here, and not in the 10-card layout). Holds up to 10 double-wide / 16 single-wide; a node is sized at 8 |
| ProLiant Compute DL380 Gen12 · 2U · 2 × Xeon 6 | L4 (8), L40S (3), RTX PRO 4500 (5), RTX PRO 6000 (3\*), H200 NVL (3\*) | no NVLink bridge listed for H200 NVL here, so a split copy runs over PCIe |
| ProLiant DL385 Gen11 · 2U · 2 × EPYC 9004/9005 | L4 (8), L40S (4), RTX PRO 4500 (4), RTX PRO 6000 (4), H200 NVL (4), MI350P (2) | H200 NVL with the 2-way S4A90C bridge only: a copy spans 2, or 4 as TP 2 × 2 stages |
| ProLiant Compute DL345 Gen12 · 2U · 1 × EPYC 9005 | L4 (4), L40S (4), RTX PRO 4500 (4), RTX PRO 6000 (2) | GPU base unit with the Front 4xGPU kit |
| ProLiant Compute DL340 Gen12 · 2U · 1 × Xeon 6 | L4 (6) | |
| ProLiant Compute ML350 Gen12 · 4U tower · 2 × Xeon 6 | L4 (4), L40S (2), RTX PRO 4500 (5) | RTX PRO 4500 limited to 2 at 30 °C ambient |
| ProLiant ML110 Gen11 · tower · 1 × Xeon | L4 (2) | |
| ProLiant Compute DL320 Gen12 · 1U · 1 × Xeon 6 | L4 (4), L40S (2) | GPU base unit |
| ProLiant Compute DL325 Gen12 · 1U · 1 × EPYC 9005 | L4 (4) | GPU base unit; 5 at 25 °C ambient or below |
| ProLiant Compute DL360 Gen12 · 1U · 2 × Xeon 6 | L4 (3) | rear slots |
| ProLiant DL365 Gen11 · 1U · 2 × EPYC 9004/9005 | L4 (2), RTX PRO 4500 (2) | GPU base unit |
| ProLiant DL145 Gen11 · 2U edge · 1 × EPYC 8004 | L4 (2), L40S (1) | |
| ProLiant Compute XD685 · 5U DLC / 6U air · 2 × EPYC 9005 | HGX H200, HGX B200, HGX B300, MI355X | whole 8-GPU systems |

\* HPE added these to the DL380 Gen12 in September 2026 without a count; 3 is its double-wide limit (one per riser). The page marks them "count to confirm".

On **Auto** each board sits in the first server above that lists it. That matches the board's own figures, except the MI350P, which is now sized at the DL385 Gen11's 2 cards. Gen11 servers with a Gen12 successor that takes the same GPUs are left out. HPE also lists the RTX PRO 4500 for the DL145 Gen11, and the RTX PRO 4500 and 6000 for the EL2000 EL240 Gen12 and Edge Server e930t, without a card count, so those are named beside the picker but not sized. Storage, Superdome, Compute Scale-up and EL8000 systems are out of scope. Sources: HPE's *NVIDIA Accelerators for HPE* QuickSpecs (version 83, 8 September 2026) for which servers take which GPU and bridge, and each server's QuickSpecs or HPE Japan configuration guide for card counts; confirm a build against the current QuickSpecs.

Separately, one copy of a model is split across 1, 2, 4 or 8 GPUs. Serving engines need the split to divide the model's attention heads evenly, so this is a software rule that applies to every board. On a 5-card RTX PRO 4500 server, one copy uses at most 4 cards and the fifth can run another copy. When the GPUs bought are more than one copy needs, as with a small model on an 8-GPU HGX system, they run more copies rather than one copy split wider, unless only the wider split reaches the speed goal: copies leave room for far more users, and it is how small models are served in production.

**Speed under load:** each concurrent user is taken to send a new request as soon as the last is answered, so new prompts keep arriving while replies stream. Reading them takes a share of the engine's time, in steps shared with everyone's next token, so speed per user is the average over prompt reads and the time between them, and the first token includes the average wait for prompts queued ahead. Speed on this setup shows the request rate, how busy reading prompts keeps the engine, and the speed between prompt reads. Long new prompts (RAG, long documents) cost the most; chat barely notices.

**Prompt caching** is modelled the way vLLM and SGLang run by default: the repeated part of each prompt (a shared system prompt, earlier turns) is reused, so only the new part is read before the first reply, and the shared part is stored once per GPU. It can be turned off in Advanced settings.

**Several workloads:** one model often serves more than one kind of request at once, such as 16 chat users alongside 10 RAG users. **Add another workload** gives each its own number of concurrent users and its own prompt and reply lengths, up to four in all. They share the same copies of the model: memory adds up across them, and because one decode step serves the whole batch, every user streams at the same speed. Each workload has its own first-token time: the headline shows the slowest, and the speed goal is checked against it. Every workload's prompts join one queue, so each first token is the same average wait for prompts ahead plus the time to read that workload's own prompt. Workloads that need a different model are a second model (below).

**Several models:** **Add another model** sizes up to four models together, each with its own users, workloads, precision and speed goal. Tabs above the controls switch between them; overhead, prompt caching and Server apply to all. Each model runs as copies of its own on GPUs of its own, sized as it would be alone but before rounding up to a server size. The copies of every model are then packed into servers of one kind of board, the widest first, so every server but the last is full; a server that isn't full shrinks to the smallest size the board comes in, and its spare GPUs run extra copies of the models closest to their goals. Each model then uses the fewest GPUs per copy that still meet its goal, and still gets at most one node. The suggestion is the board where every model meets its goal with the least GPU memory in all. A Models panel shows how each model runs and the GPUs of every server; with Server set to NVIDIA guidance, each server is sized for the models on it, its storage holding each one's weights once.

**Thinking models:** the Thinking switch under Reply length sets how many tokens a model thinks before answering. **Users see the thinking** says which wait the speed goal is judged on. On (the default), the UI streams the thinking, so it's the first token. Off, users wait for the answer, so the tile becomes **Answer starts** and the goal's limit applies to that: within 15 s for Fast, 30 s for Smooth chat and 60 s for Readable, limits of the estimator's own, since a first reply in 1 to 5 s is out of reach once thousands of thinking tokens come first.

**RAG pipelines:** the estimate covers the LLM only. RAG also needs an embedding model and usually a reranker, which aren't sized. Small ones (under ~1B parameters) add a few GB and can share a GPU or run on CPU at low traffic; a 4–8B reranker, or heavy traffic, needs its own GPU.

Not sure how many concurrent users to plan for? **Estimate from daily traffic** turns people, requests per day and hours of use into a concurrent-user count.

## Server, by NVIDIA's guidance (beta)

This part is in beta: its rules and figures are still being worked on and may change.

Set **Server** to **NVIDIA guidance** in Advanced settings to size the whole server, following the inference server table in NVIDIA's [Certified Systems Configuration Guide](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-configuration-guide.html). GPU counts then round up to 2, 4 or 8 per server, split evenly across two CPU sockets (2 or 4 for the RTX PRO 4500). That can change which board is suggested: 56 chat users on Llama 3.1 8B move from 1 × RTX PRO 6000 to 2 × L40S. A Server panel under the answer gives what goes around those GPUs:

- **CPUs:** two sockets with PCIe Gen5 and a base clock of 2.0 GHz or more. Cores follow the AI worker nodes HPE builds for [HPE Private Cloud AI](https://www.hpe.com/us/en/collaterals/collateral.a50009216enw.html): 32 physical cores per GPU in a 2- or 4-GPU server and 16 per GPU in an 8-GPU one (2 × 32-core CPUs for 2 GPUs, 2 × 64-core for 4 or 8). NVIDIA's own floor is 6 per GPU.
- **Recommended minimum system memory:** twice the GPU memory, as NVIDIA recommends. How it's made up from memory modules is left to the vendor.
- **Slots and switches:** each GPU in an x16 slot of its own PCIe generation, behind PCIe Gen5 switches when there are 8.
- **Workload storage: the usable NVMe capacity the model needs**, not a drive configuration (drives and RAID are the vendor's choice): two copies of its weights (the version serving and the next one downloading) and 100 GB for images, caches and logs, kept under 80% full, beside a separate boot volume. What offloading KV cache to disk (NVIDIA Dynamo) would add, at least the larger of 100 GB and the GPUs' KV cache, is in the tooltip, since few single servers do it. NVIDIA's own guidance, one NVMe drive of at least 1 TB per socket, is in the tooltip.

NVIDIA's guide also covers network cards, remote management and TPM; the page leaves those to the conversation with the customer. The HGX H200, HGX B200 and HGX B300 come as whole systems from their manufacturer, so their CPUs, memory and storage are left to its spec sheet. The AMD boards are left out while Server is set to NVIDIA guidance, since the guide is written for NVIDIA GPUs; deleting `guide:false` from a board's entry in `GPUS` brings it back, sized by the same rules as a stand-in (the MI355X, a whole 8-GPU system, is left to its spec sheet like the HGX systems). These are planning minimums, not a certification: a server is NVIDIA-Certified only if it is on [NVIDIA's list](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html). With Server left on GPUs only, the GPUs are sized as if the option were not there.

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
kubectl apply -k "https://github.com/nvme101010/LLM-Inference-Estimator//deploy/k8s?ref=v1.6.0"
```

Then port-forward to try it (`kubectl port-forward svc/llm-inference-estimator 8080:80`) or set the host in `ingress.yaml` and add it to the kustomization. The image runs two nginx workers regardless of node size, so the 32 Mi limit holds everywhere.

## Share a sizing

- **Download summary** saves a one-page report of the recommendation, the inputs, every board compared and the assumptions. Open it and choose Print to save it as a PDF.
- **Copy link** appears when the page is served from the web, for example GitHub Pages. The address holds every setting, so the link opens the same scenario. With several models, model 1's settings keep their keys, models 2 to 4 use the same keys with the model's number appended (`m2`, `u2`, …), and `e` says which model's tab is open; a one-model link is as it always was.

Both record the estimator's version (`v=` in the link). A link made with an older version opens with a note listing what has changed in the figures since, and the image tag to run to see them as they were (`ghcr.io/nvme101010/llm-inference-estimator:<version>`). Links made before versions were recorded are treated as 1.0.3.

When a release changes what the figures come to, bump `VERSION` in `index.html`, add a line to `CHANGES` saying what changed, and tag the release with the same version.

## Accuracy

Model and board data as of October 2026. These are planning estimates, not benchmarks. Memory figures follow each model's published architecture, and board memory is the size on the box, with 90% treated as usable (vLLM's default limit).

Speed and first-token times come from a roofline model: each board's datasheet memory bandwidth and dense compute, times the shares of them serving reaches, plus vLLM's fixed costs per step and per request. Those shares and costs are fitted to 295 published benchmark points (NVIDIA NIM and SemiAnalysis InferenceX: Llama 3.1 8B, Llama 3.3 70B and gpt-oss-120b on H200, B200 and L40S), with a further factor for AMD's software stack fitted to MI355X runs. The figures are averages over steady traffic at the server: they leave out the network round trip, and bursts of traffic make the wait for a first token longer.

**Measured error.** Every figure here comes from benchmarks the fit did not see: each group of benchmarks was predicted by a fit made without it.

- **Speed per user** lands at 1.02 × measured (80% of points between 0.80 and 1.31, 77% within ±25%). The page shows that spread as the **likely** range, and judges the speed goal at its low end, so a setup it passes over-promised by more than 25% on 3 of 295 benchmark points.
- **First token** with nothing queued lands at 1.02 × measured (80% between 0.66 and 1.49), and the goal is judged at the slow end of that range. The wait for other prompts under load is modelled; no benchmark here tests it.
- **AMD:** with the ROCm factor fitted on one model's MI355X runs, the other model's land at 0.99 × measured (80% between 0.85 and 1.25).
- **Memory:** where the page says a load won't fit, the engine held requests back for want of cache space in 18 of 20 cases.

Boards with no benchmark of their own (L4, the RTX PRO cards, H200 NVL, B300, MI350P) rest on the same fitted shares applied to their datasheets. The page marks their speed and first-token figures **extrapolated**, in the headline and the board table, and the summary stars them. Real results vary with the serving engine and its settings: before committing to hardware, confirm with a load test at your request rate (for example `vllm bench serve` against vLLM or SGLang).

Method, the fitted constants, sources, caveats and every point: [validation](validation/README.md). After a change to the model, refit with `node validation/fit.mjs --apply` and recheck with `node validation/compare.mjs --write`. `node validation/sweep.mjs` lists which sizings changed since the last release.
