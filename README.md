# LLM Inference Sizer

How many GPUs your model and users need: memory, speed and capacity. Does not cover NVL rack-scale systems.

A single-page tool for sizing LLM inference. Pick a model, the number of concurrent users, a workload and a speed goal, and it recommends the setup that meets the goal with the least GPU memory in total:

- **Memory:** model weights, KV cache, activations and overhead, per GPU and in total, in GB or GiB.
- **Speed:** estimated tokens per second per user and time to first reply, including thinking time for models that reason before answering.
- **Capacity:** how many GPUs of each board it takes, and how many users the setup can handle.

It covers 40+ popular open models (Qwen, Gemma, Llama, DeepSeek, gpt-oss, GLM, Kimi, Mistral and others) on NVIDIA L4, L40S, H100, H200, B200, B300, RTX PRO 4500/6000 Blackwell, and AMD MI350P.

GPU counts round up to what can be bought. H100 and H200 servers take 1, 2, 4 or 8 GPUs, with NVLink across 2 or 4 cards or all 8 in an 8-GPU system; beyond 8 they are counted in whole 8-GPU systems. The B200 and B300 are sold only as 8-GPU systems, so the tool counts them in whole systems.

Not sure how many concurrent users to plan for? **Estimate from daily traffic** turns people, requests per day and hours of use into a concurrent-user count.

## Use it

Open `index.html` in a browser. It is one self-contained file with no build step and no server.

## Share a sizing

- **Download summary** saves a one-page report of the recommendation, the inputs, every board compared and the assumptions. Open it and choose Print to save it as a PDF.
- **Copy link** appears when the page is served from the web, for example GitHub Pages. The address holds every setting, so the link opens the same scenario.

## Accuracy

These are planning estimates, not benchmarks. Memory figures follow each model's published architecture, and board memory is the size on the box, with about 90% treated as usable. Speed comes from a simple model of memory bandwidth, compute and interconnect; real serving engines usually land within about ±30%. Before committing to hardware, confirm with a real serving run (for example vLLM or SGLang).
