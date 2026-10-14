// Rebuilds benchmarks.json: published serving benchmarks the speed model is checked against.
// Usage: node validation/fetch-benchmarks.mjs   (Node 18+, no dependencies)
//
// Both sources hold a fixed number of requests in flight (each sends the next as soon as the last is
// answered), with synthetic prompts that never repeat, which is what the page models with prompt
// caching off. Only boards the page sizes are kept.

import { writeFileSync } from "node:fs";

const NIM_URL = "https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html";
const IX_API  = "https://inferencex.semianalysis.com/api/v1/benchmarks?model=";

/* NVIDIA NIM for Llama 3.1 8B Instruct, version 1.8.0 tables. The page carries its Markdown source,
   escaped, with tables nested in tabs: board > precision > lengths. */
const NIM_BOARDS = { "1 x H200 141GB": "HGX H200", "1 x L40s 48GB": "L40S" };   // DGX H200 is an HGX H200
async function nim(){
  const page = await (await fetch(NIM_URL)).text();
  const BS = "\\";
  const start = page.indexOf(`## Llama-3.1-8b-instruct Results${BS}n`);
  const end = page.indexOf(`## Llama-3.1-70b-instruct Results${BS}n`, start);
  if(start < 0 || end < 0) throw new Error("NIM page layout changed: Llama 3.1 8B section not found");
  const md = page.slice(start, end)
    .replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&")
    .split(`${BS}n`).join("\n").split(`${BS}"`).join('"');
  const points = [];
  let version = null, tabs = [];
  for(const line of md.split("\n")){
    let m;
    if((m = line.match(/^\s*### Version: (\S+)/))){ version = m[1]; tabs = []; continue; }
    if((m = line.match(/^(\s*)<Tab title="([^"]+)">/))){
      const depth = m[1].length;
      tabs = tabs.filter(t=> t.depth < depth).concat({depth, title: m[2]});
      continue;
    }
    m = line.match(/^\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/);
    if(!m || version !== "1.8.0") continue;
    const [board, prec, lens] = [tabs[0]?.title, tabs[1]?.title, tabs[2]?.title];
    if(!NIM_BOARDS[board]) continue;
    const [, isl, osl] = lens.match(/(\d+) Input Tokens \/ (\d+) Output Tokens/);
    const fp8 = /^fp8/.test(prec);
    points.push({
      src: "nim", engine: "NIM 1.8.0 (TensorRT-LLM)", model: "l31-8b", board: NIM_BOARDS[board],
      weights: fp8 ? "FP8" : "BF16", kv: fp8 ? "FP8" : "FP16", tp: +prec.match(/TP(\d+)/)[1],
      isl: +isl, osl: +osl, spread: 1, conc: +m[1], chunk: 8192,
      ttft: +m[2]/1000, tpot: +m[3]/1000, tput: +m[4],
      ref: `${board} / ${prec} / ${isl} in, ${osl} out`,
    });
  }
  return points;
}

/* SemiAnalysis InferenceX (formerly InferenceMAX): the latest single-node vLLM runs, plain tensor
   parallelism only (no expert parallelism, data-parallel attention, disaggregation or speculative
   decoding). Its runs draw each prompt and reply length uniformly from 80–100% of the stated length
   (random-range-ratio 0.8), and serve with prefix caching off and max-num-batched-tokens 8192. The
   KV cache is FP8 except gpt-oss on H200, which keeps vLLM's default (BF16): see
   benchmarks/70b_fp8_h200_slurm.sh at 9596d23 and benchmarks/single_node/gptoss_fp4_*.sh at f6de379
   in its repository. */
const IX_MODELS = { "Llama-3.3-70B-Instruct-FP8": {id: "l33-70b", precision: "fp8", weights: "FP8", kv: {h200: "FP8", b200: "FP8"}},
                    "gpt-oss-120b":               {id: "gptoss-120b", precision: "fp4", weights: "NATIVE", kv: {h200: "FP16", b200: "FP8"}} };   // MXFP4, as released
const IX_BOARDS = { h200: "HGX H200", b200: "HGX B200" };
async function inferencex(){
  const points = [];
  for(const [name, m] of Object.entries(IX_MODELS)){
    const rows = await (await fetch(IX_API + encodeURIComponent(name))).json();
    if(!Array.isArray(rows)) throw new Error(`InferenceX: ${JSON.stringify(rows)}`);
    for(const r of rows){
      if(!IX_BOARDS[r.hardware] || r.framework !== "vllm" || r.precision !== m.precision || r.disagg || r.is_multinode) continue;
      if(r.spec_method !== "none" || r.benchmark_type !== "single_turn" || (r.offload_mode && r.offload_mode !== "off")) continue;
      if(r.decode_ep !== 1 || r.decode_dp_attention || r.num_decode_gpu !== r.decode_tp) continue;
      /* Medians, not means: every request starts at once, so the first wave (a tenth of the requests,
         as each run sends 10 × concurrency) queues behind the others' prompts, and above ~32 in flight
         that burst drags the mean first token to 2–3 × the median. The median is the steady state. */
      const x = r.metrics, tp = r.decode_tp;
      points.push({
        src: "inferencex", engine: r.image || "vLLM", model: m.id, board: IX_BOARDS[r.hardware],
        weights: m.weights, kv: m.kv[r.hardware], tp, isl: r.isl, osl: r.osl, spread: 0.8, conc: r.conc, chunk: 8192,
        ttft: x.median_ttft, tpot: x.median_tpot, tput: x.output_tput_per_gpu * tp, ttft_mean: x.mean_ttft,
        ref: `${r.hardware} ${r.precision} TP${tp} / ${r.isl} in, ${r.osl} out`, date: r.date, run: r.run_url,
      });
    }
  }
  return points;
}

const points = [...await nim(), ...await inferencex()]
  .sort((a, b)=> a.src.localeCompare(b.src) || a.model.localeCompare(b.model) || a.board.localeCompare(b.board)
               || a.weights.localeCompare(b.weights) || a.tp - b.tp || a.isl - b.isl || a.osl - b.osl || a.conc - b.conc);
writeFileSync(new URL("./benchmarks.json", import.meta.url), JSON.stringify({
  fetched: new Date().toISOString().slice(0, 10),
  sources: {
    nim: `${NIM_URL} (Llama 3.1 8B Instruct, NIM 1.8.0)`,
    inferencex: `${IX_API}<model> (SemiAnalysis InferenceX, latest runs)`,
  },
  notes: "ttft and tpot are in seconds: means for NIM (all it publishes), medians for InferenceX (ttft_mean "
       + "keeps its mean); tput is output tokens a second for the whole copy (tp GPUs). Lengths are drawn from "
       + "spread × length up to length, so their mean is (1 + spread) / 2 × length.",
  points,
}, null, 1) + "\n");
console.log(`${points.length} points: ` + Object.entries(points.reduce((c, p)=> (c[`${p.src} ${p.model} ${p.board}`] = (c[`${p.src} ${p.model} ${p.board}`]||0) + 1, c), {}))
  .map(([k, v])=> `${k} ${v}`).join(", "));
