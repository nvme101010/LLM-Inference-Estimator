// Rebuilds trtllm-rtx.json: NVIDIA's own throughput tables for the RTX PRO 6000 Blackwell Server Edition,
// from the TensorRT-LLM performance overview. Usage: node validation/fetch-trtllm.mjs   (Node 18+, no dependencies)
//
// These are not curves like benchmarks.json's: each figure is the total output throughput of an offline
// run (trtllm-bench throughput) with every request queued at once, so the engine keeps as many in flight
// as its KV cache holds (95% of free GPU memory, padded CUDA-graph batches up to 8,192), with fixed
// prompt and reply lengths and random tokens. trtllm.mjs checks the page against them; nothing is fitted.

import { writeFileSync } from "node:fs";

/* the documentation versions read, pinned so a rebuild gives the same points */
const VERSIONS = ["1.4.0rc1", "1.2.0rc6"];
const url = v => `https://nvidia.github.io/TensorRT-LLM/${v}/developer-guide/perf-overview.html`;

/* the page's models, by the names NVIDIA's tables use (lower case, "LLaMA v3.3" read as "llama 3.3");
   the checkpoints are NVIDIA's ModelOpt NVFP4 ones, with an FP8 KV cache (their hf_quant_config) */
const MODELS = { "llama 3.3 70b": "l33-70b", "llama 3.1 405b": "l31-405b", "qwen3 30b a3b": "q3-30b", "llama 4 scout": "l4-scout" };
const CHECKPOINTS = { "l33-70b": "nvidia/Llama-3.3-70B-Instruct-FP4", "l31-405b": "nvidia/Llama-3.1-405B-Instruct-FP4",
                      "q3-30b": "nvidia/Qwen3-30B-A3B-FP4", "l4-scout": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4" };
const norm = s => s.toLowerCase().replace(/llama v(\d)/, "llama $1").replace(/[-_]/g, " ").replace(/\s+/g, " ").trim();
const text = h => h.replace(/<a class="headerlink"[\s\S]*?<\/a>/g, "").replace(/<br\s*\/?>/g, " ").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/¶/g, "").replace(/\s+/g, " ").trim();

/* The page in order: headings, paragraphs and tables. A table is the RTX card's when its heading or the
   paragraph just before it names the RTX 6000 Pro; its model is the heading, less that suffix. */
function tables(html){
  const out = []; let heading = "", caption = "";
  for(const m of html.matchAll(/<h[1-6][^>]*>([\s\S]*?)<\/h[1-6]>|<p>([\s\S]*?)<\/p>|<table[\s\S]*?<\/table>/g)){
    if(m[1] !== undefined){ heading = text(m[1]); caption = ""; continue; }
    if(m[2] !== undefined){ caption = text(m[2]); continue; }
    if(/RTX 6000 Pro/i.test(heading + " " + caption)){
      const head = [...m[0].matchAll(/<th[^>]*>([\s\S]*?)<\/th>/g)].map(x=> text(x[1]));
      const rows = [...m[0].matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/g)].map(r=> [...r[1].matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map(x=> text(x[1]))).filter(r=> r.length);
      if(/Sequence Length/i.test(head[0] || "")) out.push({ name: heading.replace(/\s*-\s*RTX 6000 Pro.*$/i, ""), head, rows });
    }
    caption = "";
  }
  return out;
}

const points = [];
for(const v of VERSIONS){
  const html = await (await fetch(url(v))).text();
  const found = tables(html);
  if(!found.length) throw new Error(`TensorRT-LLM ${v}: no RTX 6000 Pro tables found; the page layout changed`);
  for(const { name, head, rows } of found){
    const model = MODELS[norm(name)] ?? null;
    head.slice(1).forEach((h, col)=>{
      const c = h.match(/(\d+)\s*GPUs?\s*(TP|DEP)(\d+)\s*,\s*PP(\d+)\s*\((\w+)\)/i);
      if(!c) throw new Error(`TensorRT-LLM ${v}, ${name}: column "${h}" not understood`);
      for(const r of rows){
        const [isl, osl] = r[0].split("/").map(Number), tput = +String(r[col + 1] || "").replace(/,/g, "");
        if(!tput) continue;                                            // a blank cell: not run
        points.push({ src: "trtllm-perf", engine: `tensorrt-llm ${v}`, name, model, checkpoint: CHECKPOINTS[model] ?? null,
          board: "RTX PRO 6000 Blackwell", weights: c[5].toUpperCase(), kv: "FP8", gpus: +c[1],
          tp: c[2].toUpperCase()==="TP" ? +c[3] : null, dep: c[2].toUpperCase()==="DEP" ? +c[3] : null, pp: +c[4],
          isl, osl, tput, ref: `${name} · ${h}`, doc: url(v) });
      }
    });
  }
  console.log(`TensorRT-LLM ${v}: ${found.length} RTX tables, ${points.filter(p=> p.engine.endsWith(v)).length} figures`);
}

writeFileSync(new URL("./trtllm-rtx.json", import.meta.url), JSON.stringify({
  fetched: new Date().toISOString().slice(0, 10),
  sources: Object.fromEntries(VERSIONS.map(v=> [v, url(v)])),
  notes: "Total output tokens/s from trtllm-bench throughput: offline, every request queued at once, fixed prompt and reply lengths, random tokens, KV cache 95% of free GPU memory (90% if that ran out), CUDA-graph batches up to 8,192. NVIDIA's NVFP4 checkpoints, FP8 KV cache (Llama 4 Scout's quantization config is gated; assumed the same). model is the estimator's id, or null for a model it doesn't list. dep: attention data-parallel with expert parallel (DEP), which the estimator doesn't model.",
  points,
}, null, 1) + "\n");
console.log(`wrote validation/trtllm-rtx.json: ${points.length} figures, ${points.filter(p=> p.model && p.tp).length} on models and layouts the page sizes`);
