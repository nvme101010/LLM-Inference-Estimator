// Checks the page against NVIDIA's own RTX PRO 6000 Blackwell throughput tables (trtllm-rtx.json, from
// fetch-trtllm.mjs). Nothing is fitted: no benchmark the fit sees covers this board, so the page's
// figures for it are extrapolated, and this says how far they land from NVIDIA's.
// Usage: node validation/trtllm.mjs [--engine vllm] [--write] [--index path/to/index.html]
//   --engine vllm  uses the page's own fixed costs (vLLM's) instead of TensorRT-LLM's fitted ones
//   --write        also writes validation/trtllm-results.md (every figure)
//
// Each of NVIDIA's figures is an offline run: every request queued at once, so the engine keeps as many in
// flight as its KV cache holds, and replaces each finished request at once. That is the page's own load
// model at that many users, so for each figure on a model the page lists and a TP × PP layout it sizes:
//   - in flight: the most requests whose whole prompt and reply fit in the KV cache, which TensorRT-LLM
//     gives 95% of the memory left after weights, activations and overheads (the page's own memory
//     model, at batches of up to 8,192), and
//   - the page's output for everyone at that many, with NVIDIA's settings: NVFP4 weights, FP8 KV cache,
//     no prompt caching (random prompts), prompt pieces of 8,192 tokens.

import { readFileSync, writeFileSync } from "node:fs";
import { load, INDEX } from "./model.mjs";

const here = p => new URL(p, import.meta.url);
const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const P = load(arg("--index") ?? INDEX);
const engine = arg("--engine") === "vllm" ? "vllm" : "tensorrt-llm";
if(engine === "tensorrt-llm") Object.assign(P.CAL, JSON.parse(readFileSync(here("./calibration.json"), "utf8")).tensorrtLlm);
const { points, fetched, sources } = JSON.parse(readFileSync(here("./trtllm-rtx.json"), "utf8"));

const KV_SHARE = 0.95, BATCH_MAX = 8192, PIECE = 8192;
function check(p){
  const m = P.findModel(p.model), g = P.GPUS.find(b=> b.n===p.board);
  P.setModel(m);
  Object.assign(P.S, { wqLbl: "FP4", wq: 0.5625, kq: 1, cache: 0, thinking: 0, extra: [], workload: "custom",
    prompt: p.isl, reply: p.osl, shared: 0, fresh: null, chunk: PIECE, users: 1, goal: "any", ovpct: 7 });
  const holds = u => { const r = P.compute(p.tp, u, p.pp), rest = r.perGPU - r.kv_pg; return r.kv_pg <= KV_SHARE*(g.cap*P.GB - rest); };
  if(!holds(1)) return { fits: false };
  let lo = 1, hi = BATCH_MAX;
  if(holds(hi)) lo = hi;
  else while(hi - lo > 1){ const mid = (lo + hi) >> 1; if(holds(mid)) lo = mid; else hi = mid; }
  const t = P.throughput(g, p.tp, PIECE, lo, p.pp);
  return { fits: true, inFlight: lo, pred: t.total, ratio: t.total / p.tput, bottleneck: t.bottleneck };
}

const rows = points.filter(p=> p.model && p.tp).map(p=> ({ p, ...check(p) }));
const ok = rows.filter(r=> r.fits);

const q = (xs, f) => { const s = [...xs].sort((a, b)=> a - b), i = (s.length - 1)*f, lo = Math.floor(i);
  return s.length ? s[lo] + (s[Math.ceil(i)] - s[lo])*(i - lo) : NaN; };
const within = (xs, k) => xs.filter(r=> r >= 1/(1 + k) && r <= 1 + k).length / xs.length;
const f2 = x => x.toFixed(2);
const stat = xs => xs.length ? `${f2(q(xs, 0.5))} (${f2(q(xs, 0.1))}–${f2(q(xs, 0.9))}), ${Math.round(within(xs, 0.25)*100)}%` : "—";
const layout = p => `${p.gpus} GPU${p.gpus > 1 ? "s" : ""}, TP${p.tp} × PP${p.pp}`;
const line = (k, rs) => `| ${k} | ${rs.length} | ${stat(rs.map(r=> r.ratio))} |`;
const by = (key, rs = ok) => [...new Set(rs.map(key))].map(k=> line(k, rs.filter(r=> key(r)===k)));
const thead = `| Group | Figures | Output, all users: predicted ÷ NVIDIA |\n|---|---|---|\n`;
const kind = r => r.p.gpus === 1 ? "1 GPU" : r.p.pp > 1 && r.p.tp > 1 ? "TP and PP over PCIe" : r.p.pp > 1 ? "pipeline stages over PCIe (PP)" : "tensor-parallel over PCIe (TP)";
const head = `Estimator ${P.VERSION}, with ${engine === "vllm" ? "the page's own (vLLM's) fixed costs" : "TensorRT-LLM's fitted step and request times"}; NVIDIA's tables fetched ${fetched}. `
  + `${ok.length} of ${rows.length} figures on models and layouts the page sizes hold at least one request, and are summarized`
  + `${rows.length - ok.length ? ` (${rows.length - ok.length} don't fit by the page's memory model)` : ""}.\n\n`
  + `Each cell: median of predicted ÷ NVIDIA's figure (10th–90th percentile), and the share within ±25%. Below 1, the page predicts less output than NVIDIA measured.\n\n`
  + thead + [line("All", ok), ...by(r=> `TensorRT-LLM ${r.p.engine.split(" ")[1]}`)].join("\n")
  + `\n\nBy how the GPUs are used:\n\n` + thead + by(kind).sort().join("\n")
  + `\n\nBy model and layout:\n\n` + thead + by(r=> `${r.p.engine.split(" ")[1]} · ${r.p.name} · ${layout(r.p)}`).join("\n");
console.log(head);

if(process.argv.includes("--write")){
  const table = rows.map(r=> `| ${r.p.engine.split(" ")[1]} | ${r.p.name} | ${layout(r.p)} | ${r.p.isl}/${r.p.osl} | ${r.fits ? r.inFlight : "—"} `
    + `| ${commas(r.p.tput)} | ${r.fits ? commas(Math.round(r.pred)) : "—"} | ${r.fits ? f2(r.ratio) : "doesn't fit"} | ${r.fits ? r.bottleneck : ""} |`);
  writeFileSync(here("./trtllm-results.md"), `# The page against NVIDIA's RTX PRO 6000 throughput tables\n\n`
    + `Generated by \`node validation/trtllm.mjs --write\`. Source: NVIDIA's TensorRT-LLM performance overview, ${Object.entries(sources).map(([v, u])=> `[${v}](${u})`).join(" and ")}. `
    + `Check only: nothing here is fitted.\n\n${head}\n\n## Every figure\n\n`
    + `NVIDIA's figure is total output tokens/s of an offline run with every request queued at once. "In flight" is how many requests the page's memory model fits in TensorRT-LLM's KV cache; "predicted" is the page's output at that many.\n\n`
    + `| TensorRT-LLM | Model | Layout | In/out | In flight | NVIDIA | Predicted | Ratio | Limit |\n|---|---|---|---|---|---|---|---|---|\n` + table.join("\n") + "\n");
  console.log("\nwrote validation/trtllm-results.md");
}
function commas(x){ return x.toLocaleString("en-US"); }
