// The page's own speed and memory model, loaded from index.html, and one benchmark point run through it.
// Its script, from the start up to the display helpers, runs here against a stand-in for the page, so
// what is checked is exactly what the page shows. CAL (the efficiency constants) is returned live:
// change its fields and the next prediction uses them.

import { readFileSync } from "node:fs";

export const INDEX = new URL("../index.html", import.meta.url);

/* Boards the page doesn't size but benchmarks cover, used only to check the model: AMD's MI355X (vLLM on
   ROCm). The page's MI350X is the same chip at a lower power limit, and its MI350P half of one. Figures
   from AMD's MI355X brochure: 288 GB HBM3E at 8 TB/s, 2.5 PFLOPS dense BF16. Its 8 GPUs are wired to each
   other directly (no switch), one Infinity Fabric link per pair at 76.8 GB/s each way (AMD: 153.6 GB/s
   per link), so a copy on tp of them uses tp − 1 links (mesh). */
export const CHECK_ONLY = [
  {n:"MI355X", cap:288, mem:"HBM3E", bw:8000, flops:2500, fp8:true, fp4:true, link:true, lbw:76.8, mesh:true, lname:"Infinity Fabric", stack:"rocm", checkOnly:true},
];
export const checkOnly = p => CHECK_ONLY.some(b=> b.n===p.board);

export function load(path = INDEX){
  const html = readFileSync(path, "utf8").split(/\r?\n/);
  const from = html.findIndex(l => l.startsWith("<script>")) + 1, to = html.findIndex(l => l.startsWith("const fmtTps"));
  if(from < 1 || to < from) throw new Error("index.html layout changed: script or fmtTps line not found");
  const el = () => new Proxy(function(){}, { get: (t, k)=> k==="value" ? "" : k==="style" || k==="dataset" ? {} : el(), apply: ()=> el(), set: ()=> true });
  globalThis.document = { getElementById: el, querySelector: el, querySelectorAll: ()=> [], createElement: el, documentElement: el() };
  globalThis.localStorage = { getItem: ()=> null, setItem: ()=> {} };
  const P = new Function(html.slice(from, to).join("\n")
    + ";return {S, GPUS, GB, USABLE, compute, throughput, findModel, setModel: m=>{ M = m; },"
    + " VERSION: typeof VERSION==='undefined' ? '1.0.3 or earlier' : VERSION, CAL: typeof CAL==='undefined' ? null : CAL};")();
  P.GPUS.push(...CHECK_ONLY.map(b=> ({...b})));      // after the page's own boards, and only here
  return P;
}

const BYTES = { FP8: 1, BF16: 2, FP16: 2 };

/* one benchmark point: prompt caching off (the prompts are random), no thinking, the run's mean lengths,
   its split, its piece size and its number of requests in flight */
export function predict(P, p){
  const m = P.findModel(p.model), g = P.GPUS.find(b=> b.n===p.board);
  if(!m || !g) throw new Error(`unknown model or board in ${JSON.stringify(p)}`);
  P.setModel(m);
  const mean = len => Math.round(len*(1 + p.spread)/2);
  Object.assign(P.S, { wqLbl: p.weights, wq: p.weights==="NATIVE" ? m.nat.gb/m.p : BYTES[p.weights], kq: BYTES[p.kv],
    cache: 0, thinking: 0, extra: [], workload: "custom", prompt: mean(p.isl), reply: mean(p.osl), shared: 0, fresh: null,
    chunk: p.chunk, users: p.conc, goal: "any", ovpct: 7 });
  const t = P.throughput(g, p.tp, p.chunk, p.conc);
  const perGB = P.compute(p.tp, p.conc).perGPU / P.GB;
  const rate = t.rate ?? p.conc / (t.ttft + P.S.reply/t.perUser);      // estimators before 1.1.0 give no rate
  return { speed: t.perUser, ttft: t.ttft, tput: rate * P.S.reply, fits: perGB <= g.cap*P.USABLE, perGB, busy: t.busy };
}

/* each benchmark curve's lowest concurrency: its first token is the prompt read with almost nothing queued */
export function markIdle(points){
  const key = p => [p.src, p.model, p.board, p.weights, p.kv, p.tp, p.isl, p.osl].join("|"), lowest = {};
  points.forEach(p=> { lowest[key(p)] = Math.min(lowest[key(p)] ?? Infinity, p.conc); });
  return points.map(p=> ({...p, idle: p.conc === lowest[key(p)]}));
}

export const group = p => `${p.src} · ${p.model} · ${p.board}`;
