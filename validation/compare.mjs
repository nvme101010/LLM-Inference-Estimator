// Runs the page's own speed model on every point in benchmarks.json and reports how far it lands from
// the measured figures. Usage: node validation/compare.mjs [--write]
//   --write  also writes validation/results.md (the full table)
//
// The model is taken from index.html itself: its script, from the start up to the display helpers,
// runs here against a stand-in for the page, so this checks exactly what the page shows.

import { readFileSync, writeFileSync } from "node:fs";

const here = p => new URL(p, import.meta.url);
const html = readFileSync(here("../index.html"), "utf8").split(/\r?\n/);
const from = html.findIndex(l => l.startsWith("<script>")) + 1, to = html.findIndex(l => l.startsWith("const fmtTps"));
if(from < 1 || to < from) throw new Error("index.html layout changed: script or fmtTps line not found");
const el = () => new Proxy(function(){}, { get: (t, k)=> k==="value" ? "" : k==="style" || k==="dataset" ? {} : el(), apply: ()=> el(), set: ()=> true });
globalThis.document = { getElementById: el, querySelector: el, querySelectorAll: ()=> [], createElement: el, documentElement: el() };
globalThis.localStorage = { getItem: ()=> null, setItem: ()=> {} };
const P = new Function(html.slice(from, to).join("\n")
  + ";return {S, GPUS, GB, USABLE, VERSION, compute, throughput, findModel, setModel: m=>{ M = m; }};")();

const { points, fetched, sources } = JSON.parse(readFileSync(here("./benchmarks.json"), "utf8"));
const BYTES = { FP8: 1, BF16: 2, FP16: 2 };

/* one benchmark point through the model: prompt caching off (the prompts are random), no thinking,
   the run's mean lengths, its split, its piece size and its number of requests in flight */
function predict(p){
  const m = P.findModel(p.model), g = P.GPUS.find(b=> b.n===p.board);
  if(!m || !g) throw new Error(`unknown model or board in ${JSON.stringify(p)}`);
  P.setModel(m);
  const mean = len => Math.round(len*(1 + p.spread)/2);
  Object.assign(P.S, { wqLbl: p.weights, wq: p.weights==="NATIVE" ? m.nat.gb/m.p : BYTES[p.weights], kq: BYTES[p.kv],
    cache: 0, thinking: 0, extra: [], workload: "custom", prompt: mean(p.isl), reply: mean(p.osl), shared: 0, fresh: null,
    chunk: p.chunk, users: p.conc, goal: "any", ovpct: 7 });
  const t = P.throughput(g, p.tp, p.chunk, p.conc);
  const perGB = P.compute(p.tp, p.conc).perGPU / P.GB;
  return { speed: t.perUser, ttft: t.ttft, tput: t.rate * P.S.reply, fits: perGB <= g.cap*P.USABLE, perGB, busy: t.busy };
}

const rows = points.map(p=> {
  const x = predict(p);
  return { p, x, speed: x.speed * p.tpot, ttft: x.ttft / p.ttft, tput: x.tput / p.tput };   // predicted ÷ measured
});

/* The first token at each curve's lowest concurrency (1 for NIM, 4 for InferenceX) is the prompt read
   with almost nothing queued; above it, the measured first token also holds the benchmark's queueing,
   which the page models for random arrivals (see validation/README.md). */
const curve = p => [p.src, p.model, p.board, p.weights, p.kv, p.tp, p.isl, p.osl].join("|");
const lowest = {};
rows.forEach(({p})=> { lowest[curve(p)] = Math.min(lowest[curve(p)] ?? Infinity, p.conc); });
rows.forEach(r=> { r.idle = r.p.conc === lowest[curve(r.p)]; });

/* summaries over the points that fit in memory by the page's own sizing (the others it would never
   suggest): the median ratio, the middle 80% of ratios, and the share within ±25% */
const q = (xs, f) => { const s = [...xs].sort((a, b)=> a - b), i = (s.length - 1)*f, lo = Math.floor(i);
  return s.length ? s[lo] + (s[Math.ceil(i)] - s[lo])*(i - lo) : NaN; };
const within = (xs, k) => xs.filter(r=> r >= 1/(1 + k) && r <= 1 + k).length / xs.length;
const f2 = x => x.toFixed(2);
const stat = xs => xs.length ? `${f2(q(xs, 0.5))} (${f2(q(xs, 0.1))}–${f2(q(xs, 0.9))}), ${Math.round(within(xs, 0.25)*100)}%` : "—";
const line = (k, rs) => `| ${k} | ${rs.length} | ${stat(rs.map(r=> r.speed))} | ${stat(rs.map(r=> r.tput))} `
  + `| ${stat(rs.filter(r=> r.idle).map(r=> r.ttft))} | ${stat(rs.filter(r=> !r.idle).map(r=> r.ttft))} |`;
const fit = rows.filter(r=> r.x.fits);
const by = (label, key) => [...new Set(fit.map(key))].map(k=> line(label(k), fit.filter(r=> key(r)===k)));
const thead = `| Group | Points | Speed per user | Output, all users | First token, lowest concurrency | First token, more in flight |\n|---|---|---|---|---|---|\n`;
const lengths = r => r.p.isl <= 1024 ? "a: prompts up to 1K" : r.p.isl <= 8192 ? "b: prompts of 5K–8K" : "c: prompts of 20K";
const head = `Estimator ${P.VERSION}, benchmarks fetched ${fetched}. ${fit.length} of ${rows.length} points fit in memory by the page's sizing and are summarized.\n\n`
  + `Each cell: median of predicted ÷ measured (10th–90th percentile), and the share of points within ±25%. Below 1, the page predicts slower speed, lower output or a shorter first token than measured.\n\n`
  + thead + [line("All", fit), ...by(k=> k, r=> `${r.p.src} · ${r.p.model} · ${r.p.board}`)].join("\n")
  + `\n\nBy prompt length:\n\n` + thead + by(k=> k.slice(3), lengths).sort().join("\n");
console.log(head);

if(process.argv.includes("--write")){
  const ms = s => s >= 10 ? `${s.toFixed(1)} s` : s >= 1 ? `${s.toFixed(2)} s` : `${Math.round(s*1000)} ms`;
  const table = rows.map(({p, x, speed, ttft, tput})=> `| ${p.src} | ${p.model} | ${p.board} | ${p.weights}/${p.kv} KV | ${p.tp} | ${p.isl}/${p.osl} | ${p.conc} `
    + `| ${(1/p.tpot).toFixed(1)} | ${x.speed.toFixed(1)} | ${f2(speed)} | ${ms(p.ttft)} | ${ms(x.ttft)} | ${f2(ttft)} `
    + `| ${Math.round(p.tput)} | ${Math.round(x.tput)} | ${f2(tput)} | ${x.fits ? "" : "no"} |`);
  writeFileSync(here("./results.md"), `# Speed model against published benchmarks\n\n`
    + `Generated by \`node validation/compare.mjs --write\`. Sources: ${Object.values(sources).join("; ")}.\n\n${head}\n\n`
    + `## Every point\n\nRatios are predicted ÷ measured. Speed is tok/s per user (1 ÷ time per output token) and first token is per request, both means for NIM and medians for InferenceX; output is tok/s for the whole copy. "Fits: no" marks a point the page would not size onto that many GPUs (left out of the summary).\n\n`
    + `| Source | Model | Board | Precision | TP | In/out | Conc | Speed meas. | Speed pred. | Ratio | TTFT meas. | TTFT pred. | Ratio | Output meas. | Output pred. | Ratio | Fits |\n`
    + `|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n` + table.join("\n") + "\n");
  console.log("\nwrote validation/results.md");
}
