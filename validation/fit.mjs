// Fits the page's efficiency constants (CAL in index.html) to the published benchmarks, and measures
// the fit on data it did not see. Usage: node validation/fit.mjs [--apply]
//   --apply  rewrites the CAL line in index.html with the fitted values (rounded)
//
// What is fitted: the share of memory bandwidth and of dense compute that serving reaches, the fixed
// time per all-reduce over NVLink, and FP8's realized savings and speed, shared by every benchmark; and
// the engine's own fixed costs, the time per step and the time to take a request in, separately for
// vLLM (InferenceX) and TensorRT-LLM (NIM), since they differ several-fold. The page ships vLLM's, as it
// models vLLM. Board figures stay the vendors' own; PCIe all-reduce latency and the other precisions stay
// as they were, since no benchmark here exercises them.
//
// The loss, per group of points (one source, model and board), is the mean pseudo-Huber log error of
// speed per user over every point that fits in memory, plus that of the first token at each curve's
// lowest concurrency (the prompt read with almost nothing queued; see README.md for why the rest of the
// first tokens are left out). Errors past ±50% count linearly, not squared, so no one benchmark's quirks
// set the constants. Groups count equally, so the largest doesn't dominate.
//
// Honesty check: leave one group out, fit on the rest, predict the group left out; repeat for each. Those
// held-out errors are the ones to quote. The constants shipped are then fitted on every group.

import { readFileSync, writeFileSync } from "node:fs";
import { load, predict, markIdle, group, checkOnly, INDEX } from "./model.mjs";

const P = load();
const { points: all } = JSON.parse(readFileSync(new URL("./benchmarks.json", import.meta.url), "utf8"));
const fitting = markIdle(all).filter(p=> predict(P, p).fits);      // memory doesn't depend on CAL
const points = fitting.filter(p=> !checkOnly(p));                    // the page's boards: fitted and cross-validated
const amd = fitting.filter(checkOnly);                               // MI355X: never fitted, only predicted
const groups = [...new Set(points.map(group))];

/* fitted constants, each mapped into a plausible range so the search can't wander outside it */
const SHARED = {
  mbu:         [0.3, 0.95],       // share of memory bandwidth (real memory peaks out below 100%)
  mfuDecode:   [0.1, 1],          // share of dense compute, decode
  mfuPrefill:  [0.1, 1],          // share of dense compute, prompt reading
  linkLatency: [1e-6, 100e-6],    // s per all-reduce over NVLink
  fp8d:        [0.6, 1],          // share of FP8's halved bytes realized
  fp8p:        [1, 2],            // FP8 compute speed against BF16 (2 on the datasheets)
};
const ENGINE = { step: [0.1e-3, 5e-3], request: [0, 0.1] };          // s per step, s to take a request in
const ENGINE_OF = { inferencex: "vllm", nim: "trtllm" };
const DIMS = [...Object.keys(SHARED).map(k=> ({k, range: SHARED[k]})),
              ...["vllm", "trtllm"].flatMap(e=> Object.keys(ENGINE).map(k=> ({k, e, range: ENGINE[k]})))];
const decode = u => { const t = {shared: {}, vllm: {}, trtllm: {}};
  DIMS.forEach(({k, e, range: [lo, hi]}, i)=> { (e ? t[e] : t.shared)[k] = lo + (hi - lo)/(1 + Math.exp(-u[i])); });
  return t; };
const encode = t => DIMS.map(({k, e, range: [lo, hi]})=> { const f = Math.min(0.999, Math.max(0.001, ((e ? t[e] : t.shared)[k] - lo)/(hi - lo))); return Math.log(f/(1 - f)); });
const calFor = (t, p) => ({...t.shared, ...t[ENGINE_OF[p.src]]});

const DELTA = Math.log(1.5);                                        // quadratic within ±50%, linear beyond
const huber = e => DELTA*DELTA*(Math.sqrt(1 + (e/DELTA)**2) - 1);
function loss(t, pts){
  let sum = 0, n = 0;
  for(const g of new Set(pts.map(group))){
    const gp = pts.filter(p=> group(p)===g), idle = gp.filter(p=> p.idle);
    let s = 0, f = 0;
    for(const p of gp){
      Object.assign(P.CAL, calFor(t, p));
      const x = predict(P, p);
      s += huber(Math.log(x.speed*p.tpot));
      if(p.idle) f += huber(Math.log(x.ttft/p.ttft));
    }
    sum += s/gp.length + (idle.length ? f/idle.length : 0); n++;
  }
  return sum / n;
}

/* Nelder–Mead in the mapped space, restarted from its own best until it stops improving */
function minimize(f, x0, iters = 1500){
  let best = { x: x0, v: f(x0) };
  for(let restart = 0; restart < 5; restart++){
    const n = x0.length;
    let simplex = [best.x, ...best.x.map((_, i)=> best.x.map((v, j)=> j===i ? v + 0.6 : v))].map(x=> ({x, v: f(x)}));
    for(let it = 0; it < iters; it++){
      simplex.sort((a, b)=> a.v - b.v);
      if(simplex[n].v - simplex[0].v < 1e-9) break;
      const c = simplex[0].x.map((_, j)=> simplex.slice(0, n).reduce((s, p)=> s + p.x[j], 0)/n);
      const at = k => c.map((cj, j)=> cj + k*(simplex[n].x[j] - cj)), worst = simplex[n];
      const r = {x: at(-1)}; r.v = f(r.x);
      if(r.v < simplex[0].v){ const e = {x: at(-2)}; e.v = f(e.x); simplex[n] = e.v < r.v ? e : r; }
      else if(r.v < simplex[n - 1].v) simplex[n] = r;
      else {
        const k = {x: at(r.v < worst.v ? -0.5 : 0.5)}; k.v = f(k.x);
        if(k.v < Math.min(r.v, worst.v)) simplex[n] = k;
        else simplex = simplex.map((p, i)=> i===0 ? p : (q=> ({x: q, v: f(q)}))(p.x.map((v, j)=> simplex[0].x[j] + 0.5*(v - simplex[0].x[j]))));
      }
    }
    simplex.sort((a, b)=> a.v - b.v);
    const better = simplex[0].v < best.v - 1e-7;
    if(simplex[0].v < best.v) best = simplex[0];
    if(!better) break;
  }
  return best;
}
/* three starting points: the page's constants, one near published serving overheads, one in between;
   the best of the three wins */
const START = {...P.CAL};
const STARTS = [
  { shared: {mbu: START.mbu, mfuDecode: START.mfuDecode, mfuPrefill: START.mfuPrefill, linkLatency: START.linkLatency, fp8d: START.fp8d, fp8p: START.fp8p},
    vllm: {step: START.step, request: START.request || 1e-3}, trtllm: {step: START.step, request: START.request || 1e-3} },
  { shared: {mbu: 0.8, mfuDecode: 0.3, mfuPrefill: 0.5, linkLatency: 15e-6, fp8d: 0.9, fp8p: 1.7},
    vllm: {step: 1e-3, request: 0.02}, trtllm: {step: 0.5e-3, request: 0.005} },
  { shared: {mbu: 0.7, mfuDecode: 0.5, mfuPrefill: 0.35, linkLatency: 30e-6, fp8d: 0.95, fp8p: 1.5},
    vllm: {step: 2e-3, request: 0.01}, trtllm: {step: 1e-3, request: 0.01} },
];
const fit = pts => STARTS.map(s=> minimize(u=> loss(decode(u), pts), encode(s)))
  .reduce((a, b)=> b.v < a.v ? b : a);
const BEFORE = STARTS[0];

/* ratios predicted ÷ measured for some points under some constants */
function ratios(t, pts){
  return pts.map(p=> { Object.assign(P.CAL, calFor(t, p)); const x = predict(P, p);
    return { p, speed: x.speed*p.tpot, ttft: x.ttft/p.ttft, tput: x.tput/p.tput }; });
}
const q = (xs, f) => { const s = [...xs].sort((a, b)=> a - b), i = (s.length - 1)*f, lo = Math.floor(i);
  return s.length ? s[lo] + (s[Math.ceil(i)] - s[lo])*(i - lo) : NaN; };
const within = (xs, k) => xs.filter(r=> r >= 1/(1 + k) && r <= 1 + k).length / xs.length;
const summary = xs => xs.length ? { median: q(xs, 0.5), p10: q(xs, 0.1), p90: q(xs, 0.9), within25: within(xs, 0.25), n: xs.length } : null;
const f2 = x => x.toFixed(2);
const cell = s => s ? `${f2(s.median)} (${f2(s.p10)}–${f2(s.p90)}), ${Math.round(s.within25*100)}%` : "—";
const table = (rows, label) => {
  const line = (k, rs)=> `| ${k} | ${rs.length} | ${cell(summary(rs.map(r=> r.speed)))} | ${cell(summary(rs.map(r=> r.tput)))} | ${cell(summary(rs.filter(r=> r.p.idle).map(r=> r.ttft)))} |`;
  return `| ${label} | Points | Speed per user | Output, all users | First token, lowest concurrency |\n|---|---|---|---|---|\n`
    + [line("All", rows), ...[...new Set(rows.map(r=> group(r.p)))].map(g=> line(g, rows.filter(r=> group(r.p)===g)))].join("\n");
};
const unit = (k, v) => k==="step" || k==="request" ? `${(v*1e3).toFixed(2)} ms` : k==="linkLatency" ? `${(v*1e6).toFixed(1)} µs` : v.toFixed(3);
const show = t => Object.entries(t.shared).map(([k, v])=> `${k} ${unit(k, v)}`).join(" · ")
  + ` | vLLM: step ${unit("step", t.vllm.step)}, request ${unit("request", t.vllm.request)}`
  + ` | TensorRT-LLM: step ${unit("step", t.trtllm.step)}, request ${unit("request", t.trtllm.request)}`;

console.log(`${points.length} points in ${groups.length} groups.\n`);
console.log("Before fitting (the page's constants, board figures from the datasheets):\n\n" + table(ratios(BEFORE, points), "Group") + "\n");

/* leave one group out */
const heldOut = [];
for(const g of groups){
  const t = decode(fit(points.filter(p=> group(p)!==g)).x);
  heldOut.push(...ratios(t, points.filter(p=> group(p)===g)));
  console.log(`  without ${g}:\n    ${show(t)}`);
}
console.log("\nHeld out (each group predicted by a fit that never saw it):\n\n" + table(heldOut, "Group left out") + "\n");
const over = (rs, at = 1) => { const xs = rs.map(r=> r.speed*at); return Object.fromEntries([1.1, 1.25, 1.5].map(k=> [`above${k}`, xs.filter(x=> x > k).length])); };
const lowEnd = 1/q(heldOut.map(r=> r.speed), 0.9);                 // the speed goal is judged at estimate × this
console.log(`Held-out speed over-promised by more than 10% / 25% / 50%: ${Object.values(over(heldOut)).join(" / ")} of ${heldOut.length} points`
  + `; at the low end of the likely range (× ${lowEnd.toFixed(2)}), which the speed goal is judged on: ${Object.values(over(heldOut, lowEnd)).join(" / ")}\n`);

/* the shipped constants: fitted on everything, rounded as they'll appear in index.html; the page models
   vLLM, so it takes vLLM's fixed costs */
const best = fit(points), final = decode(best.x);
const round = (x, k) => Number(x.toPrecision(k));
const r2 = o => Object.fromEntries(Object.entries(o).map(([k, v])=> [k, round(v, 2)]));
const rounded = { shared: r2(final.shared), vllm: r2(final.vllm), trtllm: r2(final.trtllm) };
const shipped = { ...START, ...rounded.shared, ...rounded.vllm };
console.log(`Fitted on every group (loss ${best.v.toFixed(4)}):\n  ${show(final)}\nRounded:\n  ${show(rounded)}\n`);
console.log("In sample, with the rounded constants:\n\n" + table(ratios(rounded, points), "Group") + "\n");

/* AMD: constants fitted on NVIDIA boards predicting MI355X runs (vLLM on ROCm) they never saw, first as
   they are; then with one factor for AMD's stack (CAL.rocm, scaling the bandwidth and compute shares)
   fitted on those runs, cross-checked by fitting it on one model's runs and predicting the other's */
const ROCM = [0.4, 1.2];
const amdRatios = (rocm, pts) => { P.CAL.rocm = rocm; return ratios(rounded, pts); };
const fitRocm = pts => { const at = u => ROCM[0] + (ROCM[1] - ROCM[0])/(1 + Math.exp(-u[0]));
  const b = minimize(u=> { P.CAL.rocm = at(u); return loss(rounded, pts); }, [0]); return round(at(b.x), 2); };
console.log("AMD check (MI355X, vLLM on ROCm; never fitted), with the NVIDIA-fitted constants as they are:\n\n" + table(amdRatios(1, amd), "Group") + "\n");
const amdHeld = [];
for(const m of new Set(amd.map(p=> p.model))){
  const r = fitRocm(amd.filter(p=> p.model!==m));
  amdHeld.push(...amdRatios(r, amd.filter(p=> p.model===m)));
  console.log(`  ROCm factor without ${m}: ${r}`);
}
console.log("\nWith a ROCm factor fitted on the other model's runs:\n\n" + table(amdHeld, "Model left out") + "\n");
const rocm = fitRocm(amd), amdRows = amdRatios(rocm, amd);
console.log(`ROCm factor fitted on every MI355X run: ${rocm}\n\n` + table(amdRows, "Group") + "\n");
shipped.rocm = rocm;

const s = rs => ({ speed: summary(rs.map(r=> r.speed)), tput: summary(rs.map(r=> r.tput)), ttft: summary(rs.filter(r=> r.p.idle).map(r=> r.ttft)) });
writeFileSync(new URL("./calibration.json", import.meta.url), JSON.stringify({
  fitted: new Date().toISOString().slice(0, 10), points: points.length, groups,
  constants: shipped, tensorrtLlm: rounded.trtllm,
  heldOutOverPromises: { central: over(heldOut), lowEnd: over(heldOut, lowEnd) },
  heldOut: { all: s(heldOut), ...Object.fromEntries(groups.map(g=> [g, s(heldOut.filter(r=> group(r.p)===g))])) },
  inSample: s(ratios(rounded, points)),
  amdCheck: { rocm, asIs: s(amdRatios(1, amd)), heldOutByModel: s(amdHeld), fitted: s(amdRows) },
}, null, 1) + "\n");
console.log("wrote validation/calibration.json");

/* --apply: the constants, and the range measured results fell in around the estimate for the middle 80%
   of held-out points (measured = estimate ÷ ratio, so the 90th-percentile ratio gives the low end) */
if(process.argv.includes("--apply")){
  const num = (k, v) => /step|request/.test(k) ? `${Number((v*1e3).toPrecision(3))}e-3` : /Latency/.test(k) ? `${Number((v*1e6).toPrecision(3))}e-6` : `${v}`;
  const cal = `const CAL = {${Object.entries(shipped).map(([k, v])=> `${k}:${num(k, v)}`).join(", ")}};`;
  const band = xs => [round(1/q(xs, 0.9), 2), round(1/q(xs, 0.1), 2)];
  const range = `const RANGE = {speed:[${band(heldOut.map(r=> r.speed)).join(", ")}], ttft:[${band(heldOut.filter(r=> r.p.idle).map(r=> r.ttft)).join(", ")}]};`;
  let html = readFileSync(INDEX, "utf8");
  for(const [re, line] of [[/^const CAL = \{.*\};$/m, cal], [/^const RANGE = \{.*\};$/m, range]]){
    if(!re.test(html)) throw new Error(`${line.slice(0, 11)} line not found in index.html`);
    html = html.replace(re, line);
    console.log(`index.html: ${line}`);
  }
  writeFileSync(INDEX, html);
}
