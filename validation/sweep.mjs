// Sizes a fixed grid of scenarios on every board with the page's own layout search, and reports where this
// page and a release disagree. A change meant to leave the figures alone, such as a refactor, should report
// nothing; one that moves them says how many scenarios move and which suggestions change.
// Usage: node validation/sweep.mjs [--base <git ref>] [--index path/to/index.html] [--out file.json]
//   --base   the release to compare against: any git ref (default: the latest tag at or before HEAD)
//   --index  sweeps another copy of the page instead of this one
//   --out    also writes every scenario's result here, as JSON
// Exits 1 when anything differs, so it can gate a change.
//
// Each scenario starts as the page does on picking a model: weights as released, FP16 cache, prompt caching
// on, thinking at the model's default, GPUs per copy left to the page. It then sets the workload preset,
// users, speed goal and Server switch. "chat+rag" adds RAG users to the chat users on the same model, as
// many again. A smaller grid tries the other settings a model carries, at 32 users and the default goal:
// FP8 weights and cache, thinking hidden from users, 2 GPUs per copy, and prompt caching off.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { load, INDEX } from "./model.mjs";

const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const ROOT = fileURLToPath(new URL("..", import.meta.url));
const git = (...a) => execFileSync("git", a, { cwd: ROOT, encoding: "utf8", maxBuffer: 64 << 20 });

const WORKLOADS = ["chat", "rag", "agent", "summary", "chat+rag"];
const USERS = [1, 8, 32, 128, 512];
const GOALS = ["any", "read", "smooth", "fast"];
const SERVER = ["GPUs only", "NVIDIA guidance"];
const VARIANTS = {
  fp8:     {wqLbl:"FP8", wq:1, kq:1},
  hidden:  {seeThink:false},           // tried only on models that think by default
  tp2:     {tpo:2},
  nocache: {cache:0},
};

function* scenarios(models, thinks){
  for(const m of models) for(const workload of WORKLOADS) for(const srv of [0, 1]){
    for(const users of USERS) for(const goal of GOALS) yield {m, workload, users, goal, srv};
    for(const v in VARIANTS) if(v!=="hidden" || thinks(m)) yield {m, workload, users:32, goal:"smooth", srv, v};
  }
}
const plural = (n, what) => `${n} ${what}${n===1 ? "" : "s"}`;
const key = ({m, workload, users, goal, srv, v}) => [m.id, workload, plural(users, "user"), goal, SERVER[srv], ...(v ? [v] : [])].join(" · ");

/* the page's own steps on picking a model and a preset (resetThink, applyWorkload, setLengths and fitExtra
   in index.html), on its state alone */
const preset = (P, k) => { const w = P.WORKLOADS[k];
  return {workload:k, wantPrompt:w.prompt, wantReply:w.reply, shared:w.shared, fresh:w.fresh}; };
function fitExtra(w, m, thinking){
  w.reply  = Math.max(1, Math.min(w.wantReply, m.ctx - 512 - thinking));
  w.prompt = Math.max(1, Math.min(w.wantPrompt, Math.max(512, m.ctx - thinking - w.reply)));
  return w;
}
function setUp(P, {m, workload, users, goal, srv, v}){
  const [first, added] = workload.split("+"), mode = P.thinksOf(m), S = P.S, chunk = P.WORKLOADS[first].chunk;
  P.setModel(m);
  Object.assign(S, {wqLbl:"NATIVE", wq:m.nat.gb/m.p, kq:2, tp:1, tpo:"auto", ovpct:7, cache:1, board:"auto", goal, srv, users,
    think: mode==="always" || mode==="default", thinkTok:4096, seeThink:true, trafficApplied:null, chunk, chunkBase:chunk},
    preset(P, first), VARIANTS[v]);
  S.thinking = S.think ? Math.min(S.thinkTok, Math.floor(m.ctx/2)) : 0;
  S.reply  = Math.max(1, Math.min(S.wantReply, m.ctx - 512 - S.thinking));
  S.prompt = Math.min(S.wantPrompt, Math.max(512, m.ctx - S.thinking - S.reply));   // a preset's prompt is a slider stop, so it never snaps
  S.extra  = added ? [fitExtra({...preset(P, added), users}, m, S.thinking)] : [];
}

/* what the comparison looks at: each board's layout, fit, speed and first token, the suggestion, and the
   server around it. Figures to 6 significant digits, so only a real change shows. */
const f6 = x => +x.toPrecision(6);
const record = d => !d.fits ? {fits:false, tp:d.tp, perGB:f6(d.perGB)}
  : {gpus:d.gpus, tp:d.tp, reps:d.reps, meets:d.meets, partial:!!d.partial, served:d.served, need:d.need,
     perGB:f6(d.perGB), perUser:f6(d.t.perUser), ttft:f6(d.t.ttft)};
const server = s => s.fixed ? {fixed:true, n:s.n} : {n:s.n, cores:s.cores, ram:s.ram, disk:f6(s.diskGB), kvOff:f6(s.kvOffGB)};

function sweep(path){
  const P = load(path);
  if(!P.deploymentFor || !P.suggestDeployment || !P.thinksOf) throw new Error(`version ${P.VERSION} predates what the sweep runs`);
  const models = P.MODELS.flatMap(g=> g.items).filter(m=> !m.custom);
  const thinks = m => ["always", "default"].includes(P.thinksOf(m));
  const out = new Map(), boards = new Set();
  for(const sc of scenarios(models, thinks)){
    setUp(P, sc);
    const inPlay = P.boardsInPlay ? P.boardsInPlay() : P.GPUS;
    const deps = new Map(inPlay.map(g=> [g.n, P.deploymentFor(g)])), s = P.suggestDeployment(deps);
    inPlay.forEach(g=> boards.add(g.n));
    out.set(key(sc), {suggestion: s ? s.g.n : null, server: sc.srv && s && P.serverFor ? server(P.serverFor(s)) : null,
                      boards: Object.fromEntries([...deps].map(([n, d])=> [n, record(d)]))});
  }
  return {version: P.VERSION, models: models.map(m=> m.id), boards: [...boards], out};
}

const ref = arg("--base") ?? git("describe", "--tags", "--abbrev=0", "HEAD").trim();
const dir = mkdtempSync(join(tmpdir(), "sweep-")), basePath = join(dir, "index.html");
const t0 = Date.now();
let cur, base;
try {
  writeFileSync(basePath, git("show", `${ref}:index.html`));
  cur = sweep(arg("--index") ?? INDEX);
  base = sweep(basePath);
} finally { rmSync(dir, {recursive:true, force:true}); }
if(arg("--out")) writeFileSync(arg("--out"), JSON.stringify({version: cur.version, scenarios: Object.fromEntries(cur.out)}, null, 1));

const fmt = n => n.toLocaleString("en-US");
const same = (a, b) => JSON.stringify(a)===JSON.stringify(b);
const both = [...cur.out.keys()].filter(k=> base.out.has(k));
const diffs = both.flatMap(k=>{
  const a = base.out.get(k), b = cur.out.get(k);
  const boards = Object.keys(b.boards).filter(n=> n in a.boards && !same(a.boards[n], b.boards[n]));
  const moved = a.suggestion!==b.suggestion, srv = !same(a.server, b.server);
  return boards.length || moved || srv ? [{k, a, b, boards, moved, srv}] : [];
});

console.log(`Swept ${fmt(both.length)} scenarios: this page (${cur.version}) against ${ref} (${base.version}), in ${Math.round((Date.now() - t0)/1000)} s.`);
const only = (what, a, b, where) => { const x = a.filter(n=> !b.includes(n)); if(x.length) console.log(`${what} only ${where}: ${x.join(", ")}.`); };
only("Boards", cur.boards, base.boards, "here");  only("Boards", base.boards, cur.boards, `in ${ref}`);
only("Models", cur.models, base.models, "here");  only("Models", base.models, cur.models, `in ${ref}`);
if(!diffs.length){ console.log("No differences."); process.exit(0); }

const tally = xs => Object.entries(xs.reduce((t, x)=> (t[x] = (t[x] || 0) + 1, t), {})).sort((a, b)=> b[1] - a[1]);
const moves = diffs.filter(d=> d.moved);
console.log(`${fmt(diffs.length)} differ, by board: ${tally(diffs.flatMap(d=> d.boards)).map(([n, c])=> `${n} ${fmt(c)}`).join(", ") || "none"}.`);
console.log(`The suggestion changes in ${fmt(moves.length)}${moves.length ? ":" : "."}`);
tally(moves.map(d=> `${d.a.suggestion ?? "none"} → ${d.b.suggestion ?? "none"}`)).forEach(([m, c])=> console.log(`  ${m}: ${fmt(c)}`));

const describe = r => r.fits===false ? `doesn't fit (${r.perGB} GB per GPU at TP${r.tp})`
  : `${plural(r.gpus, "GPU")} (${r.reps} × TP${r.tp})${r.partial ? `, ${r.meets ? "serves" : "holds"} ${r.served}` : r.meets ? "" : ", below goal"}, ${r.perUser} tok/s, ${r.ttft} s`;
const SHOW = 15;   // changed suggestions first
console.log(`\n${diffs.length > SHOW ? `The first ${SHOW}` : "All of them"}:`);
[...moves, ...diffs.filter(d=> !d.moved)].slice(0, SHOW).forEach(d=>{
  console.log(`  ${d.k}`);
  if(d.moved) console.log(`    suggestion: ${d.a.suggestion ?? "none"} → ${d.b.suggestion ?? "none"}`);
  d.boards.forEach(n=> console.log(`    ${n}: ${describe(d.a.boards[n])} → ${describe(d.b.boards[n])}`));
  if(d.srv && !d.moved) console.log(`    server: ${JSON.stringify(d.a.server)} → ${JSON.stringify(d.b.server)}`);
});
process.exit(1);
