// Sizes a fixed grid of scenarios on every board with the page's own layout search, and reports where this
// page and a release disagree. A change meant to leave the figures alone, such as a refactor, should report
// nothing; one that moves them says how many scenarios move and which suggestions change.
// Usage: node validation/sweep.mjs [--base <git ref>] [--index path/to/index.html] [--out file.json]
//   --base   the release to compare against: any git ref (default: the latest tag at or before HEAD)
//   --index  sweeps another copy of the page instead of this one
//   --out    also writes every scenario's result here, as JSON
// Exits 1 when anything differs, so it can gate a change. The scenarios are scenarios.mjs's.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { load, INDEX } from "./model.mjs";
import { scenarios, key, setUp, plural, record, server } from "./scenarios.mjs";

const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const ROOT = fileURLToPath(new URL("..", import.meta.url));
const git = (...a) => execFileSync("git", a, { cwd: ROOT, encoding: "utf8", maxBuffer: 64 << 20 });

function sweep(path){
  const P = load(path);
  if(!P.deploymentFor || !P.suggestDeployment || !P.thinksOf) throw new Error(`version ${P.VERSION} predates what the sweep runs`);
  const models = P.MODELS.flatMap(g=> g.items).filter(m=> !m.custom);
  const out = new Map(), boards = new Set();
  for(const sc of scenarios(P)){
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
