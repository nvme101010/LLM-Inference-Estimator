// Sizes the scenarios of scenarios.mjs on every HPE server in the catalog with the Servers view's own search
// (configFor, allConfigs and serverAnswer in index.html), and reports where this page and a release disagree:
// which server is recommended, and for every configuration (a server with one of its boards) whether it
// holds the model, meets the goal, how many users it serves, its best layout, speed and first token. The GPU
// view's sweep (sweep.mjs) never runs this search, so a change here is invisible to it.
// Usage: node validation/servers.mjs [--base <git ref>] [--index path/to/index.html] [--quick] [--out file.json]
//   --base   the release to compare against: any git ref (default: the latest tag at or before HEAD)
//   --index  sweeps another copy of the page instead of this one
//   --quick  one scenario in seven
//   --out    also writes every scenario's result here, as JSON
// Two invariants are also checked on this page alone, whatever the base says:
//   1. a configuration listed as not holding the model holds no layout for even one user, so "Won't hold it"
//      never hides a server whose caches merely can't take everyone;
//   2. a board the GPU view sizes (deploymentFor, as in its default HPE server) is held by that server's
//      configuration here too, so the two views agree on what fits. (A server that holds the model but not
//      every user's cache was listed as not holding it until 1.8.0 was fixed; the GPU view said "serves N".)
// Exits 1 when anything differs or an invariant fails.

import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { load, INDEX } from "./model.mjs";
import { scenarios, key, setUp, plural, f6 } from "./scenarios.mjs";

const arg = k => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : null; };
const quick = process.argv.includes("--quick");
const ROOT = fileURLToPath(new URL("..", import.meta.url));
const git = (...a) => execFileSync("git", a, { cwd: ROOT, encoding: "utf8", maxBuffer: 64 << 20 });

/* what a comparison looks at on one configuration: whether it holds the model, its best layout and how
   that does; figures to 6 significant digits, so only a real change shows */
const layout = d => `${d.gpus} × ${d.L.pp > 1 ? (d.L.tp > 1 ? `TP${d.L.tp}×PP${d.L.pp}` : `PP${d.L.pp}`) : `TP${d.L.tp}`}`;
const record = c => !c.fits ? {fits:false, usable:f6(c.usable)}
  : {layout:layout(c.best), reps:c.best.reps, ok:c.ok, meets:c.best.meets, partial:!!c.best.partial, served:c.best.served,
     room:c.best.room, memRoom:c.best.memRoom, perGB:f6(c.best.perGB), perUser:f6(c.best.t.perUser), ttft:f6(c.best.t.ttft), options:c.opts.length};
const cls = r => r.fits===false ? "can't hold" : r.ok ? "meets" : "below goal";

function sweep(path, invariants){
  const P = load(path);
  if(!P.allConfigs || !P.serverAnswer || !P.configFor) throw new Error(`version ${P.VERSION} predates the Servers view`);
  const every = new Set(P.CATALOG.servers.map(sv=> sv.id)), out = new Map(), fails = [];
  let i = 0;
  for(const sc of scenarios(P)){
    if(quick && i++ % 7) continue;
    setUp(P, sc);
    P.S.scope = new Set(every);
    const cfgs = P.allConfigs(), A = P.serverAnswer(cfgs), k = key(sc);
    out.set(k, {rec: A.rec ? A.rec.id : null, cfgs: Object.fromEntries(cfgs.map(c=> [c.id, record(c)]))});
    if(!invariants) continue;
    for(const c of cfgs) if(!c.fits && P.layoutsOn(c.b).some(L=> P.evalLayout(c.b, L, 1, 1).fits))
      fails.push(`${k}: ${c.id} is listed as not holding the model, but a layout holds it for one user`);
    for(const g of P.boardsInPlay()){
      const d = P.deploymentFor(g), h = P.hostOf(g);
      if(!d || !d.fits || !h) continue;
      const c = cfgs.find(x=> x.id===`${h.id}|${g.n}`);
      if(c && !c.fits) fails.push(`${k}: the GPU view holds ${g.n} on ${d.gpus} GPUs (${d.partial ? `serves ${d.served}` : "everyone"}), but ${c.id} is listed as not holding the model`);
    }
  }
  return {version: P.VERSION, out, fails};
}

const ref = arg("--base") ?? git("describe", "--tags", "--abbrev=0", "HEAD").trim();
const dir = mkdtempSync(join(tmpdir(), "servers-")), basePath = join(dir, "index.html");
const t0 = Date.now();
let cur, base;
try {
  writeFileSync(basePath, git("show", `${ref}:index.html`));
  cur = sweep(arg("--index") ?? INDEX, true);
  base = sweep(basePath, false);
} finally { rmSync(dir, {recursive:true, force:true}); }
if(arg("--out")) writeFileSync(arg("--out"), JSON.stringify({version: cur.version, scenarios: Object.fromEntries(cur.out)}, null, 1));

const fmt = n => n.toLocaleString("en-US");
const same = (a, b) => JSON.stringify(a)===JSON.stringify(b);
const both = [...cur.out.keys()].filter(k=> base.out.has(k));
const diffs = both.flatMap(k=>{
  const a = base.out.get(k), b = cur.out.get(k);
  const cfgs = Object.keys(b.cfgs).filter(id=> id in a.cfgs && !same(a.cfgs[id], b.cfgs[id]));
  const moved = a.rec!==b.rec;
  return cfgs.length || moved ? [{k, a, b, cfgs, moved}] : [];
});

console.log(`Sized ${fmt(both.length)} scenarios on every server: this page (${cur.version}) against ${ref} (${base.version}), in ${Math.round((Date.now() - t0)/1000)} s.`);
cur.fails.slice(0, 20).forEach(f=> console.log("FAIL", f));
if(cur.fails.length) console.log(`${fmt(cur.fails.length)} invariant ${cur.fails.length===1 ? "failure" : "failures"}.`);
else console.log("Both invariants hold: every server listed as not holding the model holds no layout for one user, and every board the GPU view holds is held by its server here.");
if(!diffs.length){ console.log("No differences."); process.exit(cur.fails.length ? 1 : 0); }

const tally = xs => Object.entries(xs.reduce((t, x)=> (t[x] = (t[x] || 0) + 1, t), {})).sort((a, b)=> b[1] - a[1]);
const moves = diffs.filter(d=> d.moved);
const classMoves = diffs.flatMap(d=> d.cfgs.map(id=> `${cls(d.a.cfgs[id])} → ${cls(d.b.cfgs[id])}`)).filter(m=> m.split(" → ")[0]!==m.split(" → ")[1]);
console.log(`${fmt(diffs.length)} differ, ${fmt(diffs.reduce((n, d)=> n + d.cfgs.length, 0))} configurations in all.`);
if(classMoves.length) console.log(`Configurations changing class: ${tally(classMoves).map(([m, c])=> `${m} ${fmt(c)}`).join(", ")}.`);
console.log(`The recommended server changes in ${fmt(moves.length)}${moves.length ? ":" : "."}`);
tally(moves.map(d=> `${d.a.rec ?? "none"} → ${d.b.rec ?? "none"}`)).forEach(([m, c])=> console.log(`  ${m}: ${fmt(c)}`));

const describe = r => r.fits===false ? `can't hold it (${r.usable} GB usable)`
  : `${r.layout}${r.partial ? `, ${r.meets ? "serves" : "holds"} ${r.served}` : r.meets ? "" : ", below goal"}, ${r.perUser} tok/s, ${r.ttft} s`;
const SHOW = 15;   // changed recommendations first
console.log(`\n${diffs.length > SHOW ? `The first ${SHOW}` : "All of them"}:`);
[...moves, ...diffs.filter(d=> !d.moved)].slice(0, SHOW).forEach(d=>{
  console.log(`  ${d.k}`);
  if(d.moved) console.log(`    recommended: ${d.a.rec ?? "none"} → ${d.b.rec ?? "none"}`);
  d.cfgs.slice(0, 6).forEach(id=> console.log(`    ${id}: ${describe(d.a.cfgs[id])} → ${describe(d.b.cfgs[id])}`));
  if(d.cfgs.length > 6) console.log(`    … and ${d.cfgs.length - 6} more configurations`);
});
process.exit(1);
