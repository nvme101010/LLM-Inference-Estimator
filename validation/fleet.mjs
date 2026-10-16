// Checks the sizing of several models on one kind of board (fleetFor in index.html).
// Usage: node validation/fleet.mjs [--quick]
//   --quick  checks one scenario in seven for the one-model part (about 10 s instead of a minute)
// Exits 1 if any check fails.
//
// 1. One model: on every board in every scenario of scenarios.mjs, fleetFor gives deploymentFor's layout
//    (GPUs, split, copies, fit, goal, partial, memory, speed, first token) on one server of that many
//    GPUs, suggestFleet picks suggestDeployment's board, its server is serverFor's, and S is untouched.
// 2. Packing: random sets of copies on every board, under both Server settings. Every copy sits whole on
//    one server at a multiple of its width (so within one bridged group on H200 NVL), every server is a
//    size the board is bought in and holds what is on it, and no packing takes more servers than the
//    fewest possible (by brute force where servers aren't a power of two).
// 3. Several models: random sets of 2 to 4 scenarios sharing the page's settings. S is untouched; a model
//    that meets its goal alone meets it here, on at least the GPUs it needs alone before rounding; each
//    model holds exactly its copies' GPUs; the GPUs bought are the servers' sizes, at least what the
//    models need; the order of the models changes neither the servers nor the memory; each server's
//    storage counts each model on it once; and the suggestion meets every goal on the least memory
//    whenever some board does.

import { load } from "./model.mjs";
import { scenarios, key, setUp, record, server } from "./scenarios.mjs";

const quick = process.argv.includes("--quick");
const P = load(), S = P.S, json = JSON.stringify;
const failures = [];
const check = (ok, what) => { if(!ok && failures.push(what) <= 20) console.log("FAIL", what); };
let rand = 0x9e3779b9;                     // a fixed seed, so a failure repeats
const rnd = () => { rand = (rand + 0x6d2b79f5) | 0; let t = Math.imul(rand ^ (rand >>> 15), 1 | rand);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const pick = xs => xs[Math.floor(rnd()*xs.length)];
const t0 = Date.now();

/* 1. one model */
let one = 0, i = 0;
for(const sc of scenarios(P)){
  if(quick && i++ % 7) continue;
  setUp(P, sc);
  const spec = P.specOf(), before = json(spec), inPlay = P.boardsInPlay(), k = key(sc);
  const deps = new Map(inPlay.map(g=> [g.n, P.deploymentFor(g)])), fleets = inPlay.map(g=> P.fleetFor(g, [spec]));
  check(json(P.specOf())===before, `${k}: S changed`);
  for(const f of fleets){
    const d = deps.get(f.g.n), m = f.models[0];
    const want = record(d), got = record(m.d); delete want.need; delete got.need;
    check(json(got)===json(want), `${k} · ${f.g.n}: ${json(got)} ≠ ${json(want)}`);
    if(d.fits) check(f.servers.length===1 && f.gpus===d.gpus && f.memGB===d.memGB, `${k} · ${f.g.n}: ${f.servers.length} servers, ${f.gpus} GPUs ≠ ${d.gpus}`);
  }
  const s = P.suggestDeployment(deps), sf = P.suggestFleet(fleets);
  check((s && s.g.n)===(sf && sf.g.n), `${k}: suggestion ${sf && sf.g.n} ≠ ${s && s.g.n}`);
  if(sc.srv && s) check(json(server(P.fleetServers(sf)[0]))===json(server(P.serverFor(s))), `${k}: server differs`);
  one++;
}
console.log(`1. One model: ${one.toLocaleString("en-US")} scenarios on every board.`);

/* 2. packing */
function fewestServers(sizes, cap){        // brute force: the fewest bins of cap holding sizes
  const s = sizes.slice().sort((a, b)=> b - a);
  let best = s.length;
  const go = (j, bins)=> {
    if(bins.length >= best) return;
    if(j===s.length){ best = bins.length; return; }
    const seen = new Set();
    for(let b = 0; b < bins.length; b++) if(bins[b] + s[j] <= cap && !seen.has(bins[b])){ seen.add(bins[b]); bins[b] += s[j]; go(j + 1, bins); bins[b] -= s[j]; }
    bins.push(s[j]); go(j + 1, bins); bins.pop();
  };
  go(0, []);
  return best;
}
let packs = 0;
for(const srv of [0, 1]){
  S.srv = srv;
  for(const g of P.boardsInPlay()){
    const tps = [1, 2, 4, 8].filter(t=> t <= P.splitMax(g)), max = P.nodeMax(g), pop = P.popOf(g);
    for(let trial = 0; trial < 300; trial++){
      const copies = Array.from({length: 1 + Math.floor(rnd()*(trial < 150 ? 8 : 16))}, (_, i)=> ({i, tp: pick(tps)}));
      const servers = P.packServers(g, copies), w = `${g.n}, Server ${srv}: ${json(copies.map(c=> c.tp))}`;
      for(const c of copies){
        const on = servers.filter(s=> s.slots.includes(c.i));
        check(on.length===1, `${w}: copy ${c.i} on ${on.length} servers`);
        if(on.length!==1) continue;
        const at = on[0].slots.indexOf(c.i), n = on[0].slots.filter(x=> x===c.i).length;
        check(n===c.tp && at % c.tp===0 && on[0].slots.lastIndexOf(c.i)===at + c.tp - 1, `${w}: copy ${c.i} (TP${c.tp}) at ${at}, ${n} GPUs`);
        if(g.split) check(Math.floor(at/g.split)===Math.floor((at + c.tp - 1)/g.split), `${w}: copy ${c.i} crosses a bridge`);
      }
      for(const s of servers) check((pop ? pop.includes(s.n) : s.n <= max) && s.slots.length===s.n, `${w}: a server of ${s.n}`);
      const total = copies.reduce((n, c)=> n + c.tp, 0);
      const fewest = (max & (max - 1))===0 ? Math.ceil(total/max) : copies.length <= 10 ? fewestServers(copies.map(c=> c.tp), max) : null;
      if(fewest!==null) check(servers.length===fewest, `${w}: ${servers.length} servers, ${fewest} would do`);
      packs++;
    }
  }
}
console.log(`2. Packing: ${packs.toLocaleString("en-US")} random sets of copies on every board, under both Server settings.`);

/* 3. several models */
const base = [...scenarios(P)].filter(sc=> !sc.v);   // the per-model variants; nocache would change the page's setting
let fleetsChecked = 0, multi = 0;
for(let trial = 0; trial < (quick ? 40 : 160); trial++){
  const srv = trial % 2, scs = Array.from({length: 2 + Math.floor(rnd()*3)}, ()=> ({...pick(base), srv}));
  const specs = scs.map(sc=> { setUp(P, sc); return P.specOf(); });
  Object.assign(S, {srv, cache:1, ovpct:7});
  const before = json(P.specOf()), names = scs.map(key).join(" + "), fleets = [];
  for(const g of P.boardsInPlay()){
    const alone = specs.map(spec=> P.withSpec(spec, ()=> P.deploymentFor(g)));
    const f = P.fleetFor(g, specs), w = `${names} on ${g.n}`;
    check(json(P.specOf())===before, `${w}: S changed`);
    fleets.push(f);
    check(f.fits===alone.every(d=> d.fits), `${w}: fits ${f.fits}`);
    if(!f.fits) continue;
    const need = alone.map(d=> d.options.length ? Math.min(...d.options.map(o=> o.need)) : d.gpus);
    f.models.forEach((m, i)=>{
      if(alone[i].meets && !alone[i].partial) check(m.d.meets && m.d.gpus >= need[i], `${w}: model ${i + 1} ${m.d.meets ? "" : "misses its goal "}on ${m.d.gpus} GPUs (needs ${need[i]})`);
      check(f.servers.reduce((n, s)=> n + s.slots.filter(x=> x===i).length, 0)===m.d.gpus, `${w}: model ${i + 1}'s GPUs don't add up`);
    });
    check(f.gpus===f.servers.reduce((n, s)=> n + s.n, 0) && f.gpus >= need.reduce((a, b)=> a + b, 0), `${w}: ${f.gpus} GPUs`);
    const r = P.fleetFor(g, specs.slice().reverse());
    check(r.servers.length===f.servers.length && r.memGB===f.memGB, `${w}: reversed, ${r.servers.length} servers and ${r.memGB} GB ≠ ${f.servers.length} and ${f.memGB}`);
    P.fleetServers(f).forEach((s, k)=>{
      if(s.fixed) return;
      const on = [...new Set(f.servers[k].slots.filter(x=> x>=0))], wGB = on.reduce((t, i)=> t + f.models[i].d.r.weights/1e9, 0);
      check(Math.abs(s.wGB - wGB) < 1e-9*wGB && Math.abs(s.diskGB - (2*wGB + 100)/0.8) < 1e-6, `${w}: server ${k + 1} storage`);
    });
    fleetsChecked++;
  }
  const sf = P.suggestFleet(fleets), meeting = fleets.filter(f=> f.fits && f.meets && !f.models.some(m=> m.d.tp > 1 && !f.g.link));
  check(!meeting.length || (sf && sf.meets && sf.memGB===Math.min(...meeting.map(f=> f.memGB))), `${names}: suggestion ${sf && sf.g.n} isn't the leanest that meets every goal`);
  multi++;
}
console.log(`3. Several models: ${multi} sets of 2 to 4 models, ${fleetsChecked.toLocaleString("en-US")} fleets.`);

console.log(failures.length ? `${failures.length.toLocaleString("en-US")} checks failed.` : `All checks pass (${Math.round((Date.now() - t0)/1000)} s).`);
process.exit(failures.length ? 1 : 0);
