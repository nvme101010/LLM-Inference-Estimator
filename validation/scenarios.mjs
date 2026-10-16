// The grid of scenarios sweep.mjs and fleet.mjs size, and how each is set up on the page's state.
//
// Each scenario starts as the page does on picking a model: weights as released, FP16 cache, prompt caching
// on, thinking at the model's default, GPUs per copy left to the page. It then sets the workload preset,
// users, speed goal and Server switch. "chat+rag" adds RAG users to the chat users on the same model, as
// many again. A smaller grid tries the other settings a model carries, at 32 users and the default goal:
// FP8 weights and cache, thinking hidden from users, 2 GPUs per copy, and prompt caching off.

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

/* every scenario, for the models of page P */
export function* scenarios(P){
  const models = P.MODELS.flatMap(g=> g.items).filter(m=> !m.custom);
  const thinks = m => ["always", "default"].includes(P.thinksOf(m));
  for(const m of models) for(const workload of WORKLOADS) for(const srv of [0, 1]){
    for(const users of USERS) for(const goal of GOALS) yield {m, workload, users, goal, srv};
    for(const v in VARIANTS) if(v!=="hidden" || thinks(m)) yield {m, workload, users:32, goal:"smooth", srv, v};
  }
}
export const plural = (n, what) => `${n} ${what}${n===1 ? "" : "s"}`;
export const key = ({m, workload, users, goal, srv, v}) => [m.id, workload, plural(users, "user"), goal, SERVER[srv], ...(v ? [v] : [])].join(" · ");

/* the page's own steps on picking a model and a preset (resetThink, presetWorkload and fitLengths in
   index.html), on its state alone */
const preset = (P, k) => { const w = P.WORKLOADS[k];
  return {workload:k, wantPrompt:w.prompt, wantReply:w.reply, shared:w.shared, fresh:w.fresh}; };
function fitExtra(w, m, thinking){
  w.reply  = Math.max(1, Math.min(w.wantReply, m.ctx - 512 - thinking));
  w.prompt = Math.max(1, Math.min(w.wantPrompt, Math.max(512, m.ctx - thinking - w.reply)));
  return w;
}
export function setUp(P, {m, workload, users, goal, srv, v}){
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

/* what a comparison looks at: a board's layout, fit, speed and first token, and the server around it.
   Figures to 6 significant digits, so only a real change shows. */
export const f6 = x => +x.toPrecision(6);
export const record = d => !d.fits ? {fits:false, tp:d.tp, perGB:f6(d.perGB)}
  : {gpus:d.gpus, tp:d.tp, reps:d.reps, meets:d.meets, partial:!!d.partial, served:d.served, need:d.need,
     perGB:f6(d.perGB), perUser:f6(d.t.perUser), ttft:f6(d.t.ttft)};
export const server = s => s.fixed ? {fixed:true, n:s.n} : {n:s.n, cores:s.cores, ram:s.ram, disk:f6(s.diskGB), kvOff:f6(s.kvOffGB)};
