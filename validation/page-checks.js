// Regression checks for the page's own behaviour: links, the Servers view's panels, and the notes. They need a
// browser, since the sizing scripts here (model.mjs) load only the model, not the page. Open index.html, then
// paste this file into the browser console (or run it there with a headless browser) and read the table it
// prints: every row should say PASS. Each check names the bug it guards against.
(async function pageChecks(){
  const out = [];
  const check = (name, ok, detail = "") => out.push({check: name, result: ok ? "PASS" : "FAIL", detail: String(detail).slice(0, 120)});
  const fire = (el, type) => el.dispatchEvent(new Event(type, {bubbles: true}));
  const text = () => document.body.innerText;
  const broken = () => (text().match(/\bNaN\b|\bundefined\b|\[object/g) || []).length;
  const open = h => { history.replaceState(null, "", "#" + h); applyHash(); render(); };   // as a hashchange does, without waiting for it
  const errs = []; const onErr = e => errs.push(e.message); addEventListener('error', onErr);

  /* 1. a link naming an inherited object property as a workload, goal, precision or added workload: the page used
        to size nothing (NaN throughout, from w=constructor) or throw inside render (g=constructor) */
  for(const [h, key, want] of [["m=l31-8b&w=constructor", "workload", "chat"], ["m=l31-8b&g=constructor", "goal", "smooth"],
                               ["m=l31-8b&wq=__proto__", "wqLbl", "NATIVE"], ["m=l31-8b&x=constructor_8", "extra", "[]"]]){
    let threw = null;
    try { open(h); } catch(e){ threw = e; }
    const got = key==="extra" ? JSON.stringify(S.extra) : S[key];
    check(`link ${h} is ignored (${key} = ${want})`, !threw && got===want && broken()===0, threw || `${key} = ${got}, ${broken()} broken figures`);
  }

  /* 2. the version note shows in the Servers view too: it used to sit inside the GPU view's panel, so a link made
        with 1.7.0 to 1.11.7 (which open in the Servers view) never said that the figures had changed */
  open("v=1.10.2&m=l31-8b&u=16&w=chat");
  const note = document.getElementById('vnote');
  check("old link's version note is visible in the Servers view", S.view==="srv" && !note.hidden && note.getBoundingClientRect().height > 0, `view ${S.view}, hidden ${note.hidden}, height ${note.getBoundingClientRect().height}`);
  note.querySelector('[data-act="vnote"]').click();
  check("the note's Dismiss hides it", note.hidden);

  /* 3. the Servers view's Show the math and the notes under Advanced settings follow every render: they used to be
        left as they were when no configuration showed (no server in scope, or none holds the model), and the GPUs
        per copy and Pipeline notes were only drawn by the GPU view */
  open("v=1.11.8&m=l31-8b&u=16&w=chat");
  const mathDefault = "Every figure worked out · speed on this setup · model architecture · memory breakdown · how it's calculated";
  const tpnote = () => document.getElementById('tpnote').textContent, pipenote = () => document.getElementById('pipenote').textContent;
  check("Servers view: GPUs per copy note is drawn", /^Auto/.test(tpnote()), tpnote());
  check("Servers view: Pipeline note is drawn", /^Off/.test(pipenote()), pipenote());
  document.querySelector('#tp [data-v="4"]').click();
  check("Servers view: GPUs per copy note follows the setting", /^Fixed at 4 GPUs/.test(tpnote()), tpnote());
  document.querySelector('#pipe [data-v="1"]').click();
  check("Servers view: Pipeline note follows the setting", /^On/.test(pipenote()), pipenote());
  document.querySelector('#tp [data-v="auto"]').click(); document.querySelector('#pipe [data-v="0"]').click();
  const before = document.getElementById('mathState').textContent;
  document.querySelector('[data-scope-none]').click();
  const ms = document.getElementById('mathState').textContent;
  check("Servers view, no server in scope: Show the math no longer names the last recommendation", ms===mathDefault && ms!==before, ms);
  check("Servers view, no server in scope: worked figures are cleared", document.getElementById('workedScope').textContent==="" && /Worked out once/.test(document.getElementById('worked').textContent));
  document.querySelector('[data-scope-all]').click();
  check("Servers view, scope restored: Show the math names the recommendation again", document.getElementById('mathState').textContent===before, document.getElementById('mathState').textContent);
  modelSel.value = "custom"; fire(modelSel, 'change');
  const params = document.getElementById('c_params'); params.value = 5000; fire(params, 'input');
  check("Servers view, model too big for every server: Show the math shows the model alone", document.getElementById('mathState').textContent===mathDefault && /Too big/.test(document.getElementById('svRec').textContent), document.getElementById('mathState').textContent);
  check("Servers view, model too big: the speed panel says so", /once the model fits/.test(document.getElementById('tputbody').textContent));

  /* 4. a link without tx returns to this viewer's own wording, rather than keeping an earlier link's technical one */
  open("v=1.11.8&m=l31-8b&u=16&w=chat&tx=on");
  const tech = S.words;
  open("v=1.11.8&m=l31-8b&u=16&w=chat");
  check("wording: tx=on applies, and the next link without it goes back to the viewer's choice", tech==="tech" && S.words===loadWords(), `${tech} then ${S.words}`);

  /* 5. a link round trip: what the page writes, it reads back the same */
  open("v=1.11.8&m=q3-8b&u=40&w=custom&p=8192&r=777&cb=512&x=rag_8.custom_4_3000_512_512_512&th=2048&tv=hidden&wq=FP8&g=fast&kv=FP8&tp=2&pp=on&ck=8192&ov=12&pc=off&sv=nv&mm=least&vw=gpu&sh=512&fr=512&b=L40S&m2=l33-70b&u2=16&w2=chat&e=2");
  const h1 = stateToHash(); open(h1); const h2 = stateToHash();
  check("a complex link survives a round trip", h1===h2 && broken()===0, h1===h2 ? "" : `${h1}\n${h2}`);

  /* 6. a board whose largest node can't serve everyone: its pill names the full node and that the count is within the
        goal, with a tooltip in both wordings, and its Why row says what limits it. "Serves 31 of 150" beside "8 × GPU"
        used to read as the layout of an 8-GPU node that supports only 31 users */
  open("v=1.11.9&m=glm53-flash&u=150&w=agent&th=4096&g=smooth&vw=gpu");
  const prow = [...document.querySelectorAll('.boards tbody tr.brow')].find(r=> r.textContent.includes("RTX PRO 6000"));
  const ppill = prow && prow.querySelector('.pill.warn'), pwhy = prow && prow.nextElementSibling;
  check("partial node: the pill says it is a ceiling", !!ppill && /^Tops out at [\d,]+ of 150 users$/.test(ppill.textContent) && ppill.dataset.tip==="partial", ppill ? ppill.textContent : "no pill on the RTX PRO 6000 row");
  const plainWhy = S.words!=="tech";
  check(`partial node: the Why row says what limits it (${plainWhy ? "plain" : "technical"} wording)`, !!pwhy && pwhy.classList.contains("bwhy") && (plainWhy ? /the most one server takes/.test(pwhy.textContent) && /slow each other down/.test(pwhy.textContent) : /largest node this board allows/.test(pwhy.textContent) && /the limit is speed/.test(pwhy.textContent)), pwhy ? pwhy.textContent : "no Why row");
  check("partial node: the tooltip exists in both wordings", typeof TIPS.partial==="string" && typeof TIPS_PLAIN.partial==="string");
  ppill && ppill.dispatchEvent(new FocusEvent('focusin', {bubbles: true}));   // as keyboard focus does; a real focus() fires nothing while the window is in the background
  const tipEl = document.getElementById('tip');
  check("partial node: hovering the pill shows the row's own explanation", !!ppill && tipEl.classList.contains('on') && tipEl.textContent===pwhy.textContent.replace(/ Not recommended:.*$/s, "") && /more than one server|more than one node/.test(tipEl.textContent), tipEl.textContent);
  hideTip();
  const wasWords = S.words;
  document.querySelector('#words [data-v="tech"]').click();
  const trow = [...document.querySelectorAll('.boards tbody tr.brow')].find(r=> r.textContent.includes("RTX PRO 6000")), tpill = trow && trow.querySelector('.pill.warn');
  check("partial node: technical wording keeps the detailed explanation", !!tpill && /largest node this board allows/.test(tpill.dataset.tiptext) && /the limit is speed/.test(tpill.dataset.tiptext), tpill ? tpill.dataset.tiptext : "no pill");
  document.querySelector(`#words [data-v="${wasWords}"]`).click();

  removeEventListener('error', onErr);
  check("no uncaught errors during the checks", errs.length===0, errs.join("; "));
  open("v=1.11.8&m=l31-8b&u=16&w=chat");   // leave the page at its defaults
  console.table(out);
  const failed = out.filter(r=> r.result==="FAIL").length;
  console.log(failed ? `${failed} of ${out.length} checks FAILED` : `All ${out.length} checks pass`);
  return out;
})();
