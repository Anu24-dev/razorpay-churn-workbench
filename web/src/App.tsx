import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { run, NC, CAPACITY, PB3_RENEWAL_DAYS, type Row, type Check, type Playbook } from './engine/core'
import { RISK, RISK_HELP, risk, tierKey, DRIVER_NAME, FIX, PICK, channelFor, SENDER, plural, rs, pct, ncShort, whyShort, whyPoints, whyLeft,
  GROUPS, callQuestions, templateDraft } from './lib/text'
import { checkEdit } from './lib/guard'
import { readMerchants, DataError, downloadCsv, store } from './lib/io'
import { liveAvailable, draftWithRetry, answerQuestion, type Draft } from './lib/ai'

type AI = { note_category?: string; reason?: string; qa?: Record<string, string>
  draft?: { main_language: string; regional_language?: string; english: { subject?: string; message: string }; regional?: { subject?: string; message: string } | null; passed: boolean; checks: [string, boolean, string][] } }
type Data = { file: string; isPack: boolean; merchants: Row[]; summary: any; value: any; playbooks: Record<string, Playbook>; ai: Record<string, AI> }
type Decision = { status: 'Approved' | 'Skipped' | 'Rewriting'; text?: string; language?: string; edited?: boolean; warnings?: string[]; reason?: string; time: string }
type LogEntry = { time: string; by: string; merchant_id: string; action: string; detail: string }

const PRESETS: [string, string][] = [['Why are they on the list?', 'Why is this merchant on the call list?'],
  ['What should I focus on in the call?', 'What should I focus on in the call?'], ['What is not known yet?', 'What do we not know about this merchant?']]
const TESTED_LEAVE_RATE = 88 / 96   // share of flagged (declining) merchants that did leave, across four datasets with answer keys
// Share of would-be leavers a call keeps. Randomised trial where likely churners were listed to be called: churn fell 1.9 points from 17.2%
// (Godinho de Matos, Ferreira & Belo, Marketing Science 2018). 30% is the default in churn-profit research (Verbraken et al. 2013), shown only as the optimistic case.
const SAVE_RATE = 1.9 / 17.2, SAVE_RATE_OPTIMISTIC = 0.3
const now = () => new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
const nameOf = (r: Row) => r.business_name ?? r.merchant_id

// ---------------------------------------------------------------- small building blocks
function useCountUp(target: number, ms = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    let raf = 0; const t0 = performance.now()
    const tick = (t: number) => { const k = Math.min(1, (t - t0) / ms); setV(target * (1 - Math.pow(1 - k, 3))); if (k < 1) raf = requestAnimationFrame(tick) }
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}
function Pill({ tone = 'grey', children, title }: { tone?: string; children: ReactNode; title?: string }) {
  return <span className={`pill pill-${tone}`} title={title}>{children}</span>
}
function Kpi({ label, value, format, sub, delay }: { label: string; value: number; format: (n: number) => string; sub: string; delay: number }) {
  const v = useCountUp(value)
  return <div className="kpi rise" style={{ animationDelay: `${delay}ms` }}><div className="kpi-label">{label}</div><div className="kpi-value">{format(v)}</div><div className="kpi-sub">{sub}</div></div>
}
function Collapse({ title, children, open = false }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  const [o, setO] = useState(open)
  return <div className={`collapse ${o ? 'open' : ''}`}><button className="collapse-head" onClick={() => setO(!o)} aria-expanded={o}><span>{title}</span><span className="chev">⌄</span></button>{o && <div className="collapse-body">{children}</div>}</div>
}

// ---------------------------------------------------------------- the app
export default function App() {
  const [pack, setPack] = useState<Data | null>(null)
  const [data, setData] = useState<Data | null>(null)
  const [tab, setTab] = useState<'calls' | 'left'>('calls')
  const [openId, setOpenId] = useState<string | null>(null)
  const [drawer, setDrawer] = useState(false)
  const [how, setHow] = useState(false)
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([])
  const [upload, setUpload] = useState<{ state: 'idle' | 'reading' | 'error' | 'done'; message?: string; checks?: Check[] }>({ state: 'idle' })
  const key = data ? `rzp-retention:${data.file}:${data.merchants.length}` : ''
  const [decisions, setDecisions] = useState<Record<string, Decision>>({})
  const [confirms, setConfirms] = useState<Record<string, string>>({})
  const [log, setLog] = useState<LogEntry[]>([])
  const [celebrated, setCelebrated] = useState(false)

  useEffect(() => {
    fetch('/data/datapack.json').then(r => r.json()).then((d: any) => { const p = { ...d, isPack: true }; setPack(p); setData(p) })
  }, [])
  useEffect(() => {   // decisions are kept per file in this browser, so a refresh does not lose them
    if (!key) return
    const s = store.get(key, { decisions: {}, confirms: {}, log: [] as LogEntry[] })
    setDecisions(s.decisions); setConfirms(s.confirms); setLog(s.log); setCelebrated(false)
  }, [key])
  useEffect(() => { if (key) store.set(key, { decisions, confirms, log }) }, [key, decisions, confirms, log])

  const toast = (text: string) => { const id = Date.now() + Math.random(); setToasts(t => [...t, { id, text }]); setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 3200) }
  const addLog = (merchant_id: string, action: string, detail = '') => setLog(l => [...l, { time: new Date().toLocaleString('en-IN'), by: SENDER, merchant_id, action, detail }])

  const M = data?.merchants ?? []
  const byId = useMemo(() => new Map(M.map(r => [r.merchant_id, r])), [M])
  const calls = useMemo(() => M.filter(r => r.csm_call_list).sort((a, b) => a.revenue_rank - b.revenue_rank), [M])
  const review = useMemo(() => M.filter(r => r.declining && r.needs_person).sort((a, b) => a.revenue_rank - b.revenue_rank), [M])
  const nextUp = useMemo(() => M.filter(r => r.declining && r.revenue_rank > calls.length).sort((a, b) => a.revenue_rank - b.revenue_rank), [M, calls])
  const done = calls.filter(r => decisions[r.merchant_id] && decisions[r.merchant_id].status !== 'Rewriting').length
  const [aiOn, setAiOn] = useState<boolean>(() => store.get('rzp-retention:ai-on', true))
  useEffect(() => store.set('rzp-retention:ai-on', aiOn), [aiOn])
  const [live, setLive] = useState(false)
  useEffect(() => { liveAvailable().then(setLive) }, [])
  const [liveDrafts, setLiveDrafts] = useState<Record<string, Draft>>({})
  useEffect(() => { if (key) setLiveDrafts(store.get(key + ':drafts', {})) }, [key])
  useEffect(() => { if (key) store.set(key + ':drafts', liveDrafts) }, [key, liveDrafts])
  const ai = (id: string): AI => {
    if (!aiOn) return {}
    const saved = data?.ai?.[id] ?? {}
    return liveDrafts[id] ? { ...saved, draft: liveDrafts[id] as any } : saved
  }

  useEffect(() => {
    if (calls.length && done === calls.length && !celebrated) { setCelebrated(true); toast(`All ${calls.length} calls reviewed. Nice work.`) }
  }, [done, calls.length, celebrated])

  async function onFile(file: File) {
    setUpload({ state: 'reading' })
    try {
      const rows = await readMerchants(file)
      const res = run(rows)
      if (!res.ok) { setUpload({ state: 'error', message: `${file.name} is missing a few things. Fix the rows marked FAIL and upload it again.`, checks: res.checks }); return }
      const same = pack && res.merchants.length === pack.merchants.length && res.merchants.every(r => { const p = pack.merchants.find(x => x.merchant_id === r.merchant_id); return p && p.tpv_m6_inr_lakh === r.tpv_m6_inr_lakh && p.tpv_m1_inr_lakh === r.tpv_m1_inr_lakh })
      setData(same ? pack : { file: file.name, isPack: false, merchants: res.merchants, summary: res.summary, value: res.value, playbooks: res.playbooks, ai: {} })
      setOpenId(null); setTab('calls')
      setUpload({ state: 'done', message: `${file.name}: ${res.merchants.length} merchants scored, ${plural(res.merchants.filter(r => r.csm_call_list).length, 'call')}.` })
      toast(`Loaded ${file.name}`)
      setTimeout(() => setDrawer(false), 900)
    } catch (e: any) {
      setUpload({ state: 'error', message: e instanceof DataError ? e.message : 'This file could not be read. Upload an .xlsx or .csv with the P2_Merchants columns.' })
    }
  }

  if (!data) return <div className="loading"><div className="spinner" /><div>Loading this month's merchants…</div></div>
  const S = data.summary, V = data.value, PB = data.playbooks
  const cover = V.at_risk_value ? V.call_list_value / V.at_risk_value : 0
  // Saving: 9 in 10 flagged merchants left in testing (88 of 96 across four answer-key datasets); a call keeps about 1 in 9 of them (field trial)
  const saving = V.call_list_value * TESTED_LEAVE_RATE * SAVE_RATE - V.outreach_cost
  const open = openId ? byId.get(openId) ?? null : null
  const callIdx = open ? calls.findIndex(r => r.merchant_id === open.merchant_id) : -1

  function decide(id: string, d: Decision | null, action: string, detail = '') {
    setDecisions(prev => { const n = { ...prev }; if (d) n[id] = d; else delete n[id]; return n })
    addLog(id, action, detail)
  }
  function goNextToReview(fromId: string) {
    const i = calls.findIndex(r => r.merchant_id === fromId)
    const order = [...calls.slice(i + 1), ...calls.slice(0, i)]
    const nxt = order.find(r => !decisions[r.merchant_id] || decisions[r.merchant_id].status === 'Rewriting')
    setOpenId(nxt ? nxt.merchant_id : null)
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><img src="/razorpay-logo.png" alt="Razorpay" className="logo" /><span className="divider" /><span className="product">Merchant Retention</span></div>
        <div className="top-actions">
          <span className="file-chip" title={data.file}>Data: {data.isPack ? 'Build-a-thon data pack' : data.file} · {M.length} merchants</span>
          <button className={`ai-switch ${aiOn ? 'on' : ''}`} role="switch" aria-checked={aiOn} onClick={() => { setAiOn(!aiOn); toast(aiOn ? 'AI off: showing rule-based messages only' : 'AI on: showing Claude-written drafts and summaries') }}
            title="On: drafts, summaries and answers written by Claude and checked by code. Off: rule-based content only.">
            <span className="knob" /><span>AI {aiOn ? (live ? 'on · live' : 'on') : 'off'}</span></button>
          <button className="btn btn-upload" onClick={() => setDrawer(true)}><span aria-hidden>⤒</span> Upload data</button>
        </div>
      </header>

      <section className="hero">
        <div className="hero-inner">
          <div className="eyebrow">Customer Success · Monthly outreach</div>
          <h1>This month's outreach</h1>
          <p>Merchants whose payment volume is falling, ranked by revenue at risk, with the recommended playbook and a first message for your review.</p>
        </div>
        <div className="kpis">
          <Kpi label="Merchants at risk" value={S.at_risk} format={n => String(Math.round(n))} sub={`${S.imminent} critical`} delay={0} />
          <Kpi label="Annual revenue at risk" value={V.at_risk_value} format={rs} sub={`if all ${S.at_risk} declining merchants left`} delay={60} />
          <Kpi label={`Reached by the ${plural(calls.length, 'call')}`} value={V.call_list_value} format={rs} sub={calls.length ? `${pct(cover)} of the revenue at risk` : 'no calls needed'} delay={120} />
          <div className="kpi rise" style={{ animationDelay: '180ms' }}><div className="kpi-label">Likely saving from the calls</div>
            <div className="kpi-value">{calls.length ? rs(saving) : '–'}</div>
            <div className="kpi-sub">a year, if 1 in 9 calls keeps the merchant, as in a published trial <button className="link" onClick={() => setHow(true)}>How?</button></div></div>
        </div>
      </section>

      <main className="main">
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'calls'} className={tab === 'calls' ? 'on' : ''} onClick={() => setTab('calls')}>This month's calls <span className="count">{calls.length}</span></button>
          <button role="tab" aria-selected={tab === 'left'} className={tab === 'left' ? 'on' : ''} onClick={() => setTab('left')}>Not flagged <span className="count">{M.filter(r => !r.declining).length}</span></button>
        </nav>

        {tab === 'calls' ? <>
          <Insights M={M} />

          <section className="section">
            <div className="section-head">
              <div><h2>{calls.length ? `This month's ${plural(calls.length, 'call')}` : "This month's calls"}</h2>
                <p className="lead">{!calls.length ? 'No merchant in this file is declining, so no calls are needed this month.'
                  : calls.length < CAPACITY ? `Only ${plural(calls.length, 'merchant is', 'merchants are')} declining this month, so there is spare call capacity.`
                  : 'Select a merchant to see the reasons, the recommended playbook and a draft message.'}</p></div>
              <button className="btn btn-ghost" onClick={() => setHow(true)}>How these are picked</button>
            </div>
            {calls.length > 0 && <>
              <div className="legend">
                {(['1', '2', '3'] as const).map(k => <span key={k}><Pill tone={RISK[k].tone}>{RISK[k].label}</Pill>{RISK_HELP[k]}</span>)}
              </div>
              <div className="progress">
                <div className="dots">{calls.map(r => { const d = decisions[r.merchant_id]; return <button key={r.merchant_id} title={nameOf(r)} onClick={() => setOpenId(r.merchant_id)}
                  className={`dot ${d?.status === 'Approved' ? 'ok' : d?.status === 'Skipped' ? 'skip' : ''} ${!d && calls.find(c => !decisions[c.merchant_id])?.merchant_id === r.merchant_id ? 'next' : ''}`} /> })}</div>
                <span><b>{done} of {calls.length} reviewed</b>{calls.filter(r => decisions[r.merchant_id]?.status === 'Approved').length ? ` · ${calls.filter(r => decisions[r.merchant_id]?.status === 'Approved').length} approved` : ''}</span>
              </div>
              {celebrated && <div className="celebrate rise"><Confetti /><b>All {calls.length} calls reviewed.</b> Approved messages are ready to send below.</div>}
              <div className="calls">
                <div className="calls-head"><span>#</span><span>Merchant</span><span>Risk</span><span>Main reason</span><span>Playbook</span><span>Renews in</span><span>Pick</span><span>Status</span></div>
                {calls.map((r, i) => { const d = decisions[r.merchant_id]; const k = tierKey(r.tier); return (
                  <button key={r.merchant_id} className={`call rise ${openId === r.merchant_id ? 'sel' : ''} ${d?.status === 'Approved' ? 'is-ok' : d?.status === 'Skipped' ? 'is-skip' : ''}`}
                    style={{ animationDelay: `${i * 25}ms` }} onClick={() => setOpenId(r.merchant_id)}>
                    <span className="rank">{r.revenue_rank}</span>
                    <span className="mname"><b>{nameOf(r)}</b><small>{r.segment} · {r.city ?? r.industry}</small></span>
                    <span><Pill tone={RISK[k].tone} title={RISK_HELP[k]}>{RISK[k].label}</Pill>{r.call_this_week && <Pill tone="red">Call this week</Pill>}</span>
                    <span>{whyShort(r)}{r.driver_main === NC ? <em className="q"> · confirm</em> : null}</span>
                    <span className="pb">{r.playbook_id}</span>
                    <span>{r.days_to_contract_renewal} days</span>
                    <span>{r.call_list_status === 'Borderline' ? <Pill tone="amber" title="Drops out of the top 15 under some estimates, so the CSM lead decides.">Close call</Pill> : <span className="muted">{PICK[r.call_list_status] ?? ''}</span>}</span>
                    <span className="status">{d?.status === 'Approved' ? <Pill tone="green">✓ Approved</Pill> : d?.status === 'Skipped' ? <Pill tone="grey">Skipped</Pill> : d?.status === 'Rewriting' ? <Pill tone="blue">Rewriting</Pill> : <span className="muted">To review</span>}</span>
                  </button>) })}
              </div>
            </>}
          </section>

          {review.length > 0 && <ConfirmSection review={review} ai={ai} PB={PB} confirms={confirms}
            onConfirm={(id, v) => { setConfirms(c => ({ ...c, [id]: v })); addLog(id, 'Cause confirmed', v); toast(`Saved: ${nameOf(byId.get(id)!)} · ${v}`) }} />}

          <Outbox calls={calls} decisions={decisions} byId={byId} log={log} M={M} toast={toast} />
        </> : <NotFlagged M={M} />}
      </main>

      <footer className="footer">Built for the Razorpay Campus AI Build-a-thon · synthetic data from the build-a-thon data pack · nothing is sent from this app</footer>

      {open && <Panel key={open.merchant_id} r={open} ai={ai(open.merchant_id)} PB={PB} isPack={data.isPack} aiOn={aiOn} live={live && aiOn}
        onDraft={(d: Draft) => { setLiveDrafts(x => ({ ...x, [open.merchant_id]: d })); addLog(open.merchant_id, 'AI draft written', d.passed ? 'passed all checks' : 'stopped by a check') }} decision={decisions[open.merchant_id]}
        idx={callIdx} total={calls.length} nextUp={nextUp} decisions={decisions} calls={calls}
        onClose={() => setOpenId(null)} onPrev={() => callIdx > 0 && setOpenId(calls[callIdx - 1].merchant_id)} onNext={() => callIdx < calls.length - 1 && setOpenId(calls[callIdx + 1].merchant_id)}
        onDecide={(d, action, detail) => { decide(open.merchant_id, d, action, detail)
          if (d?.status === 'Approved') { toast(`Approved: ${nameOf(open)} is ready to send.`); setTimeout(() => goNextToReview(open.merchant_id), 1300) }
          if (d?.status === 'Skipped') toast(`${nameOf(open)} skipped this month.`) }} />}

      {drawer && <UploadDrawer upload={upload} isPack={data.isPack} onClose={() => setDrawer(false)} onFile={onFile}
        onReset={() => { if (pack) { setData(pack); setOpenId(null); setUpload({ state: 'idle' }); toast('Back to the data pack'); setDrawer(false) } }} />}
      {how && <HowModal S={S} V={V} onClose={() => setHow(false)} />}
      <div className="toasts">{toasts.map(t => <div key={t.id} className="toast">{t.text}</div>)}</div>
    </div>
  )
}

// ---------------------------------------------------------------- insights: main reasons and the largest fix
function Insights({ M }: { M: Row[] }) {
  const dec = M.filter(r => r.declining)
  const counts: Record<string, number> = {}
  dec.forEach(r => { counts[r.driver_main] = (counts[r.driver_main] ?? 0) + 1 })
  const mix: [string, number][] = Object.entries(counts).sort((a, b) => b[1] - a[1])
  if (!mix.length) return null
  const max = mix[0][1]
  const top = mix.find(([d]) => d !== NC) ?? mix[0]
  const nc = dec.filter(r => r.needs_person)
  const soon = nc.filter(r => r.days_to_contract_renewal <= PB3_RENEWAL_DAYS).length
  return (
    <section className="insights">
      <div className="card rise chart-card">
        <h3>Main reason for decline</h3>
        <div className="bars">{mix.map(([d, n], i) => (
          <div key={d} className="bar-row"><span className="bar-label">{DRIVER_NAME[d] ?? d}</span>
            <span className="bar-track"><span className={`bar ${d === NC ? 'grey' : ''}`} style={{ width: `${(n / max) * 100}%`, animationDelay: `${150 + i * 90}ms` }} /></span>
            <span className="bar-n">{plural(n, 'merchant')}</span></div>))}</div>
        <p className="muted tiny foot">Across the {plural(dec.length, 'merchant')} at risk. The main reason is the strongest problem in each merchant's own numbers.</p>
      </div>
      <div className="stack">
        <div className="card accent rise" style={{ animationDelay: '80ms' }}>
          <div className="label">Largest fixable reason · suggested change</div>
          <div className="title">{DRIVER_NAME[top[0]]} · {plural(top[1], 'merchant')}</div>
          <p>{FIX[top[0]]}</p>
        </div>
        {nc.length > 0 && <div className="card rise" style={{ animationDelay: '160ms' }}>
          <p><b>{plural(nc.length, 'merchant')} show{nc.length === 1 ? 's' : ''} no operational cause.</b> {soon === nc.length ? 'All' : soon} renew within 60 days, so pricing is the likely reason.</p>
          <p className="muted">Suggested change: {FIX[NC]}</p>
        </div>}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- the merchant panel
function Panel(p: { r: Row; ai: AI; PB: Record<string, Playbook>; isPack: boolean; aiOn: boolean; live: boolean; onDraft: (d: Draft) => void; decision?: Decision; idx: number; total: number; nextUp: Row[]; decisions: Record<string, Decision>; calls: Row[]
  onClose: () => void; onPrev: () => void; onNext: () => void; onDecide: (d: Decision | null, action: string, detail?: string) => void }) {
  const { r, ai, PB } = p
  const [tab, setTab] = useState<'why' | 'msg' | 'trend' | 'ask'>(p.decision ? 'msg' : 'why')
  const k = tierKey(r.tier)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if ((e.target as HTMLElement)?.tagName === 'TEXTAREA') return; if (e.key === 'Escape') p.onClose(); if (e.key === 'ArrowRight') p.onNext(); if (e.key === 'ArrowLeft') p.onPrev() }
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey)
  })
  return (
    <div className="overlay" onClick={p.onClose}>
      <aside className="panel" onClick={e => e.stopPropagation()} role="dialog" aria-label={nameOf(r)}>
        <div className="panel-head">
          <div className="nav">
            {p.idx >= 0 && <><button className="icon" onClick={p.onPrev} disabled={p.idx <= 0} aria-label="Previous merchant">‹</button>
              <span className="muted">{p.idx + 1} of {p.total}</span>
              <button className="icon" onClick={p.onNext} disabled={p.idx >= p.total - 1} aria-label="Next merchant">›</button></>}
            <button className="icon close" onClick={p.onClose} aria-label="Close">✕</button>
          </div>
          <h2>{nameOf(r)}</h2>
          <div className="meta">{r.city ? `${r.city} · ` : ''}{r.industry} · {r.segment} · customer for {r.tenure_months} months</div>
          <div className="pills"><Pill tone={RISK[k].tone} title={RISK_HELP[k]}>{RISK[k].label} risk</Pill>
            {r.call_this_week && <Pill tone="red">Call this week</Pill>}
            {r.call_list_status === 'Borderline' && <Pill tone="amber">Close call</Pill>}
            {r.needs_person && <Pill tone="amber">{ncShort(r)}: confirm on the call</Pill>}</div>
          <div className="ptabs">{([['why', 'Why'], ['msg', 'Message'], ['trend', 'Trend'], ['ask', 'Ask']] as const).map(([v, l]) =>
            <button key={v} className={tab === v ? 'on' : ''} onClick={() => setTab(v)}>{l}</button>)}</div>
        </div>
        <div className="panel-body">
          {tab === 'why' && <div className="fade">
            {RISK_HELP[k] && <p className="muted small">{RISK[k].label}: {RISK_HELP[k]}</p>}
            <ul className="why">{whyPoints(r).map((x, i) => <li key={i} className={x.hot ? 'hot' : ''} style={{ animationDelay: `${i * 60}ms` }}>{x.text}</li>)}</ul>
            {r.needs_person ? <div className="note warn"><b>Why {ncShort(r).toLowerCase()}?</b> Nothing in their numbers explains the decline, and the contract renews in {r.days_to_contract_renewal} days.
              {r.account_manager_note ? <> Their account manager noted: “{r.account_manager_note}”.</> : null} Confirm the cause on the call before acting.
              <div className="tiny">The note supports this but does not decide anything. Volume and revenue put the merchant on the list.</div></div>
              : r.account_manager_note ? <p className="muted">Account manager note: “{r.account_manager_note}”</p> : null}
            <div className="plan"><div className="label">Recommended playbook</div><div className="pb-name">{r.playbook_id} · {PB[r.playbook_id].name}</div>
              <p>{PB[r.playbook_id].what_happens}</p><div className="muted">Run by: {r.delivery}</div></div>
            {ai.reason && <Collapse title="Summary by AI"><p>{ai.reason}</p></Collapse>}
            <Collapse title="All the numbers"><div className="tiles">{[
              ['Volume, last 3 months', `${r.tpv_change > 0 ? '+' : ''}${pct(r.tpv_change)}`], ['By month 9, on the current trend', `-${pct(r.projected_loss_m9)}`],
              ['Payment success', `${r.success_rate_m1_m3_pct.toFixed(1)}% → ${r.success_rate_m4_m6_pct.toFixed(1)}%`], ['Open tickets', `${r.unresolved_tickets} of ${r.support_tickets_last_90d}`],
              ['Last support rating', r.last_csat_1_to_5 == null ? '–' : `${Math.round(r.last_csat_1_to_5)} out of 5`], ['Logins this month', String(r.logins_last_30d)],
              ['Products used', String(r.products_used)], ['Annual revenue', rs(r.yearly_value_rs)]].map(([l, v]) => <div key={l} className="tile"><div className="label">{l}</div><b>{v}</b></div>)}</div></Collapse>
          </div>}
          {tab === 'msg' && <Message key={`${p.aiOn}-${Boolean(ai.draft)}`} {...p} />}
          {tab === 'trend' && <Trend r={r} />}
          {tab === 'ask' && <div className="fade">
            <p className="muted small">Answers use only this merchant's data, and say so when the data is silent.</p>
            <Ask ai={ai} isPack={p.isPack} aiOn={p.aiOn} live={p.live} r={r} playbookName={PB[r.playbook_id].name} />
          </div>}
        </div>
      </aside>
    </div>
  )
}

function Message(p: { r: Row; ai: AI; aiOn?: boolean; live?: boolean; onDraft?: (d: Draft) => void; decision?: Decision; nextUp: Row[]; decisions: Record<string, Decision>; onDecide: (d: Decision | null, action: string, detail?: string) => void }) {
  const { r, ai, decision } = p
  const versions = useMemo(() => {
    const v: Record<string, string> = {}
    const fmt = (x?: { subject?: string; message: string } | null) => (x ? (x.subject ? `Subject: ${x.subject}\n\n` : '') + x.message : '')
    if (ai.draft) {
      v[ai.draft.main_language] = ai.draft.main_language === 'English' ? fmt(ai.draft.english) : fmt(ai.draft.regional)
      if (ai.draft.main_language !== 'English') v.English = fmt(ai.draft.english)
      else if (ai.draft.regional && ai.draft.regional_language) v[ai.draft.regional_language] = fmt(ai.draft.regional)
    } else v.English = templateDraft(r)
    return v
  }, [ai, r])
  const langs = Object.keys(versions)
  const [lang, setLang] = useState(langs[0])
  const [texts, setTexts] = useState<Record<string, string>>(versions)
  const [blocked, setBlocked] = useState<{ label: string; words: string[] }[] | null>(null)
  const [rejecting, setRejecting] = useState(false)
  const [why, setWhy] = useState('Wrong cause or playbook')
  const [next, setNext] = useState<'rewrite' | 'skip'>('rewrite')
  const [stamp, setStamp] = useState(false)
  const [writing, setWriting] = useState<string[] | null>(null)
  const [aiError, setAiError] = useState<string | null>(null)
  async function writeWithAI() {
    setWriting([]); setAiError(null)
    try { const d = await draftWithRetry(r, m => setWriting(w => [...(w ?? []), m])); p.onDraft?.(d) }
    catch (e: any) { setAiError(e.message ?? 'The AI request failed.') }
    finally { setWriting(null) }
  }
  const text = texts[lang]
  const edited = text.trim() !== versions[lang].trim()
  const checks = ai.draft?.checks ?? []
  const passed = checks.filter(c => c[1]).length

  function approve() {
    const res = edited ? checkEdit(text, lang === 'English') : checkEdit(text, lang === 'English').filter(c => !c.hard)
    const hard = res.filter(c => c.hard)
    if (hard.length) { setBlocked(hard); return }
    setBlocked(null); setStamp(true)
    setTimeout(() => p.onDecide({ status: 'Approved', text, language: lang, edited, warnings: res.filter(c => !c.hard).map(c => `${c.label}: ${c.words.join(', ')}`), time: now() },
      'Approved', `${lang}${edited ? ' (edited, re-checked)' : ''}`), 650)
  }

  if (decision?.status === 'Approved') return (
    <div className="fade">
      <div className="done-card"><svg className="tick" viewBox="0 0 52 52"><circle cx="26" cy="26" r="24" /><path d="M14 27l8 8 16-17" /></svg>
        <div><b>Approved</b> by {SENDER} at {decision.time}{decision.edited ? ' · edited' : ''}<div className="muted">It is in Ready to send below. Copy it and send it by email or WhatsApp.</div></div></div>
      <pre className="msg">{decision.text}</pre>
      {decision.text?.includes('[') && <div className="note warn">Fill in the two time slots in [ ] before you send it.</div>}
      {decision.warnings?.filter(w => w.startsWith('English words')).map(w => <div key={w} className="note warn">{w}. Fine for a product name; otherwise write it in {decision.language}.</div>)}
      <div className="row"><button className="btn" onClick={() => navigator.clipboard?.writeText(decision.text ?? '')}>Copy message</button>
        <button className="btn btn-ghost" onClick={() => p.onDecide(null, 'Approval undone')}>Undo</button></div>
    </div>)
  if (decision?.status === 'Skipped') {
    const skipped = Object.entries(p.decisions).filter(([, d]) => d.status === 'Skipped').map(([id]) => id)
    const n = p.nextUp[Math.max(0, skipped.indexOf(r.merchant_id))]
    return <div className="fade"><div className="skip-card"><b>Skipped this month</b> · {SENDER} at {decision.time}<div>Reason: {decision.reason}</div>
      <div className="muted">{n ? <>{nameOf(n)} is next up for this slot.</> : 'No other merchant is waiting for a call.'}</div></div>
      <button className="btn btn-ghost" onClick={() => p.onDecide(null, 'Skip undone')}>Undo</button></div>
  }
  return (
    <div className={`fade ${stamp ? 'stamping' : ''}`}>
      <p className="muted small">{channelFor(r.segment)}. The language is based on their city, so confirm it with the merchant first.</p>
      {decision?.status === 'Rewriting' && <div className="note info">You are rewriting this one. Edit the message, then approve it.</div>}
      {ai.draft && !ai.draft.passed && <div className="note bad">The safety checks stopped this draft. Rewrite it below, then approve it.</div>}
      {!ai.draft && <div className="note info">{p.aiOn === false ? 'AI is off, so this is a rule-based starting message.' : 'No AI draft for this merchant yet.'} It follows the same rules: no numbers, no promises.
        {p.live && !writing && <div className="row"><button className="btn btn-primary" onClick={writeWithAI}>Write it with AI</button><span className="muted small">About a minute. Every check runs before you see it.</span></div>}</div>}
      {writing && <div className="note info progress-steps"><b><span className="spinner small" />Claude is writing the message…</b><ul>{writing.map((m, i) => <li key={i} className="fade">{m}</li>)}</ul></div>}
      {aiError && <div className="note bad">{aiError}</div>}
      {langs.length > 1 && <div className="seg">{langs.map(l => <button key={l} className={l === lang ? 'on' : ''} onClick={() => setLang(l)}>{l}</button>)}</div>}
      <textarea className="msg-edit" value={text} onChange={e => { setTexts(t => ({ ...t, [lang]: e.target.value })); setBlocked(null) }} rows={12} aria-label="Message" />
      {checks.length > 0 && <Collapse title={<>{ai.draft?.passed ? <span className="ok">✓ Passed all {checks.length} safety checks</span> : <span className="bad">✗ {checks.length - passed} of {checks.length} safety checks failed</span>}{edited ? <span className="muted"> · your edits are checked when you approve</span> : null}</>}>
        <ul className="checks">{checks.map(([n, ok, d]) => <li key={n} className={ok ? 'ok' : 'bad'}>{ok ? '✓' : '✗'} {n}{d ? ` · ${d}` : ''}</li>)}</ul></Collapse>}
      {blocked && <div className="note bad shake"><b>This can't be approved yet.</b> Your edit includes something that is never promised or shared with merchants:
        <ul>{blocked.map(b => <li key={b.label}>{b.label}: <b>{b.words.join(', ')}</b></li>)}</ul>Take it out and approve again.</div>}
      {!rejecting ? <div className="row">
        <button className="btn btn-primary" onClick={approve}>{stamp ? '✓ Approved' : '✓ Approve'}</button>
        <button className="btn btn-ghost" onClick={() => setRejecting(true)}>Not now…</button></div>
        : <div className="reject">
          <div className="label">What's not right?</div>
          {['Wrong cause or playbook', 'The wording or tone is off', 'Already in touch with them', 'Not worth a call this month', 'Something else'].map(o =>
            <label key={o} className="radio"><input type="radio" checked={why === o} onChange={() => setWhy(o)} />{o}</label>)}
          <div className="label">What next?</div>
          <label className="radio"><input type="radio" checked={next === 'rewrite'} onChange={() => setNext('rewrite')} />I'll rewrite it (they stay on the list)</label>
          <label className="radio"><input type="radio" checked={next === 'skip'} onChange={() => setNext('skip')} />Skip them this month (the next merchant gets the slot)</label>
          <div className="row"><button className="btn btn-primary" onClick={() => { setRejecting(false); p.onDecide({ status: next === 'skip' ? 'Skipped' : 'Rewriting', reason: why, time: now() }, next === 'skip' ? 'Skipped this month' : 'Rewriting', why) }}>Done</button>
            <button className="btn btn-ghost" onClick={() => setRejecting(false)}>Cancel</button></div>
        </div>}
      <p className="muted tiny">Signed as {SENDER}. Approved messages wait in Ready to send; nothing is sent from here.</p>
    </div>
  )
}

function Trend({ r }: { r: Row }) {
  const T = [1, 2, 3, 4, 5, 6].map(i => r[`tpv_m${i}_inr_lakh`] as number)
  const g = Math.sqrt(T[5] / T[3])
  const proj = [T[5], T[5] * g, T[5] * g * g, T[5] * g ** 3]
  const all = [...T, ...proj.slice(1)], max = Math.max(...all) * 1.12
  const W = 520, H = 220, L = 36, R = 16, Tp = 16, B = 30
  const x = (i: number) => L + (i * (W - L - R)) / 8, y = (v: number) => Tp + (1 - v / max) * (H - Tp - B)
  const base = (T[0] + T[1] + T[2]) / 3
  const line = T.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ')
  const pline = proj.map((v, i) => `${i ? 'L' : 'M'}${x(5 + i)},${y(v)}`).join(' ')
  return (
    <div className="fade">
      <div className="label">Monthly payment volume (Rs lakh), with the current trend carried to month 9</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Monthly payment volume">
        <line x1={L} x2={W - R} y1={y(base)} y2={y(base)} className="baseline" />
        <text x={W - R} y={y(base) - 6} textAnchor="end" className="axis">first three months, average</text>
        <rect x={x(6) - 6} y={Tp} width={x(8) - x(6) + 12} height={H - Tp - B} className="future" />
        <path d={`${line} L${x(5)},${H - B} L${x(0)},${H - B} Z`} className="area" />
        <path d={line} className="line draw" />
        <path d={pline} className="proj" />
        {T.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={3.5} className="pt" />)}
        {Array.from({ length: 9 }, (_, i) => <text key={i} x={x(i)} y={H - 10} textAnchor="middle" className="axis">M{i + 1}</text>)}
        <text x={x(8)} y={Math.min(H - B - 8, y(proj[3]) + 22)} textAnchor="end" className="axis strong">{rs(proj[3] * 1e5).replace('Rs ', 'Rs ')} by month 9</text>
      </svg>
      <div className="label" style={{ marginTop: 18 }}>Payments that went through</div>
      <div className="sr">{[['First three months', r.success_rate_m1_m3_pct], ['Last three months', r.success_rate_m4_m6_pct]].map(([l, v], i) =>
        <div key={l} className="sr-row"><span>{l}</span><span className="bar-track"><span className={`bar ${i === 1 && r.success_rate_m1_m3_pct - r.success_rate_m4_m6_pct > 2 ? 'red' : i ? '' : 'grey'}`} style={{ width: `${Math.max(0, ((v as number) - 70) / 30) * 100}%` }} /></span><b>{(v as number).toFixed(1)}%</b></div>)}</div>
      <p className="muted tiny">For your review only. Messages to merchants never mention numbers.</p>
    </div>
  )
}

function Ask({ ai, isPack, aiOn, live, r, playbookName }: { ai: AI; isPack: boolean; aiOn: boolean; live: boolean; r: Row; playbookName: string }) {
  const [q, setQ] = useState<string | null>(null)
  const [free, setFree] = useState('')
  const [liveAns, setLiveAns] = useState<Record<string, { text: string; ok: boolean } | 'loading' | { error: string }>>({})
  const ask = async (label: string, question: string) => {
    setQ(label)
    if (ai.qa?.[question] || liveAns[label] || !live) return
    setLiveAns(a => ({ ...a, [label]: 'loading' }))
    try { const res = await answerQuestion(r, question, playbookName); setLiveAns(a => ({ ...a, [label]: res })) }
    catch (e: any) { setLiveAns(a => ({ ...a, [label]: { error: e.message ?? 'The AI request failed.' } })) }
  }
  const saved = q ? ai.qa?.[PRESETS.find(p => p[0] === q)?.[1] ?? ''] : null
  const la = q ? liveAns[q] : undefined
  const body = saved ?? (la === 'loading' ? null : la && 'error' in la ? la.error : la && !la.ok ? "This answer was hidden because it quoted a number that is not in the merchant's data. Try asking another way."
    : la ? la.text : !aiOn ? 'AI is off. Turn it on in the header to see answers written by Claude.' : isPack ? 'No saved answer for this question.' : 'Ask Claude a question once AI is live on the server.')
  return <>
    <div className="asks">{PRESETS.map(([l, qq]) => <button key={l} className={`chip ${q === l ? 'on' : ''}`} onClick={() => ask(l, qq)}>{l}</button>)}</div>
    {live && <form className="ask-free" onSubmit={e => { e.preventDefault(); if (free.trim()) ask(free.trim(), free.trim()) }}>
      <input value={free} onChange={e => setFree(e.target.value)} placeholder="Or ask your own question, e.g. Is this a seasonal dip?" maxLength={300} />
      <button className="btn" type="submit">Ask</button></form>}
    {q && <div className="answer fade"><b>{q}</b>{la === 'loading' ? <p className="muted"><span className="spinner small" />Asking Claude…</p> : <p>{body}</p>}</div>}
  </>
}

// ---------------------------------------------------------------- confirm the cause
function ConfirmSection({ review, ai, PB, confirms, onConfirm }: { review: Row[]; ai: (id: string) => AI; PB: Record<string, Playbook>; confirms: Record<string, string>; onConfirm: (id: string, v: string) => void }) {
  const opts = ['Pricing (PB-3)', 'Payment failures (PB-1)', 'Support issues (PB-2)', 'Low engagement (PB-4)', 'Other (PB-6)']
  const allPricing = review.every(r => String(r.review_likely_bucket).startsWith('Likely commercial (renews'))
  return (
    <section className="section">
      <h2>Confirm the cause before acting</h2>
      <p className="lead">These merchants are declining, but their payments, support and usage look normal.{allPricing ? ' All renew within 60 days, so pricing is the likely cause.' : ''} Confirm it on the call.</p>
      <div className="confirm-grid">{review.map((r, i) => { const pricingNote = ai(r.merchant_id).note_category === 'pricing'; const c = confirms[r.merchant_id]; return (
        <div key={r.merchant_id} className={`card confirm rise ${c ? 'confirmed' : ''}`} style={{ animationDelay: `${i * 50}ms` }}>
          <div className="confirm-top"><b>{nameOf(r)}</b><span className="muted">{r.csm_call_list ? "On this month's calls" : r.am_alert ? 'Passed to their account manager' : 'Automated message'}</span></div>
          <div className="pills"><Pill tone="blue">{ncShort(r)}</Pill><Pill>{r.playbook_id} · {PB[r.playbook_id].name}</Pill></div>
          <div className="mini-stats">
            <div className={r.days_to_contract_renewal <= PB3_RENEWAL_DAYS ? 'hot' : ''}><span className="label">Renews in</span><b>{r.days_to_contract_renewal} days</b></div>
            <div><span className="label">Annual revenue</span><b>{rs(r.yearly_value_rs)}</b></div>
          </div>
          <blockquote className={`am-note ${pricingNote ? 'hot' : ''}`}><span className="label">Account manager note</span>{r.account_manager_note ? r.account_manager_note : 'No note'}</blockquote>
          <div className="label">Ask on the call</div>
          <ul className="asklist">{callQuestions(r, pricingNote).map(a => <li key={a}>{a}</li>)}</ul>
          <label className="select-wrap"><span className="label">Cause confirmed on the call</span>
            <select value={c ?? ''} onChange={e => e.target.value && onConfirm(r.merchant_id, e.target.value)}>
              <option value="">Not confirmed yet</option>{opts.map(o => <option key={o}>{o}</option>)}</select></label>
          {c && <div className="confirmed-tag">✓ Confirmed: {c}</div>}
        </div>) })}</div>
    </section>
  )
}

// ---------------------------------------------------------------- ready to send, downloads and activity log
function Outbox({ calls, decisions, byId, log, M, toast }: { calls: Row[]; decisions: Record<string, Decision>; byId: Map<string, Row>; log: LogEntry[]; M: Row[]; toast: (t: string) => void }) {
  const appr = Object.entries(decisions).filter(([id, d]) => d.status === 'Approved' && byId.has(id))
  const skipped = Object.entries(decisions).filter(([id, d]) => d.status === 'Skipped' && byId.has(id))
  const ranked = [...M].sort((a, b) => (a.revenue_rank ?? 999) - (b.revenue_rank ?? 999) || String(a.tier).localeCompare(String(b.tier)))
    .map(r => ({ merchant_id: r.merchant_id, business_name: r.business_name, segment: r.segment, industry: r.industry, risk: risk(r).label, tier: r.tier,
      main_reason: r.driver_main ? whyShort(r) : '', playbook_id: r.playbook_id, delivery: r.delivery,
      revenue_rank: r.revenue_rank ?? '', pick: PICK[r.call_list_status] ?? '', call_this_week: r.call_this_week, account_manager_alert: r.am_alert,
      automated_track: r.automated_track, expected_saving_rs: r.expected_saving_rs == null ? '' : Math.round(r.expected_saving_rs), why_not_flagged: r.declining ? '' : whyLeft(r) }))
  return (
    <section className="section">
      <h2>Ready to send</h2>
      {!appr.length ? <p className="muted">Nothing here yet. Approve a message and it lands here.</p>
        : <div className="outbox">{appr.map(([id, d]) => { const r = byId.get(id)!; return (
          <div key={id} className="out rise"><div className="out-top"><b>✓ {nameOf(r)}</b><span className="muted">{channelFor(r.segment)} · {d.language} · approved {d.time}</span>
            <button className="btn btn-small" onClick={() => { navigator.clipboard?.writeText(d.text ?? ''); toast('Message copied') }}>Copy</button></div>
            <pre className="msg small">{d.text}</pre></div>) })}</div>}
      {skipped.length > 0 && <p className="muted">Skipped this month: {skipped.map(([id, d]) => `${nameOf(byId.get(id)!)} (${d.reason?.toLowerCase()})`).join('; ')}</p>}
      <div className="row wrap">
        <button className="btn" disabled={!appr.length} onClick={() => downloadCsv('approved_messages.csv', appr.map(([id, d]) => ({ merchant_id: id, business_name: nameOf(byId.get(id)!), channel: channelFor(byId.get(id)!.segment), language: d.language, approved_by: SENDER, edited: d.edited, message: d.text })))}>Download messages ({appr.length})</button>
        <button className="btn" onClick={() => downloadCsv('ranked_merchants.csv', ranked)}>Download the full ranked list</button>
        <button className="btn" disabled={!log.length} onClick={() => downloadCsv('activity_log.csv', log)}>Download activity log ({log.length})</button>
      </div>
      {log.length > 0 && <Collapse title={`Activity log (${log.length})`}><table className="table small"><thead><tr><th>Time</th><th>Merchant</th><th>Action</th><th>Detail</th></tr></thead>
        <tbody>{[...log].reverse().map((l, i) => <tr key={i}><td>{l.time}</td><td>{byId.get(l.merchant_id) ? nameOf(byId.get(l.merchant_id)!) : l.merchant_id}</td><td>{l.action}</td><td>{l.detail}</td></tr>)}</tbody></table></Collapse>}
      <p className="muted tiny">Decisions are saved in this browser, so a refresh keeps them.</p>
    </section>
  )
}

// ---------------------------------------------------------------- not flagged
function NotFlagged({ M }: { M: Row[] }) {
  const nf = M.filter(r => !r.declining)
  const groups = GROUPS.filter(([g]) => nf.some(r => r.why_group === g))
  const [g, setG] = useState(groups[0]?.[0])
  const show = nf.filter(r => r.why_group === g)
  return (
    <section className="section fade">
      <h2>Merchants that look at risk but are not</h2>
      <p className="lead">Seasonal businesses, small merchants and busy, growing merchants can look at risk. Each was checked and not flagged for the reason shown.</p>
      {!groups.length ? <p className="muted">Every merchant in this file is declining, so none were left out.</p> : <>
        <div className="seg big">{groups.map(([k, label]) => <button key={k} className={g === k ? 'on' : ''} onClick={() => setG(k)}>{label} <span className="count">{nf.filter(r => r.why_group === k).length}</span></button>)}</div>
        <div className="table-wrap"><table className="table"><thead><tr><th>Merchant</th><th>Business</th><th>Size</th><th>Last 3 months</th><th>Why not flagged</th><th>Account manager note</th></tr></thead>
          <tbody>{show.map(r => <tr key={r.merchant_id} className="fade"><td><b>{nameOf(r)}</b></td><td>{r.industry}</td><td>{r.segment}</td>
            <td className={r.tpv_change < 0 ? 'neg' : 'posv'}>{r.tpv_change > 0 ? '+' : ''}{pct(r.tpv_change)}</td><td>{whyLeft(r)}</td><td className="muted">{r.account_manager_note ?? ''}</td></tr>)}</tbody></table></div>
      </>}
    </section>
  )
}

// ---------------------------------------------------------------- upload drawer, method, confetti
function UploadDrawer({ upload, isPack, onClose, onFile, onReset }: { upload: any; isPack: boolean; onClose: () => void; onFile: (f: File) => void; onReset: () => void }) {
  const [over, setOver] = useState(false)
  const inp = useRef<HTMLInputElement>(null)
  return (
    <div className="overlay left" onClick={onClose}>
      <aside className="drawer" onClick={e => e.stopPropagation()} role="dialog" aria-label="Upload data">
        <div className="drawer-head"><h2>Score a different month</h2><button className="icon close" onClick={onClose} aria-label="Close">✕</button></div>
        <p className="muted">Upload the data pack, or any .xlsx or .csv with the same columns as its P2_Merchants tab. The file is read in your browser and is not uploaded anywhere.</p>
        <div className={`drop ${over ? 'over' : ''}`} onClick={() => inp.current?.click()} onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) onFile(f) }}>
          <div className="drop-icon">⤒</div><b>Drop a file here</b><span className="muted">or click to choose one · .xlsx or .csv</span>
          <input ref={inp} type="file" accept=".xlsx,.csv" hidden onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = '' }} />
        </div>
        {upload.state === 'reading' && <div className="note info"><span className="spinner small" /> Scoring the file…</div>}
        {upload.state === 'done' && <div className="note ok">✓ {upload.message}</div>}
        {upload.state === 'error' && <div className="note bad"><b>{upload.message}</b>{upload.checks && <ul>{upload.checks.filter((c: Check) => c.status !== 'PASS').map((c: Check) => <li key={c.check}>{c.status === 'FAIL' ? '✗' : '!'} {c.check}{c.detail ? `: ${c.detail}` : ''}</li>)}</ul>}</div>}
        {!isPack && <button className="btn" onClick={onReset}>Back to the data pack</button>}
        <p className="muted tiny">AI-written drafts and answers are saved for the data pack. For a new file, each merchant gets a simple starting message that follows the same safety rules.</p>
      </aside>
    </div>
  )
}

function HowModal({ S, V, onClose }: { S: any; V: any; onClose: () => void }) {
  return (
    <div className="overlay center" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()} role="dialog" aria-label="How these are picked">
        <div className="drawer-head"><h2>How these are picked</h2><button className="icon close" onClick={onClose} aria-label="Close">✕</button></div>
        <h3 className="how-h">How the money is worked out</h3>
        <ol className="how">
          <li><b>Annual value of a merchant.</b> The data pack's rule: 12 months of net revenue at the latest month's volume, with net revenue at 1% of payment volume.</li>
          <li><b>Revenue at risk ({rs(V.at_risk_value)}).</b> The annual value of every declining merchant, as if all of them left.</li>
          <li><b>Reached by the calls ({rs(V.call_list_value)}).</b> The annual value of the merchants on this month's calls.</li>
          <li><b>Likely saving ({rs(V.call_list_value * TESTED_LEAVE_RATE * SAVE_RATE - V.outreach_cost)} a year).</b> In testing on four datasets with answer keys, 88 of 96 flagged merchants did leave, so about 9 in 10. The data has no record of what happened after a call, so the share a call keeps comes from a published randomised trial in which likely leavers were listed to be called: their leaving rate fell from 17.2% to 15.3%, so about 1 in 9 were kept (Godinho de Matos, Ferreira and Belo, Marketing Science, 2018). The cost of the calls, {rs(V.outreach_cost)}, is taken off. Each call pays for itself if about 2 in 100 calls work.</li>
          <li><b>Optimistic case ({rs(V.call_list_value * TESTED_LEAVE_RATE * SAVE_RATE_OPTIMISTIC - V.outreach_cost)} a year).</b> If 3 in 10 calls keep the merchant, the default used in churn research (Verbraken, Verbeke and Baesens, 2013). This is an assumption, not a measurement.</li>
          <li><b>Limits.</b> The trial was with telecom customers, not merchants with a dedicated CSM. Outreach can also backfire: in another trial, a retention campaign raised leaving from about 6% to 10% (Ascarza, Iyengar and Schleicher, 2016). Recording who stayed after each call would replace these borrowed figures with Razorpay's own.</li>
        </ol>
        <h3 className="how-h">How the calls are picked</h3>
        <ol className="how">
          <li><b>Declining.</b> Payment volume over the last three months is more than {pct(Math.abs(S.decline_line), 1)} below the first three months, and it has not recovered. That threshold is three times the normal month-to-month variation ({pct(S.typical_wobble, 1)}), so ordinary ups and downs are not flagged.</li>
          <li><b>Risk level.</b> <i>Critical</i>: on the current trend, more than half of the volume is gone within about 90 days (by month 9). <i>High</i>: declining, and the contract renews within 90 days. <i>Moderate</i>: declining, but not yet on track to lose half.</li>
          <li><b>Main reason.</b> Each merchant's payment success rate, unresolved tickets, support rating, logins and products used are compared with all other merchants. If none stands out, the likely cause is pricing, to be confirmed on the call. Account manager notes support this but do not decide it.</li>
          <li><b>Playbook.</b> One of the six playbooks in the data pack, matched to the main reason. Merchant size decides who runs it.</li>
          <li><b>Ranking.</b> Merchants are ranked by urgency × how fixable the cause is × annual revenue, minus Rs 1,500 per call. Critical merchants count fully and others half, because they are losing volume faster. Fixability is an estimate (payment failures and support issues highest, then pricing, then low engagement). These weights only set the order, not the money figures above, and the list is re-checked under 11 different sets of them. <i>Close call</i> marks merchants that drop out of the top 15 under some.</li>
          <li><b>Role of AI.</b> AI drafts the summaries and messages and checks translations. It does not decide who is on the list. Every message is approved by a CSM before it is sent.</li>
        </ol>
      </div>
    </div>
  )
}

function Confetti() {
  const bits = useMemo(() => Array.from({ length: 28 }, (_, i) => ({ left: Math.random() * 100, delay: Math.random() * 0.4, rot: Math.random() * 360, c: ['#3395FF', '#012652', '#22C55E', '#F59E0B'][i % 4] })), [])
  return <div className="confetti" aria-hidden>{bits.map((b, i) => <i key={i} style={{ left: `${b.left}%`, animationDelay: `${b.delay}s`, background: b.c, transform: `rotate(${b.rot}deg)` }} />)}</div>
}
