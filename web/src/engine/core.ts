/*
 * Churn early-warning engine: a line-by-line port of engine/core.py (the tested Python engine).
 * Rules and statistics only, no AI. `npm run parity` checks every decision against the Python results on four datasets.
 * Not ported: the experimental watch list (tier 5), which needs scikit-learn. It never changes the calls; the data pack's
 * own results (with the watch list) are loaded from public/data/datapack.json.
 */

export type Row = Record<string, any>

export const WOBBLES_FOR_DECLINE = 3
export const POSSIBLE_DECLINE_WOBBLES = 2
export const RECOVERY_VOTES_NEEDED = 3
export const IMMINENT_LOSS = 0.5
export const SWITCH_WINDOW_DAYS = 90
export const CALL_THIS_WEEK_DAYS = 30
export const PB3_RENEWAL_DAYS = 60
export const DRIVER_LINE_SD = 1.0
export const SIZE_GAP_P = 0.05
export const CAPACITY = 15
export const OUTREACH_COST = 1500
export const NET_REVENUE_RATE = 0.01
export const LAKH = 100000
export const P_LEAVE = { imminent: 1.0, other: 0.5 }
export const NC = 'no operational cause found'
export const SAVE_CHANCE: Record<string, number> = { performance: 0.6, service: 0.6, [NC]: 0.4, engagement: 0.3 }
export const SEASONAL_INDUSTRIES = ['Gifting', 'Education', 'Travel']
export const TPV = [1, 2, 3, 4, 5, 6].map(i => `tpv_m${i}_inr_lakh`)
export const REQUIRED = ['merchant_id', 'segment', 'industry', 'tenure_months', ...TPV,
  'success_rate_m1_m3_pct', 'success_rate_m4_m6_pct', 'support_tickets_last_90d', 'unresolved_tickets',
  'last_csat_1_to_5', 'products_used', 'logins_last_30d', 'days_to_contract_renewal', 'account_manager_note']
const NUMERIC = [...TPV, 'tenure_months', 'success_rate_m1_m3_pct', 'success_rate_m4_m6_pct', 'support_tickets_last_90d', 'unresolved_tickets',
  'last_csat_1_to_5', 'products_used', 'logins_last_30d', 'days_to_contract_renewal']

export type Playbook = { name: string; what_happens: string; when_it_applies: string }
export const PLAYBOOKS: Record<string, Playbook> = {
  'PB-1': { name: 'Performance deep-dive', what_happens: 'Joint call with merchant and Payments Ops to review success-rate drop by method/bank; share fix plan within 3 days.', when_it_applies: 'Success rate fell sharply or merchant complains of failures' },
  'PB-2': { name: 'Support recovery', what_happens: 'CSM takes ownership of all open tickets, gives a single point of contact and a closure date; senior callback within 48 hours.', when_it_applies: 'Multiple unresolved tickets, low CSAT' },
  'PB-3': { name: 'Commercial review', what_happens: 'Account Manager reviews rate card, volume tiers and renewal terms before the renewal date.', when_it_applies: 'Competitor mention, renewal within ~60 days' },
  'PB-4': { name: 'Re-engagement and adoption', what_happens: 'Executive-sponsor outreach plus a short workshop on unused products and reports; re-confirm the right contact person.', when_it_applies: 'Low logins, few products used, no response to outreach' },
  'PB-5': { name: 'Growth / expansion check-in', what_happens: 'Positive check-in; offer new products or higher-limit plans.', when_it_applies: 'Healthy and growing' },
  'PB-6': { name: 'Monitor only', what_happens: 'No outreach; keep on watchlist and re-check next month.', when_it_applies: 'Seasonal dip, low-volume stable, or no real risk signal' },
}
export const DRIVER_PB_ID: Record<string, string> = { performance: 'PB-1', service: 'PB-2', engagement: 'PB-4', pricing: 'PB-3' }
export const DELIVERY: Record<string, string> = { Enterprise: 'Account manager, with a senior sponsor', Mid: 'CSM call', Small: 'CSM via WhatsApp or email' }
export const SEGMENTS = Object.keys(DELIVERY)
export const deliveryFor = (seg: string) => DELIVERY[seg] ?? DELIVERY.Mid

// ---------------------------------------------------------------- small statistics helpers (pandas conventions: std with ddof = 1, NaN skipped)
const isNum = (v: any) => typeof v === 'number' && !Number.isNaN(v)
const mean = (xs: number[]) => { const v = xs.filter(isNum); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN }
const std = (xs: number[]) => { const v = xs.filter(isNum); if (v.length < 2) return NaN; const m = mean(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) }
const median = (xs: number[]) => { const v = xs.filter(isNum).sort((a, b) => a - b); const n = v.length; return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2 }
const quantile = (xs: number[], q: number) => { const v = xs.filter(isNum).sort((a, b) => a - b); const pos = (v.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos); return v[lo] + (v[hi] - v[lo]) * (pos - lo) }
const zs = (xs: number[]) => { const m = mean(xs), s = std(xs); return xs.map(x => (x - m) / s) }

function segRelative(rows: Row[], col: string): number[] {
  const groups = new Map<string, number[]>()
  rows.forEach(r => { const g = groups.get(r.segment) ?? []; g.push(r[col]); groups.set(r.segment, g) })
  const stats = new Map<string, [number, number]>()
  groups.forEach((v, k) => { let s = std(v); if (s === 0) s = 1; stats.set(k, [mean(v), s]) })
  return rows.map(r => { const [m, s] = stats.get(r.segment)!; return (r[col] - m) / s })
}

// Kruskal-Wallis H test with tie correction, as scipy.stats.kruskal (p from the chi-square distribution)
function lnGamma(z: number): number {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z)
  z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i)
  const t = z + g + 0.5
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x)
}
function gammaQ(a: number, x: number): number {   // regularised upper incomplete gamma
  if (x <= 0) return 1
  const gln = lnGamma(a)
  if (x < a + 1) {
    let sum = 1 / a, del = sum, ap = a
    for (let n = 0; n < 500; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln)
  }
  let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a); b += 2
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300
    d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h
}
export function kruskalP(groups: number[][]): number {
  const gs = groups.map(g => g.filter(isNum)).filter(g => g.length > 0)
  if (gs.length < 2) return NaN
  const all = gs.flatMap((g, gi) => g.map(v => ({ v, gi }))).sort((a, b) => a.v - b.v)
  const N = all.length, ranks = new Array(N).fill(0)
  let tieSum = 0
  for (let i = 0; i < N;) {
    let j = i; while (j + 1 < N && all[j + 1].v === all[i].v) j++
    const r = (i + j) / 2 + 1; for (let k = i; k <= j; k++) ranks[k] = r
    const t = j - i + 1; tieSum += t ** 3 - t; i = j + 1
  }
  const C = 1 - tieSum / (N ** 3 - N)
  if (C === 0) return NaN                          // every value identical: scipy raises, the engine treats it as "no size gap"
  const sumR = gs.map(() => 0), n = gs.map(g => g.length)
  all.forEach((e, i) => { sumR[e.gi] += ranks[i] })
  let H = 12 / (N * (N + 1)) * sumR.reduce((a, R, i) => a + R * R / n[i], 0) - 3 * (N + 1)
  H /= C
  return gammaQ((gs.length - 1) / 2, H / 2)
}

// ---------------------------------------------------------------- loading and checks
export type Check = { check: string; status: 'PASS' | 'WARN' | 'FAIL'; detail: string }

export function tidy(input: Row[]): Row[] {
  return input.map(raw => {
    const r: Row = {}
    Object.keys(raw).forEach(k => { r[String(k).trim()] = raw[k] })
    if ('merchant_id' in r) r.merchant_id = String(r.merchant_id ?? '').trim()
    for (const c of NUMERIC) {
      if (!(c in r)) continue
      const v = r[c]
      if (v === null || v === undefined || v === '' || (typeof v === 'string' && ['nan', ''].includes(v.trim().toLowerCase()))) { r[c] = null; continue }
      if (typeof v === 'number') continue
      const n = Number(String(v).trim()); r[c] = Number.isNaN(n) ? v : n
    }
    if (typeof r.segment === 'string') { const s = r.segment.trim().toLowerCase(); const hit = SEGMENTS.find(x => x.toLowerCase() === s); if (hit) r.segment = hit }
    return r
  })
}

export function checkData(rows: Row[]): Check[] {
  const out: Check[] = []
  const check = (name: string, ok: boolean, detail = '', warn = false) => out.push({ check: name, status: ok ? 'PASS' : warn ? 'WARN' : 'FAIL', detail })
  const cols = new Set(rows.flatMap(r => Object.keys(r)))
  const miss = REQUIRED.filter(c => !cols.has(c))
  check('Required columns present', !miss.length, miss.length ? `missing: ${miss.join(', ')}` : '')
  if (miss.length) return out
  const textIn = NUMERIC.filter(c => rows.some(r => r[c] !== null && r[c] !== undefined && typeof r[c] !== 'number'))
  check('Number columns hold only numbers', !textIn.length, textIn.map(c => `${c}: ${rows.filter(r => r[c] !== null && typeof r[c] !== 'number').slice(0, 3).map(r => r.merchant_id).join(', ')}`).join('; '))
  if (textIn.length) return out
  check('At least 30 merchants', rows.length >= 30, `${rows.length} rows`)
  const seen = new Set<string>(), dups = new Set<string>()
  rows.forEach(r => { if (seen.has(r.merchant_id)) dups.add(r.merchant_id); seen.add(r.merchant_id) })
  check('Merchant IDs are unique', !dups.size, dups.size ? `repeated: ${[...dups].slice(0, 5).join(', ')}` : '')
  const blanks = NUMERIC.filter(c => c !== 'last_csat_1_to_5' && rows.some(r => r[c] === null || r[c] === undefined))
  check('No blanks (except CSAT)', !blanks.length, blanks.map(c => `${c}: ${rows.filter(r => r[c] === null || r[c] === undefined).length} blank`).join('; '))
  const odd = [...new Set(rows.map(r => r.segment).filter(s => s && !SEGMENTS.includes(s)))]
  check('Segments are Enterprise, Mid or Small', !odd.length, odd.length ? `other values: ${odd.join(', ')} (handled like Mid)` : '', true)
  if (blanks.length) return out
  const zero = rows.filter(r => TPV.some(c => !(r[c] > 0))).length
  check('TPV is above zero', zero === 0, zero ? `${zero} merchants with a zero or negative month` : '')
  check('Success rates between 0 and 100', rows.every(r => ['success_rate_m1_m3_pct', 'success_rate_m4_m6_pct'].every(c => r[c] >= 0 && r[c] <= 100)))
  check('CSAT is 1 to 5 where present', rows.every(r => r.last_csat_1_to_5 === null || r.last_csat_1_to_5 === undefined || (r.last_csat_1_to_5 >= 1 && r.last_csat_1_to_5 <= 5)))
  const cnt = ['support_tickets_last_90d', 'unresolved_tickets', 'products_used', 'logins_last_30d', 'days_to_contract_renewal', 'tenure_months']
  check('Counts are whole and not negative', rows.every(r => cnt.every(c => r[c] >= 0 && r[c] % 1 === 0)))
  return out
}

// ---------------------------------------------------------------- stage 1: volume
function project(win: number[], h: number, method: string): number {
  if (method === 'straight line') return Math.max(win[2] + (win[2] - win[0]) / 2 * h, 0)
  const r = Math.sqrt(win[2] / win[0])
  if (method === 'compound') return win[2] * r ** h
  let s = 0; for (let k = 1; k <= h; k++) s += 0.8 ** k
  return win[2] * Math.exp(Math.log(r) * s)                 // damped
}

function volume(df: Row[]) {
  df.forEach(r => {
    const T = TPV.map(c => r[c] as number)
    r._T = T
    r.tpv_base = mean(T.slice(0, 3)); r.tpv_peak = Math.max(...T.slice(0, 3)); r.tpv_recent = mean(T.slice(3))
    r.tpv_change = r.tpv_recent / r.tpv_base - 1
    r._own_var = (std(T.slice(0, 3)) / r.tpv_base) ** 2
  })
  const wobble = Math.sqrt(median(df.map(r => r._own_var)))
  const declineLine = -WOBBLES_FOR_DECLINE * wobble
  const sharpLine = declineLine * Math.sqrt(1 + 1 / 3)
  const pl = -POSSIBLE_DECLINE_WOBBLES * wobble
  df.forEach(r => {
    const [, , , m4, m5, m6] = r._T
    const bounce = [m6 > m5, m6 > Math.min(m4, m5), m6 > m4, m6 > (m4 + m5) / 2]
    r.no_recovery_votes = bounce.filter(b => !b).length
    r.m6_change = m6 / r.tpv_base - 1
    const dropped = (avg: number, m6l: number) => (r.tpv_change <= avg && r.tpv_change < 0) || (r.m6_change <= m6l && r.m6_change < 0)
    const big = dropped(declineLine, sharpLine)
    r.declining = big && r.no_recovery_votes >= RECOVERY_VOTES_NEEDED
    r.bounced_dip = big && !r.declining
    r.possible_decline = !big && dropped(pl, pl * Math.sqrt(4 / 3)) && r.no_recovery_votes >= RECOVERY_VOTES_NEEDED
    r.growing = r.tpv_change >= -declineLine && r.tpv_change > 0
    r.tpv_m9_projected = project(r._T.slice(3), 3, 'compound')
    r.projected_loss_m9 = 1 - r.tpv_m9_projected / r.tpv_peak
    r.imminent = r.declining && r.projected_loss_m9 > IMMINENT_LOSS
  })
  return { typical_wobble: wobble, decline_line: declineLine }
}

// ---------------------------------------------------------------- stage 2: size check
function sizeCheck(df: Row[]) {
  const adjust: Record<string, boolean> = {}
  for (const col of ['logins_last_30d', 'products_used']) {
    const groups = new Map<string, number[]>()
    df.forEach(r => { const g = groups.get(r.segment) ?? []; g.push(r[col]); groups.set(r.segment, g) })
    const p = kruskalP([...groups.values()])
    adjust[col] = p < SIZE_GAP_P
  }
  return adjust
}

// ---------------------------------------------------------------- stage 3: drivers
function drivers(df: Row[], adjust: Record<string, boolean>) {
  const col = (c: string) => df.map(r => r[c] as number)
  const csat = df.map(r => (isNum(r.last_csat_1_to_5) ? r.last_csat_1_to_5 : NaN))
  const cm = mean(csat), cs = std(csat)
  const csatZ = csat.map(v => (isNum(v) ? (v - cm) / cs : 0))
  const srDrop = df.map(r => r.success_rate_m1_m3_pct - r.success_rate_m4_m6_pct)
  const zDrop = zs(srDrop), zRecent = zs(col('success_rate_m4_m6_pct'))
  const perf = zs(zDrop.map((v, i) => v - zRecent[i]))
  const zUnres = zs(col('unresolved_tickets'))
  const serv = zs(zUnres.map((v, i) => v - csatZ[i]))
  const eng = (c: string) => (adjust[c] ? segRelative(df, c) : col(c))
  const zL = zs(eng('logins_last_30d')), zP = zs(eng('products_used'))
  const engagement = zs(zL.map((v, i) => -v - zP[i]))
  const FAM = ['performance', 'service', 'engagement']
  df.forEach((r, i) => {
    const f: Record<string, number> = { performance: perf[i], service: serv[i], engagement: engagement[i] }
    const order = FAM.filter(k => f[k] > DRIVER_LINE_SD).sort((a, b) => f[b] - f[a])
    let main: string | null = order[0] ?? null
    const second = order[1] ?? null
    const flag = r.declining || r.possible_decline
    if (flag && main === null) main = NC
    r.signal_to_watch = !flag && main !== null ? main : ''
    r.driver_main = flag ? main : null
    r.driver_second = flag ? second : null
    let strongest = FAM[0], sval = f[FAM[0]]
    FAM.forEach(k => { if (f[k] > sval) { sval = f[k]; strongest = k } })
    if (r.driver_main !== NC) r.review_likely_bucket = ''
    else if (r.days_to_contract_renewal <= PB3_RENEWAL_DAYS) r.review_likely_bucket = `Likely commercial (renews in ${Math.trunc(r.days_to_contract_renewal)} days)`
    else if (sval >= DRIVER_LINE_SD / 2) r.review_likely_bucket = `Likely ${strongest} (weak signal, score ${sval.toFixed(2)})`
    else r.review_likely_bucket = 'Likely commercial or external (no operational signal)'
    r.fam_performance = Math.round(f.performance * 100) / 100
    r.fam_service = Math.round(f.service * 100) / 100
    r.fam_engagement = Math.round(f.engagement * 100) / 100
  })
}

// ---------------------------------------------------------------- stages 4-5: tiers and playbooks
function tiersPlaybooks(df: Row[]) {
  df.forEach(r => {
    r.tier = r.imminent ? '1 Imminent fade'
      : r.declining && r.days_to_contract_renewal <= SWITCH_WINDOW_DAYS ? '2 Declining, renewal within 90 days'
      : r.declining ? '3 Declining'
      : r.possible_decline ? '3P Possible decline (automated track)'
      : r.bounced_dip ? '4 Bounced-back dip'
      : r.growing ? '6 Growing' : '6 Stable'
  })
  const bucketPb = (b: string) => {
    if (b.includes('commercial')) return 'PB-3'
    const k = ['performance', 'service', 'engagement'].find(x => b.includes(x))
    return k ? DRIVER_PB_ID[k] : 'PB-3'
  }
  df.forEach(r => {
    if (r.declining || r.possible_decline) r.playbook_id = r.driver_main === NC ? bucketPb(String(r.review_likely_bucket)) : DRIVER_PB_ID[r.driver_main]
    else if (r.growing && !(r.tier.startsWith('4') || r.tier.startsWith('5'))) r.playbook_id = 'PB-5'
    else r.playbook_id = 'PB-6'
    r.needs_person = (r.declining || r.possible_decline) && r.driver_main === NC
    r.delivery = r.declining ? deliveryFor(r.segment) : r.possible_decline ? 'Automated digital message' : ''
  })
}

// ---------------------------------------------------------------- stage 6: ranking
function topIds(dec: Row[], ev: number[]): Set<string> {
  return new Set(dec.map((r, i) => ({ id: r.merchant_id, ev: ev[i], i })).sort((a, b) => b.ev - a.ev || a.i - b.i).slice(0, CAPACITY).map(x => x.id))
}

function ranking(df: Row[]) {
  df.forEach(r => { r.yearly_value_rs = r[TPV[5]] * LAKH * NET_REVENUE_RATE * 12 })
  const dec = df.filter(r => r.declining)
  const loss = (m: string) => dec.map(r => 1 - project(r._T.slice(3), 3, m) / r.tpv_peak)
  const pLeave = (l = dec.map(r => r.projected_loss_m9), line = IMMINENT_LOSS, pOther = P_LEAVE.other, renewalP: number | null = null) =>
    dec.map((r, i) => {
      let p = l[i] > line ? P_LEAVE.imminent : pOther
      if (renewalP !== null && l[i] <= line && r.days_to_contract_renewal <= SWITCH_WINDOW_DAYS) p = renewalP
      return p
    })
  const EQUAL = { performance: 0.5, service: 0.5, [NC]: 0.5, engagement: 0.5 }
  const versions: [string, number[], Record<string, number>][] = [
    ['Main', pLeave(), SAVE_CHANCE],
    ['Other declining: 30% leave', pLeave(undefined, undefined, 0.3), SAVE_CHANCE],
    ['Other declining: 70% leave', pLeave(undefined, undefined, 0.7), SAVE_CHANCE],
    ['Equal 50% save chance', pLeave(), EQUAL],
    ['Unclear cause easier (often pricing)', pLeave(), { performance: 0.5, service: 0.5, [NC]: 0.6, engagement: 0.3 }],
    ['Engagement hardest', pLeave(), { performance: 0.6, service: 0.5, [NC]: 0.5, engagement: 0.2 }],
    ['Imminent line at 40%', pLeave(undefined, 0.4), SAVE_CHANCE],
    ['Imminent line at 60%', pLeave(undefined, 0.6), SAVE_CHANCE],
    ['Straight-line projection', pLeave(loss('straight line')), SAVE_CHANCE],
    ['Damped projection', pLeave(loss('damped')), SAVE_CHANCE],
    ['Renewal within 90 days: 70% leave', pLeave(undefined, undefined, undefined, 0.7), SAVE_CHANCE],
  ]
  const tops = versions.map(([, p, save]) => topIds(dec, dec.map((r, i) => p[i] * save[r.driver_main] * r.yearly_value_rs - OUTREACH_COST)))
  dec.forEach(r => {
    r.p_leave = r.imminent ? P_LEAVE.imminent : P_LEAVE.other
    r.save_chance = SAVE_CHANCE[r.driver_main]
    r.expected_saving_rs = r.p_leave * r.save_chance * r.yearly_value_rs - OUTREACH_COST
  })
  const sorted = [...dec].sort((a, b) => b.expected_saving_rs - a.expected_saving_rs || a.days_to_contract_renewal - b.days_to_contract_renewal)
  sorted.forEach((r, i) => { r.revenue_rank = i + 1 })
  const core = new Set([...tops[0]].filter(id => tops.every(t => t.has(id))))
  const anyv = new Set(tops.flatMap(t => [...t]))
  df.forEach(r => {
    if (!r.declining) { r.p_leave = null; r.save_chance = null; r.expected_saving_rs = null; r.revenue_rank = null }
    r.csm_call_list = r.revenue_rank !== null && r.revenue_rank <= CAPACITY
    r.call_list_status = r.csm_call_list && core.has(r.merchant_id) ? 'Firm' : r.csm_call_list ? 'Borderline'
      : r.declining && anyv.has(r.merchant_id) ? 'Next in line' : ''
    r.call_this_week = r.csm_call_list && r.days_to_contract_renewal <= CALL_THIS_WEEK_DAYS
    r.automated_track = (r.declining && !r.csm_call_list) || r.possible_decline
    r.am_alert = r.automated_track && r.driver_main === NC && r.days_to_contract_renewal <= PB3_RENEWAL_DAYS
  })
}

function valueCost(df: Row[]) {
  const c15 = df.filter(r => r.csm_call_list), atRisk = df.filter(r => r.declining)
  const saving = (save: Record<string, number> | number) =>
    c15.reduce((a, r) => a + r.p_leave * (typeof save === 'number' ? save : save[r.driver_main]) * r.yearly_value_rs - OUTREACH_COST, 0)
  const be = c15.length ? Math.max(...c15.map(r => OUTREACH_COST / (r.p_leave * r.yearly_value_rs))) : NaN
  return {
    at_risk_value: atRisk.reduce((a, r) => a + r.yearly_value_rs, 0), call_list_value: c15.reduce((a, r) => a + r.yearly_value_rs, 0),
    saving_ours: saving(SAVE_CHANCE), saving_30: saving(0.3), saving_10: saving(0.1), outreach_cost: c15.length * OUTREACH_COST,
    false_alarm_cost: c15.reduce((a, r) => a + (1 - r.p_leave) * OUTREACH_COST, 0), break_even_max: be,
  }
}

// ---------------------------------------------------------------- why not flagged
function whyNotFlagged(df: Row[]) {
  const tickHi = quantile(df.map(r => r.support_tickets_last_90d), 0.75)
  df.forEach(r => {
    if (r.declining) { r.why_group = null; r.why_not_flagged = null; return }
    const seasonal = SEASONAL_INDUSTRIES.some(x => String(r.industry).includes(x))
    let g: string, t: string
    if (r.tier.startsWith('3P')) { g = 'Possible decline'; t = 'Smaller drop, no bounce-back yet: automated digital check-in, not a call.' }
    else if (r.tier.startsWith('4')) { g = seasonal ? 'Seasonal dip' : 'Bounced-back dip'; t = seasonal ? `Dipped, then came back by month 6. Seasonal industry (${r.industry}).` : 'Dipped, then came back by month 6.' }
    else if (r.tier.startsWith('5')) { g = 'Watch list'; t = 'Volume steady, but behaviour looks like merchants who later declined. Experimental watch, no outreach.' }
    else if (r.growing) {
      if (r.support_tickets_last_90d >= tickHi && r.support_tickets_last_90d > 0) { g = 'Busy and growing'; t = 'Busy, not at risk.' }
      else { g = 'Growing'; t = 'Growing.' }
    }
    else if (r.segment === 'Small') { g = 'Small and steady'; t = 'Small merchant: low volume is normal, and it has not fallen.' }
    else { g = 'Steady'; t = 'Volume within normal month-to-month movement.' }
    r.why_group = g; r.why_not_flagged = t
  })
}

// ---------------------------------------------------------------- run everything
export type Result = {
  ok: true; checks: Check[]; merchants: Row[]
  summary: { merchants: number; at_risk: number; imminent: number; review: number; typical_wobble: number; decline_line: number }
  value: ReturnType<typeof valueCost>; playbooks: Record<string, Playbook>
} | { ok: false; checks: Check[]; error?: string }

export function run(input: Row[]): Result {
  const rows = tidy(input)
  const checks = checkData(rows)
  if (checks.some(c => c.status === 'FAIL')) return { ok: false, checks }
  const df = rows.map(r => ({ ...r }))
  const vol = volume(df)
  const adjust = sizeCheck(df)
  drivers(df, adjust)
  tiersPlaybooks(df)
  ranking(df)
  whyNotFlagged(df)
  df.forEach(r => { delete r._T; delete r._own_var })
  return {
    ok: true, checks, merchants: df, playbooks: PLAYBOOKS, value: valueCost(df),
    summary: { merchants: df.length, at_risk: df.filter(r => r.declining).length, imminent: df.filter(r => r.imminent).length,
      review: df.filter(r => r.declining && r.driver_main === NC).length, typical_wobble: vol.typical_wobble, decline_line: vol.decline_line },
  }
}
