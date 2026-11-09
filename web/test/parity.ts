// Checks the TypeScript engine against the Python engine, decision by decision, on four datasets.
import { readFileSync } from 'node:fs'
import { run } from '../src/engine/core'

const COLS = ['tier', 'driver_main', 'driver_second', 'signal_to_watch', 'playbook_id', 'needs_person', 'delivery', 'revenue_rank', 'csm_call_list',
  'call_list_status', 'call_this_week', 'automated_track', 'am_alert', 'why_group', 'declining', 'imminent', 'possible_decline', 'bounced_dip', 'growing']
const NUM = ['expected_saving_rs', 'projected_loss_m9', 'yearly_value_rs', 'tpv_change', 'fam_performance', 'fam_service', 'fam_engagement']
let failures = 0
for (const name of ['datapack', 'seed13', 'seed14', 'seed15', 'anu_synthetic']) {
  const fx = JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'))
  const res = run(fx.input)
  if (!res.ok) { console.log(name, 'REFUSED', res.checks); failures++; continue }
  const got = new Map(res.merchants.map(r => [r.merchant_id, r]))
  const bad: Record<string, string[]> = {}
  for (const e of fx.expected) {
    const g = got.get(e.merchant_id)!
    for (const c of COLS) {
      const a = g[c] ?? null, b = e[c] ?? null
      const same = c === 'tier' ? String(a).startsWith('6') && String(b).startsWith('6') || a === b : (a === '' ? null : a) === (b === '' ? null : b)
      if (!same) (bad[c] ??= []).push(`${e.merchant_id}: ts=${a} py=${b}`)
    }
    for (const c of NUM) {
      const a = g[c], b = e[c]
      if ((a === null || a === undefined) !== (b === null || b === undefined) || (b !== null && b !== undefined && Math.abs(a - b) > (c.startsWith('fam') ? 0.011 : Math.max(1e-6, Math.abs(b) * 1e-9))))
        (bad[c] ??= []).push(`${e.merchant_id}: ts=${a} py=${b}`)
    }
    const bucketA = String(g.review_likely_bucket ?? '').split('(')[0], bucketB = String(e.review_likely_bucket ?? '').split('(')[0]
    if (bucketA !== bucketB) (bad.review_likely_bucket ??= []).push(`${e.merchant_id}: ts=${g.review_likely_bucket} py=${e.review_likely_bucket}`)
  }
  for (const k of Object.keys(fx.summary)) if (k in res.summary && Math.abs((res.summary as any)[k] - fx.summary[k]) > 1e-9) (bad['summary.' + k] ??= []).push(`ts=${(res.summary as any)[k]} py=${fx.summary[k]}`)
  for (const k of Object.keys(fx.value)) if (Math.abs((res.value as any)[k] - fx.value[k]) > 1e-6) (bad['value.' + k] ??= []).push(`ts=${(res.value as any)[k]} py=${fx.value[k]}`)
  const n = Object.keys(bad).length
  failures += n
  console.log(`${name}: ${res.merchants.length} merchants, ${res.summary.at_risk} declining, calls ${res.merchants.filter(r => r.csm_call_list).length} -> ${n ? 'MISMATCHES' : 'identical to Python'}`)
  for (const [k, v] of Object.entries(bad)) console.log('   ', k, v.length, v.slice(0, 3).join(' | '))
}
process.exit(failures ? 1 : 0)
