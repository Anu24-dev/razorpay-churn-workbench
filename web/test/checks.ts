import { readFileSync } from 'node:fs'
import { checkDraft } from '../src/lib/guard'
import { merchantFacts } from '../src/lib/facts'
import { channelFor } from '../src/lib/text'
const d = JSON.parse(readFileSync(new URL('../public/data/datapack.json', import.meta.url), 'utf8'))
let same = 0, total = 0
for (const r of d.merchants.filter((m: any) => m.csm_call_list)) {
  const dr = d.ai[r.merchant_id].draft
  const ts = checkDraft(dr.english.message, new Set(merchantFacts(r).map(f => f.topic)), r.driver_main === 'no operational cause found', r.business_name, channelFor(r.segment), dr.english.subject ?? '')
  for (const [name, ok] of ts) {
    const py = dr.checks.find((c: any) => c[0] === name)
    if (!py) continue
    total++; if (py[1] === ok) same++; else console.log('DIFF', r.merchant_id, name, 'ts', ok, 'py', py[1])
  }
}
console.log(`English draft checks: ${same} of ${total} agree with Python`)
