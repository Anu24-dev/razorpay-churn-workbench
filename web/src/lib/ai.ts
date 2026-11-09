// Live AI (only when the server has an API key). Each step is one short request to /api/ai; the checks run here, in code.
import type { Row } from '../engine/core'
import { NC } from '../engine/core'
import { merchantFacts, factText, internalFacts, languages, SENDER_IN_SCRIPT, channel } from './facts'
import { checkDraft, hasDigits, numbersMatch, type CheckRow } from './guard'
import { SENDER } from './text'

export async function liveAvailable(): Promise<boolean> {
  try { const r = await fetch('/api/ai'); if (!r.ok) return false; return Boolean((await r.json()).live) } catch { return false }
}
async function step(body: Record<string, unknown>) {
  const r = await fetch('/api/ai', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  const j = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(j.error ?? `AI request failed (${r.status})`)
  return j
}

export type Draft = { main_language: string; regional_language?: string | null; english: { subject?: string; message: string }; regional?: { subject?: string; message: string } | null; passed: boolean; checks: CheckRow[] }

async function draftOnce(r: Row, feedback: string | null, say: (m: string) => void): Promise<Draft> {
  const facts = merchantFacts(r), [main, alt] = languages(r), regional = main !== 'English' ? main : alt
  const ch = channel(r), unclear = r.driver_main === NC, name = String(r.business_name ?? r.merchant_id)
  let user = `Business name: ${name}\nChannel: ${ch}\nCause: ${unclear ? 'unclear (neutral check-in only)' : 'known'}\nFacts you may use:\n${factText(facts)}`
  if (feedback) user += `\n\nYour previous draft broke these rules, fix them: ${feedback}`
  say(feedback ? 'Writing the English message again, fixing what the checks found' : 'Writing the English message')
  const en = await step({ step: 'draft', sender: SENDER, length: ch.includes('WhatsApp') ? 'Keep the message under 70 words.' : 'Keep the message under 120 words, plus a short subject line.', user })
  let reg: any = null, back: string | null = null, same: any = null
  if (regional) {
    say(`Writing the ${regional} version`)
    let tIn = JSON.stringify({ subject: en.subject ?? '', message: en.message ?? '', sender_name_spelt_in_this_script: SENDER_IN_SCRIPT[regional] ?? SENDER })
    if (feedback) tIn += `\n\nThe previous version broke these rules, fix them: ${feedback}`
    reg = await step({ step: 'translate', language: regional, user: tIn })
    if (reg?.message) {
      say(`Translating the ${regional} version back to English to check it`)
      back = (await step({ step: 'back', language: regional, user: `${reg.subject ?? ''}\n${reg.message}`.trim() })).text
      same = await step({ step: 'same', user: `FIRST:\n${en.subject ?? ''}\n${en.message ?? ''}\n\nSECOND:\n${back}` })
    }
  }
  say("Checking every claim against the merchant's data")
  const ver = await step({ step: 'verify', user: `Allowed facts:\n${factText(facts)}\n\nMessage:\n${en.message ?? ''}` })
  const checks = checkDraft(String(en.message ?? ''), new Set(facts.map(f => f.topic)), unclear, name, ch, en.subject ?? '', back,
    same ? Boolean(same.same_meaning) : null, Array.isArray(ver?.unsupported_claims) ? ver.unsupported_claims : null)
  if (regional) {
    checks.push([`${regional} draft produced and translated back for checking`, Boolean(reg?.message && back), ''])
    if (reg?.message) {
      const t = `${reg.subject ?? ''}\n${reg.message}`
      checks.push([`No numbers in the ${regional} draft itself`, !hasDigits(t), ''])
      const latin = [...new Set(t.match(/[A-Za-z][A-Za-z.'-]*/g) ?? [])]
      checks.push([`${regional} draft has no English words in it`, !latin.length, latin.slice(0, 8).join(', ')])
    }
  }
  return { main_language: main, regional_language: regional, english: { subject: en.subject ?? '', message: en.message ?? '' }, regional: reg?.message ? reg : null,
    passed: checks.every(c => c[1]), checks }
}

// One retry if a check fails; if it fails again the draft is marked as stopped and the CSM rewrites it.
export async function draftWithRetry(r: Row, say: (m: string) => void): Promise<Draft> {
  const d = await draftOnce(r, null, say)
  if (d.passed) return d
  const feedback = d.checks.filter(c => !c[1]).map(c => (c[2] ? `${c[0]}: ${c[2]}` : c[0])).join('; ')
  return draftOnce(r, feedback, say)
}

function recordNumbers(rec: any): number[] {
  const nums: number[] = []
  const add = (x: any) => { if (typeof x === 'number' && !Number.isNaN(x)) nums.push(x, Math.abs(x), Math.round(Math.abs(x))) }
  Object.values(rec).forEach((v: any) => (Array.isArray(v) ? v.forEach(add) : v && typeof v === 'object' ? Object.values(v).forEach(add) : add(v)))
  return [...nums, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

// An answer that quotes a number not in the merchant's record is hidden, as in the Python version.
export async function answerQuestion(r: Row, question: string, playbookName: string): Promise<{ text: string; ok: boolean }> {
  const rec = internalFacts(r, playbookName)
  const text = (await step({ step: 'ask', record: JSON.stringify(rec), question })).text as string
  return { text, ok: numbersMatch(text, [...recordNumbers(rec), 1500, 15, 50, 90, 60, 30, 12, 100]) }
}
