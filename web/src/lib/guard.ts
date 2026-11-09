// Re-check of a message a person edited, at the moment they approve it (port of ai/guardrails.py check_edit).
// The CSM fills in the two time options, so numbers and day names are allowed; promises and internal information are not.
const PROMISES = [/refund/g, /waiv/g, /discount/g, /\bfree\b/g, /cash ?back/g, /compensat/g, /guarantee/g, /\bpromise/g, /rebate/g,
  /\bcredit(?! card)/g, /lower (your )?(rate|price|fee|mdr)/g, /reduc\w* (your )?(rate|price|fee|charges|mdr)/g, /\bmdr\b/g,
  /better (rate|price|deal|pricing)/g, /special (rate|price|offer|pricing)/g]
const TIMELINES = [/\bwithin\b/g, /\b(by|before) (monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|end of|the end)/g,
  /\b(a|an|one|two|three|four|five|six|seven|a few|few|couple of) (business |working )?(hours?|days?|weeks?)\b/g, /\bdeadline/g,
  /\bimmediately\b/g, /\bright away\b/g, /\bsame day\b/g, /\bovernight\b/g, /\bturnaround/g,
  /will (be )?(fixed|resolved|closed|sorted|done) (by|in|within)/g, /\bclosure date/g]
const INTERNAL = [/churn/g, /\brisk/g, /at-risk/g, /\bscore/g, /\btier/g, /\bflag/g, /\bleav(e|ing)\b/g, /\bswitch/g, /moving (your )?(business|volume|payments)/g,
  /\bvolume/g, /\btpv\b/g, /drop in (your )?(business|sales|revenue)/g, /\bsales\b/g, /\brevenue/g, /competitor/g, /other provider/g,
  /(account manager|internal|our|account) notes?\b/g, /\bnotes? (say|says|show|shows|suggest|mention)/g, /our (records|systems?) show/g, /\bmodel\b/g, /\banalytics?\b/g,
  /payu/g, /cashfree/g, /paytm/g, /phonepe/g, /stripe/g, /ccavenue/g, /juspay/g, /instamojo/g, /billdesk/g, /easebuzz/g, /pine ?labs/g]

const find = (pats: RegExp[], text: string) => {
  const t = text.toLowerCase(), hits = new Set<string>()
  pats.forEach(p => { for (const m of t.matchAll(new RegExp(p.source, 'g'))) hits.add(m[0]) })
  return [...hits].sort()
}

export type EditCheck = { label: string; words: string[]; hard: boolean }

export function checkEdit(text: string, english: boolean): EditCheck[] {
  const t = text.replace(/^subject:\s*/i, '')
  const out: EditCheck[] = []
  if (english) {
    out.push({ label: 'a promise (refund, waiver, discount or price change)', words: find(PROMISES, t), hard: true })
    out.push({ label: 'a promised deadline', words: find(TIMELINES, t).filter(h => !/^(by|before) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)$/.test(h)), hard: true })
    out.push({ label: 'internal information (volume, leaving, risk, notes or competitors)', words: find(INTERNAL, t), hard: true })
  } else {
    out.push({ label: 'English words in the regional message', words: [...new Set(t.match(/[A-Za-z][A-Za-z.'-]*/g) ?? [])].slice(0, 8), hard: false })
  }
  out.push({ label: 'time slots still to fill in', words: [...new Set(t.match(/\[[^\]]+\]/g) ?? [])], hard: false })
  return out.filter(c => c.words.length)
}

// ---------------------------------------------------------------- full draft checks (port of check_draft in ai/guardrails.py)
const TOPIC_WORDS: Record<string, RegExp[]> = {
  performance: [/fail/g, /success rate/g, /declin\w* (payment|transaction)/g, /payments? (are|were|have been) not going through/g, /drop-?offs?/g],
  service: [/ticket/g, /support request/g, /open request/g, /complain/g, /feedback on (our |razorpay )?support/g, /unresolved/g],
  engagement: [/dashboard/g, /log ?in/g, /products?\b/g, /features?\b/g, /workshop/g, /walkthrough/g],
  renewal: [/renewal/g, /renew/g, /contract/g],
}
const DIAGNOSIS = [...TOPIC_WORDS.performance, ...TOPIC_WORDS.service, /dashboard/g, /log ?in/g]
export const hasDigits = (t: string) => /\p{Nd}/u.test(t)
export type CheckRow = [string, boolean, string]

export function checkDraft(english: string, topics: Set<string>, unclear: boolean, businessName: string, channel: string, subject = '',
  back: string | null = null, translationOk: boolean | null = null, unsupported: string[] | null = null): CheckRow[] {
  const out: CheckRow[] = []
  const texts: [string, string][] = [['English draft', `${subject}\n${english}`.trim()]]
  if (back) texts.push(['regional draft (translated back)', back])
  for (const [label, t] of texts) {
    const nameFree = businessName ? t.split(businessName).join('') : t    // a merchant's own name may contain digits
    out.push([`No numbers (${label})`, !hasDigits(nameFree.replace(/\[option [12]\]/g, '')), ''])
    let h = find(PROMISES, t); out.push([`No promises of refunds, waivers, discounts or price changes (${label})`, !h.length, h.join(', ')])
    h = find(TIMELINES, t); out.push([`No timelines or deadlines (${label})`, !h.length, h.join(', ')])
    h = find(INTERNAL, t); out.push([`No internal information: volume, leaving, risk, scores, notes, competitors (${label})`, !h.length, h.join(', ')])
    const stray = Object.keys(TOPIC_WORDS).filter(tp => find(TOPIC_WORDS[tp], t).length && !topics.has(tp))
    out.push([`Every topic raised is backed by the data (${label})`, !stray.length, stray.length ? 'not in fact sheet: ' + stray.join(', ') : ''])
    if (unclear) { h = find(DIAGNOSIS, t); out.push([`Cause unknown, so neutral check-in only (${label})`, !h.length, h.join(', ')]) }
  }
  const brackets = [...new Set(english.match(/\[[^\]]+\]/g) ?? [])].filter(b => !['[option 1]', '[option 2]'].includes(b))
  out.push(['Only allowed placeholders left for the CSM to fill', !brackets.length, brackets.join(', ')])
  out.push(['Addresses the right merchant by name', english.toLowerCase().includes(businessName.toLowerCase()), ''])
  const limit = channel.includes('WhatsApp') ? 80 : 140, n = english.split(/\s+/).filter(Boolean).length
  out.push([`Short enough for the channel (max ${limit} words)`, n <= limit, `${n} words`])
  if (translationOk !== null) out.push(['Regional and English versions say the same thing', translationOk, ''])
  if (unsupported !== null) out.push(['Independent check found no claim outside the fact sheet', !unsupported.length, unsupported.join('; ')])
  return out
}

export function numbersMatch(text: string, allowed: number[]): boolean {
  const found = [...text.matchAll(/(?<![A-Za-z\d])\d[\d,]*\.?\d*/g)].map(m => parseFloat(m[0].replace(/,/g, '').replace(/\.$/, '')))
  return found.every(x => allowed.some(a => Math.abs(x - a) <= Math.max(0.6, Math.abs(a) * 0.01)))
}
