// Server-side Claude calls for the web version. The API key lives only here (Vercel environment variable ANTHROPIC_API_KEY);
// the page never sees it. Each request is one short step, so no call runs into the function time limit.
// The prompts are the same as ai/pipeline.py in the Python version.

const MODEL_WRITE = 'claude-sonnet-5-5'
const MAX_CALLS_PER_INSTANCE = 300          // a cap per running server instance; the Anthropic console spend limit is the hard cap
let used = 0

const SYS_DRAFT = "You write a short first message from a Razorpay team member to a merchant, for a human to review before sending. Rules you must follow:\n" +
  "1. Use ONLY the numbered facts provided. Do not add any other claim about the merchant.\n" +
  "2. No numbers or digits of any kind, and no percentages.\n" +
  "3. Never mention payment volume, sales, business falling, the merchant leaving or switching, risk, scores, internal notes or competitors.\n" +
  "4. Never promise refunds, waivers, discounts, credits, price or fee changes, and never give a timeline or deadline " +
  "(no 'within', no days or hours, no 'by' a date).\n" +
  "5. Open with the merchant's business name. Say plainly that their account matters to Razorpay. Lead with the problem Razorpay can fix " +
  "(the first fact listed), then at most one more fact. If the cause is unclear, write a warm, neutral check-in: ask how things are going and offer " +
  "to review whether their current plan still fits their needs, with no diagnosis of any problem.\n" +
  "6. End with one easy ask: a short call, offering two slots written exactly as [option 1] and [option 2]. Sign off as {sender} from Razorpay.\n" +
  "7. Warm, direct, respectful. Write mostly as 'we' (the Razorpay team). No em dashes. {length}\n" +
  'Return JSON only: {"subject": "<email subject, or empty for WhatsApp>", "message": "...", "facts_used": ["F1", ...]}'

const SYS_TRANSLATE = "Translate this message from a Razorpay team member to a merchant into natural, polite, business {language}, the way a local " +
  "customer success manager would actually write it. Keep the meaning exactly. Write both the subject and the message in {language} script " +
  "(not in Roman letters). Prefer 'we' over 'I' so the text reads naturally whatever the sender's gender, and never use slashes for alternative word forms. " +
  "There must be NO English or Roman letters anywhere: write the business name, the word Razorpay and the sender's name in {language} script as they sound, " +
  "and write the two time placeholders [option 1] and [option 2] as short bracketed {language} words meaning 'first time option' and 'second time option', " +
  "with no digits. Do not add numbers, promises or new claims. Return exactly one JSON object and nothing else: " +
  '{"subject": "...", "message": "..."}'

const SYS_BACK = 'Translate this {language} text into plain English, word for word as far as possible. Do not improve or correct it. Return only the English translation.'

const SYS_SAME = 'Compare two English messages. The second is a back-translation of a translated version of the first. Do they say the same thing: ' +
  'same facts, same ask, no added or missing promises, claims or numbers? Small wording differences are fine. ' +
  'Return JSON only: {"same_meaning": true/false, "differences": ["..."]}'

const SYS_VERIFY = 'You check a message against a list of allowed facts. List every statement in the message about the merchant\'s situation that is NOT ' +
  'supported by the allowed facts. Courtesy phrases, the request for a call, and the offer to help or to review their plan are fine. ' +
  'Return JSON only: {"unsupported_claims": ["..."]}'

const SYS_QA = "You answer a Razorpay customer success manager's question about one merchant. Use ONLY the JSON record and the method notes given. " +
  'If the answer is not in them, say that the data does not show it. Copy numbers exactly. Never promise anything to the merchant, ' +
  'never invent facts, keep it to a few sentences, plain words, no em dashes.'

const METHOD_NOTES = "Method notes: A merchant is 'declining' if its average volume in months 4-6 fell more than 3 typical monthly wobbles below months 1-3 " +
  "(or month 6 alone fell sharply) and it did not bounce back. 'Imminent fade' (shown as Critical) = projected to lose more than half its healthy volume by month 9. " +
  'The driver is the strongest problem family (performance, service, engagement) more than 1 standard deviation above average; if none, the ' +
  'cause is unclear, the playbook follows the likely bucket, and a person confirms it before acting. Every playbook is one of the six in the data pack (PB-1 to PB-6). ' +
  'The 15 calls are ranked by chance of leaving x chance of saving x yearly value minus Rs 1,500; ' +
  "'Borderline' (shown as Close call) means the merchant drops out of the 15 under some reasonable assumptions. Chance of leaving and chance of saving are assumptions, not measured."

const SYS_REASON = 'You write a one or two sentence reason for a Razorpay customer success manager (internal, never sent to the merchant), ' +
  'explaining why this merchant is on the list and what the main problem is. Use only the facts in the JSON record. ' +
  'Numbers are allowed but must be copied exactly from the record (round percentages to whole numbers if you like). ' +
  "If driver_main is 'no operational cause found', say the numbers show no operational problem, mention the renewal timing, " +
  'and say what the account manager note suggests if there is one. Plain words, no jargon, no em dashes. Return only the reason.'

async function claude(system: string, user: string, maxTokens: number, model = MODEL_WRITE): Promise<string> {
  const call = async (mt: number) => {
    const body: any = { model, max_tokens: mt, system, messages: [{ role: 'user', content: user }] }
    if (model === MODEL_WRITE) body.thinking = { type: 'between_tools' }   // answer directly within max_tokens, as in the Python version
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', headers: { 'content-type': 'application/json', 'x-api-key': process.env.ANTHROPIC_API_KEY as string, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify(body),
    })
    if (!r.ok) throw new Error(`Claude API ${r.status}: ${(await r.text()).slice(0, 200)}`)
    return r.json()
  }
  let resp = await call(maxTokens)
  if (resp.stop_reason === 'max_tokens') resp = await call(maxTokens * 4)
  used++
  return (resp.content ?? []).map((b: any) => b.text ?? '').join('').trim()
}

function parseJson(text: string): any {
  // the last complete JSON object in the reply (models sometimes correct themselves mid-answer), as in the Python version
  let found: any = null, i = text.indexOf('{')
  while (i >= 0) {
    let depth = 0, inStr = false, esc = false, end = -1
    for (let j = i; j < text.length; j++) {
      const c = text[j]
      if (inStr) { if (esc) esc = false; else if (c === '\\') esc = true; else if (c === '"') inStr = false; continue }
      if (c === '"') inStr = true; else if (c === '{') depth++; else if (c === '}') { depth--; if (depth === 0) { end = j; break } }
    }
    if (end < 0) break
    try { const o = JSON.parse(text.slice(i, end + 1)); if (o && typeof o === 'object') found = o; i = text.indexOf('{', end + 1) } catch { i = text.indexOf('{', i + 1) }
  }
  return found ?? { _unparsed: text }
}

const fill = (t: string, v: Record<string, string>) => t.replace(/\{(\w+)\}/g, (m, k) => (k in v ? v[k] : m))

export default async function handler(req: any, res: any) {
  const live = Boolean(process.env.ANTHROPIC_API_KEY)
  if (req.method === 'GET') return res.status(200).json({ live })
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' })
  if (!live) return res.status(503).json({ error: 'No API key on the server' })
  const origin = String(req.headers.origin ?? ''), host = String(req.headers.host ?? '')
  if (origin && !origin.endsWith(host)) return res.status(403).json({ error: 'Calls are only accepted from this site' })
  if (used >= MAX_CALLS_PER_INSTANCE) return res.status(429).json({ error: 'The AI usage cap for this server has been reached' })
  const b = typeof req.body === 'string' ? JSON.parse(req.body) : req.body ?? {}
  if (JSON.stringify(b).length > 20000) return res.status(413).json({ error: 'Request too large' })
  try {
    switch (b.step) {
      case 'draft': return res.json(parseJson(await claude(fill(SYS_DRAFT, { sender: b.sender, length: b.length }), b.user, 500)))
      case 'translate': return res.json(parseJson(await claude(fill(SYS_TRANSLATE, { language: b.language }), b.user, 900)))
      case 'back': return res.json({ text: await claude(fill(SYS_BACK, { language: b.language }), b.user, 600) })
      case 'same': return res.json(parseJson(await claude(SYS_SAME, b.user, 300)))
      case 'verify': return res.json(parseJson(await claude(SYS_VERIFY, b.user, 300)))
      case 'ask': return res.json({ text: await claude(SYS_QA, `${METHOD_NOTES}\n\nRecord:\n${b.record}\n\nQuestion: ${String(b.question).slice(0, 500)}`, 400) })
      case 'reason': return res.json({ text: await claude(SYS_REASON, b.record, 200) })
      default: return res.status(400).json({ error: 'Unknown step' })
    }
  } catch (e: any) {
    return res.status(502).json({ error: String(e?.message ?? e).slice(0, 300) })
  }
}
