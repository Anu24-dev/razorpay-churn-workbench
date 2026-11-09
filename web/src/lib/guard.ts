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
