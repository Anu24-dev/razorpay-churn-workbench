// One vocabulary for the whole page (the engine's own labels stay as they are, because they are tested against the notebook).
import { NC, PB3_RENEWAL_DAYS, type Row } from '../engine/core'

export const tierKey = (t: string) => (String(t).startsWith('3P') ? '3P' : String(t)[0])

export const RISK: Record<string, { label: string; tone: string }> = {
  '1': { label: 'Critical', tone: 'red' }, '2': { label: 'High', tone: 'amber' }, '3': { label: 'Moderate', tone: 'yellow' },
  '3P': { label: 'Small dip', tone: 'blue' }, '4': { label: 'Recovered dip', tone: 'grey' }, '5': { label: 'Early signs', tone: 'purple' }, '6': { label: 'Healthy', tone: 'green' },
}
export const RISK_HELP: Record<string, string> = {
  '1': 'On the current trend, this merchant loses more than half its payment volume within about 90 days.',
  '2': 'Payment volume is falling beyond normal ups and downs, and the contract renews within 90 days.',
  '3': 'Payment volume is falling beyond normal ups and downs, but it is not yet on track to lose half within 90 days.',
}
export const risk = (r: Row) => RISK[tierKey(r.tier)]

export const DRIVER_NAME: Record<string, string> = { performance: 'Payment failures', service: 'Support issues', engagement: 'Low engagement', [NC]: 'Likely pricing' }
export const FIX: Record<string, string> = {
  performance: "Set up an alert that tells you when a merchant's payment success rate drops below its usual level, so you can bring in Payments Ops before the merchant notices.",
  service: 'Take ownership of every open ticket with a closing date, and follow up on any low support rating within the week.',
  engagement: "Re-confirm your main contact whenever a merchant's ownership changes, and walk them through products they are not using yet.",
  [NC]: 'Start a pricing review with the account manager 60 days before each renewal, before the merchant compares providers.',
}
export const PICK: Record<string, string> = { Firm: 'Solid', Borderline: 'Close call', 'Next in line': 'Next up' }
export const CHANNEL: Record<string, string> = { Enterprise: 'Email from the account manager', Mid: 'Email from the CSM, followed by a call', Small: 'WhatsApp message from the CSM' }
export const channelFor = (seg: string) => CHANNEL[seg] ?? CHANNEL.Mid
export const SENDER = 'Anoushka'

export const plural = (n: number, w: string, many?: string) => `${n} ${n === 1 ? w : many ?? w + 's'}`
export const rs = (x: number | null | undefined) => (x === null || x === undefined || Number.isNaN(x) ? '–' : Math.abs(x) >= 1e5 ? `Rs ${(x / 1e5).toFixed(1)} lakh` : `Rs ${Math.round(x).toLocaleString('en-IN')}`)
export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`

export function ncShort(r: Row) {
  const b = String(r.review_likely_bucket ?? '')
  if (b.startsWith('Likely commercial (renews')) return 'Likely pricing'
  if (b.includes('commercial or external')) return 'Pricing or outside factor'
  const m = b.match(/^Likely (\w+) \(weak/)
  return m ? `Possibly ${(DRIVER_NAME[m[1]] ?? m[1]).toLowerCase()}` : 'Cause unclear'
}
export const whyShort = (r: Row) => (r.driver_main === NC ? ncShort(r) : DRIVER_NAME[r.driver_main] ?? '')

export function driverLine(r: Row, d: string) {
  if (d === 'performance') return `More payments are failing: the success rate fell from ${r.success_rate_m1_m3_pct.toFixed(1)}% to ${r.success_rate_m4_m6_pct.toFixed(1)}%`
  if (d === 'service') {
    const rating = r.last_csat_1_to_5 === null || r.last_csat_1_to_5 === undefined ? '' : `, and their last support rating was ${Math.round(r.last_csat_1_to_5)} out of 5`
    return `${plural(r.unresolved_tickets, 'support ticket')} still open${rating}`
  }
  if (d === 'engagement') return `They logged in ${plural(r.logins_last_30d, 'time')} this month and use ${plural(r.products_used, 'product')}`
  return 'No payment, support or usage problem shows up in their numbers'
}

export function whyPoints(r: Row): { text: string; hot: boolean }[] {
  const pts = [{ text: `Payment volume is down ${pct(Math.abs(r.tpv_change))} over the last three months`, hot: true }]
  pts.push(r.imminent ? { text: 'On the current trend, more than half of it is gone by month 9', hot: true }
    : { text: `On the current trend, it will be about ${pct(r.projected_loss_m9)} below its best month by month 9`, hot: false })
  pts.push({ text: driverLine(r, r.driver_main), hot: r.driver_main !== NC })
  if (r.driver_second) { const l = driverLine(r, r.driver_second); pts.push({ text: 'Also: ' + l[0].toLowerCase() + l.slice(1), hot: false }) }
  pts.push({ text: `The contract renews in ${r.days_to_contract_renewal} days`, hot: r.days_to_contract_renewal <= PB3_RENEWAL_DAYS })
  return pts
}

export const GROUPS: [string, string][] = [['Seasonal dip', 'Seasonal dip'], ['Bounced-back dip', 'Recovered dip'], ['Small and steady', 'Small and steady'],
  ['Busy and growing', 'Busy and growing'], ['Growing', 'Growing'], ['Steady', 'Steady'], ['Possible decline', 'Small dip'], ['Watch list', 'Early signs']]

export function whyLeft(r: Row) {
  const g = r.why_group
  const t = g === 'Seasonal dip' ? 'A seasonal dip. Volume picked up again in month 6.'
    : g === 'Bounced-back dip' ? 'Dipped, then picked up again in month 6.'
    : g === 'Small and steady' ? 'A small business. Low volume is normal for them, and it is not falling.'
    : g === 'Busy and growing' ? `Up ${pct(r.tpv_change)} and raising a lot of tickets: a busy business, not one that is leaving.`
    : g === 'Growing' ? `Up ${pct(r.tpv_change)}.`
    : g === 'Steady' ? 'Normal month-to-month movement.'
    : g === 'Possible decline' ? 'A small drop that has not recovered yet. Gets an automated check-in, not a call.'
    : g === 'Watch list' ? 'Steady for now, but showing early signs seen in merchants who later declined. Review next month.'
    : String(r.why_not_flagged ?? '')
  return t + (r.signal_to_watch ? ` Watch: ${(DRIVER_NAME[r.signal_to_watch] ?? r.signal_to_watch).toLowerCase()}.` : '')
}

export function callQuestions(r: Row, noteIsPricing: boolean) {
  const f: Record<string, number> = { performance: r.fam_performance, service: r.fam_service, engagement: r.fam_engagement }
  const top = Object.keys(f).reduce((a, b) => (f[b] > f[a] ? b : a))
  const asks = ['Has anything changed in how you take payments, or in who looks after the account?']
  if (r.days_to_contract_renewal <= PB3_RENEWAL_DAYS || noteIsPricing) asks.unshift('Are you reviewing pricing or comparing providers before your renewal?')
  if (['Gifting', 'Education', 'Travel'].some(x => String(r.industry).includes(x))) asks.push('Is this dip part of your usual season?')
  asks.push(({ performance: 'Have your customers mentioned failed payments?', service: 'Is anything still open with our support team?',
    engagement: 'Is the dashboard still giving your team what it needs?' } as Record<string, string>)[top])
  return asks
}

// The fallback first message for an uploaded file (no saved AI draft). Same rules as the AI drafts: no numbers, no promises.
export function templateDraft(r: Row) {
  const name = r.business_name ?? r.merchant_id
  const body = r.driver_main === 'performance'
    ? `Hi ${name} team,\n\nYour account matters to us at Razorpay. We have noticed more of your customers' payments failing recently, and we would like to fix this with you alongside our payments team.`
    : r.driver_main === 'service'
    ? `Hi ${name} team,\n\nYour account matters to us at Razorpay. You have support requests with us that are still open, and I would like to take ownership of them personally and be your single point of contact.`
    : r.driver_main === 'engagement'
    ? `Hi ${name} team,\n\nYour account matters to us at Razorpay. I would like to make sure you are getting full value from your account and that we have the right contact on your side.`
    : `Hi ${name} team,\n\nYour account matters to us at Razorpay, and I would like to check in on how things are going and whether your current plan still fits your needs.`
  return `${body}\n\nCould we set up a short call at [option 1] or [option 2]?\n\nWarm regards,\n${SENDER}, Razorpay`
}
