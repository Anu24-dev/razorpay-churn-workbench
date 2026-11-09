// Fact sheets: the only material the AI may use (port of ai/facts.py). Built by code from the engine output, not by AI.
import { NC, type Row } from '../engine/core'
import { channelFor } from './text'

export const CITY_LANGUAGE: Record<string, string> = { Mumbai: 'Marathi', Pune: 'Marathi', Hyderabad: 'Telugu', Jaipur: 'Hindi', Lucknow: 'Hindi',
  Indore: 'Hindi', Chandigarh: 'Hindi', 'Delhi NCR': 'Hindi', Bengaluru: 'Kannada', Ahmedabad: 'Gujarati', Surat: 'Gujarati', Kochi: 'Malayalam', Kolkata: 'Bengali', Chennai: 'Tamil' }
export const SENDER_IN_SCRIPT: Record<string, string> = { Hindi: 'अनुष्का', Marathi: 'अनुष्का', Gujarati: 'અનુષ્કા', Bengali: 'অনুষ্কা', Tamil: 'அனுஷ்கா', Telugu: 'అనుష్క', Kannada: 'ಅನುಷ್ಕಾ', Malayalam: 'അനുഷ്ക' }

export function languages(r: Row): [string, string | null] {
  const regional = CITY_LANGUAGE[String(r.city ?? '')]
  if (!regional) return ['English', null]
  return r.segment === 'Enterprise' ? ['English', regional] : [regional, 'English']
}

export type Fact = { id: string; topic: string; text: string }
export function merchantFacts(r: Row): Fact[] {
  const facts: Fact[] = []
  const add = (topic: string, text: string) => facts.push({ id: `F${facts.length + 1}`, topic, text })
  const d = r.driver_main === NC ? new Set<string>() : new Set([r.driver_main, r.driver_second])
  if (d.has('performance')) add('performance', "More of the merchant's customers' payments have been failing in recent months than before.")
  if (d.has('service')) {
    if (r.unresolved_tickets > 0) add('service', 'The merchant has support requests with Razorpay that are still open.')
    if (r.last_csat_1_to_5 != null && r.last_csat_1_to_5 <= 2) add('service', "The merchant's last feedback on Razorpay support was poor.")
  }
  if (d.has('engagement')) {
    add('engagement', 'The merchant has been using the Razorpay dashboard much less recently.')
    if (r.products_used <= 2) add('engagement', 'The merchant uses only a few of the Razorpay products available to it.')
  }
  if (r.days_to_contract_renewal <= 90) add('renewal', "The merchant's contract renewal is coming up in the next few months.")
  if (r.tenure_months >= 24) add('relationship', 'The merchant is a long-standing Razorpay customer.')
  return facts
}
export const factText = (f: Fact[]) => f.map(x => `${x.id} (${x.topic}): ${x.text}`).join('\n') || '(no specific facts: write a neutral check-in)'

const one = (v: any, nd = 1) => (v === null || v === undefined || Number.isNaN(v) ? null : Math.round(v * 10 ** nd) / 10 ** nd)
export function internalFacts(r: Row, playbookName: string) {
  return {
    merchant_id: r.merchant_id, business_name: r.business_name ?? null, segment: r.segment, industry: r.industry, city: r.city ?? null, tenure_months: r.tenure_months,
    tpv_lakh_by_month_1_to_6: [1, 2, 3, 4, 5, 6].map(i => one(r[`tpv_m${i}_inr_lakh`])),
    tpv_change_months_4_6_vs_1_3_pct: one(r.tpv_change * 100), projected_loss_by_month_9_pct: one(r.projected_loss_m9 * 100),
    success_rate_m1_m3_pct: one(r.success_rate_m1_m3_pct), success_rate_m4_m6_pct: one(r.success_rate_m4_m6_pct),
    support_tickets_last_90d: r.support_tickets_last_90d, unresolved_tickets: r.unresolved_tickets, last_csat_1_to_5: one(r.last_csat_1_to_5),
    logins_last_30d: r.logins_last_30d, products_used: r.products_used, days_to_contract_renewal: r.days_to_contract_renewal,
    tier: r.tier, driver_main: r.driver_main ?? null, driver_second: r.driver_second ?? null, review_likely_bucket: r.review_likely_bucket || null,
    playbook: `${r.playbook_id} ${playbookName}`, delivery: r.delivery || null, cause_needs_person_to_confirm: Boolean(r.needs_person),
    yearly_value_rs: one(r.yearly_value_rs, 0), expected_saving_rs: one(r.expected_saving_rs, 0), revenue_rank: r.revenue_rank ?? null,
    call_list_status: r.call_list_status || null,
    problem_scores_sd: { performance: one(r.fam_performance, 2), service: one(r.fam_service, 2), engagement: one(r.fam_engagement, 2) },
    account_manager_note: r.account_manager_note ?? null,
  }
}
export const channel = (r: Row) => channelFor(r.segment)
