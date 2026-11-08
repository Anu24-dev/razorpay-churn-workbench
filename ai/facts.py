"""
Fact sheets: the ONLY material the AI may use. Built by code from the engine output, not by AI.

- merchant-facing facts are qualitative (no numbers), each with an id and a topic, so a guardrail can
  trace every claim in a draft back to a field in the data;
- internal facts (with numbers) are for the CSM's reason line and the merchant Q&A only.
"""
from __future__ import annotations

import math

from engine.core import NC

CITY_LANGUAGE = {'Mumbai': 'Marathi', 'Pune': 'Marathi', 'Hyderabad': 'Telugu', 'Jaipur': 'Hindi', 'Lucknow': 'Hindi',
                 'Indore': 'Hindi', 'Chandigarh': 'Hindi', 'Delhi NCR': 'Hindi', 'Bengaluru': 'Kannada', 'Ahmedabad': 'Gujarati',
                 'Surat': 'Gujarati', 'Kochi': 'Malayalam', 'Kolkata': 'Bengali', 'Chennai': 'Tamil'}
CHANNEL = {'Enterprise': 'email from the account manager', 'Mid': 'email from the CSM, followed by a call',
           'Small': 'WhatsApp message from the CSM'}


def channel_for(segment):
    """A segment the data pack does not have is handled like Mid."""
    return CHANNEL.get(segment, CHANNEL['Mid'])


def languages(row):
    """Main and alternative draft language. Enterprise: English first. Others: the city's main language first.
    The city is only a guess at the merchant's language, so the CSM always confirms."""
    regional = CITY_LANGUAGE.get(str(row.get('city', '')))
    if regional is None:
        return 'English', None
    if row['segment'] == 'Enterprise':
        return 'English', regional
    return regional, 'English'


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def merchant_facts(row):
    """Qualitative facts a merchant-facing message may use."""
    facts = []
    def add(topic, text):
        facts.append({'id': f'F{len(facts) + 1}', 'topic': topic, 'text': text})
    drivers = {row.get('driver_main'), row.get('driver_second')}
    if row.get('driver_main') == NC:
        drivers = set()                        # cause unknown: neutral check-in only
    if 'performance' in drivers:
        add('performance', 'More of the merchant\'s customers\' payments have been failing in recent months than before.')
    if 'service' in drivers:
        if row['unresolved_tickets'] > 0:
            add('service', 'The merchant has support requests with Razorpay that are still open.')
        if not _blank(row['last_csat_1_to_5']) and row['last_csat_1_to_5'] <= 2:
            add('service', 'The merchant\'s last feedback on Razorpay support was poor.')
    if 'engagement' in drivers:
        add('engagement', 'The merchant has been using the Razorpay dashboard much less recently.')
        if row['products_used'] <= 2:
            add('engagement', 'The merchant uses only a few of the Razorpay products available to it.')
    if row['days_to_contract_renewal'] <= 90:
        add('renewal', 'The merchant\'s contract renewal is coming up in the next few months.')
    if row['tenure_months'] >= 24:
        add('relationship', 'The merchant is a long-standing Razorpay customer.')
    return facts


def internal_facts(row):
    """Facts with numbers, for the CSM only (reason line and Q&A). Never shown to the merchant."""
    def f(v, nd=1):
        return None if _blank(v) else round(float(v), nd)
    return {
        'merchant_id': row['merchant_id'], 'business_name': row.get('business_name'), 'segment': row['segment'],
        'industry': row['industry'], 'city': row.get('city'), 'tenure_months': int(row['tenure_months']),
        'tpv_lakh_by_month_1_to_6': [f(row[f'tpv_m{i}_inr_lakh']) for i in range(1, 7)],
        'tpv_change_months_4_6_vs_1_3_pct': f(row['tpv_change'] * 100),
        'projected_loss_by_month_9_pct': f(row['projected_loss_m9'] * 100),
        'success_rate_m1_m3_pct': f(row['success_rate_m1_m3_pct']), 'success_rate_m4_m6_pct': f(row['success_rate_m4_m6_pct']),
        'support_tickets_last_90d': int(row['support_tickets_last_90d']), 'unresolved_tickets': int(row['unresolved_tickets']),
        'last_csat_1_to_5': f(row['last_csat_1_to_5']), 'logins_last_30d': int(row['logins_last_30d']),
        'products_used': int(row['products_used']), 'days_to_contract_renewal': int(row['days_to_contract_renewal']),
        'tier': row['tier'], 'driver_main': row.get('driver_main'), 'driver_second': row.get('driver_second'),
        'review_likely_bucket': row.get('review_likely_bucket') or None, 'playbook': row['playbook'],
        'playbook_also': row.get('playbook_also') or None, 'delivery': row.get('delivery') or None,
        'cause_needs_person_to_confirm': bool(row.get('needs_person', False)),
        'yearly_value_rs': f(row['yearly_value_rs'], 0), 'expected_saving_rs': f(row.get('expected_saving_rs'), 0),
        'revenue_rank': None if _blank(row.get('revenue_rank')) else int(row['revenue_rank']),
        'call_list_status': row.get('call_list_status') or None,
        'problem_scores_sd': {'performance': f(row['fam_performance'], 2), 'service': f(row['fam_service'], 2), 'engagement': f(row['fam_engagement'], 2)},
        'account_manager_note': None if _blank(row.get('account_manager_note')) else row['account_manager_note'],
    }
