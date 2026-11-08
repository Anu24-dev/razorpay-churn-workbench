"""
Merchant Retention Workbench (Razorpay Campus AI Build-a-thon, Problem 2).

Input: the data pack (or any file with the same P2_Merchants columns).
Engine (rules + statistics, no AI): who is declining, how fast, why, which playbook, who gets the 15 calls.
AI (Claude): reads account-manager notes, writes the CSM's reason line, drafts the outreach (regional + English),
checks the translation, answers questions about one merchant. Code guardrails check every draft.
People: approve, edit or reject every draft; review unclear causes; account managers own pricing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai import llm
from ai import pipeline as P
from ai.facts import CHANNEL, languages
from engine import core

st.set_page_config(page_title='Merchant Retention Workbench', page_icon='📉', layout='wide')

# ------------------------------------------------------------------ style
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.kpi {border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; background: var(--card); height: 100%;}
.kpi .label {font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 1.32rem; font-weight: 650; margin-top: 2px; color: var(--ink);}
.kpi .sub {font-size: 0.82rem; color: var(--muted); margin-top: 2px;}
.card {border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; background: var(--card); margin-bottom: 10px;}
.pill {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-right: 4px; margin-bottom: 4px;}
.p-red {background: #FDECEC; color: #B42318;} .p-amber {background: #FEF3E2; color: #9A5B00;} .p-blue {background: #E8F0FE; color: #1A4FB8;}
.p-grey {background: #EEF0F3; color: #475467;} .p-green {background: #E7F6EC; color: #1E7B3A;} .p-purple {background: #F1EAFE; color: #5B2DB3;}
.muted {color: var(--muted); font-size: 0.86rem;}
.check-ok {color: #1E7B3A;} .check-bad {color: #B42318;}
:root {--line: #E4E7EC; --card: #FFFFFF; --muted: #667085; --ink: #101828;}
@media (prefers-color-scheme: dark) { :root {--line: #2B313B; --card: #151A21; --muted: #98A2B3; --ink: #F2F4F7;} }
</style>
""", unsafe_allow_html=True)


def rs(x):
    """Rupees in lakh, Indian style."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'Rs {x / 1e5:,.1f} lakh' if abs(x) >= 1e5 else f'Rs {x:,.0f}'


def pill(text, kind='grey'):
    return f'<span class="pill p-{kind}">{text}</span>'


TIER_SHORT = {'1': 'Imminent fade', '2': 'Declining · renews soon', '3': 'Declining', '3P': 'Possible decline', '4': 'Bounced back', '5': 'Watch', '6': 'Stable / growing'}
TIER_PILL = {'1': ('Imminent fade', 'red'), '2': ('Declining, renews within 90 days', 'amber'), '3': ('Declining', 'amber'),
             '3P': ('Possible decline', 'blue'), '4': ('Bounced back', 'grey'), '5': ('Watch (experimental)', 'purple'), '6': ('Stable / growing', 'green')}
DRIVER_NAME = {'performance': 'Performance', 'service': 'Service', 'engagement': 'Engagement', core.NC: 'Unclear: human review'}


def tier_key(t):
    return '3P' if str(t).startswith('3P') else str(t)[0]


# ------------------------------------------------------------------ data + engine (cached by file content)
DEFAULT_FILE = 'data/Buildathon_Student_Data_Pack.xlsx'


@st.cache_data(show_spinner=False)
def run_engine(file_bytes: bytes, name: str):
    merchants = core.load_merchants(io.BytesIO(file_bytes)) if not name.lower().endswith('.csv') else pd.read_csv(io.BytesIO(file_bytes))
    return core.run(merchants)


with st.sidebar:
    st.markdown('### Merchant Retention Workbench')
    st.caption('Problem 2: churn early-warning and proactive outreach')
    up = st.file_uploader('Merchant file (data pack .xlsx or .csv with the same columns)', type=['xlsx', 'csv'])
    if up is not None:
        file_bytes, file_name = up.getvalue(), up.name
    else:
        file_bytes, file_name = open(DEFAULT_FILE, 'rb').read(), DEFAULT_FILE.split('/')[-1]
    st.caption(f'Using: **{file_name}**')
    csm = st.text_input('Your name (for the approval log)', value=st.session_state.get('csm', ''), placeholder='e.g. Anu')
    st.session_state.csm = csm
    if llm.live():
        st.success('AI: live (Claude). New drafts and answers can be generated.', icon='✅')
    else:
        st.info('AI: stored results only. Everything for the data pack works without a key.', icon='💾')
    st.divider()
    st.caption('Nothing is sent to merchants from this app. Approved messages are exported for the CSM to send.')

file_id = hashlib.sha1(file_bytes).hexdigest()[:10]
with st.spinner('Scoring merchants...'):
    res = run_engine(file_bytes, file_name)

if not res['ok']:
    st.error('This file cannot be scored. Fix the checks marked FAIL and upload again.')
    st.dataframe(res['checks'], hide_index=True, width='stretch')
    st.stop()

df = res['df'].copy()
merchants_by_id = df.set_index('merchant_id', drop=False)
calls = df[df.csm_call_list].sort_values('revenue_rank')
review = df[df.declining & (df.driver_main == core.NC)].sort_values('revenue_rank')
alerts = df[df.am_alert].sort_values('revenue_rank')
V, S = res['value'], res['summary']

# session state: approvals and audit log
ss = st.session_state
ss.setdefault('decisions', {})        # (file_id, merchant_id) -> dict
ss.setdefault('audit', [])            # list of dicts
ss.setdefault('review_decisions', {})
ss.setdefault('qa', {})
ss.setdefault('selected', None)


def log(action, merchant_id, detail=''):
    ss.audit.append({'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'by': ss.csm or '(no name)', 'file': file_name,
                     'merchant_id': merchant_id, 'action': action, 'detail': detail})


# ------------------------------------------------------------------ AI results (stored or live)
@st.cache_data(show_spinner=False)
def ai_reason(_row, key):
    return P.write_reason(_row)


@st.cache_data(show_spinner=False)
def ai_draft(_row, key):
    return P.draft_with_retry(_row)


@st.cache_data(show_spinner=False)
def ai_note(note):
    return P.read_note(note)


def reason_for(row):
    if row.declining or row.possible_decline:
        text, chk = ai_reason(row, f"{file_id}-{row.merchant_id}")
        if text:
            return text, chk
        return (f"{DRIVER_NAME.get(row.driver_main, row.driver_main)}. Volume months 4-6 vs 1-3: {row.tpv_change:+.0%}; "
                f"projected loss by month 9: {row.projected_loss_m9:.0%}; renews in {int(row.days_to_contract_renewal)} days."), None
    return row.why_not_flagged, None


# ------------------------------------------------------------------ header
st.markdown('## Merchant Retention Workbench')
st.markdown(f'<span class="muted">{S["merchants"]} merchants scored from <b>{file_name}</b> · month-6 snapshot · '
            f'typical monthly wobble {S["typical_wobble"]:.1%}, so a decline means a drop beyond {abs(S["decline_line"]):.1%}</span>',
            unsafe_allow_html=True)

tab_work, tab_why = st.tabs(['Workbench', 'Why not flagged'])

# ================================================================== WORKBENCH
with tab_work:
    # ---- money panel
    k = st.columns(5)
    cards = [('Merchants at risk', f"{S['at_risk']}", f"{S['imminent']} imminent fades, {S['review']} need human review"),
             ('Yearly revenue at risk', rs(V['at_risk_value']), 'net revenue at 1% of TPV'),
             ('Covered by the 15 calls', rs(V['call_list_value']), f"{V['call_list_value'] / V['at_risk_value']:.0%} of revenue at risk" if V['at_risk_value'] else ''),
             ('Expected saving', f"Rs {V['saving_10'] / 1e5:,.1f}-{V['saving_ours'] / 1e5:,.1f}L", 'a range, because the save chance is an assumption'),
             ('Cost of the 15 calls', rs(V['outreach_cost']), f"each call pays off above a {V['break_even_max']:.1%} save chance" if V['outreach_cost'] else '')]
    for col, (lab, val, sub) in zip(k, cards):
        col.markdown(f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    st.write('')

    # ---- driver mix and the fix
    mix = res['driver_mix']
    if len(mix):
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Bar(x=mix.merchants, y=[DRIVER_NAME.get(d, d) for d in mix.driver], orientation='h',
                                   marker_color=['#1A4FB8' if d != core.NC else '#98A2B3' for d in mix.driver],
                                   text=[f"{m} merchants · {rs(v)}/yr" for m, v in zip(mix.merchants, mix.yearly_value_rs)], textposition='auto',
                                   hovertemplate='%{y}: %{x} merchants<extra></extra>'))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Why merchants are at risk', font=dict(size=14)),
                              xaxis=dict(visible=False), yaxis=dict(autorange='reversed'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        with c2:
            top = mix[mix.driver != core.NC].iloc[0] if (mix.driver != core.NC).any() else mix.iloc[0]
            st.markdown(f"""<div class="card"><div class="label muted">BIGGEST FIXABLE DRIVER</div>
            <div style="font-weight:650;font-size:1.05rem;margin:4px 0 6px 0">{DRIVER_NAME.get(top.driver, top.driver)}: {top.merchants} merchants, {rs(top.yearly_value_rs)}/yr</div>
            <div class="muted">Suggested process fix</div><div>{res['fix_ideas'][top.driver]}</div></div>""", unsafe_allow_html=True)
            if (mix.driver == core.NC).any():
                n_nc = int(mix.loc[mix.driver == core.NC, 'merchants'].iloc[0])
                nc_df = df[df.declining & (df.driver_main == core.NC)]
                soon = int((nc_df.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS).sum())
                st.markdown(f"""<div class="card"><div class="muted">{n_nc} merchants show no operational cause ({soon} of them renew within 60 days).
                Suggested fix: {res['fix_ideas'][core.NC]}</div></div>""", unsafe_allow_html=True)

    # ---- call list + side panel
    st.markdown('### This month\'s 15 calls')
    st.caption('Ranked by chance of leaving x chance of saving x yearly value. Borderline = drops out of the 15 under some reasonable assumptions; '
               'the CSM lead decides those slots. Click a row to open the merchant.')
    _prev = ss.get(f'calls-{file_id}')
    _has_sel = bool(_prev and getattr(_prev, 'selection', None) and _prev.selection.rows)
    if _has_sel:
        left, right = st.columns([12, 11], gap='medium')
    else:
        left = st.container(); right = st.container()
    with left:
        def status_of(mid):
            d = ss.decisions.get((file_id, mid))
            return d['status'] if d else 'To review'
        tbl = pd.DataFrame({
            '#': calls.revenue_rank.astype(int).values,
            'Merchant': [f"{r.business_name}" if 'business_name' in calls else r.merchant_id for r in calls.itertuples()],
            'ID': calls.merchant_id.values,
            'Size': calls.segment.values,
            'Risk': [TIER_SHORT[tier_key(t)] for t in calls.tier],
            'Why': [DRIVER_NAME.get(d, d) + (f' + {s}' if isinstance(s, str) and s else '') for d, s in zip(calls.driver_main, calls.driver_second)],
            'Renews in': [f"{int(x)} days" for x in calls.days_to_contract_renewal],
            'Slot': calls.call_list_status.values,
            'Expected saving': [rs(x) for x in calls.expected_saving_rs],
            'Draft': [status_of(m) for m in calls.merchant_id]})
        sel = st.dataframe(tbl, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row',
                           key=f'calls-{file_id}', height=min(36 * (len(tbl) + 1) + 4, 600),
                           column_order=(['#', 'Merchant', 'Risk', 'Why', 'Slot', 'Draft'] if _has_sel else list(tbl.columns)),
                           column_config={'#': st.column_config.NumberColumn(width='small'), 'ID': st.column_config.TextColumn(width='small')})
        if sel and sel.selection.rows:
            ss.selected = tbl.iloc[sel.selection.rows[0]]['ID']
        with st.expander('See how every other merchant is ranked (all 100, with risk, reason and playbook)'):
            allm = df.copy()
            allm['order'] = allm.revenue_rank.fillna(999)
            allm['tk'] = allm.tier.map(tier_key)
            allm = allm.sort_values(['order', 'tk', 'tpv_change'])
            full = pd.DataFrame({'Rank': allm.revenue_rank.map(lambda x: '' if pd.isna(x) else str(int(x))), 'ID': allm.merchant_id,
                                 'Merchant': allm.get('business_name', allm.merchant_id), 'Size': allm.segment, 'Risk': [TIER_PILL[tier_key(t)][0] for t in allm.tier],
                                 'Reason': [DRIVER_NAME.get(d, '') if isinstance(d, str) else w for d, w in zip(allm.driver_main, allm.why_not_flagged)],
                                 'Playbook': allm.playbook, 'Track': np.where(allm.csm_call_list, 'CSM call', np.where(allm.am_alert, 'Account-manager alert',
                                                                       np.where(allm.automated_track, 'Automated digital', 'No outreach')))})
            st.dataframe(full, hide_index=True, width='stretch', height=420)

    # ---- merchant side panel
    with right:
        mid = ss.selected if (_has_sel or (sel and sel.selection.rows)) and ss.selected in set(calls.merchant_id) else None
        if mid is None:
            pass
        else:
            r = merchants_by_id.loc[mid]
            tk = tier_key(r.tier); tl, tc = TIER_PILL[tk]
            pills = pill(tl, tc) + pill(r.segment, 'grey') + (pill(r.call_list_status, 'amber' if r.call_list_status == 'Borderline' else 'green'))
            if r.call_this_week: pills += pill('Call this week', 'red')
            st.markdown(f"#### {r.get('business_name', mid)} <span class='muted'>· {mid} · {r.get('city', '')} · {r.industry}</span>", unsafe_allow_html=True)
            st.markdown(pills, unsafe_allow_html=True)
            reason, chk = reason_for(r)
            st.markdown(f"<div class='card'><div class='muted'>Why this merchant</div>{reason}</div>", unsafe_allow_html=True)

            # trend charts (internal only)
            months = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
            tpv = [r[f'tpv_m{i}_inr_lakh'] for i in range(1, 7)]
            g1, g2 = st.columns([3, 2])
            with g1:
                f1 = go.Figure(go.Scatter(x=months, y=tpv, mode='lines+markers', line=dict(color='#1A4FB8', width=2.5), hovertemplate='%{x}: Rs %{y} lakh<extra></extra>'))
                f1.add_hline(y=float(np.mean(tpv[:3])), line_dash='dot', line_color='#98A2B3', annotation_text='months 1-3 average', annotation_font_size=10)
                f1.update_layout(height=170, margin=dict(l=0, r=0, t=26, b=0), title=dict(text='Monthly volume (Rs lakh)', font=dict(size=12)),
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(rangemode='tozero', gridcolor='#EEF0F3'))
                st.plotly_chart(f1, width='stretch', config={'displayModeBar': False})
            with g2:
                f2 = go.Figure(go.Bar(x=['Months 1-3', 'Months 4-6'], y=[r.success_rate_m1_m3_pct, r.success_rate_m4_m6_pct],
                                      marker_color=['#98A2B3', '#B42318' if r.success_rate_m4_m6_pct < r.success_rate_m1_m3_pct - 2 else '#1A4FB8'],
                                      text=[f'{r.success_rate_m1_m3_pct:.1f}%', f'{r.success_rate_m4_m6_pct:.1f}%'], textposition='outside'))
                f2.update_layout(height=170, margin=dict(l=0, r=0, t=26, b=0), title=dict(text='Payment success rate', font=dict(size=12)),
                                 yaxis=dict(range=[min(70, r.success_rate_m4_m6_pct - 5), 100], visible=False), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(f2, width='stretch', config={'displayModeBar': False})
            csat = '–' if pd.isna(r.last_csat_1_to_5) else f'{r.last_csat_1_to_5:.0f}/5'
            st.caption(f"Tickets (90 days): {int(r.support_tickets_last_90d)} · unresolved: {int(r.unresolved_tickets)} · last CSAT: {csat} · "
                       f"logins (30 days): {int(r.logins_last_30d)} · products: {int(r.products_used)} · with Razorpay {int(r.tenure_months)} months")

            # playbook
            st.markdown(f"<div class='card'><div class='muted'>Playbook · {r.delivery}</div><b>{r.playbook}</b>"
                        f"{'<br>' + r.playbook_also if r.playbook_also else ''}</div>", unsafe_allow_html=True)
            note = r.get('account_manager_note')
            if isinstance(note, str):
                nr = ai_note(note)
                tag = f" · AI reads it as <b>{nr['category']}</b>" if nr and nr.get('category') else ''
                st.markdown(f"<div class='muted'>Account manager note: “{note}”{tag}</div>", unsafe_allow_html=True)

            # drafts
            st.markdown('##### Draft message')
            d = ai_draft(r, f'{file_id}-{mid}')
            key = (file_id, mid)
            if d is None:
                st.info('No stored draft for this file and no API key, so here is the rule-based English draft.')
                text_en = P.template_draft(r); drafts = {'English': ('', text_en)}; checks = []; ok = True
            else:
                drafts = {'English': (d['english'].get('subject', ''), d['english'].get('message', ''))}
                if d.get('regional') and 'message' in d['regional']:
                    drafts[d['regional_language']] = (d['regional'].get('subject', ''), d['regional']['message'])
                order = [d['main_language']] + [l for l in drafts if l != d['main_language']]
                drafts = {l: drafts[l] for l in order if l in drafts}
                checks, ok = d['checks'], d['passed']
            st.caption(f"Channel: {CHANNEL[r.segment]}. Language from the merchant's city ({r.get('city', '')}); please confirm with the merchant. "
                       + ('Enterprise accounts get English first.' if r.segment == 'Enterprise' else ''))
            if not ok:
                st.error('Blocked by the guardrails after one retry. Write this message yourself, or fix the draft below and approve it.')
            lang = st.radio('Version', list(drafts), horizontal=True, key=f'lang-{file_id}-{mid}', label_visibility='collapsed')
            subj, body = drafts[lang]
            edited = st.text_area('Message (you can edit before approving)', value=(f'Subject: {subj}\n\n' if subj else '') + body,
                                  height=230, key=f'txt-{file_id}-{mid}-{lang}')
            if checks:
                n_ok = sum(1 for _, okk, _ in checks if okk)
                with st.expander(f"Guardrail checks: {n_ok} of {len(checks)} passed"):
                    for name, okk, det in checks:
                        st.markdown(f"<span class='{'check-ok' if okk else 'check-bad'}'>{'✓' if okk else '✗'}</span> {name}"
                                    f"{' · ' + det if det else ''}", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            if b1.button('Approve', type='primary', key=f'ap-{file_id}-{mid}', disabled=not ss.csm):
                ss.decisions[key] = {'status': 'Approved', 'language': lang, 'text': edited, 'edited': edited.strip() != ((f'Subject: {subj}\n\n' if subj else '') + body).strip()}
                log('Approved', mid, f"{lang}{' (edited)' if ss.decisions[key]['edited'] else ''}"); st.rerun()
            if b2.button('Reject', key=f'rj-{file_id}-{mid}', disabled=not ss.csm):
                ss.decisions[key] = {'status': 'Rejected', 'language': lang, 'text': '', 'edited': False}
                log('Rejected', mid, lang); st.rerun()
            if b3.button('Reset', key=f'rs-{file_id}-{mid}'):
                ss.decisions.pop(key, None); log('Reset', mid); st.rerun()
            if not ss.csm:
                st.caption('Enter your name in the sidebar to approve or reject. Every decision is logged.')
            elif key in ss.decisions:
                st.caption(f"Status: {ss.decisions[key]['status']}")

            # ask about this merchant
            st.markdown('##### Ask about this merchant')
            st.caption('Answers come only from this merchant\'s data and the method. If the data does not show it, it says so.')
            qcols = st.columns(3)
            for i, (lab, q) in enumerate([('Why on the list?', 'Why is this merchant on the call list?'), ('Focus for the call', 'What should I focus on in the call?'),
                                          ('What we don\'t know', 'What do we not know about this merchant?')]):
                if qcols[i].button(lab, key=f'q{i}-{file_id}-{mid}', width='stretch'):
                    ss.qa[(file_id, mid)] = q
            free = st.chat_input('Ask a question about this merchant', key=f'chat-{file_id}-{mid}')
            if free:
                ss.qa[(file_id, mid)] = free
            q = ss.qa.get((file_id, mid))
            if q:
                ans, achk = P.answer_question(r, q)
                with st.chat_message('user'):
                    st.write(q)
                with st.chat_message('assistant'):
                    if ans is None:
                        st.write('Live answers need the API key. The three suggested questions have stored answers for the data pack.')
                    elif achk and not achk[1]:
                        st.warning('This answer quoted a number that is not in the merchant\'s data, so it is hidden. Try rephrasing.')
                    else:
                        st.write(ans)

    # ---- human review
    st.markdown('### Needs a person: cause unclear')
    st.caption('The numbers show no performance, service or engagement problem for these declining merchants. A person confirms the cause '
               'before any playbook beyond a neutral check-in. The likely bucket and the AI\'s reading of the account manager note are hints, not decisions. '
               'Merchants outside the 15 calls that renew within 60 days go to their account manager, who owns any pricing conversation; the AI never offers terms.')
    if len(review) == 0:
        st.markdown('<div class="muted">None this month.</div>', unsafe_allow_html=True)
    for _, r in review.iterrows():
        note = r.get('account_manager_note'); nr = ai_note(note) if isinstance(note, str) else None
        cA, cB = st.columns([3, 2])
        with cA:
            where = 'On the call list' if r.csm_call_list else ('Account-manager alert: review rate card and renewal terms before renewal (PB-3)' if r.am_alert else 'Automated track')
            st.markdown(f"<div class='card'><b>{r.get('business_name', r.merchant_id)}</b> <span class='muted'>· {r.merchant_id} · {r.segment} · renews in {int(r.days_to_contract_renewal)} days · {where}</span><br>"
                        f"{pill(r.review_likely_bucket, 'blue')}"
                        f"{pill('Note reads as ' + nr['category'], 'purple') if nr and nr.get('category') else ''}"
                        f"<div class='muted' style='margin-top:6px'>Note: “{note if isinstance(note, str) else 'no note'}”</div></div>", unsafe_allow_html=True)
        with cB:
            opts = ['Not reviewed yet', 'Pricing / commercial (PB-3)', 'Performance (PB-1)', 'Service (PB-2)', 'Engagement (PB-4)', 'Other / external: monitor (PB-6)']
            prev = ss.review_decisions.get((file_id, r.merchant_id), opts[0])
            choice = st.selectbox('Confirmed cause', opts, index=opts.index(prev), key=f'rv-{file_id}-{r.merchant_id}', disabled=not ss.csm)
            if choice != prev:
                ss.review_decisions[(file_id, r.merchant_id)] = choice; log('Cause confirmed', r.merchant_id, choice)

    # ---- export and audit log
    st.markdown('### Export and approval log')
    e1, e2, e3 = st.columns(3)
    appr = [{'merchant_id': m, 'business_name': merchants_by_id.loc[m].get('business_name', m), 'channel': CHANNEL[merchants_by_id.loc[m].segment],
             'language': v['language'], 'edited_by_csm': v['edited'], 'message': v['text']}
            for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Approved']
    e1.download_button(f'Approved messages ({len(appr)})', pd.DataFrame(appr).to_csv(index=False).encode('utf-8-sig') if appr else b'',
                       file_name='approved_messages.csv', disabled=not appr, width='stretch')
    out_cols = ['merchant_id', 'segment', 'industry', 'tier', 'driver_main', 'driver_second', 'review_likely_bucket', 'playbook', 'playbook_also',
                'delivery', 'revenue_rank', 'call_list_status', 'call_this_week', 'am_alert', 'automated_track', 'expected_saving_rs', 'why_not_flagged']
    ranked = df.assign(_o=df.revenue_rank.fillna(999)).sort_values(['_o', 'tier'])[[c for c in out_cols if c in df]]
    e2.download_button('Ranked list of all merchants', ranked.to_csv(index=False).encode('utf-8-sig'), file_name='ranked_merchants.csv', width='stretch')
    e3.download_button(f'Approval log ({len(ss.audit)} entries)', pd.DataFrame(ss.audit).to_csv(index=False).encode('utf-8-sig') if ss.audit else b'',
                       file_name='approval_log.csv', disabled=not ss.audit, width='stretch')
    if ss.audit:
        st.dataframe(pd.DataFrame(ss.audit).iloc[::-1], hide_index=True, width='stretch', height=200)

    with st.expander('How the ranking works'):
        st.markdown(f"""
1. **Is volume falling and staying down?** We measure how much a merchant's volume normally moves month to month ({S['typical_wobble']:.1%} on this file). A merchant is *declining* if months 4-6 fell more than three of those wobbles below months 1-3 (or month 6 alone fell sharply) and volume did not bounce back. Dips that recover are not flagged; see the *Why not flagged* tab.
2. **How fast?** The months 4-6 trend is projected to month 9. More than half of healthy volume gone = *imminent fade*, the brief's definition of churn.
3. **Why?** Three problem scores from the merchant's own signals: performance (success rate), service (unresolved tickets, CSAT), engagement (logins, products). A score more than 1 standard deviation above average names the driver. If none is high, a person reviews it. Blank CSAT is never used to name a driver.
4. **Who gets the 15 calls?** Chance of leaving x chance of saving if we call x yearly value, minus Rs 1,500 per call. Both chances are assumptions (the data has no outcomes), so the list is re-run under 11 versions of them; merchants that drop out under some versions are marked *Borderline*.
5. **AI's role:** reading notes, writing the reason line and the drafts, checking translations and answering questions. AI never decides who is flagged, the driver, the playbook or the ranking. Code checks every draft; a person approves every message.
""")

# ================================================================== WHY NOT FLAGGED
with tab_why:
    st.markdown('### Why these merchants are not on the list')
    st.caption('The brief warns that seasonal businesses, small merchants and busy growing merchants can look risky without being at risk. '
               'This view shows how each of them was told apart.')
    nf = df[~df.declining].copy()
    groups = ['Seasonal dip', 'Bounced-back dip', 'Small and steady', 'Busy and growing', 'Growing', 'Steady', 'Possible decline', 'Watch list']
    counts = nf.why_group.value_counts()
    gc = st.columns(len([g for g in groups if g in counts]))
    for col, g in zip(gc, [g for g in groups if g in counts]):
        col.markdown(f'<div class="kpi"><div class="label">{g}</div><div class="value">{int(counts[g])}</div></div>', unsafe_allow_html=True)
    st.write('')
    pick = st.segmented_control('Show', [g for g in groups if g in counts], default=[g for g in groups if g in counts][0], key='why-pick')
    show = nf[nf.why_group == pick] if pick else nf
    st.dataframe(pd.DataFrame({'ID': show.merchant_id, 'Merchant': show.get('business_name', show.merchant_id), 'Size': show.segment,
                               'Volume 4-6 vs 1-3': [f'{x:+.0%}' for x in show.tpv_change], 'Why not flagged': show.why_not_flagged,
                               'Account manager note': show.account_manager_note.fillna('')}),
                 hide_index=True, width='stretch', height=min(36 * (len(show) + 1) + 4, 560),
                 column_config={'Why not flagged': st.column_config.TextColumn(width='large'), 'Account manager note': st.column_config.TextColumn(width='medium')})
