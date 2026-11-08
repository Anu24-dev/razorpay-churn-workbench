"""
Merchant Retention Workbench (Razorpay Campus AI Build-a-thon, Problem 2).

Input: the data pack (or any file with the same P2_Merchants columns).
Engine (rules + statistics, no AI): who is declining, how fast, why, which of the six data-pack playbooks, who gets the 15 calls.
AI (Claude): reads account-manager notes, writes the CSM's reason line, drafts the outreach (regional + English),
checks the translation, answers questions about one merchant. Code guardrails check every draft. AI can be switched off.
People: approve, edit or reject every draft; confirm unclear causes; account managers own pricing.

Design: built for a CSM preparing calls. The answer comes first (who, why, what to do), details are one click away,
the list stays visible while a merchant is open, and every action sits next to the thing it acts on.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai import llm
from ai import pipeline as P
from ai.facts import CHANNEL
from engine import core

st.set_page_config(page_title='Merchant Retention Workbench', page_icon='📉', layout='wide')

# ------------------------------------------------------------------ style (works in light and dark themes)
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
.kpi, .card, .tile {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; background: rgba(128,128,128,.05); color: inherit;}
.kpi {padding: 14px 16px; height: 100%;}
.kpi .label, .tile .label {font-size: .74rem; opacity: .65; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 1.32rem; font-weight: 650; margin-top: 2px;}
.kpi .sub {font-size: .8rem; opacity: .7; margin-top: 2px;}
.card {padding: 12px 14px; margin-bottom: 10px;}
.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0 10px 0;}
.tile {padding: 8px 10px;}
.tile .value {font-size: 1.02rem; font-weight: 650; margin-top: 1px;}
.tile.flag {border-color: rgba(180,35,24,.55); background: rgba(180,35,24,.07);}
.tile.warn {border-color: rgba(154,91,0,.5); background: rgba(154,91,0,.07);}
.pill {display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .74rem; font-weight: 600; margin: 0 4px 4px 0;}
.p-red {background: #FDECEC; color: #B42318;} .p-amber {background: #FEF3E2; color: #9A5B00;} .p-blue {background: #E8F0FE; color: #1A4FB8;}
.p-grey {background: #EEF0F3; color: #475467;} .p-green {background: #E7F6EC; color: #1E7B3A;} .p-purple {background: #F1EAFE; color: #5B2DB3;}
.muted {opacity: .68; font-size: .86rem;}
.pb-id {font-weight: 700;} .pb-text {font-size: .86rem; opacity: .8; margin: 2px 0 6px 0;}
.banner {border-left: 4px solid #9A5B00; background: rgba(154,91,0,.08); padding: 8px 12px; border-radius: 6px; margin-bottom: 8px; font-size: .9rem;}
.check-ok {color: #1E7B3A;} .check-bad {color: #B42318;}
</style>
""", unsafe_allow_html=True)


def rs(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'Rs {x / 1e5:,.1f} lakh' if abs(x) >= 1e5 else f'Rs {x:,.0f}'


def pill(text, kind='grey'):
    return f'<span class="pill p-{kind}">{text}</span>'


TIER_SHORT = {'1': 'Imminent fade', '2': 'Declining · renews soon', '3': 'Declining', '3P': 'Possible decline', '4': 'Bounced back', '5': 'Watch', '6': 'Stable / growing'}
TIER_PILL = {'1': ('Imminent fade', 'red'), '2': ('Declining, renews within 90 days', 'amber'), '3': ('Declining', 'amber'),
             '3P': ('Possible decline', 'blue'), '4': ('Bounced back', 'grey'), '5': ('Watch (experimental)', 'purple'), '6': ('Stable / growing', 'green')}
DRIVER_NAME = {'performance': 'Performance', 'service': 'Service', 'engagement': 'Engagement', core.NC: 'Unclear'}


def tier_key(t):
    return '3P' if str(t).startswith('3P') else str(t)[0]


# ------------------------------------------------------------------ sidebar: file, name, AI switch
DEFAULT_FILE = 'data/Buildathon_Student_Data_Pack.xlsx'


@st.cache_data(show_spinner=False)
def run_engine(file_bytes: bytes, name: str):
    if name.lower().endswith('.csv'):
        return core.run(pd.read_csv(io.BytesIO(file_bytes)))
    return core.run(core.load_merchants(io.BytesIO(file_bytes)), playbooks=core.load_playbooks(io.BytesIO(file_bytes)))


ss = st.session_state
with st.sidebar:
    st.markdown('### Merchant Retention Workbench')
    st.caption('Problem 2: churn early-warning and proactive outreach')
    up = st.file_uploader('Merchant file (data pack .xlsx, or .csv with the same columns)', type=['xlsx', 'csv'])
    if up is not None:
        file_bytes, file_name = up.getvalue(), up.name
    else:
        file_bytes, file_name = open(DEFAULT_FILE, 'rb').read(), DEFAULT_FILE.split('/')[-1]
    st.caption(f'Using: **{file_name}**')
    ss.csm = st.text_input('Your name (for the approval log)', value=ss.get('csm', ''), placeholder='e.g. Anu')
    st.divider()
    ai_on = st.toggle('AI on (Claude)', value=ss.get('ai_on', llm.has_key()), disabled=not llm.has_key(), key='ai_on',
                      help='On: Claude can write new drafts and answer new questions. Off: no calls to Claude at all. '
                           'Stored AI results for the data pack are still shown, and anything else falls back to rule-based text.')
    llm.set_enabled(ai_on)
    if not llm.has_key():
        st.caption('No API key on this server, so AI is off. Stored AI results for the data pack are shown.')
    elif ai_on:
        st.caption('AI is on. New calls to Claude are capped to protect the budget.')
    else:
        st.caption('AI is off: no calls to Claude. Stored results only.')
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
PB = res['playbooks']
by_id = df.set_index('merchant_id', drop=False)
calls = df[df.csm_call_list].sort_values('revenue_rank')
review = df[df.declining & df.needs_person].sort_values('revenue_rank')
V, S = res['value'], res['summary']
ss.setdefault('decisions', {}); ss.setdefault('audit', []); ss.setdefault('review_decisions', {}); ss.setdefault('qa', {}); ss.setdefault('selected', None)


def log(action, merchant_id, detail=''):
    ss.audit.append({'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'by': ss.csm or '(no name)', 'file': file_name,
                     'merchant_id': merchant_id, 'action': action, 'detail': detail})


# ------------------------------------------------------------------ AI results (stored, or live when AI is on)
@st.cache_data(show_spinner=False)
def ai_reason(_row, key, live):
    return P.write_reason(_row)


@st.cache_data(show_spinner=False)
def ai_draft(_row, key, live):
    return P.draft_with_retry(_row)


@st.cache_data(show_spinner=False)
def ai_note(note, live):
    return P.read_note(note)


def reason_for(r):
    if r.declining or r.possible_decline:
        text, _ = ai_reason(r, f'{file_id}-{r.merchant_id}', llm.live())
        if text:
            return text, 'AI'
        return (f"Volume in months 4-6 is {abs(r.tpv_change):.0%} below months 1-3 and projected to be {r.projected_loss_m9:.0%} below its healthy level by month 9. "
                f"Main driver: {DRIVER_NAME.get(r.driver_main, r.driver_main).lower()}. Renews in {int(r.days_to_contract_renewal)} days."), 'rule-based'
    return r.why_not_flagged, 'rule-based'


def pb_label(r):
    return r.playbook_id + (f' + {r.playbook_also_id}' if r.playbook_also_id else '')


# ------------------------------------------------------------------ header
st.markdown('## Merchant Retention Workbench')
st.markdown(f'<span class="muted">{S["merchants"]} merchants scored from <b>{file_name}</b> · month-6 snapshot · '
            f'typical monthly wobble {S["typical_wobble"]:.1%}, so a decline means a drop beyond {abs(S["decline_line"]):.1%} that does not bounce back</span>',
            unsafe_allow_html=True)
tab_work, tab_why = st.tabs(['Workbench', 'Why not flagged'])

# ================================================================== WORKBENCH
with tab_work:
    # ---- money panel
    cards = [('Merchants at risk', f"{S['at_risk']}", f"{S['imminent']} imminent fades · {S['review']} need a person"),
             ('Yearly revenue at risk', rs(V['at_risk_value']), 'net revenue at 1% of TPV'),
             ('Covered by the 15 calls', rs(V['call_list_value']), f"{V['call_list_value'] / V['at_risk_value']:.0%} of revenue at risk" if V['at_risk_value'] else ''),
             ('Expected saving', f"Rs {V['saving_10'] / 1e5:,.1f}-{V['saving_ours'] / 1e5:,.1f}L", 'a range: the save chance is an assumption'),
             ('Cost of the 15 calls', rs(V['outreach_cost']), f"each call pays off above a {V['break_even_max']:.1%} save chance" if V['outreach_cost'] else '')]
    for col, (lab, val, sub) in zip(st.columns(5), cards):
        col.markdown(f'<div class="kpi"><div class="label">{lab}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    st.write('')

    # ---- driver mix and the fix
    mix = res['driver_mix']
    if len(mix):
        c1, c2 = st.columns([3, 2])
        with c1:
            names = [DRIVER_NAME.get(d, d) + (' (needs a person)' if d == core.NC else '') for d in mix.driver]
            fig = go.Figure(go.Bar(x=mix.merchants, y=names, orientation='h', marker_color=['#1A4FB8' if d != core.NC else '#98A2B3' for d in mix.driver],
                                   text=[f"{m} merchants · {rs(v)}/yr" for m, v in zip(mix.merchants, mix.yearly_value_rs)], textposition='auto',
                                   hovertemplate='%{y}: %{x} merchants<extra></extra>'))
            fig.update_layout(height=190, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Why merchants are at risk', font=dict(size=14)),
                              xaxis=dict(visible=False), yaxis=dict(autorange='reversed'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        with c2:
            top = mix[mix.driver != core.NC].iloc[0] if (mix.driver != core.NC).any() else mix.iloc[0]
            st.markdown(f"""<div class="card"><div class="label muted">BIGGEST FIXABLE DRIVER · SUGGESTED PROCESS FIX</div>
            <div style="font-weight:650;margin:4px 0">{DRIVER_NAME.get(top.driver, top.driver)}: {top.merchants} merchants, {rs(top.yearly_value_rs)}/yr</div>
            <div style="font-size:.9rem">{res['fix_ideas'][top.driver]}</div></div>""", unsafe_allow_html=True)
            if (mix.driver == core.NC).any():
                nc_df = df[df.declining & df.needs_person]
                soon = int((nc_df.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS).sum())
                st.markdown(f"""<div class="card" style="font-size:.88rem">{len(nc_df)} merchants show no operational cause; {soon} renew within 60 days.
                Suggested fix: {res['fix_ideas'][core.NC]}</div>""", unsafe_allow_html=True)

    # ---- the 15 calls (list) + merchant panel
    st.markdown("### This month's 15 calls")
    st.caption('Ranked by chance of leaving x chance of saving x yearly value. Borderline = drops out of the 15 under some reasonable assumptions, '
               'so the CSM lead decides those slots. Click a row to open the merchant.')

    def status_of(mid):
        d = ss.decisions.get((file_id, mid))
        return d['status'] if d else 'Pending'
    tbl = pd.DataFrame({
        '#': calls.revenue_rank.astype(int).values,
        'Merchant': calls.get('business_name', calls.merchant_id).values,
        'ID': calls.merchant_id.values,
        'Size': calls.segment.values,
        'Risk': [TIER_SHORT[tier_key(t)] for t in calls.tier],
        'Why': [('Unclear (person checks)' if d == core.NC else DRIVER_NAME[d] + (f' + {DRIVER_NAME[s].lower()}' if isinstance(s, str) and s else ''))
                for d, s in zip(calls.driver_main, calls.driver_second)],
        'Playbook': [pb_label(r) for r in calls.itertuples()],
        'Renews in': [f"{int(x)} d" for x in calls.days_to_contract_renewal],
        'Slot': calls.call_list_status.values,
        'Draft': [status_of(m) for m in calls.merchant_id]})
    prev = ss.get(f'calls-{file_id}')
    has_sel = bool(prev and getattr(prev, 'selection', None) and prev.selection.rows)
    if has_sel:
        left, right = st.columns([12, 11], gap='medium')
    else:
        left, right = st.container(), st.container()
    with left:
        sel = st.dataframe(tbl, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key=f'calls-{file_id}',
                           height=36 * (len(tbl) + 1) + 4,
                           column_order=(['#', 'Merchant', 'Why', 'Playbook', 'Draft'] if has_sel else list(tbl.columns)),
                           column_config={'#': st.column_config.NumberColumn(width=36), 'Merchant': st.column_config.TextColumn(width=135),
                                          'Why': st.column_config.TextColumn(width=150), 'Playbook': st.column_config.TextColumn(width=88),
                                          'Draft': st.column_config.TextColumn(width=78)})
        if sel and sel.selection.rows:
            ss.selected = tbl.iloc[sel.selection.rows[0]]['ID']
        if not has_sel:
            st.caption('Tip: the playbook codes are the six in the data pack. Open a merchant to read what each one involves.')

    with right:
        mid = ss.selected if (has_sel or (sel and sel.selection.rows)) and ss.selected in set(calls.merchant_id) else None
        if mid is not None:
            r = by_id.loc[mid]
            tl, tc = TIER_PILL[tier_key(r.tier)]
            with st.container(height=36 * (len(tbl) + 1) + 4, border=True):
                pills = pill(tl, tc) + pill(r.segment, 'grey') + pill(r.call_list_status, 'amber' if r.call_list_status == 'Borderline' else 'green')
                if r.call_this_week: pills += pill('Call this week', 'red')
                if r.needs_person: pills += pill('Cause unclear: confirm first', 'amber')
                st.markdown(f"<div style='font-size:1.15rem;font-weight:650'>{r.get('business_name', mid)}</div>"
                            f"<div class='muted'>{mid} · {r.get('city', '')} · {r.industry} · renews in {int(r.days_to_contract_renewal)} days · "
                            f"with Razorpay {int(r.tenure_months)} months</div>{pills}", unsafe_allow_html=True)
                t_over, t_msg, t_trend, t_ask = st.tabs(['Overview', 'Message', 'Trends', 'Ask'])

                with t_over:
                    reason, src = reason_for(r)
                    st.markdown(f"<div class='card'><div class='label muted'>WHY THIS MERCHANT ({src})</div>{reason}</div>", unsafe_allow_html=True)
                    drv = {r.driver_main, r.driver_second}
                    sr_drop = r.success_rate_m1_m3_pct - r.success_rate_m4_m6_pct
                    csat = '–' if pd.isna(r.last_csat_1_to_5) else f'{r.last_csat_1_to_5:.0f}/5'
                    tiles = [('Volume, months 4-6 vs 1-3', f'{r.tpv_change:+.0%}', 'flag'),
                             ('Projected by month 9', f'-{r.projected_loss_m9:.0%}', 'flag' if r.imminent else 'warn'),
                             ('Success rate', f'{r.success_rate_m1_m3_pct:.1f}% → {r.success_rate_m4_m6_pct:.1f}%', 'flag' if 'performance' in drv else ''),
                             ('Unresolved tickets', f'{int(r.unresolved_tickets)} of {int(r.support_tickets_last_90d)}', 'flag' if 'service' in drv else ''),
                             ('Last CSAT', csat, 'flag' if ('service' in drv and csat != '–') else ''),
                             ('Logins, last 30 days', f'{int(r.logins_last_30d)}', 'flag' if 'engagement' in drv else ''),
                             ('Products used', f'{int(r.products_used)}', 'flag' if 'engagement' in drv else ''),
                             ('Renews in', f'{int(r.days_to_contract_renewal)} days', 'warn' if r.days_to_contract_renewal <= 60 else '')]
                    st.markdown('<div class="tiles">' + ''.join(f'<div class="tile {c}"><div class="label">{l}</div><div class="value">{v}</div></div>' for l, v, c in tiles)
                                + '</div>', unsafe_allow_html=True)
                    if r.needs_person:
                        st.markdown(f"<div class='banner'>The numbers show no operational problem, so a person confirms the cause before this playbook runs. "
                                    f"Hint: {r.review_likely_bucket}.</div>", unsafe_allow_html=True)
                    pb_html = (f"<div class='pb-id'>{r.playbook_id} {PB[r.playbook_id]['name']}</div><div class='pb-text'>{PB[r.playbook_id]['what_happens']}</div>")
                    if r.playbook_also_id:
                        pb_html += (f"<div class='pb-id'>+ {r.playbook_also_id} {PB[r.playbook_also_id]['name']}</div>"
                                    f"<div class='pb-text'>{PB[r.playbook_also_id]['what_happens']}</div>")
                    st.markdown(f"<div class='card'><div class='label muted'>PLAYBOOK (FROM THE DATA PACK)</div>{pb_html}"
                                f"<div class='muted'>Who runs it ({r.segment} merchant): {r.delivery}</div></div>", unsafe_allow_html=True)
                    note = r.get('account_manager_note')
                    if isinstance(note, str):
                        nr = ai_note(note, llm.live())
                        tag = f" · AI reads it as <b>{nr['category']}</b>" if nr and nr.get('category') else ''
                        st.markdown(f"<div class='muted'>Account manager note: “{note}”{tag}</div>", unsafe_allow_html=True)

                with t_msg:
                    key = (file_id, mid)
                    d = ai_draft(r, f'{file_id}-{mid}', llm.live())
                    if d is None:
                        st.info('No stored draft for this file and AI is off, so this is the rule-based English draft.')
                        drafts, checks, ok = {'English': ('', P.template_draft(r))}, [], True
                    else:
                        drafts = {'English': (d['english'].get('subject', ''), d['english'].get('message', ''))}
                        if d.get('regional') and 'message' in d['regional']:
                            drafts[d['regional_language']] = (d['regional'].get('subject', ''), d['regional']['message'])
                        order = [d['main_language']] + [l for l in drafts if l != d['main_language']]
                        drafts = {l: drafts[l] for l in order if l in drafts}
                        checks, ok = d['checks'], d['passed']
                    ch = CHANNEL[r.segment]
                    st.caption(f"{ch[0].upper() + ch[1:]}. Language guessed from the city ({r.get('city', '')}): confirm with the merchant."
                               + (' Enterprise: English first.' if r.segment == 'Enterprise' else ''))
                    if not ok:
                        st.error('Blocked by the guardrails after one retry. Write this one yourself, or fix the text below before approving.')
                    lang = st.radio('Version', list(drafts), horizontal=True, key=f'lang-{file_id}-{mid}', label_visibility='collapsed')
                    subj, body = drafts[lang]
                    original = (f'Subject: {subj}\n\n' if subj else '') + body
                    edited = st.text_area('Message', value=original, height=250, key=f'txt-{file_id}-{mid}-{lang}', label_visibility='collapsed')
                    if checks:
                        n_ok = sum(1 for _, okk, _ in checks if okk)
                        with st.expander(f"{'✓' if ok else '✗'} Guardrails: {n_ok} of {len(checks)} checks passed"):
                            for name, okk, det in checks:
                                st.markdown(f"<span class='{'check-ok' if okk else 'check-bad'}'>{'✓' if okk else '✗'}</span> {name}{' · ' + det if det else ''}",
                                            unsafe_allow_html=True)
                    b1, b2, b3 = st.columns(3)
                    if b1.button('Approve', type='primary', key=f'ap-{file_id}-{mid}', disabled=not ss.csm, width='stretch'):
                        ss.decisions[key] = {'status': 'Approved', 'language': lang, 'text': edited, 'edited': edited.strip() != original.strip()}
                        log('Approved', mid, f"{lang}{' (edited)' if ss.decisions[key]['edited'] else ''}"); st.rerun()
                    if b2.button('Reject', key=f'rj-{file_id}-{mid}', disabled=not ss.csm, width='stretch'):
                        ss.decisions[key] = {'status': 'Rejected', 'language': lang, 'text': '', 'edited': False}
                        log('Rejected', mid, lang); st.rerun()
                    if b3.button('Reset', key=f'rs-{file_id}-{mid}', width='stretch'):
                        ss.decisions.pop(key, None); log('Reset', mid); st.rerun()
                    st.caption(('Enter your name in the sidebar to approve or reject. ' if not ss.csm else '')
                               + f"Status: {ss.decisions[key]['status'] if key in ss.decisions else 'to review'}. Every decision is logged.")

                with t_trend:
                    months = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
                    tpv = [r[f'tpv_m{i}_inr_lakh'] for i in range(1, 7)]
                    f1 = go.Figure(go.Scatter(x=months, y=tpv, mode='lines+markers', line=dict(color='#1A4FB8', width=2.5), hovertemplate='%{x}: Rs %{y} lakh<extra></extra>'))
                    f1.add_hline(y=float(np.mean(tpv[:3])), line_dash='dot', line_color='#98A2B3', annotation_text='months 1-3 average', annotation_font_size=10)
                    f1.update_layout(height=220, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Monthly volume (Rs lakh)', font=dict(size=13)),
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(rangemode='tozero', gridcolor='rgba(128,128,128,.2)'))
                    st.plotly_chart(f1, width='stretch', config={'displayModeBar': False})
                    f2 = go.Figure(go.Bar(x=['Months 1-3', 'Months 4-6'], y=[r.success_rate_m1_m3_pct, r.success_rate_m4_m6_pct],
                                          marker_color=['#98A2B3', '#B42318' if sr_drop > 2 else '#1A4FB8'],
                                          text=[f'{r.success_rate_m1_m3_pct:.1f}%', f'{r.success_rate_m4_m6_pct:.1f}%'], textposition='outside'))
                    f2.update_layout(height=200, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Payment success rate', font=dict(size=13)),
                                     yaxis=dict(range=[min(70, r.success_rate_m4_m6_pct - 5), 100], visible=False), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(f2, width='stretch', config={'displayModeBar': False})
                    st.caption('Internal only. Drafts never mention volume or numbers.')

                with t_ask:
                    st.caption("Answers come only from this merchant's data and the method. If the data does not show it, it says so.")
                    for i, (lab, q) in enumerate([('Why is it on the list?', 'Why is this merchant on the call list?'),
                                                  ('What to focus on in the call', 'What should I focus on in the call?'),
                                                  ("What we don't know", 'What do we not know about this merchant?')]):
                        if st.button(lab, key=f'q{i}-{file_id}-{mid}', width='stretch'):
                            ss.qa[(file_id, mid)] = q
                    free = st.text_input('Or ask your own question', key=f'free-{file_id}-{mid}', placeholder='e.g. Is the drop seasonal?')
                    if free and free != ss.get(f'last-free-{file_id}-{mid}'):
                        ss.qa[(file_id, mid)] = free; ss[f'last-free-{file_id}-{mid}'] = free
                    q = ss.qa.get((file_id, mid))
                    if q:
                        ans, achk = P.answer_question(r, q)
                        st.markdown(f'**{q}**')
                        if ans is None:
                            st.info('No stored answer for this question, and AI is off. Switch AI on in the sidebar to ask new questions.')
                        elif achk and not achk[1]:
                            st.warning("This answer quoted a number that is not in the merchant's data, so it is hidden. Try rephrasing.")
                        else:
                            st.write(ans)

    with st.expander('See how every merchant is ranked (all of them, with risk, reason and playbook)'):
        allm = df.assign(_o=df.revenue_rank.fillna(999), _t=df.tier.map(tier_key)).sort_values(['_o', '_t', 'tpv_change'])
        full = pd.DataFrame({'Rank': allm.revenue_rank.map(lambda x: '' if pd.isna(x) else str(int(x))), 'ID': allm.merchant_id,
                             'Merchant': allm.get('business_name', allm.merchant_id), 'Size': allm.segment, 'Risk': [TIER_SHORT[tier_key(t)] for t in allm.tier],
                             'Reason': [('Unclear: needs a person' if d == core.NC else DRIVER_NAME[d]) if isinstance(d, str) else w for d, w in zip(allm.driver_main, allm.why_not_flagged)],
                             'Playbook': [pb_label(r) + ' ' + PB[r.playbook_id]['name'] for r in allm.itertuples()],
                             'Track': np.where(allm.csm_call_list, 'CSM call', np.where(allm.am_alert, 'Account-manager alert', np.where(allm.automated_track, 'Automated digital', 'No outreach')))})
        st.dataframe(full, hide_index=True, width='stretch', height=420)

    # ---- needs a person
    st.markdown('### Needs a person: cause unclear')
    st.caption('The numbers show no performance, service or engagement problem for these declining merchants. The playbook comes from the likely bucket, '
               'and a person confirms the cause before it runs. Merchants outside the 15 calls that renew within 60 days are alerted to their account manager, '
               'who owns any pricing conversation; the AI never offers terms.')
    if len(review) == 0:
        st.markdown('<div class="muted">None this month.</div>', unsafe_allow_html=True)
    for _, r in review.iterrows():
        note = r.get('account_manager_note'); nr = ai_note(note, llm.live()) if isinstance(note, str) else None
        cA, cB = st.columns([3, 2])
        with cA:
            where = 'On the call list' if r.csm_call_list else ('Account-manager alert' if r.am_alert else 'Automated track')
            st.markdown(f"<div class='card'><b>{r.get('business_name', r.merchant_id)}</b> <span class='muted'>· {r.merchant_id} · {r.segment} · renews in {int(r.days_to_contract_renewal)} days · {where}</span><br>"
                        f"{pill(r.review_likely_bucket, 'blue')}{pill('Playbook: ' + r.playbook, 'grey')}"
                        f"{pill('Note reads as ' + nr['category'], 'purple') if nr and nr.get('category') else ''}"
                        f"<div class='muted' style='margin-top:6px'>Note: “{note if isinstance(note, str) else 'no note'}”</div></div>", unsafe_allow_html=True)
        with cB:
            opts = ['Not confirmed yet', 'Pricing / commercial: PB-3', 'Performance: PB-1', 'Service: PB-2', 'Engagement: PB-4', 'Other or external: PB-6']
            prev_choice = ss.review_decisions.get((file_id, r.merchant_id), opts[0])
            choice = st.selectbox('Confirmed cause and playbook', opts, index=opts.index(prev_choice), key=f'rv-{file_id}-{r.merchant_id}', disabled=not ss.csm)
            if choice != prev_choice:
                ss.review_decisions[(file_id, r.merchant_id)] = choice; log('Cause confirmed', r.merchant_id, choice)

    # ---- export and log
    st.markdown('### Export and approval log')
    e1, e2, e3 = st.columns(3)
    appr = [{'merchant_id': m, 'business_name': by_id.loc[m].get('business_name', m), 'channel': CHANNEL[by_id.loc[m].segment],
             'language': v['language'], 'edited_by_csm': v['edited'], 'message': v['text']}
            for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Approved']
    e1.download_button(f'Approved messages ({len(appr)})', pd.DataFrame(appr).to_csv(index=False).encode('utf-8-sig') if appr else b'',
                       file_name='approved_messages.csv', disabled=not appr, width='stretch')
    out_cols = ['merchant_id', 'segment', 'industry', 'tier', 'driver_main', 'driver_second', 'review_likely_bucket', 'needs_person', 'playbook', 'playbook_also',
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
3. **Why?** Three problem scores from the merchant's own signals: performance (success rate), service (unresolved tickets, CSAT), engagement (logins, products). A score more than 1 standard deviation above average names the driver. If none is high, a person confirms the cause. Blank CSAT is never used to name a driver.
4. **Which playbook?** Always one of the six in the data pack: performance PB-1, service PB-2, engagement PB-4, unclear cause follows the likely bucket (PB-3 when renewal is within 60 days). A second problem adds its playbook. Merchant size sets who runs it.
5. **Who gets the 15 calls?** Chance of leaving x chance of saving if we call x yearly value, minus Rs 1,500 per call. Both chances are assumptions (the data has no outcomes), so the list is re-run under 11 versions of them; merchants that drop out under some are marked *Borderline*.
6. **AI's role:** reading notes, writing the reason line and the drafts, checking translations and answering questions. AI never decides who is flagged, the driver, the playbook or the ranking. Code checks every draft; a person approves every message. AI can be switched off in the sidebar.
""")

# ================================================================== WHY NOT FLAGGED
with tab_why:
    st.markdown('### Why these merchants are not on the list')
    st.caption('The brief warns that seasonal businesses, small merchants and busy growing merchants can look risky without being at risk. '
               'This view shows how each of them was told apart.')
    nf = df[~df.declining].copy()
    groups = [g for g in ['Seasonal dip', 'Bounced-back dip', 'Small and steady', 'Busy and growing', 'Growing', 'Steady', 'Possible decline', 'Watch list']
              if g in set(nf.why_group)]
    counts = nf.why_group.value_counts()
    for col, g in zip(st.columns(len(groups)), groups):
        col.markdown(f'<div class="kpi"><div class="label">{g}</div><div class="value">{int(counts[g])}</div></div>', unsafe_allow_html=True)
    st.write('')
    pick = st.segmented_control('Show', groups, default=groups[0], key='why-pick')
    show = nf[nf.why_group == pick] if pick else nf
    st.dataframe(pd.DataFrame({'ID': show.merchant_id, 'Merchant': show.get('business_name', show.merchant_id), 'Size': show.segment,
                               'Volume 4-6 vs 1-3': [f'{x:+.0%}' for x in show.tpv_change], 'Why not flagged': show.why_not_flagged,
                               'Playbook': show.playbook, 'Account manager note': show.account_manager_note.fillna('')}),
                 hide_index=True, width='stretch', height=min(36 * (len(show) + 1) + 4, 560),
                 column_config={'Why not flagged': st.column_config.TextColumn(width='large')})
