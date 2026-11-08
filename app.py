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
NAVY, BLUE, BLUE_SOFT, GREY = '#012652', '#3395FF', '#EAF3FF', '#98A2B3'   # Razorpay brand navy and blue
st.markdown("""
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1440px;}
h1, h2, h3, h4 {letter-spacing: -0.015em;}
h3 {color: inherit; margin-top: .6rem;}
.hero {background: linear-gradient(120deg, #012652 0%, #06397A 55%, #3395FF 130%); color: #fff; border-radius: 16px; padding: 22px 26px 18px 26px; margin-bottom: 14px;
       box-shadow: 0 6px 24px rgba(1,38,82,.18);}
.hero .eyebrow {font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; opacity: .75;}
.hero .title {font-size: 1.85rem; font-weight: 750; margin: 2px 0 4px 0; letter-spacing: -0.02em;}
.hero .sub {font-size: .92rem; opacity: .85;}
.hero .chip {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 3px 11px;
             font-size: .78rem; margin: 10px 6px 0 0;}
.kpi, .card, .tile {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; background: rgba(255,255,255,.72); color: inherit;}
@media (prefers-color-scheme: dark) { .stApp[data-theme="dark"] .kpi, .stApp[data-theme="dark"] .card, .stApp[data-theme="dark"] .tile {background: rgba(255,255,255,.04);} }
.kpi {padding: 14px 16px 12px 16px; height: 100%; border-top: 4px solid #3395FF; box-shadow: 0 2px 10px rgba(1,38,82,.06);}
.kpi.accent {border-top-color: #012652;}
.kpi .label, .tile .label {font-size: .72rem; opacity: .62; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}
.kpi .value {font-size: 1.45rem; font-weight: 750; margin-top: 2px; color: #012652;}
.kpi .sub {font-size: .79rem; opacity: .7; margin-top: 2px;}
.card {padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 1px 6px rgba(1,38,82,.05);}
.card.fix {border-left: 4px solid #3395FF;}
.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0 10px 0;}
.tile {padding: 8px 10px;}
.tile .value {font-size: 1.02rem; font-weight: 700; margin-top: 1px;}
.tile.flag {border-color: rgba(229,72,77,.55); background: rgba(229,72,77,.07);}
.tile.warn {border-color: rgba(245,158,11,.6); background: rgba(245,158,11,.08);}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 650; margin: 0 4px 4px 0;}
.p-red {background: #FDECEC; color: #C2262E;} .p-amber {background: #FFF4E0; color: #9A5B00;} .p-blue {background: #EAF3FF; color: #0B5ED7;}
.p-grey {background: #EEF1F5; color: #475467;} .p-green {background: #E6F7EE; color: #11804A;} .p-purple {background: #F1EAFE; color: #5B2DB3;}
.p-navy {background: #012652; color: #fff;}
.muted {opacity: .68; font-size: .86rem;}
.pb-id {font-weight: 750; color: #0B5ED7;} .pb-text {font-size: .86rem; opacity: .82; margin: 2px 0 6px 0;}
.banner {border-left: 4px solid #F59E0B; background: rgba(245,158,11,.09); padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: .9rem;}
.check-ok {color: #11804A; font-weight: 700;} .check-bad {color: #C2262E; font-weight: 700;}
.done {border-left: 4px solid #11804A; background: rgba(17,128,74,.07); padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;}
.skipped {border-left: 4px solid #98A2B3; background: rgba(128,128,128,.08); padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;}
.facts {display: grid; grid-template-columns: 150px 1fr; gap: 5px 12px; font-size: .88rem; margin-top: 8px;}
.facts .k {opacity: .62; font-weight: 600;} .facts .v.hot {color: #9A5B00; font-weight: 650;}
.asks {margin: 6px 0 0 0; padding-left: 18px; font-size: .88rem;}
section[data-testid="stSidebar"] > div {background: linear-gradient(180deg, #EAF3FF 0%, #F7FAFF 40%);}
.side-brand {font-weight: 750; color: #012652; font-size: 1.05rem;}
div[data-testid="stTabs"] button[aria-selected="true"] p {color: #0B5ED7; font-weight: 650;}
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


PRECOMPUTED = 'data/datapack_results.pkl'   # full results for the data pack, so the page opens instantly


@st.cache_data(show_spinner=False)
def run_engine(file_bytes: bytes, name: str, watch: bool):
    """watch=False is instant (0.1 s) and already has every decision; watch=True adds the experimental watch list (the slow part)."""
    if watch and hashlib.sha1(file_bytes).hexdigest() == hashlib.sha1(open(DEFAULT_FILE, 'rb').read()).hexdigest():
        try:
            import pickle
            with open(PRECOMPUTED, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    if name.lower().endswith('.csv'):
        return core.run(pd.read_csv(io.BytesIO(file_bytes)), watch=watch)
    return core.run(core.load_merchants(io.BytesIO(file_bytes)), playbooks=core.load_playbooks(io.BytesIO(file_bytes)), watch=watch)


ss = st.session_state
ss.setdefault('up_key', 0); ss.setdefault('watch_ready', set()); ss.setdefault('last_file', None)
with st.sidebar:
    st.markdown("<div class='side-brand'>Merchant Retention Workbench</div>", unsafe_allow_html=True)
    st.caption('Razorpay Campus AI Build-a-thon · Problem 2: churn early-warning and proactive outreach')
    up = st.file_uploader('Score a different month (optional)', type=['xlsx', 'csv'], key=f"up-{ss.up_key}",
                          help='Upload the data pack or any file with the same P2_Merchants columns. It is scored as soon as it is uploaded: no button needed.')
    if up is not None:
        file_bytes, file_name = up.getvalue(), up.name
    else:
        file_bytes, file_name = open(DEFAULT_FILE, 'rb').read(), DEFAULT_FILE.split('/')[-1]
    status_box = st.empty()
    if up is not None and st.button('Back to the data pack', width='stretch'):
        ss.up_key += 1; st.rerun()
    ss.csm = st.text_input('Your name (signs the drafts and the log)', value=ss.get('csm', 'Anoushka'), placeholder='e.g. Anoushka')
    st.divider()
    ai_on = st.toggle('AI on (Claude)', value=ss.get('ai_on', llm.has_key()), disabled=not llm.has_key(), key='ai_on',
                      help='On: you can ask Claude to write new drafts and answer new questions (only when you click). '
                           'Off: no calls to Claude at all. Stored AI results are always shown.')
    llm.set_enabled(ai_on)
    if not llm.has_key():
        st.caption('No API key on this server, so AI is off. Stored AI results for the data pack are shown.')
    elif ai_on:
        st.caption('AI is on. Claude is only called when you click a button, and calls are capped to protect the budget.')
    else:
        st.caption('AI is off: no calls to Claude. Stored results only.')
    st.divider()
    st.caption('Nothing is sent to merchants from this app. Approved messages are exported for the CSM to send.')

file_id = hashlib.sha1(file_bytes).hexdigest()[:10]
t0 = dt.datetime.now()
watch_ready = file_id in ss.watch_ready or file_name == DEFAULT_FILE.split('/')[-1] and up is None
res = run_engine(file_bytes, file_name, watch_ready)
secs = (dt.datetime.now() - t0).total_seconds()
if not res['ok']:
    status_box.error(f'Could not score {file_name}. Fix the checks marked FAIL and upload again.')
    st.error(f'**{file_name}** cannot be scored. These checks failed:')
    st.dataframe(res['checks'], hide_index=True, width='stretch')
    st.stop()
n_merch = res['summary']['merchants']
status_box.success(f"Loaded **{file_name}**: {n_merch} merchants scored, {int(res['df'].csm_call_list.sum())} calls ready.", icon='✅')
if ss.last_file != file_id:
    if ss.last_file is not None:
        st.toast(f'Loaded {file_name}: {n_merch} merchants scored.', icon='✅')
    ss.last_file = file_id

df = res['df'].copy()
PB = res['playbooks']
by_id = df.set_index('merchant_id', drop=False)
calls = df[df.csm_call_list].sort_values('revenue_rank')
review = df[df.declining & df.needs_person].sort_values('revenue_rank')
V, S = res['value'], res['summary']
ss.setdefault('decisions', {}); ss.setdefault('audit', []); ss.setdefault('review_decisions', {}); ss.setdefault('qa', {}); ss.setdefault('selected', None)


def log(action, merchant_id, detail=''):
    ss.audit.append({'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'by': ss.csm or 'name not given', 'file': file_name,
                     'merchant_id': merchant_id, 'action': action, 'detail': detail})


# ------------------------------------------------------------------ AI results
# Drawing the page never waits on Claude: it only reads stored results. New AI work happens only when the user clicks,
# with a progress box saying what is happening.
def stored(fn, *a):
    with llm.stored_only():
        return fn(*a)


def reason_for(r):
    if r.declining or r.possible_decline:
        text, _ = stored(P.write_reason, r)
        if text:
            return text, 'AI'
        return (f"Volume in months 4-6 is {abs(r.tpv_change):.0%} below months 1-3 and projected to be {r.projected_loss_m9:.0%} below its healthy level by month 9. "
                f"Main driver: {DRIVER_NAME.get(r.driver_main, r.driver_main).lower()}. Renews in {int(r.days_to_contract_renewal)} days."), 'rule-based'
    return r.why_not_flagged, 'rule-based'


def pb_label(r):
    return r.playbook_id + (f' + {r.playbook_also_id}' if r.playbook_also_id else '')


# ------------------------------------------------------------------ header
st.markdown(f"""<div class="hero"><div class="eyebrow">Customer success · Churn early-warning</div>
<div class="title">Merchant Retention Workbench</div>
<div class="sub">Find merchants whose payment volume is falling and not coming back, see why, and send the right first message to the 15 that matter most.</div>
<span class="chip">{S['merchants']} merchants · {file_name}</span><span class="chip">Month-6 snapshot</span>
<span class="chip">Decline = drop beyond {abs(S['decline_line']):.1%} that does not bounce back</span>
<span class="chip">AI {'on' if llm.live() else 'off'}</span></div>""", unsafe_allow_html=True)
tab_work, tab_why = st.tabs(['Workbench', 'Why not flagged'])

# ================================================================== WORKBENCH
with tab_work:
    # ---- money panel
    cards = [('Merchants at risk', f"{S['at_risk']}", f"{S['imminent']} imminent fades · {S['review']} need a person"),
             ('Yearly revenue at risk', rs(V['at_risk_value']), 'net revenue at 1% of TPV'),
             ('Covered by the 15 calls', rs(V['call_list_value']), f"{V['call_list_value'] / V['at_risk_value']:.0%} of revenue at risk" if V['at_risk_value'] else ''),
             ('Expected saving', f"Rs {V['saving_10'] / 1e5:,.1f}-{V['saving_ours'] / 1e5:,.1f}L", 'a range: the save chance is an assumption'),
             ('Cost of the 15 calls', rs(V['outreach_cost']), f"each call pays off above a {V['break_even_max']:.1%} save chance" if V['outreach_cost'] else '')]
    for i, (col, (lab, val, sub)) in enumerate(zip(st.columns(5), cards)):
        col.markdown(f'<div class="kpi{" accent" if i in (0, 3) else ""}"><div class="label">{lab}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    st.write('')

    # ---- driver mix and the fix
    mix = res['driver_mix']
    if len(mix):
        c1, c2 = st.columns([3, 2])
        with c1:
            names = [DRIVER_NAME.get(d, d) + (' (needs a person)' if d == core.NC else '') for d in mix.driver]
            fig = go.Figure(go.Bar(x=mix.merchants, y=names, orientation='h', marker_color=[BLUE if d != core.NC else GREY for d in mix.driver],
                                   text=[f"{m} merchant{'s' if m != 1 else ''} · {rs(v)}/yr" for m, v in zip(mix.merchants, mix.yearly_value_rs)],
                                   textposition='outside', cliponaxis=False, textfont=dict(color=NAVY, size=12),
                                   hovertemplate='%{y}: %{x} merchants<extra></extra>'))
            fig.update_layout(height=190, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Why merchants are at risk', font=dict(size=14)),
                              xaxis=dict(visible=False, range=[0, float(mix.merchants.max()) * 1.75]), yaxis=dict(autorange='reversed'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        with c2:
            top = mix[mix.driver != core.NC].iloc[0] if (mix.driver != core.NC).any() else mix.iloc[0]
            st.markdown(f"""<div class="card fix"><div class="label muted">BIGGEST FIXABLE DRIVER · SUGGESTED PROCESS FIX</div>
            <div style="font-weight:650;margin:4px 0">{DRIVER_NAME.get(top.driver, top.driver)}: {top.merchants} merchants, {rs(top.yearly_value_rs)}/yr</div>
            <div style="font-size:.9rem">{res['fix_ideas'][top.driver]}</div></div>""", unsafe_allow_html=True)
            if (mix.driver == core.NC).any():
                nc_df = df[df.declining & df.needs_person]
                soon = int((nc_df.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS).sum())
                st.markdown(f"""<div class="card" style="font-size:.88rem">{len(nc_df)} merchant{'s' if len(nc_df) != 1 else ''} show{'' if len(nc_df) != 1 else 's'} no operational cause; {soon} renew{'s' if soon == 1 else ''} within 60 days.
                Suggested fix: {res['fix_ideas'][core.NC]}</div>""", unsafe_allow_html=True)

    # ---- the 15 calls (list) + merchant panel
    st.markdown("### This month's 15 calls")
    st.caption('Ranked by chance of leaving x chance of saving x yearly value. Borderline = drops out of the 15 under some reasonable assumptions, '
               'so the CSM lead decides those slots. Click a row to open the merchant.')

    STATUS_LABEL = {'Approved': '✓ Approved', 'Not this month': '✗ Not this month', 'Writing it myself': '✎ Rewriting'}

    def status_of(mid):
        d = ss.decisions.get((file_id, mid))
        return STATUS_LABEL.get(d['status'], d['status']) if d else 'To review'
    n_ok = sum(1 for m in calls.merchant_id if (ss.decisions.get((file_id, m)) or {}).get('status') == 'Approved')
    n_skip = sum(1 for m in calls.merchant_id if (ss.decisions.get((file_id, m)) or {}).get('status') == 'Not this month')
    st.progress((n_ok + n_skip) / max(len(calls), 1),
                text=f"**{n_ok} approved** and in the outbox · {n_skip} not contacted this month · {len(calls) - n_ok - n_skip} still to review")
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
                    if src == 'rule-based' and (r.declining or r.possible_decline) and llm.live():
                        if st.button('Explain it in plain words with AI (a few seconds)', key=f'why-{file_id}-{mid}'):
                            with st.spinner('Asking Claude to explain this merchant...'):
                                P.write_reason(r)
                            st.rerun()
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
                        nr = stored(P.read_note, note)
                        tag = f" · AI reads it as <b>{nr['category']}</b>" if nr and nr.get('category') else ''
                        st.markdown(f"<div class='muted'>Account manager note: “{note}”{tag}</div>", unsafe_allow_html=True)

                with t_msg:
                    key = (file_id, mid)
                    d = stored(P.draft_with_retry, r)
                    if d is None:
                        if llm.live():
                            st.info('This merchant has no AI draft yet. Below is a rule-based English draft you can use now, or let Claude write the '
                                    'regional and English drafts and run every guardrail check (about 30 to 60 seconds).')
                            if st.button('Write the drafts with AI', type='primary', key=f'gen-{file_id}-{mid}'):
                                with st.status('Writing the drafts with Claude...', expanded=True) as stat:
                                    out = P.draft_with_retry(r, progress=lambda m: stat.write(m))
                                    stat.update(label='Drafts ready' if out and out.get('passed') else 'Done: see the guardrail result', state='complete')
                                st.rerun()
                        else:
                            st.info('No AI draft for this merchant and AI is off, so this is the rule-based English draft.')
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
                    dec = ss.decisions.get(key)
                    if dec and dec['status'] == 'Approved':
                        st.markdown(f"<div class='done'><b>✓ Approved</b> by {dec['by']} at {dec['time']} · {dec['language']}"
                                    f"{' · edited before approving' if dec['edited'] else ''}<br><span class='muted'>It is in the Outbox at the bottom of the page. "
                                    "Copy it with the icon on the message and send it from email or WhatsApp.</span></div>", unsafe_allow_html=True)
                        st.code(dec['text'], language=None, wrap_lines=True)
                        if '[' in dec['text']:
                            st.warning('Fill in the two time options in [ ] before you send it.')
                        if st.button('Undo approval', key=f'undo-{file_id}-{mid}'):
                            ss.decisions.pop(key); log('Approval undone', mid); st.toast(f'Approval undone: {r.get("business_name", mid)} is back to review.'); st.rerun()
                    elif dec and dec['status'] == 'Not this month':
                        nxt = df[df.declining & (df.revenue_rank > len(calls))].sort_values('revenue_rank').head(1)
                        nxt_txt = (f"Slot freed. Next in line: <b>{nxt.iloc[0].get('business_name', nxt.iloc[0].merchant_id)}</b> "
                                   f"({nxt.iloc[0].merchant_id}, rank {int(nxt.iloc[0].revenue_rank)})." if len(nxt) else 'Slot freed.')
                        st.markdown(f"<div class='skipped'><b>✗ Not contacting this month</b> · decided by {dec['by']} at {dec['time']}<br>"
                                    f"Reason: {dec['reason']}<br><span class='muted'>{nxt_txt}</span></div>", unsafe_allow_html=True)
                        if st.button('Undo', key=f'undo-{file_id}-{mid}'):
                            ss.decisions.pop(key); log('Rejection undone', mid); st.rerun()
                    else:
                        if dec and dec['status'] == 'Writing it myself':
                            st.markdown(f"<div class='banner'>You chose to rewrite this message ({dec['reason'].lower()}). Edit it below, then approve it.</div>",
                                        unsafe_allow_html=True)
                        elif not ok:
                            st.error('The safety checks blocked this draft after one retry. Rewrite it below, then approve it.')
                        lang = st.radio('Version', list(drafts), horizontal=True, key=f'lang-{file_id}-{mid}', label_visibility='collapsed')
                        subj, body = drafts[lang]
                        original = (f'Subject: {subj}\n\n' if subj else '') + body
                        edited = st.text_area('Message (edit if you like)', value=original, height=230, key=f'txt-{file_id}-{mid}-{lang}')
                        if checks:
                            n_pass = sum(1 for _, okk, _ in checks if okk)
                            with st.expander(f"{'✓' if ok else '✗'} Safety checks: {n_pass} of {len(checks)} passed"):
                                for name, okk, det in checks:
                                    st.markdown(f"<span class='{'check-ok' if okk else 'check-bad'}'>{'✓' if okk else '✗'}</span> {name}{' · ' + det if det else ''}",
                                                unsafe_allow_html=True)
                        rj_key = f'rj-open-{file_id}-{mid}'
                        b1, b2 = st.columns(2)
                        if b1.button('✓ Approve and add to outbox', type='primary', key=f'ap-{file_id}-{mid}', width='stretch'):
                            ss.decisions[key] = {'status': 'Approved', 'language': lang, 'text': edited, 'edited': edited.strip() != original.strip(),
                                                 'by': ss.csm or 'name not given', 'time': dt.datetime.now().strftime('%H:%M'), 'reason': ''}
                            log('Approved', mid, f"{lang}{' (edited)' if ss.decisions[key]['edited'] else ''}")
                            st.toast(f'Approved: {r.get("business_name", mid)} is in the outbox.', icon='✅'); st.rerun()
                        if b2.button('✗ Reject...', key=f'rj-{file_id}-{mid}', width='stretch'):
                            ss[rj_key] = not ss.get(rj_key, False)
                        if ss.get(rj_key):
                            with st.container(border=True):
                                why = st.radio('Why are you rejecting this draft?', ['The cause or playbook looks wrong', 'The wording or tone is not right',
                                               'This merchant was already contacted', 'Not worth a call this month', 'Other'], key=f'rjwhy-{file_id}-{mid}')
                                nxt_step = st.radio('What happens next?', ['I will rewrite the message myself (it stays on the list)',
                                                    'Do not contact this merchant this month (frees the slot)'], key=f'rjnext-{file_id}-{mid}')
                                c1, c2 = st.columns(2)
                                if c1.button('Confirm', type='primary', key=f'rjok-{file_id}-{mid}', width='stretch'):
                                    status = 'Writing it myself' if nxt_step.startswith('I will') else 'Not this month'
                                    ss.decisions[key] = {'status': status, 'language': lang, 'text': '', 'edited': False, 'reason': why,
                                                         'by': ss.csm or 'name not given', 'time': dt.datetime.now().strftime('%H:%M')}
                                    ss[rj_key] = False; log('Rejected: ' + status.lower(), mid, why); st.rerun()
                                if c2.button('Cancel', key=f'rjno-{file_id}-{mid}', width='stretch'):
                                    ss[rj_key] = False; st.rerun()
                        st.caption(f"Approving puts the message in the Outbox for you to send. The app never sends anything itself. Signed as {ss.csm or 'the CSM'}.")

                with t_trend:
                    months = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
                    tpv = [r[f'tpv_m{i}_inr_lakh'] for i in range(1, 7)]
                    f1 = go.Figure(go.Scatter(x=months, y=tpv, mode='lines+markers', line=dict(color=BLUE, width=3), fill='tozeroy', fillcolor='rgba(51,149,255,.10)', hovertemplate='%{x}: Rs %{y} lakh<extra></extra>'))
                    f1.add_hline(y=float(np.mean(tpv[:3])), line_dash='dot', line_color=NAVY, annotation_text='months 1-3 average', annotation_font_size=10)
                    f1.update_layout(height=220, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Monthly volume (Rs lakh)', font=dict(size=13)),
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(rangemode='tozero', gridcolor='rgba(128,128,128,.2)'))
                    st.plotly_chart(f1, width='stretch', config={'displayModeBar': False})
                    f2 = go.Figure(go.Bar(x=['Months 1-3', 'Months 4-6'], y=[r.success_rate_m1_m3_pct, r.success_rate_m4_m6_pct],
                                          marker_color=[GREY, '#E5484D' if sr_drop > 2 else BLUE],
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
                        ans, achk = stored(P.answer_question, r, q)
                        if ans is None and llm.live():
                            with st.spinner('Asking Claude (a few seconds)...'):
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
    st.caption('The numbers show no performance, service or engagement problem for these declining merchants. Everything to weigh is on each card. '
               'Each has a suggested playbook (PB-3 by default, because pricing is the most common hidden cause). Confirm the cause after the call and '
               'the playbook follows. Merchants outside the 15 calls that renew within 60 days go to their account manager, who owns any pricing talk.')
    if len(review) == 0:
        st.markdown('<div class="muted">None this month.</div>', unsafe_allow_html=True)
    FAM_NAME = {'performance': 'performance (success rate)', 'service': 'service (tickets, CSAT)', 'engagement': 'engagement (logins, products)'}
    for _, r in review.iterrows():
        note = r.get('account_manager_note'); nr = stored(P.read_note, note) if isinstance(note, str) else None
        scores = {k: r[f'fam_{k}'] for k in ['performance', 'service', 'engagement']}
        top_fam = max(scores, key=scores.get)
        seasonal = any(x in str(r.industry) for x in core.SEASONAL_INDUSTRIES)
        renew_soon = r.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS
        csat = '–' if pd.isna(r.last_csat_1_to_5) else f'{r.last_csat_1_to_5:.0f}/5'
        where = 'On the call list' if r.csm_call_list else ('Account-manager alert' if r.am_alert else 'Automated track')
        facts = [('Renewal', f"Renews in {int(r.days_to_contract_renewal)} days: " + ("inside PB-3's 60-day window, so a pricing review is due now" if renew_soon
                  else "outside PB-3's 60-day window"), renew_soon),
                 ('Account manager note', (f"“{note}”" + (f" (AI reads it as {nr['category']})" if nr and nr.get('category') else '')) if isinstance(note, str) else 'None',
                  bool(nr and nr.get('category') == 'pricing')),
                 ('Volume', f"Down {abs(r.tpv_change):.0%} (months 4-6 vs 1-3); projected {r.projected_loss_m9:.0%} below its healthy level by month 9", r.imminent),
                 ('Operational signals', f"Success rate {r.success_rate_m1_m3_pct:.1f}% → {r.success_rate_m4_m6_pct:.1f}% · unresolved tickets {int(r.unresolved_tickets)} of "
                  f"{int(r.support_tickets_last_90d)} · CSAT {csat} · logins (30 days) {int(r.logins_last_30d)} · products {int(r.products_used)}. "
                  f"None is high enough to name a cause; the strongest is {FAM_NAME[top_fam]} ({scores[top_fam]:+.1f} SD).", False),
                 ('Seasonality', f"{r.industry} is a seasonal industry: compare with last year's pattern before acting" if seasonal else f"{r.industry} is not a seasonal industry", seasonal),
                 ('At stake', f"{rs(r.yearly_value_rs)} a year · {r.segment} merchant · with Razorpay {int(r.tenure_months)} months", False)]
        asks = ['Has anything changed in how you take payments, or in who manages the account?']
        if renew_soon or (nr and nr.get('category') == 'pricing'):
            asks.insert(0, 'Are you reviewing pricing or comparing providers ahead of renewal?')
        if seasonal:
            asks.append('Is this dip part of your usual seasonal pattern?')
        asks.append({'performance': 'Have your customers mentioned failed payments?', 'service': 'Is anything still open with our support team?',
                     'engagement': 'Is the dashboard still giving your team what it needs?'}[top_fam])
        cA, cB = st.columns([3, 1.3])
        with cA:
            rows = ''.join(f"<div class='k'>{k}</div><div class='v{' hot' if hot else ''}'>{v}</div>" for k, v, hot in facts)
            st.markdown(f"<div class='card'><b>{r.get('business_name', r.merchant_id)}</b> <span class='muted'>· {r.merchant_id} · {where}</span><br>"
                        f"{pill(r.review_likely_bucket, 'blue')}{pill('Suggested: ' + r.playbook, 'grey')}"
                        f"<div class='facts'>{rows}</div>"
                        f"<div class='k' style='margin-top:8px;opacity:.62;font-weight:600;font-size:.88rem'>Ask on the call</div>"
                        f"<ul class='asks'>{''.join(f'<li>{a}</li>' for a in asks)}</ul></div>", unsafe_allow_html=True)
        with cB:
            opts = ['Not confirmed yet', 'Pricing / commercial: PB-3', 'Performance: PB-1', 'Service: PB-2', 'Engagement: PB-4', 'Other or external: PB-6']
            prev_choice = ss.review_decisions.get((file_id, r.merchant_id), opts[0])
            choice = st.selectbox('After the call, confirm the cause', opts, index=opts.index(prev_choice), key=f'rv-{file_id}-{r.merchant_id}')
            if choice != prev_choice:
                ss.review_decisions[(file_id, r.merchant_id)] = choice; log('Cause confirmed', r.merchant_id, choice)
                st.toast(f"Confirmed for {r.get('business_name', r.merchant_id)}: {choice}", icon='✅')
            if choice != opts[0]:
                st.markdown(f"<div class='done' style='font-size:.86rem'>✓ Confirmed: <b>{choice}</b><br><span class='muted'>by {ss.csm or 'name not given'}; "
                            "logged</span></div>", unsafe_allow_html=True)
            else:
                st.caption('Pick the cause once you have spoken to the merchant. The playbook then follows from it.')

    # ---- outbox, exports and log
    st.markdown('### Outbox: approved messages ready to send')
    approved = [(m, v) for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Approved']
    if not approved:
        st.markdown("<div class='muted'>Nothing approved yet. Open a merchant in the call list, check the message and click <b>Approve and add to outbox</b>.</div>",
                    unsafe_allow_html=True)
    for m, v in approved:
        rr = by_id.loc[m]
        with st.expander(f"✓ {rr.get('business_name', m)} · {CHANNEL[rr.segment]} · {v['language']} · approved by {v['by']} at {v['time']}"):
            st.code(v['text'], language=None, wrap_lines=True)
    skipped = [(m, v) for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Not this month']
    if skipped:
        st.markdown('<div class="muted">Not contacting this month: ' + '; '.join(f"{by_id.loc[m].get('business_name', m)} ({v['reason'].lower()})" for m, v in skipped)
                    + '</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    appr = [{'merchant_id': m, 'business_name': by_id.loc[m].get('business_name', m), 'channel': CHANNEL[by_id.loc[m].segment],
             'language': v['language'], 'approved_by': v['by'], 'edited_by_csm': v['edited'], 'message': v['text']} for m, v in approved]
    e1.download_button(f'Download approved messages ({len(appr)})', pd.DataFrame(appr).to_csv(index=False).encode('utf-8-sig') if appr else b'',
                       file_name='approved_messages.csv', disabled=not appr, width='stretch')
    out_cols = ['merchant_id', 'segment', 'industry', 'tier', 'driver_main', 'driver_second', 'review_likely_bucket', 'needs_person', 'playbook_id', 'playbook_also_id', 'playbook', 'playbook_also',
                'delivery', 'revenue_rank', 'call_list_status', 'call_this_week', 'am_alert', 'automated_track', 'expected_saving_rs', 'why_not_flagged']
    ranked = df.assign(_o=df.revenue_rank.fillna(999)).sort_values(['_o', 'tier'])[[c for c in out_cols if c in df]]
    e2.download_button('Download ranked list of all merchants', ranked.to_csv(index=False).encode('utf-8-sig'), file_name='ranked_merchants.csv', width='stretch')
    e3.download_button(f'Download decision log ({len(ss.audit)} entries)', pd.DataFrame(ss.audit).to_csv(index=False).encode('utf-8-sig') if ss.audit else b'',
                       file_name='decision_log.csv', disabled=not ss.audit, width='stretch')
    if ss.audit:
        with st.expander(f'Decision log: who did what ({len(ss.audit)} entries)'):
            st.dataframe(pd.DataFrame(ss.audit).iloc[::-1], hide_index=True, width='stretch', height=220)


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

# ------------------------------------------------------------------ finish the experimental watch list after the page is drawn
# Everything above is already on screen. The watch list is the only slow step (it does not change the 15 calls),
# so it runs last, with a note in the sidebar, and the page refreshes when it is done.
if not res.get('watch_done', True):
    status_box.info(f"Loaded **{file_name}**: {n_merch} merchants scored, {int(df.csm_call_list.sum())} calls ready. "
                    "Finishing the experimental watch list in the background...", icon='⏳')
    run_engine(file_bytes, file_name, True)
    ss.watch_ready.add(file_id)
    st.rerun()
