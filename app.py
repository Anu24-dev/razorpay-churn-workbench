"""
Merchant Retention Workbench (Razorpay Campus AI Build-a-thon, Problem 2).

Input: the data pack (or any file with the same P2_Merchants columns).
Engine (rules + statistics, no AI): who is declining, how fast, why, which of the six data-pack playbooks, who gets the 15 calls.
AI (Claude): reads account-manager notes, writes the CSM's reason line, drafts the outreach (regional + English),
checks the translation, answers questions about one merchant. Code guardrails check every draft. AI can be switched off.
People: approve, edit or reject every draft; confirm likely-pricing causes the numbers cannot show; account managers own pricing.

Design: built for a CSM preparing calls. The answer comes first (who, why, what to do), details are one click away,
the list stays visible while a merchant is open, and every action sits next to the thing it acts on.
"""
from __future__ import annotations

import datetime as dt
import re
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib
import os

import engine.core
import ai.llm, ai.facts, ai.guardrails, ai.pipeline


def _fresh_code():
    """Streamlit Cloud pulls new commits without restarting, so helper modules already in memory can be older than app.py.
    Reload them (in dependency order) whenever a file on disk is newer than the copy in memory."""
    mods = [engine.core, ai.llm, ai.facts, ai.guardrails, ai.pipeline]
    stale = [m for m in mods if getattr(m, '_loaded_mtime', None) != os.path.getmtime(m.__file__)]
    if stale:
        for m in mods:
            importlib.reload(m)
            m._loaded_mtime = os.path.getmtime(m.__file__)
        st.cache_data.clear()
_fresh_code()

from ai import llm
from ai import guardrails as G
from ai import pipeline as P
from ai.facts import CHANNEL, channel_for
from engine import core

st.set_page_config(page_title='Merchant Retention · Razorpay Build-a-thon', page_icon='📉', layout='wide', initial_sidebar_state='collapsed')

# ------------------------------------------------------------------ style (works in light and dark themes)
NAVY, BLUE, BLUE_SOFT, GREY = '#012652', '#0D94FB', '#EAF3FF', '#98A2B3'   # Razorpay brand navy and blue
st.markdown("""
<style>
.block-container {padding-top: 3.6rem; padding-bottom: 2rem; max-width: 1440px;}
h1, h2, h3, h4 {letter-spacing: -0.015em;}
h3 {color: inherit; margin-top: .6rem;}
.hero {background: linear-gradient(120deg, #012652 0%, #06397A 55%, #0D94FB 130%); color: #fff; border-radius: 16px; padding: 22px 26px 18px 26px; margin-bottom: 14px;
       box-shadow: 0 6px 24px rgba(1,38,82,.18);}
.hero .eyebrow {font-size: .82rem; letter-spacing: .12em; text-transform: uppercase; opacity: .75;}
.hero .title {font-size: 1.85rem; font-weight: 750; margin: 2px 0 4px 0; letter-spacing: -0.02em;}
.hero .sub {font-size: 1.05rem; opacity: .85;}
.hero .chip {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 3px 11px;
             font-size: .88rem; margin: 10px 6px 0 0;}
.kpi, .card, .tile {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; background: rgba(255,255,255,.72); color: inherit;}
@media (prefers-color-scheme: dark) { .stApp[data-theme="dark"] .kpi, .stApp[data-theme="dark"] .card, .stApp[data-theme="dark"] .tile {background: rgba(255,255,255,.04);} }
.kpi {padding: 14px 16px 12px 16px; height: 100%; border-top: 4px solid #0D94FB; box-shadow: 0 2px 10px rgba(1,38,82,.06);}
.kpi.accent {border-top-color: #012652;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 14px;}
.kpi .value {white-space: nowrap;}
.kpi .label, .tile .label {font-size: .86rem; opacity: .62; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}
.kpi .value {font-size: 1.75rem; font-weight: 750; margin-top: 2px; color: #012652;}
.kpi .sub {font-size: 1.08rem; opacity: .85; margin-top: 4px; line-height: 1.45;}
.card {padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 1px 6px rgba(1,38,82,.05);}
.card.fix {border-left: 4px solid #0D94FB;}
.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 6px 0 10px 0;}
.tile {padding: 8px 10px;}
.tile .value {font-size: 1.12rem; font-weight: 700; margin-top: 1px;}
.tile.flag {border-color: rgba(229,72,77,.55); background: rgba(229,72,77,.07);}
.tile.warn {border-color: rgba(245,158,11,.6); background: rgba(245,158,11,.08);}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .88rem; font-weight: 650; margin: 0 4px 4px 0;}
.p-red {background: #FDECEC; color: #C2262E;} .p-amber {background: #FFF4E0; color: #9A5B00;} .p-blue {background: #EAF3FF; color: #0B5ED7;}
.p-grey {background: #EEF1F5; color: #475467;} .p-green {background: #E6F7EE; color: #11804A;} .p-purple {background: #F1EAFE; color: #5B2DB3;}
.p-navy {background: #012652; color: #fff;}
.muted {opacity: .72; font-size: .96rem;}
.pb-id {font-weight: 750; color: #0B5ED7;} .pb-text {font-size: .96rem; opacity: .82; margin: 2px 0 6px 0;}
.banner {border-left: 4px solid #F59E0B; background: rgba(245,158,11,.09); padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: 1rem;}
.check-ok {color: #11804A; font-weight: 700;} .check-bad {color: #C2262E; font-weight: 700;}
.done {border-left: 4px solid #11804A; background: rgba(17,128,74,.07); padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;}
.skipped {border-left: 4px solid #98A2B3; background: rgba(128,128,128,.08); padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;}
.facts {display: grid; grid-template-columns: 150px 1fr; gap: 6px 12px; font-size: 1rem; margin-top: 8px;}
.facts .k {opacity: .62; font-weight: 600;} .facts .v.hot {color: #9A5B00; font-weight: 650;}
.asks {margin: 6px 0 0 0; padding-left: 18px; font-size: 1rem;}
section[data-testid="stSidebar"] > div {background: linear-gradient(180deg, #EAF3FF 0%, #F7FAFF 40%);}
.side-brand {font-weight: 750; color: #012652; font-size: 1.05rem;}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button {background: #0D94FB !important; color: #fff !important; border-radius: 10px !important;
  padding: 6px 12px !important; width: auto !important; box-shadow: 0 2px 8px rgba(13,148,251,.35);}
[data-testid="stExpandSidebarButton"]::after, [data-testid="stSidebarCollapsedControl"] button::after {content: 'Upload data'; font-weight: 650; font-size: .92rem; margin-left: 6px;}
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapsedControl"] svg {color: #fff !important; fill: #fff !important;}
div[data-testid="stTabs"] button[aria-selected="true"] p {color: #0B5ED7; font-weight: 650;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.why-list {list-style: none; padding: 0; margin: 4px 0 10px 0;}
.why-list li {padding: 7px 0 7px 26px; position: relative; font-size: 1.06rem; border-bottom: 1px solid rgba(128,128,128,.14);}
.why-list li:last-child {border-bottom: none;}
.why-list li::before {content: '•'; position: absolute; left: 8px; color: #0D94FB; font-weight: 900;}
.why-list li.hot::before {color: #E5484D;}
.plan {border-left: 4px solid #0D94FB;}
.plan .what {font-size: 1rem; opacity: .85; margin: 3px 0 6px 0;}
.lead {font-size: 1.1rem; margin: -4px 0 10px 0; opacity: .85;}
.hero .title {font-size: 1.7rem;}
</style>
""", unsafe_allow_html=True)


def rs(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'Rs {x / 1e5:,.1f} lakh' if abs(x) >= 1e5 else f'Rs {x:,.0f}'


def pill(text, kind='grey'):
    return f'<span class="pill p-{kind}">{text}</span>'


def plural(n, word, many=None):
    return f"{n} {word if n == 1 else (many or word + 's')}"


# One set of words, used everywhere on the page. The engine's own labels stay as they are (they are tested against the notebook).
URGENCY = {'1': ('Critical', 'red'), '2': ('High', 'amber'), '3': ('Moderate', 'amber'), '3P': ('Small dip', 'blue'),
           '4': ('Recovered dip', 'grey'), '5': ('Early signs', 'purple'), '6': ('Healthy', 'green')}
RISK_HELP = {'1': 'Critical: on the current trend, this merchant loses more than half its payment volume within about 90 days (by month 9).',
             '2': 'High: payment volume is falling beyond normal ups and downs, and the contract renews within 90 days.',
             '3': 'Moderate: payment volume is falling beyond normal ups and downs, but it is not yet on track to lose half within 90 days.'}
DRIVER_NAME = {'performance': 'Payment failures', 'service': 'Support issues', 'engagement': 'Low engagement', core.NC: 'Likely pricing'}
SHORT2 = {'performance': 'payments', 'service': 'support', 'engagement': 'usage'}
PICK = {'Firm': 'Solid', 'Borderline': 'Close call', 'Next in line': 'Next up'}
FIX = {'performance': "Set up an alert that tells you when a merchant's payment success rate drops below its usual level, so you can bring in Payments Ops before the merchant notices.",
       'service': 'Take ownership of every open ticket with a closing date, and follow up on any low support rating within the week.',
       'engagement': "Re-confirm your main contact whenever a merchant's ownership changes, and walk them through products they are not using yet.",
       core.NC: 'Start a pricing review with the account manager 60 days before each renewal, before the merchant compares providers.'}
CHECK_WORDS = {'promise': 'a promise (refund, waiver, discount or price change)', 'timeline': 'a promised deadline',
               'internal': 'internal information (volume, leaving, risk, notes or competitors)'}


def tier_key(t):
    return '3P' if str(t).startswith('3P') else str(t)[0]


def urgency(r):
    return URGENCY[tier_key(r.tier)][0]


def nc_short(r):
    """Merchants whose numbers show no payment, support or usage problem. Picked and ranked like everyone else."""
    b = str(r.review_likely_bucket)
    if b.startswith('Likely commercial (renews'): return 'Likely pricing'
    if 'commercial or external' in b: return 'Pricing or outside factor'
    m = re.match(r'Likely (\w+) \(weak', b)
    return f"Possibly {DRIVER_NAME.get(m.group(1), m.group(1)).lower()}" if m else 'Cause unclear'


def why_short(r):
    if r.driver_main == core.NC:
        return nc_short(r) + '?'
    if not isinstance(r.driver_main, str):
        return ''
    return DRIVER_NAME[r.driver_main]


def stored(fn, *a):
    """Drawing the page never waits on Claude: it reads saved results. New AI work only happens on a click."""
    with llm.stored_only():
        return fn(*a)


def note_category(r):
    n = r.get('account_manager_note')
    nr = stored(P.read_note, n) if isinstance(n, str) else None
    return nr.get('category') if nr else None


def driver_line(r, d):
    """One plain sentence for a driver, from the merchant's own numbers."""
    if d == 'performance':
        return f'More payments are failing: the success rate fell from {r.success_rate_m1_m3_pct:.1f}% to {r.success_rate_m4_m6_pct:.1f}%'
    if d == 'service':
        rating = '' if pd.isna(r.last_csat_1_to_5) else f', and their last support rating was {r.last_csat_1_to_5:.0f} out of 5'
        return f"{plural(int(r.unresolved_tickets), 'support ticket')} still open{rating}"
    if d == 'engagement':
        return f"They have logged in {plural(int(r.logins_last_30d), 'time')} this month and use {plural(int(r.products_used), 'product')}"
    return 'No payment, support or usage problem shows up in their numbers'


def why_points(r):
    """The 'why' for one merchant as three to five short lines. The first line is the drop, the last is the renewal."""
    pts = [(f'Their payment volume is down {abs(r.tpv_change):.0%} over the last three months', True)]
    pts.append((('If nothing changes, they lose more than half their volume by month 9', True) if r.imminent else
                (f'If nothing changes, they will be about {r.projected_loss_m9:.0%} down by month 9', False)))
    pts.append((driver_line(r, r.driver_main), r.driver_main != core.NC))
    if isinstance(r.driver_second, str) and r.driver_second:
        pts.append(('Also: ' + driver_line(r, r.driver_second)[0].lower() + driver_line(r, r.driver_second)[1:], False))
    d = int(r.days_to_contract_renewal)
    pts.append((f'Their contract renews in {d} days', d <= core.PB3_RENEWAL_DAYS))
    return pts


def why_left(r):
    """Why a merchant is not on the list, in plain words."""
    g = r.why_group
    if g == 'Seasonal dip': t = 'A seasonal dip. It picked up again in month 6.'
    elif g == 'Bounced-back dip': t = 'Dipped, then picked up again in month 6.'
    elif g == 'Small and steady': t = 'A small business. Low volume is normal for them, and it is not falling.'
    elif g == 'Busy and growing': t = f'Up {r.tpv_change:.0%} and raising a lot of tickets. That is a busy business, not one that is leaving.'
    elif g == 'Growing': t = f'Up {r.tpv_change:.0%}.'
    elif g == 'Steady': t = 'Normal month-to-month movement.'
    elif g == 'Possible decline': t = 'A small drop that has not bounced back yet. They get an automated check-in, not a call.'
    elif g == 'Watch list': t = 'Steady for now, but showing early signs seen in merchants who later declined. Review next month.'
    else: t = str(r.why_not_flagged or '')
    sig = str(r.signal_to_watch or '')
    return t + (f' Watch: {DRIVER_NAME.get(sig, sig).lower()}.' if sig else '')


GROUP_NAME = {'Seasonal dip': 'Seasonal dip', 'Bounced-back dip': 'Recovered dip', 'Small and steady': 'Small and steady',
              'Busy and growing': 'Busy and growing', 'Growing': 'Growing', 'Steady': 'Steady', 'Possible decline': 'Small dip', 'Watch list': 'Early signs'}


def friendly_check(name):
    n = name.lower()
    key = 'promise' if 'promise' in n else 'timeline' if 'timeline' in n or 'deadline' in n else 'internal' if 'internal' in n else None
    return CHECK_WORDS.get(key, name)


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
    try:
        merchants = core.read_any(file_bytes, name)
    except core.DataError as e:
        return {'ok': False, 'error': str(e)}
    pbs = None if name.lower().endswith('.csv') else core.load_playbooks(io.BytesIO(file_bytes))
    return core.run(merchants, playbooks=pbs, watch=watch)


ss = st.session_state
ss.setdefault('up_key', 0); ss.setdefault('watch_ready', set()); ss.setdefault('last_file', None)
with st.sidebar:
    st.markdown("<div class='side-brand'>Merchant Retention Workbench</div>", unsafe_allow_html=True)
    st.caption('Razorpay Campus AI Build-a-thon · Problem 2')
    up = st.file_uploader("Score a different month", type=['xlsx', 'csv'], key=f"up-{ss.up_key}",
                          help='Drop in the data pack, or any file with the same columns. It is scored straight away.')
    if up is not None:
        file_bytes, file_name = up.getvalue(), up.name
    else:
        file_bytes, file_name = open(DEFAULT_FILE, 'rb').read(), DEFAULT_FILE.split('/')[-1]
    status_box = st.empty()
    if up is not None and st.button('Back to the data pack', width='stretch'):
        ss.up_key += 1; st.rerun()
    ss.csm = 'Anoushka'
    st.divider()
    ai_on = st.toggle('AI drafting', value=ss.get('ai_on', llm.has_key()), disabled=not llm.has_key(), key='ai_on',
                      help='On: Claude can write new drafts and answer new questions when you click. Off: you only see saved results.')
    llm.set_enabled(ai_on)
    if not llm.has_key():
        st.caption("AI is off on this server. You'll see the saved drafts.")
    elif ai_on:
        st.caption('Claude only runs when you click, and usage is capped.')
    else:
        st.caption("You're seeing saved drafts only.")
    st.divider()
    st.caption('Nothing is ever sent from this app. You send the approved messages yourself.')

file_id = hashlib.sha1(file_bytes).hexdigest()[:10]
watch_ready = file_id in ss.watch_ready or file_name == DEFAULT_FILE.split('/')[-1] and up is None
res = run_engine(file_bytes, file_name, watch_ready)
if not res['ok']:
    status_box.error("Couldn't use this file. The reason is on the right.")
    st.markdown("<div class='hero'><div class='title'>Merchant Retention Workbench</div><div class='sub'>This file couldn't be scored.</div></div>", unsafe_allow_html=True)
    if 'error' in res:
        st.error(f"**{file_name}** could not be read. {res['error']}")
    else:
        st.error(f"**{file_name}** is missing a few things. Fix the rows marked FAIL below and upload it again.")
        st.dataframe(res['checks'], hide_index=True, width='stretch')
    st.caption('Expected: the data pack, or a file with the same columns as its P2_Merchants tab and at least 30 merchants.')
    st.stop()
n_merch = res['summary']['merchants']
n_c = int(res['df'].csm_call_list.sum())
status_box.success(f"**Loaded.** {n_merch} merchants · {plural(n_c, 'call')}", icon='✅')
if ss.last_file != file_id:
    if ss.last_file is not None:
        st.toast(f'Loaded {file_name}. {n_merch} merchants scored.', icon='✅')
    ss.last_file = file_id

df = res['df'].copy()
PB = res['playbooks']
by_id = df.set_index('merchant_id', drop=False)
calls = df[df.csm_call_list].sort_values('revenue_rank')
review = df[df.declining & df.needs_person].sort_values('revenue_rank')
V, S = res['value'], res['summary']
# No saving figure: the data has no record of how many merchants a call keeps. Show the cost of the calls against what each
# merchant on the list is worth; the explainer shows how well the calls were aimed in tests with known outcomes.
MONEY = {'min': float(calls.yearly_value_rs.min()) if len(calls) else 0.0, 'max': float(calls.yearly_value_rs.max()) if len(calls) else 0.0}


def rough_rs(x):
    return f"Rs {round(x / 1000) * 1000:,.0f}" if x < 1e5 else rs(x)


def cover_line(min_value, cost, n):
    times = int(min_value // cost) if cost else 0
    if times >= 2:
        return f"a year. Keeping just this one covers all {n} calls {times} times over."
    if times == 1:
        return f"a year. Keeping just this one covers all {n} calls."
    return f"a year. Keeping one merchant worth more than {rs(cost)} covers all {n} calls."
ss.setdefault('decisions', {}); ss.setdefault('follow_ups', {}); ss.setdefault('audit', []); ss.setdefault('review_decisions', {}); ss.setdefault('qa', {}); ss.setdefault('selected', None)


# What happened after the outreach, set by the CSM for each merchant on the call list
FOLLOW_UPS = ['Not started', 'Reached out', 'In progress', 'Resolved', 'No resolution', 'Merchant left']


def follow_of(mid):
    return ss.follow_ups.get((file_id, mid), 'Not started')


def log(action, merchant_id, detail=''):
    ss.audit.append({'time': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'by': ss.csm or 'name not given', 'file': file_name,
                     'merchant_id': merchant_id, 'action': action, 'detail': detail})


def name_of(m):
    return by_id.loc[m].get('business_name', m)


def how_we_pick():
    st.markdown(f"""
**How the money is worked out**

1. **Yearly value.** 1% of the latest month's payment volume, times 12 (the data pack's rule).
2. **Revenue at risk, {rs(V['at_risk_value'])}.** What we lose if every falling merchant leaves. This is the worst case.
3. **Reached by the calls, {rs(V['call_list_value'])}.** The yearly value of the merchants on this month's calls.
4. **Cost of the calls, {rs(V['outreach_cost'])}.** Rs 1,500 of CSM time per call (the data pack's rule). The smallest merchant on the list is worth
{rough_rs(MONEY['min'])} a year and the largest {rs(MONEY['max'])}, so keeping any one of them pays for the month's calls.
5. **Why there is no saving figure.** The data does not show how many merchants a call keeps, so we do not guess it. For reference, a published trial
of calling likely leavers kept about 1 in 9 (Godinho de Matos, Ferreira and Belo, 2018, telecom customers).

| Tested on 8 datasets where we know who left (1,200 merchants) | Calls to merchants who left | Leaving revenue reached |
|---|---:|---:|
| **This engine** | **115 of 120** | **50%** |
| Biggest clients | 16 of 120 | 45% |
| Biggest volume drop | 118 of 120 | 31% |
| Most support tickets | 64 of 120 | 29% |
| Random | 25 of 120 | 9% |

Most of the leaving revenue the calls miss belongs to merchants whose volume had not started falling within the six months of data.
The test data is made up, so real results may differ. Recording who stays after each call would show what the calls are worth.

**How the calls are picked**

1. **Falling.** Payment volume is down more than {abs(S['decline_line']):.1%} and hasn't picked up. Normal monthly swings are about {S['typical_wobble']:.1%}, so small dips are ignored.
2. **Risk level.** *Critical*: at this rate, half the volume is gone by month 9. *High*: falling, and the contract renews within 90 days. *Moderate*: falling, but slower.
3. **Main reason.** Success rate, open tickets, support rating, logins and products used, compared with other merchants. If nothing stands out, we assume pricing and confirm it on the call.
4. **Playbook.** One of the six in the data pack, matched to the main reason. Merchant size decides who runs it.
5. **Order.** Urgent, fixable, high-value merchants come first. Critical counts double: in tests, 172 of 173 Critical merchants left, against 9 of 20 others. The order is re-checked 11 ways, and *Close call* means the merchant drops out in some of them.
6. **AI's role.** AI writes the drafts and checks translations. It does not pick the list, and a CSM approves every message.
""")


# ------------------------------------------------------------------ header
st.markdown(f"""<div class="hero"><div class="eyebrow">Razorpay · Customer success</div>
<div class="title">Merchant retention: this month's outreach</div>
<div class="sub">Merchants whose payment volume is falling, ranked by revenue at risk, with the recommended playbook and a first message for your review.</div>
<span class="chip">{file_name}</span><span class="chip">AI {'on' if llm.live() else 'off'}</span></div>""", unsafe_allow_html=True)
tab_work, tab_why = st.tabs(["This month's calls", 'Not flagged'])

# ================================================================== YOUR CALLS
with tab_work:
    cover = V['call_list_value'] / V['at_risk_value'] if V['at_risk_value'] else 0
    cards = [('Merchants at risk', f"{S['at_risk']}", f"{S['imminent']} critical"),
             ('Annual revenue at risk', rs(V['at_risk_value']), f"if all {S['at_risk']} declining merchants left"),
             (f"Reached by the {plural(len(calls), 'call')}", rs(V['call_list_value']) if len(calls) else '–',
              f"{cover:.0%} of the revenue at risk" if len(calls) else 'no calls needed'),
             ('Smallest merchant on the list', rough_rs(MONEY['min']) if len(calls) else '–',
              cover_line(MONEY['min'], V['outreach_cost'], len(calls)) if len(calls) else 'no calls needed')]
    st.markdown('<div class="kpis">' + ''.join(f'<div class="kpi{" accent" if i == 0 else ""}"><div class="label">{lab}</div><div class="value">{val}</div>'
                                               f'<div class="sub">{sub}</div></div>' for i, (lab, val, sub) in enumerate(cards)) + '</div>', unsafe_allow_html=True)

    # ---- why they're slipping, and the biggest fix
    mix = res['driver_mix']
    if len(mix):
        c1, c2 = st.columns([3, 2])
        with c1:
            names = [DRIVER_NAME.get(d, d) for d in mix.driver]
            fig = go.Figure(go.Bar(x=mix.merchants, y=names, orientation='h', marker_color=[BLUE if d != core.NC else GREY for d in mix.driver],
                                   text=[plural(int(m), 'merchant') for m in mix.merchants],
                                   textposition='outside', cliponaxis=False, textfont=dict(color=NAVY, size=14),
                                   customdata=[rs(v) for v in mix.yearly_value_rs],
                                   hovertemplate='%{y}: %{x} merchants, %{customdata} a year<extra></extra>'))
            fig.update_layout(height=210, yaxis_tickfont=dict(size=14), margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Main reason for decline', font=dict(size=16)),
                              xaxis=dict(visible=False, range=[0, float(mix.merchants.max()) * 1.45]), yaxis=dict(autorange='reversed'),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        with c2:
            top = mix[mix.driver != core.NC].iloc[0] if (mix.driver != core.NC).any() else mix.iloc[0]
            st.markdown(f"""<div class="card fix"><div class="label muted">LARGEST FIXABLE REASON · SUGGESTED CHANGE</div>
            <div style="font-weight:650;margin:4px 0">{DRIVER_NAME.get(top.driver, top.driver)} · {plural(int(top.merchants), 'merchant')}</div>
            <div style="font-size:1rem">{FIX[top.driver]}</div></div>""", unsafe_allow_html=True)
            nc_df = df[df.declining & df.needs_person]
            if len(nc_df):
                soon = int((nc_df.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS).sum())
                heard = sum(note_category(x) == 'pricing' for _, x in nc_df.iterrows())
                st.markdown(f"""<div class="card" style="font-size:1rem"><b>{plural(len(nc_df), 'merchant')} show{'s' if len(nc_df) == 1 else ''} no operational cause.</b>
                {'All' if soon == len(nc_df) else soon} renew within 60 days{f', and their account managers have heard about quotes or competitors' if heard else ''}.
                <br><span class="muted">Suggested change: {FIX[core.NC]}</span></div>""", unsafe_allow_html=True)

    # ---- the calls (list) + merchant panel
    if len(calls) == 0:
        st.markdown("### This month's calls")
        st.info('No merchant in this file is declining, so no calls are needed this month. The **Not flagged** tab shows how each merchant was assessed.')
    else:
        h1, h2 = st.columns([5, 1.4], vertical_alignment='bottom')
        h1.markdown(f"### This month's {plural(len(calls), 'call')}")
        with h2.popover('How these are picked'):
            how_we_pick()
        st.markdown(("<div class='lead'>Select a merchant to see the reasons, the recommended playbook and a draft message.</div>" if len(calls) == core.CAPACITY else
                     f"<div class='lead'>Only {plural(len(calls), 'merchant is', 'merchants are')} declining this month, so there is spare call capacity. "
                     "Select a merchant to see the details.</div>"), unsafe_allow_html=True)
        st.markdown("<div class='muted'><b>Critical</b>: on the current trend, loses more than half its payment volume within about 90 days. "
                    "<b>High</b>: declining, and renews within 90 days. <b>Moderate</b>: declining, not yet critical.</div>", unsafe_allow_html=True)

    STATUS_LABEL = {'Approved': '✓ Approved', 'Not this month': '– Skipped', 'Writing it myself': '✎ Rewriting'}

    def status_of(mid):
        d = ss.decisions.get((file_id, mid))
        return STATUS_LABEL.get(d['status'], d['status']) if d else 'To review'
    tbl = pd.DataFrame({
        '#': calls.revenue_rank.astype(int).values,
        'Merchant': calls.get('business_name', calls.merchant_id).values,
        'ID': calls.merchant_id.values,
        'Risk': [urgency(r) for _, r in calls.iterrows()],
        'Why': [why_short(r) for _, r in calls.iterrows()],
        'Plan': [r.playbook_id for r in calls.itertuples()],
        'Renews in': [f"{int(x)} days" for x in calls.days_to_contract_renewal],
        'Pick': [PICK.get(x, x) for x in calls.call_list_status],
        'Status': [status_of(m) for m in calls.merchant_id],
        'Follow-up': [follow_of(m) for m in calls.merchant_id]})
    ROW_H = 40
    prev = ss.get(f'calls-{file_id}')
    has_sel = bool(prev and getattr(prev, 'selection', None) and prev.selection.rows)
    if has_sel:
        left, right = st.columns([12, 11], gap='medium')
    else:
        left, right = st.container(), st.container()
    with left:
        approved = tbl['Status'].eq(STATUS_LABEL['Approved']) | tbl['Follow-up'].eq('Resolved')
        shown = tbl.style.apply(lambda row: ['background-color: #E6F6EE; color: #12804A' if approved[row.name] else '' for _ in row], axis=1)
        sel = None if len(tbl) == 0 else st.dataframe(
            shown, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row', key=f'calls-{file_id}',
            height=ROW_H * (len(tbl) + 1) + 4, row_height=ROW_H,
            column_order=(['#', 'Merchant', 'Why', 'Follow-up'] if has_sel else ['#', 'Merchant', 'Risk', 'Why', 'Plan', 'Renews in', 'Pick', 'Status', 'Follow-up']),
            column_config={'#': st.column_config.NumberColumn(width=36), 'Merchant': st.column_config.TextColumn(width=150),
                           'Plan': st.column_config.TextColumn(help='The playbook from the data pack. Open a merchant to read what it involves.'),
                           'Risk': st.column_config.TextColumn(help='Critical: loses more than half its volume within about 90 days on the current trend. High: declining and renews within 90 days. Moderate: declining, not yet critical.'),
                           'Why': st.column_config.TextColumn(width=150 if has_sel else 200, help='Main reason for the decline'),
                           'Pick': st.column_config.TextColumn(help='Solid: in the top 15 under every set of estimates. Close call: drops out under some, so the CSM lead decides.'),
                           'Status': st.column_config.TextColumn('Message', width=112),
                           'Follow-up': st.column_config.TextColumn(width=112, help='What happened after the outreach. Set it in the merchant panel.')})
        if sel and sel.selection.rows:
            ss.selected = tbl.iloc[sel.selection.rows[0]]['ID']

    with right:
        mid = ss.selected if (has_sel or (sel and sel.selection.rows)) and ss.selected in set(calls.merchant_id) else None
        if mid is not None:
            r = by_id.loc[mid]
            tl, tc = URGENCY[tier_key(r.tier)]
            with st.container(height=ROW_H * (len(tbl) + 1) + 4, border=True):
                pills = f'<span title="{RISK_HELP.get(tier_key(r.tier), "")}">' + pill(tl + ' risk' if tier_key(r.tier) in RISK_HELP else tl, tc) + '</span>'
                if r.call_this_week: pills += pill('Call this week', 'red')
                if r.call_list_status == 'Borderline': pills += pill('Close call', 'amber')
                if r.needs_person: pills += pill(nc_short(r) + '? Check on the call', 'amber')
                st.markdown(f"<div style='font-size:1.15rem;font-weight:650'>{r.get('business_name', mid)}</div>"
                            f"<div class='muted'>{r.get('city', '')} · {r.industry} · {r.segment} · customer for {int(r.tenure_months)} months</div>{pills}",
                            unsafe_allow_html=True)
                cur = follow_of(mid)
                new = st.selectbox('Follow-up', FOLLOW_UPS, index=FOLLOW_UPS.index(cur), key=f'follow-{file_id}-{mid}',
                                   help='What happened after the outreach. It is saved with your decisions and written to the activity log.')
                if new != cur:
                    ss.follow_ups[(file_id, mid)] = new
                    log('Follow-up updated', mid, new)
                    st.rerun()
                t_over, t_msg, t_trend, t_ask = st.tabs(['Why', 'Message', 'Trend', 'Ask'])

                with t_over:
                    if tier_key(r.tier) in RISK_HELP:
                        st.caption(RISK_HELP[tier_key(r.tier)])
                    st.markdown('<ul class="why-list">' + ''.join(f"<li class='{'hot' if hot else ''}'>{t}</li>" for t, hot in why_points(r)) + '</ul>',
                                unsafe_allow_html=True)
                    note = r.get('account_manager_note')
                    if r.needs_person:
                        heard = f' Their account manager heard: “{note}”.' if isinstance(note, str) else ''
                        st.markdown(f"<div class='banner'><b>Why {nc_short(r).lower()}?</b> Nothing in their numbers explains the drop, "
                                    f"and the contract renews in {int(r.days_to_contract_renewal)} days.{heard} Confirm it on the call before acting.</div>",
                                    unsafe_allow_html=True)
                        st.caption("The note backs this up, but it doesn't decide anything. Volume and value put them on the list.")
                    elif isinstance(note, str):
                        st.markdown(f"<div class='muted'>Their account manager says: “{note}”</div>", unsafe_allow_html=True)
                    plan = f"<div class='pb-id'>{r.playbook_id} · {PB[r.playbook_id]['name']}</div><div class='what'>{PB[r.playbook_id]['what_happens']}</div>"
                    st.markdown(f"<div class='card plan'><div class='label muted'>THE PLAN</div>{plan}<div class='muted'>Who: {r.delivery}</div></div>",
                                unsafe_allow_html=True)
                    ai_text, _ = stored(P.write_reason, r) if (r.declining or r.possible_decline) else (None, None)
                    if ai_text:
                        with st.expander("Claude's summary"):
                            st.write(ai_text)
                    with st.expander('All their numbers'):
                        csat = '–' if pd.isna(r.last_csat_1_to_5) else f'{r.last_csat_1_to_5:.0f} out of 5'
                        tiles = [('Volume, last 3 months', f'{r.tpv_change:+.0%}'), ('By month 9, if nothing changes', f'-{r.projected_loss_m9:.0%}'),
                                 ('Payment success', f'{r.success_rate_m1_m3_pct:.1f}% → {r.success_rate_m4_m6_pct:.1f}%'),
                                 ('Open tickets', f'{int(r.unresolved_tickets)} of {int(r.support_tickets_last_90d)}'), ('Last support rating', csat),
                                 ('Logins this month', f'{int(r.logins_last_30d)}'), ('Products used', f'{int(r.products_used)}'),
                                 ('Worth a year', rs(r.yearly_value_rs))]
                        st.markdown('<div class="tiles">' + ''.join(f'<div class="tile"><div class="label">{l}</div><div class="value">{v}</div></div>' for l, v in tiles)
                                    + '</div>', unsafe_allow_html=True)

                with t_msg:
                    key = (file_id, mid)
                    d = stored(P.draft_with_retry, r)
                    if d is None:
                        if llm.live():
                            st.info("There's no AI draft for this merchant yet. Here's a simple one to start with, or let Claude write it.")
                            if st.button('Write it with AI (about a minute)', type='primary', key=f'gen-{file_id}-{mid}'):
                                with st.status('Claude is writing the message...', expanded=True) as stat:
                                    out = P.draft_with_retry(r, progress=lambda m: stat.write(m))
                                    stat.update(label='Message ready' if out and out.get('passed') else 'Done. Check the safety result below.', state='complete')
                                st.rerun()
                        else:
                            st.info("There's no AI draft for this merchant, and AI is off. Here's a simple one to start with.")
                        drafts, checks, ok = {'English': ('', P.template_draft(r))}, [], True
                    else:
                        drafts = {'English': (d['english'].get('subject', ''), d['english'].get('message', ''))}
                        if d.get('regional') and 'message' in d['regional']:
                            drafts[d['regional_language']] = (d['regional'].get('subject', ''), d['regional']['message'])
                        order = [d['main_language']] + [l for l in drafts if l != d['main_language']]
                        drafts = {l: drafts[l] for l in order if l in drafts}
                        checks, ok = d['checks'], d['passed']
                    ch = channel_for(r.segment)
                    st.caption(f"{ch[0].upper() + ch[1:]}. The language is based on their city, so confirm it with the merchant first.")
                    dec = ss.decisions.get(key)
                    if dec and dec['status'] == 'Approved':
                        st.markdown(f"<div class='done'><b>✓ Approved</b> by {dec['by']} at {dec['time']}{' · you edited it' if dec['edited'] else ''}<br>"
                                    "<span class='muted'>It is in Ready to send below. Copy it and send it from email or WhatsApp.</span></div>",
                                    unsafe_allow_html=True)
                        st.code(dec['text'], language=None, wrap_lines=True)
                        if '[' in dec['text']:
                            st.warning('Fill in the two times in [ ] before you send it.')
                        for w in dec.get('warnings', []):
                            if 'English words' in w:
                                st.warning(f"There are English words in the {dec['language']} message ({w.split(': ', 1)[-1]}). "
                                           f"That's fine for a product name; otherwise write it in {dec['language']}.")
                        if st.button('Undo', key=f'undo-{file_id}-{mid}'):
                            ss.decisions.pop(key); log('Approval undone', mid); st.toast(f"Undone. {r.get('business_name', mid)} is back in your list."); st.rerun()
                    elif dec and dec['status'] == 'Not this month':
                        freed = [m for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Not this month']
                        pos = freed.index(mid) if mid in freed else 0
                        nxt = df[df.declining & (df.revenue_rank > len(calls))].sort_values('revenue_rank')
                        nxt_txt = (f"<b>{nxt.iloc[pos].get('business_name', nxt.iloc[pos].merchant_id)}</b> is next up for this slot." if len(nxt) > pos
                                   else 'Nobody else is waiting for a call.')
                        st.markdown(f"<div class='skipped'><b>Skipped this month</b> · {dec['by']} at {dec['time']}<br>"
                                    f"Reason: {dec['reason']}<br><span class='muted'>{nxt_txt}</span></div>", unsafe_allow_html=True)
                        if st.button('Undo', key=f'undo-{file_id}-{mid}'):
                            ss.decisions.pop(key); log('Skip undone', mid); st.rerun()
                    else:
                        if dec and dec['status'] == 'Writing it myself':
                            st.markdown("<div class='banner'>You're rewriting this one. Edit the message, then approve it.</div>", unsafe_allow_html=True)
                        elif not ok:
                            st.error("The safety checks stopped this draft. Rewrite it below, then approve it.")
                        lang = st.radio('Version', list(drafts), horizontal=True, key=f'lang-{file_id}-{mid}', label_visibility='collapsed')
                        subj, body = drafts[lang]
                        original = (f'Subject: {subj}\n\n' if subj else '') + body
                        edited = st.text_area('Message (you can edit it)', value=original, height=230, key=f'txt-{file_id}-{mid}-{lang}')
                        is_edited = edited.strip() != original.strip()
                        if checks:
                            n_pass = sum(1 for _, okk, _ in checks if okk)
                            label = (f"✓ Passed all {len(checks)} safety checks" if ok else f"✗ {len(checks) - n_pass} of {len(checks)} safety checks failed")
                            with st.expander(label + (' · your edits are checked when you approve' if is_edited else '')):
                                for cname, okk, det in checks:
                                    st.markdown(f"<span class='{'check-ok' if okk else 'check-bad'}'>{'✓' if okk else '✗'}</span> {cname}{' · ' + det if det else ''}",
                                                unsafe_allow_html=True)
                        blk_key = f'blocked-{file_id}-{mid}'
                        if ss.get(blk_key) and ss[blk_key]['text'] == edited:
                            st.error("This can't be approved yet. Your edit includes something that is never promised or shared with merchants:\n\n"
                                     + '\n'.join(f"- {friendly_check(n)}: **{d}**" for n, d in ss[blk_key]['hits']) + '\n\nTake it out and approve again.')
                        rj_key = f'rj-open-{file_id}-{mid}'
                        b1, b2 = st.columns(2)
                        if b1.button('✓ Approve', type='primary', key=f'ap-{file_id}-{mid}', width='stretch'):
                            warn, hits = [], []
                            if is_edited:      # a person's edit gets the same safety checks before it can be approved
                                chk = G.check_edit(edited, english=(lang == 'English'))
                                if lang != 'English' and llm.live():
                                    with st.spinner('Checking your edit...'):
                                        back = P.back_translate(lang, edited)
                                    if back:
                                        chk += [(n + ' (your edit, translated back)', okk, det) for n, okk, det in G.check_edit(back) if 'placeholder' not in n]
                                soft = ('placeholder', 'English words')
                                hits = [(n, det) for n, okk, det in chk if not okk and not any(w in n for w in soft)]
                                warn = [f'{n}: {det}' for n, okk, det in chk if not okk and any(w in n for w in soft)]
                            if hits:
                                ss[blk_key] = {'text': edited, 'hits': hits}; log('Approval blocked by safety check', mid, '; '.join(f'{n}: {d}' for n, d in hits)); st.rerun()
                            ss.pop(blk_key, None)
                            ss.decisions[key] = {'status': 'Approved', 'language': lang, 'text': edited, 'edited': is_edited, 'warnings': warn,
                                                 'by': ss.csm or 'name not given', 'time': dt.datetime.now().strftime('%H:%M'), 'reason': ''}
                            log('Approved', mid, f"{lang}{' (edited, re-checked)' if is_edited else ''}")
                            st.toast(f"Approved. {r.get('business_name', mid)} is in your outbox.", icon='✅'); st.rerun()
                        if b2.button('Not now...', key=f'rj-{file_id}-{mid}', width='stretch'):
                            ss[rj_key] = not ss.get(rj_key, False)
                        if ss.get(rj_key):
                            with st.container(border=True):
                                why = st.radio("What's not right?", ['Wrong cause or plan', 'The wording or tone is off', "We're already in touch with them",
                                               'Not worth a call this month', 'Something else'], key=f'rjwhy-{file_id}-{mid}')
                                nxt_step = st.radio('What next?', ["I'll rewrite it (they stay on the list)",
                                                    'Skip them this month (the next merchant gets the slot)'], key=f'rjnext-{file_id}-{mid}')
                                c1, c2 = st.columns(2)
                                if c1.button('Done', type='primary', key=f'rjok-{file_id}-{mid}', width='stretch'):
                                    status = 'Writing it myself' if nxt_step.startswith("I'll") else 'Not this month'
                                    ss.decisions[key] = {'status': status, 'language': lang, 'text': '', 'edited': False, 'reason': why,
                                                         'by': ss.csm or 'name not given', 'time': dt.datetime.now().strftime('%H:%M')}
                                    ss[rj_key] = False; log('Rewriting' if status == 'Writing it myself' else 'Skipped this month', mid, why); st.rerun()
                                if c2.button('Cancel', key=f'rjno-{file_id}-{mid}', width='stretch'):
                                    ss[rj_key] = False; st.rerun()
                        st.caption(f"Signed as {ss.csm or 'the CSM'}. Approved messages wait in your outbox; nothing is sent from here.")

                with t_trend:
                    months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
                    tpv = [r[f'tpv_m{i}_inr_lakh'] for i in range(1, 7)]
                    f1 = go.Figure(go.Scatter(x=months, y=tpv, mode='lines+markers', line=dict(color=BLUE, width=3), fill='tozeroy',
                                              fillcolor='rgba(51,149,255,.10)', hovertemplate='%{x}: Rs %{y} lakh<extra></extra>'))
                    f1.add_hline(y=float(np.mean(tpv[:3])), line_dash='dot', line_color=NAVY, annotation_text='first three months, on average', annotation_font_size=10)
                    f1.update_layout(height=220, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Payment volume each month (Rs lakh)', font=dict(size=13)),
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(rangemode='tozero', gridcolor='rgba(128,128,128,.2)'))
                    st.plotly_chart(f1, width='stretch', config={'displayModeBar': False})
                    sr_drop = r.success_rate_m1_m3_pct - r.success_rate_m4_m6_pct
                    f2 = go.Figure(go.Bar(x=['First three months', 'Last three months'], y=[r.success_rate_m1_m3_pct, r.success_rate_m4_m6_pct],
                                          marker_color=[GREY, '#E5484D' if sr_drop > 2 else BLUE],
                                          text=[f'{r.success_rate_m1_m3_pct:.1f}%', f'{r.success_rate_m4_m6_pct:.1f}%'], textposition='outside'))
                    f2.update_layout(height=200, margin=dict(l=0, r=0, t=28, b=0), title=dict(text='Payments that went through', font=dict(size=13)),
                                     yaxis=dict(range=[min(70, r.success_rate_m4_m6_pct - 5), 100], visible=False), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(f2, width='stretch', config={'displayModeBar': False})
                    st.caption('Just for you. Messages to merchants never mention numbers.')

                with t_ask:
                    st.caption('Ask anything about this merchant. Answers only use their data, and say so when the data is silent.')
                    # the question wording is what saved answers are stored under, so it stays as it is behind the friendlier labels
                    presets = [('Why are they on the list?', 'Why is this merchant on the call list?'),
                               ('What should I focus on in the call?', 'What should I focus on in the call?'),
                               ("What is not known yet?", 'What do we not know about this merchant?')]
                    for i, (lab, q) in enumerate(presets):
                        if st.button(lab, key=f'q{i}-{file_id}-{mid}', width='stretch'):
                            ss.qa[(file_id, mid)] = q
                    free = st.text_input('Or ask your own', key=f'free-{file_id}-{mid}', placeholder='e.g. Is this a seasonal dip?')
                    if free and free != ss.get(f'last-free-{file_id}-{mid}'):
                        ss.qa[(file_id, mid)] = free; ss[f'last-free-{file_id}-{mid}'] = free
                    q = ss.qa.get((file_id, mid))
                    if q:
                        ans, achk = stored(P.answer_question, r, q)
                        if ans is None and llm.live():
                            with st.spinner('Asking Claude...'):
                                ans, achk = P.answer_question(r, q)
                        st.markdown(f"**{next((lab for lab, qq in presets if qq == q), q)}**")
                        if ans is None:
                            st.info("There's no saved answer for this one. Turn on AI drafting in the sidebar to ask new questions.")
                        elif achk and not achk[1]:
                            st.warning("This answer was hidden because it quoted a number that is not in the merchant's data. Try asking another way.")
                        else:
                            st.write(ans)

    with st.expander(f'See all {n_merch} merchants'):
        allm = df.assign(_o=df.revenue_rank.fillna(999), _t=df.tier.map(tier_key)).sort_values(['_o', '_t', 'tpv_change'])
        full = pd.DataFrame({'Rank': allm.revenue_rank.map(lambda x: '' if pd.isna(x) else str(int(x))),
                             'Merchant': allm.get('business_name', allm.merchant_id), 'ID': allm.merchant_id, 'Size': allm.segment,
                             'Risk': [urgency(r) for _, r in allm.iterrows()],
                             'Why': [why_short(r) if isinstance(r.driver_main, str) else why_left(r) for _, r in allm.iterrows()],
                             'Plan': [r.playbook_id + ' · ' + PB[r.playbook_id]['name'] for r in allm.itertuples()],
                             'What happens': np.where(allm.csm_call_list, 'Call', np.where(allm.am_alert, 'Account manager alert',
                                                      np.where(allm.automated_track, 'Automated message', 'Nothing this month')))})
        st.dataframe(full, hide_index=True, width='stretch', height=420)

    # ---- check before acting (cause not in the numbers)
    if len(review):
        all_pricing = review.review_likely_bucket.str.startswith('Likely commercial (renews').all()
        st.markdown('### Confirm the cause before acting')
        st.markdown("<div class='lead'>" + ("These merchants are declining, but their payments, support and usage look normal. All renew within 60 days, "
                                            "so pricing is the likely cause. Confirm it on the call." if all_pricing else
                                            "These merchants are declining, but their payments, support and usage look normal. Confirm the cause on the call.") + "</div>",
                    unsafe_allow_html=True)
        with st.popover('Why the commercial review playbook?'):
            st.markdown("PB-3, the commercial review, applies to merchants who mention a competitor or renew within about 60 days. "
                        "When nothing in the numbers explains a decline and the renewal is close, pricing is the most likely reason.\n\n"
                        "Account manager notes support this but do not decide who is on the list. Merchants outside this month's calls who renew "
                        "within 60 days are passed to their account manager, who owns any pricing discussion.")
    opts = ['Not confirmed yet', 'Pricing (PB-3)', 'Payment failures (PB-1)', 'Support issues (PB-2)', 'Low engagement (PB-4)', 'Other (PB-6)']
    for _, r in review.iterrows():
        note = r.get('account_manager_note')
        seasonal = any(x in str(r.industry) for x in core.SEASONAL_INDUSTRIES)
        renew_soon = r.days_to_contract_renewal <= core.PB3_RENEWAL_DAYS
        where = "On this month's calls" if r.csm_call_list else ('Passed to their account manager' if r.am_alert else 'Automated message')
        facts = [('Renews in', f"{int(r.days_to_contract_renewal)} days", renew_soon),
                 ('Account manager note', f"“{note}”" if isinstance(note, str) else 'Nothing noted', note_category(r) == 'pricing'),
                 ('Annual revenue', rs(r.yearly_value_rs), False)]
        scores = {k: r[f'fam_{k}'] for k in ['performance', 'service', 'engagement']}
        top_fam = max(scores, key=scores.get)
        asks = ['Has anything changed in how you take payments, or in who looks after the account?']
        if renew_soon or note_category(r) == 'pricing':
            asks.insert(0, 'Are you reviewing pricing or comparing providers before your renewal?')
        if seasonal:
            asks.append('Is this dip part of your usual season?')
        asks.append({'performance': 'Have your customers mentioned failed payments?', 'service': 'Is anything still open with our support team?',
                     'engagement': 'Is the dashboard still giving your team what it needs?'}[top_fam])
        cA, cB = st.columns([3, 1.3])
        with cA:
            rows = ''.join(f"<div class='k'>{k}</div><div class='v{' hot' if hot else ''}'>{v}</div>" for k, v, hot in facts)
            st.markdown(f"<div class='card'><b>{r.get('business_name', r.merchant_id)}</b> <span class='muted'>· {where}</span><br>"
                        f"{pill(nc_short(r), 'blue')}{pill('Plan: ' + r.playbook_id + ' · ' + PB[r.playbook_id]['name'], 'grey')}"
                        f"<div class='facts'>{rows}</div>"
                        f"<div class='k' style='margin-top:10px;opacity:.7;font-weight:600;font-size:1rem'>Ask on the call</div>"
                        f"<ul class='asks'>{''.join(f'<li>{a}</li>' for a in asks)}</ul></div>", unsafe_allow_html=True)
            with st.expander('Their other numbers'):
                csat = '–' if pd.isna(r.last_csat_1_to_5) else f'{r.last_csat_1_to_5:.0f} out of 5'
                st.markdown(f"Volume down {abs(r.tpv_change):.0%} over the last three months. "
                            f"Payment success {r.success_rate_m1_m3_pct:.1f}% → {r.success_rate_m4_m6_pct:.1f}%, "
                            f"{plural(int(r.unresolved_tickets), 'open ticket')}, last support rating {csat}, "
                            f"{plural(int(r.logins_last_30d), 'login')} this month, {plural(int(r.products_used), 'product')} used. "
                            "None of these stands out enough to explain the drop."
                            + (f" {r.industry} is a seasonal business, so compare with last year before acting." if seasonal else ''))
        with cB:
            prev_choice = ss.review_decisions.get((file_id, r.merchant_id), opts[0])
            choice = st.selectbox('Cause confirmed on the call', opts, index=opts.index(prev_choice), key=f'rv-{file_id}-{r.merchant_id}')
            if choice != prev_choice:
                ss.review_decisions[(file_id, r.merchant_id)] = choice; log('Cause confirmed', r.merchant_id, choice)
                st.toast(f"Saved. {r.get('business_name', r.merchant_id)}: {choice}", icon='✅')
            if choice != opts[0]:
                st.markdown(f"<div class='done' style='font-size:1rem'>✓ Confirmed: <b>{choice}</b><br><span class='muted'>Saved to the activity log</span></div>",
                            unsafe_allow_html=True)
            else:
                st.caption('Pick the cause once you have spoken to them. The plan follows from it.')

    # ---- outbox, downloads and activity log
    st.markdown('### Ready to send')
    approved = [(m, v) for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Approved']
    if not approved:
        st.markdown("<div class='muted'>Nothing here yet. Approve a message and it lands here.</div>", unsafe_allow_html=True)
    for m, v in approved:
        rr = by_id.loc[m]
        with st.expander(f"✓ {rr.get('business_name', m)} · {v['language']} · approved at {v['time']}"):
            st.caption(channel_for(rr.segment).capitalize())
            st.code(v['text'], language=None, wrap_lines=True)
    skipped = [(m, v) for (f, m), v in ss.decisions.items() if f == file_id and v['status'] == 'Not this month']
    if skipped:
        st.markdown('<div class="muted">Skipped this month: ' + '; '.join(f"{name_of(m)} ({v['reason'].lower()})" for m, v in skipped) + '</div>',
                    unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    appr = [{'merchant_id': m, 'business_name': name_of(m), 'channel': channel_for(by_id.loc[m].segment),
             'language': v['language'], 'approved_by': v['by'], 'edited_by_csm': v['edited'], 'message': v['text']} for m, v in approved]
    e1.download_button(f'Download messages ({len(appr)})', pd.DataFrame(appr).to_csv(index=False).encode('utf-8-sig') if appr else b'',
                       file_name='approved_messages.csv', disabled=not appr, width='stretch')
    out_cols = ['merchant_id', 'segment', 'industry', 'tier', 'driver_main', 'driver_second', 'review_likely_bucket', 'needs_person', 'playbook_id', 'playbook',
                'delivery', 'revenue_rank', 'call_list_status', 'call_this_week', 'am_alert', 'automated_track', 'expected_saving_rs', 'why_not_flagged']
    ranked = df.assign(_o=df.revenue_rank.fillna(999)).sort_values(['_o', 'tier'])[[c for c in out_cols if c in df]]
    ranked['follow_up'] = [follow_of(m) if c else '' for m, c in zip(ranked.merchant_id, df.loc[ranked.index, 'csm_call_list'])]
    e2.download_button('Download the full ranked list', ranked.to_csv(index=False).encode('utf-8-sig'), file_name='ranked_merchants.csv', width='stretch')
    e3.download_button(f'Download activity log ({len(ss.audit)})', pd.DataFrame(ss.audit).to_csv(index=False).encode('utf-8-sig') if ss.audit else b'',
                       file_name='decision_log.csv', disabled=not ss.audit, width='stretch')
    if ss.audit:
        with st.expander(f'Activity log ({len(ss.audit)})'):
            st.dataframe(pd.DataFrame(ss.audit).iloc[::-1], hide_index=True, width='stretch', height=220)

# ================================================================== LEFT OFF, AND WHY
with tab_why:
    st.markdown('### Merchants that look at risk but are not')
    st.markdown("<div class='lead'>Seasonal businesses, small merchants and busy, growing merchants can look at risk. Each was checked and not flagged for the reason shown.</div>",
                unsafe_allow_html=True)
    nf = df[~df.declining].copy()
    order = ['Seasonal dip', 'Bounced-back dip', 'Small and steady', 'Busy and growing', 'Growing', 'Steady', 'Possible decline', 'Watch list']
    groups = [g for g in order if g in set(nf.why_group)]
    counts = nf.why_group.value_counts()
    if not groups:
        st.info('Every merchant in this file is declining, so none were left out.')
    else:
        labels = {GROUP_NAME[g]: g for g in groups}
        pick = st.segmented_control('Show', list(labels), default=GROUP_NAME[groups[0]], key='why-pick',
                                    format_func=lambda x: f'{x} ({int(counts[labels[x]])})', label_visibility='collapsed')
        show = nf[nf.why_group == labels[pick]] if pick else nf
        st.dataframe(pd.DataFrame({'Merchant': show.get('business_name', show.merchant_id), 'Business': show.industry, 'Size': show.segment,
                                   'Last 3 months': [f'{x:+.0%}' for x in show.tpv_change], 'Why not flagged': [why_left(r) for _, r in show.iterrows()],
                                   'Account manager note': show.account_manager_note.fillna('')}),
                     hide_index=True, width='stretch', height=min(36 * (len(show) + 1) + 4, 560),
                     column_config={'Why not flagged': st.column_config.TextColumn(width='large')})

# ------------------------------------------------------------------ finish the experimental watch list after the page is drawn
# Everything above is already on screen. The watch list is the only slow step (it does not change the calls),
# so it runs last, with a note in the sidebar, and the page refreshes when it is done.
if not res.get('watch_done', True):
    status_box.info(f"**Loaded.** {n_merch} merchants · {plural(n_c, 'call')}. Finishing a few last checks...", icon='⏳')
    run_engine(file_bytes, file_name, True)
    ss.watch_ready.add(file_id)
    st.rerun()
