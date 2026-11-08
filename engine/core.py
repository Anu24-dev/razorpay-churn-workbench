"""
Churn early-warning engine (rules and statistics only, no AI).

Same logic as P2_churn_engine_v3.ipynb. The notebook also runs the full test suite;
this module runs the decisions only, so the app can score an uploaded file in seconds.
Every decision (who is flagged, tier, driver, playbook, ranking) comes from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- settings (written down before any run)
SEED = 42
WOBBLES_FOR_DECLINE = 3          # a drop counts if it is bigger than 3 typical monthly wobbles
POSSIBLE_DECLINE_WOBBLES = 2     # 2-3 wobbles and no bounce-back = possible decline (automated track only)
RECOVERY_VOTES_NEEDED = 3        # at least 3 of 4 bounce-back checks must say "no recovery"
IMMINENT_LOSS = 0.50             # brief: "most of its volume"
SWITCH_WINDOW_DAYS = 90          # brief: churn within the next 90 days
CALL_THIS_WEEK_DAYS = 30
PB3_RENEWAL_DAYS = 60            # PB-3's own trigger in the playbook sheet
DRIVER_LINE_SD = 1.0             # a problem family is present if > 1 SD above average
SIZE_GAP_P = 0.05                # adjust logins/products for size only if segments differ
WATCH_N = 5
MI_DRAWS = 10
CAPACITY = 15                    # brief
OUTREACH_COST = 1500             # brief
NET_REVENUE_RATE = 0.01          # brief
LAKH = 100000
P_LEAVE = {'imminent': 1.0, 'other declining': 0.5}            # assumption (no outcome data)
NC = 'no operational cause found'
SAVE_CHANCE = {'performance': .6, 'service': .6, NC: .4, 'engagement': .3}   # assumption, by lead driver
SEASONAL_INDUSTRIES = ('Gifting', 'Education', 'Travel')       # data pack assumptions tab

TPV = [f'tpv_m{i}_inr_lakh' for i in range(1, 7)]
REQUIRED = (['merchant_id', 'segment', 'industry', 'tenure_months'] + TPV +
            ['success_rate_m1_m3_pct', 'success_rate_m4_m6_pct', 'support_tickets_last_90d', 'unresolved_tickets',
             'last_csat_1_to_5', 'products_used', 'logins_last_30d', 'days_to_contract_renewal', 'account_manager_note'])

DRIVER_PB = {'performance': 'PB-1 Performance deep-dive', 'service': 'PB-2 Support recovery',
             'engagement': 'PB-4 Re-engagement and adoption', NC: 'Human review first (diagnostic call); PB-3 if pricing is confirmed'}
SECOND_STEPS = {'performance': 'PB-1 steps (success-rate review with Payments Ops)',
                'service': 'PB-2 steps (own open tickets, single contact, closure date)',
                'engagement': 'PB-4 steps (re-confirm contact, short product walkthrough)'}
DELIVERY = {'Enterprise': 'Account manager + senior sponsor', 'Mid': 'CSM-led call',
            'Small': 'Digital-first (WhatsApp/email), CSM if escalated'}
DRIVER_LABEL = {'performance': 'Performance (payments failing more)', 'service': 'Service (unresolved tickets, low CSAT)',
                'engagement': 'Engagement (low logins, few products)', NC: 'Unclear: needs human review'}
FIX_IDEAS = {
    'performance': 'Alert the merchant and their CSM automatically when a merchant\'s success rate falls well below its own normal, and route the case to Payments Ops before the merchant notices.',
    'service': 'Give every merchant with an unresolved ticket a named owner and a closure date, and follow up low CSAT scores within the same week.',
    'engagement': 'Re-confirm the main contact whenever a merchant\'s ownership or manager changes, and show in-dashboard prompts for products the merchant is not using.',
    NC: 'Start a commercial review automatically 60 days before every renewal, so pricing questions are handled before the merchant shops around.'}


# ---------------------------------------------------------------- loading and checks
def load_merchants(source) -> pd.DataFrame:
    """Read the P2_Merchants tab (Excel) or a CSV with the same columns."""
    name = getattr(source, 'name', str(source))
    if str(name).lower().endswith('.csv'):
        return pd.read_csv(source)
    return pd.read_excel(source, sheet_name='P2_Merchants')


def check_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def check(name, ok, detail='', warn=False):
        rows.append((name, 'PASS' if ok else ('WARN' if warn else 'FAIL'), detail))
    miss = [c for c in REQUIRED if c not in df.columns]
    check('Required columns present', not miss, f'missing: {miss}' if miss else '')
    if miss:
        return pd.DataFrame(rows, columns=['check', 'status', 'detail'])
    check('At least 30 merchants with unique IDs', len(df) >= 30 and df.merchant_id.is_unique, f'{len(df)} rows')
    check('TPV has no blanks or zeros', bool(df[TPV].notna().all().all() and (df[TPV] > 0).all().all()))
    sr = df[['success_rate_m1_m3_pct', 'success_rate_m4_m6_pct']]
    check('Success rates between 0 and 100', bool(((sr >= 0) & (sr <= 100)).all().all()))
    check('CSAT is 1 to 5 where present', bool(df.last_csat_1_to_5.dropna().between(1, 5).all()))
    nb = int(df.last_csat_1_to_5.isna().sum())
    check('CSAT blanks', nb == 0, f'{nb} blank (handled by multiple imputation; never used to name a driver)', warn=True)
    cnt = ['support_tickets_last_90d', 'unresolved_tickets', 'products_used', 'logins_last_30d', 'days_to_contract_renewal', 'tenure_months']
    check('Counts are whole and not negative', bool((df[cnt] >= 0).all().all() and (df[cnt] % 1 == 0).all().all()))
    odd = df.merchant_id[df.unresolved_tickets > df.support_tickets_last_90d].tolist()
    check('Unresolved tickets <= tickets in last 90 days', not odd, f'{len(odd)} exceptions (older tickets still open)', warn=True)
    return pd.DataFrame(rows, columns=['check', 'status', 'detail'])


# ---------------------------------------------------------------- helpers
def _zs(s):
    return (s - s.mean()) / s.std()


def _seg_relative(d, col):
    g = d.groupby('segment')[col]
    return (d[col] - g.transform('mean')) / g.transform('std').replace(0, 1)


def _proj_method(win, h, method):
    if method == 'no change':
        return win[-1]
    if method == 'straight line':
        return max(win[-1] + np.polyfit(np.arange(3), win, 1)[0] * h, 0)
    r = (win[-1] / win[0]) ** 0.5
    if method == 'compound':
        return win[-1] * r ** h
    g = np.log(r)
    return win[-1] * np.exp(g * sum(0.8 ** k for k in range(1, h + 1)))   # damped


# ---------------------------------------------------------------- stage 1: volume evidence
def _volume(df):
    T = df[TPV].values
    df['tpv_base'] = T[:, :3].mean(1); df['tpv_peak'] = T[:, :3].max(1); df['tpv_recent'] = T[:, 3:].mean(1)
    df['tpv_change'] = df.tpv_recent / df.tpv_base - 1
    own_var = (T[:, :3].std(1, ddof=1) / df.tpv_base) ** 2
    wobble = float(np.sqrt(np.median(own_var)))
    decline_line = -WOBBLES_FOR_DECLINE * wobble
    sharp_line = decline_line * np.sqrt(1 + 1 / 3)
    m4, m5, m6 = T[:, 3], T[:, 4], T[:, 5]
    bounce = np.column_stack([m6 > m5, m6 > np.minimum(m4, m5), m6 > m4, m6 > (m4 + m5) / 2])
    df['no_recovery_votes'] = (~bounce).sum(axis=1)
    df['m6_change'] = m6 / df.tpv_base - 1
    def dropped(avg_line, m6_line):
        return (df.tpv_change <= avg_line) | (df.m6_change <= m6_line)
    big_drop = dropped(decline_line, sharp_line)
    df['declining'] = big_drop & (df.no_recovery_votes >= RECOVERY_VOTES_NEEDED)
    df['bounced_dip'] = big_drop & ~df.declining
    pl = -POSSIBLE_DECLINE_WOBBLES * wobble
    df['possible_decline'] = ~big_drop & dropped(pl, pl * np.sqrt(4 / 3)) & (df.no_recovery_votes >= RECOVERY_VOTES_NEEDED)
    df['growing'] = df.tpv_change >= -decline_line
    df['tpv_m9_projected'] = [_proj_method(r[3:6], 3, 'compound') for r in T]
    df['projected_loss_m9'] = 1 - df.tpv_m9_projected / df.tpv_peak
    df['imminent'] = df.declining & (df.projected_loss_m9 > IMMINENT_LOSS)
    moved = set()
    for method in ['straight line', 'damped']:
        loss = 1 - np.array([_proj_method(r[3:6], 3, method) for r in T]) / df.tpv_peak
        moved |= set(df.merchant_id[df.declining & ((loss > IMMINENT_LOSS) != df.imminent)])
    df['borderline_severity'] = df.merchant_id.isin(moved)
    return {'typical_wobble': wobble, 'decline_line': decline_line, 'sharp_line': sharp_line, 'T': T}


# ---------------------------------------------------------------- stage 2: behaviour (size check + watch list)
def _size_check(df):
    adjust, rows = {}, []
    for col in ['logins_last_30d', 'products_used']:
        p = kruskal(*[v.dropna() for _, v in df.groupby('segment')[col]]).pvalue
        adjust[col] = bool(p < SIZE_GAP_P)
        rows.append({'input': col, 'p_value': round(float(p), 4), 'used_as': 'size-relative' if adjust[col] else 'raw (no size gap)'})
    return adjust, pd.DataFrame(rows)


def _eng_input(d, col, adjust):
    return _seg_relative(d, col) if adjust[col] else d[col]


def _make_X(d, adjust):
    X = pd.DataFrame({'sr_drop': d.success_rate_m1_m3_pct - d.success_rate_m4_m6_pct, 'sr_recent': d.success_rate_m4_m6_pct,
                      'unresolved': d.unresolved_tickets, 'tickets_90d': d.support_tickets_last_90d,
                      'logins_30d': _eng_input(d, 'logins_last_30d', adjust), 'products_used': _eng_input(d, 'products_used', adjust),
                      'tenure_months': d.tenure_months})
    X['csat'] = d.last_csat_1_to_5.fillna(d.last_csat_1_to_5.median())
    X['csat_missing'] = d.last_csat_1_to_5.isna().astype(int)
    return X


def _make_X_mi(d, k, adjust):
    Xk = _make_X(d, adjust)
    tmp = Xk.drop(columns=['csat', 'csat_missing']).copy(); tmp['csat'] = d.last_csat_1_to_5.values
    Xk['csat'] = IterativeImputer(sample_posterior=True, random_state=SEED + k).fit_transform(tmp)[:, -1].clip(1, 5)
    return Xk


def _pu_bagging(X, y, rounds=200, C=0.3, seed=SEED):
    rng = np.random.default_rng(seed); P = np.where(y == 1)[0]; U = np.where(y == 0)[0]; K = len(P)
    total = np.zeros(len(y)); n = np.zeros(len(y))
    for _ in range(rounds):
        Ut = rng.choice(U, K, replace=True)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X.iloc[np.r_[P, Ut]], np.r_[np.ones(K), np.zeros(K)])
        oob = np.setdiff1d(U, Ut); total[oob] += m.predict_proba(X.iloc[oob])[:, 1]; n[oob] += 1
    out = np.full(len(y), np.nan); out[U] = total[U] / n[U]
    return out


def _watch_scores(df, adjust, pu_rounds=200):
    y = df.declining.astype(int).values
    if y.sum() < 5:
        return np.full(len(df), np.nan)
    draws = [_make_X_mi(df, k, adjust) for k in range(MI_DRAWS)]
    return np.mean([_pu_bagging(Xk, y, rounds=pu_rounds, seed=SEED + k) for k, Xk in enumerate(draws)], axis=0)


# ---------------------------------------------------------------- stage 3: drivers
def _drivers(df, adjust):
    csat = df.last_csat_1_to_5
    csat_obs_z = ((csat - csat.mean()) / csat.std()).fillna(0)     # blank = average: a filled-in value never creates a driver
    fam = pd.DataFrame({'performance': _zs(_zs(df.success_rate_m1_m3_pct - df.success_rate_m4_m6_pct) - _zs(df.success_rate_m4_m6_pct)),
                        'service': _zs(_zs(df.unresolved_tickets) - csat_obs_z),
                        'engagement': _zs(-_zs(_eng_input(df, 'logins_last_30d', adjust)) - _zs(_eng_input(df, 'products_used', adjust)))})
    present = fam > DRIVER_LINE_SD
    order = fam.where(present).apply(lambda r: list(r.dropna().sort_values(ascending=False).index), axis=1)
    main = order.apply(lambda l: l[0] if l else None); second = order.apply(lambda l: l[1] if len(l) > 1 else None)
    flag = df.declining | df.possible_decline
    main = pd.Series(np.where(flag & main.isna(), NC, main), index=df.index)
    df['driver_main'], df['driver_second'] = main, second
    df['signal_to_watch'] = np.where(~flag & df.driver_main.notna(), df.driver_main, '')
    df.loc[~flag, ['driver_main', 'driver_second']] = None
    strongest, sval = fam.idxmax(axis=1), fam.max(axis=1)

    def likely_bucket(i):
        if df.at[i, 'driver_main'] != NC:
            return ''
        if df.at[i, 'days_to_contract_renewal'] <= PB3_RENEWAL_DAYS:
            return f"Likely commercial (renews in {int(df.at[i, 'days_to_contract_renewal'])} days)"
        if sval[i] >= DRIVER_LINE_SD / 2:
            return f'Likely {strongest[i]} (weak signal, score {sval[i]:.2f})'
        return 'Likely commercial or external (no operational signal)'
    df['review_likely_bucket'] = [likely_bucket(i) for i in df.index]
    df['driver_hint'] = np.where(df.driver_main != NC, '',
                                 np.where(sval > 0, 'possible ' + strongest + ' (weak, score ' + sval.round(2).astype(str) + ')', 'no signal above average'))
    df[['fam_performance', 'fam_service', 'fam_engagement']] = fam.round(2).values


# ---------------------------------------------------------------- stage 4-5: tiers and playbooks
def _tiers_playbooks(df):
    def tier(r):
        if r.imminent: return '1 Imminent fade'
        if r.declining and r.days_to_contract_renewal <= SWITCH_WINDOW_DAYS: return '2 Declining, renewal within 90 days'
        if r.declining: return '3 Declining'
        if r.possible_decline: return '3P Possible decline (automated track)'
        if r.bounced_dip: return '4 Bounced-back dip'
        return '6 Growing' if r.growing else '6 Stable'
    df['tier'] = df.apply(tier, axis=1)
    if df.score_pu.notna().any():
        pool = df[df.tier.str.startswith('6')].sort_values('score_pu', ascending=False)
        df.loc[pool.index[:WATCH_N], 'tier'] = '5 Early-warning watch (experimental)'

    def playbook(r):
        if r.declining: return DRIVER_PB[r.driver_main]
        if r.possible_decline:
            if r.driver_main == NC: return 'Human review (light): check likely bucket, send digital check-in; PB-3 if pricing confirmed'
            return DRIVER_PB[r.driver_main] + ' (digital version)'
        if r.tier.startswith('4') or r.tier.startswith('5'): return 'PB-6 Monitor only'
        return 'PB-5 Growth check-in' if r.growing else 'PB-6 Monitor only'
    df['playbook'] = df.apply(playbook, axis=1)
    has2 = (df.declining | df.possible_decline) & df.driver_second.isin(list(SECOND_STEPS)) & (df.driver_main != NC)
    df['playbook_also'] = np.where(has2, 'Also: ' + df.driver_second.map(SECOND_STEPS).fillna(''), '')
    df['delivery'] = np.where(df.declining, df.segment.map(DELIVERY), np.where(df.possible_decline, 'Automated digital track', ''))


# ---------------------------------------------------------------- stage 6: ranking
def _ranking(df, T):
    df['yearly_value_rs'] = df.tpv_m6_inr_lakh * LAKH * NET_REVENUE_RATE * 12
    dec = df[df.declining].copy()
    EQUAL = {k: .5 for k in SAVE_CHANCE}
    loss_by = {m: pd.Series(1 - np.array([_proj_method(r[3:6], 3, m) for r in T]) / df.tpv_peak.values, index=df.index).loc[dec.index]
               for m in ['straight line', 'damped']}

    def p_leave_vec(loss=None, line=IMMINENT_LOSS, p_other=P_LEAVE['other declining'], renewal_p=None):
        loss = dec.projected_loss_m9 if loss is None else loss
        p = np.where(loss > line, P_LEAVE['imminent'], p_other)
        if renewal_p is not None:
            p = np.where((loss <= line) & (dec.days_to_contract_renewal <= SWITCH_WINDOW_DAYS), renewal_p, p)
        return p
    versions = {'Main': (p_leave_vec(), SAVE_CHANCE),
                'Other declining: 30% leave': (p_leave_vec(p_other=0.3), SAVE_CHANCE),
                'Other declining: 70% leave': (p_leave_vec(p_other=0.7), SAVE_CHANCE),
                'Equal 50% save chance': (p_leave_vec(), EQUAL),
                'Unclear cause easier (often pricing)': (p_leave_vec(), {'performance': .5, 'service': .5, NC: .6, 'engagement': .3}),
                'Engagement hardest': (p_leave_vec(), {'performance': .6, 'service': .5, NC: .5, 'engagement': .2}),
                'Imminent line at 40%': (p_leave_vec(line=0.4), SAVE_CHANCE),
                'Imminent line at 60%': (p_leave_vec(line=0.6), SAVE_CHANCE),
                'Straight-line projection': (p_leave_vec(loss=loss_by['straight line']), SAVE_CHANCE),
                'Damped projection': (p_leave_vec(loss=loss_by['damped']), SAVE_CHANCE),
                'Renewal within 90 days: 70% leave': (p_leave_vec(renewal_p=0.7), SAVE_CHANCE)}
    tops = {}
    for name, (p, save) in versions.items():
        dec[f'ev_{name}'] = p * dec.driver_main.map(save) * dec.yearly_value_rs - OUTREACH_COST
        tops[name] = set(dec.nlargest(CAPACITY, f'ev_{name}').merchant_id)
    dec['p_leave'] = np.where(dec.imminent, P_LEAVE['imminent'], P_LEAVE['other declining'])
    dec['save_chance'] = dec.driver_main.map(SAVE_CHANCE)
    dec['expected_saving_rs'] = dec['ev_Main']
    dec = dec.sort_values(['expected_saving_rs', 'days_to_contract_renewal'], ascending=[False, True])
    dec['revenue_rank'] = np.arange(1, len(dec) + 1)
    core = set.intersection(*tops.values()) if tops else set()
    anyv = set.union(*tops.values()) if tops else set()
    for c in ['p_leave', 'save_chance', 'expected_saving_rs', 'revenue_rank']:
        df[c] = dec[c]
    df['csm_call_list'] = df.revenue_rank <= CAPACITY
    df['in_top15_all_versions'] = df.merchant_id.isin(core)
    df['call_list_status'] = np.where(df.csm_call_list & df.in_top15_all_versions, 'Firm',
                              np.where(df.csm_call_list, 'Borderline',
                              np.where(df.declining & df.merchant_id.isin(anyv), 'Next in line', '')))
    df['call_this_week'] = df.csm_call_list & (df.days_to_contract_renewal <= CALL_THIS_WEEK_DAYS)
    df['automated_track'] = (df.declining & ~df.csm_call_list) | df.possible_decline
    df['automated_urgent'] = df.automated_track & (df.days_to_contract_renewal <= CALL_THIS_WEEK_DAYS)
    df['am_alert'] = df.automated_track & (df.driver_main == NC) & (df.days_to_contract_renewal <= PB3_RENEWAL_DAYS)
    versions_table = pd.DataFrame([(k, len(v & tops['Main'])) for k, v in tops.items()], columns=['assumption version', 'calls shared with main list'])
    return dec, versions_table


def _value_cost(df):
    c15 = df[df.csm_call_list]
    at_risk = df[df.declining]
    def saving(save):
        s = c15.driver_main.map(save) if isinstance(save, dict) else save
        return float((c15.p_leave * s * c15.yearly_value_rs - OUTREACH_COST).sum())
    be = (OUTREACH_COST / (c15.p_leave * c15.yearly_value_rs)).max() if len(c15) else np.nan
    return {'at_risk_value': float(at_risk.yearly_value_rs.sum()), 'call_list_value': float(c15.yearly_value_rs.sum()),
            'saving_ours': saving(SAVE_CHANCE), 'saving_30': saving(0.3), 'saving_10': saving(0.1),
            'outreach_cost': CAPACITY * OUTREACH_COST if len(c15) else 0,
            'false_alarm_cost': float(((1 - c15.p_leave) * OUTREACH_COST).sum()), 'break_even_max': float(be)}


# ---------------------------------------------------------------- why not flagged (decoy view)
def _why_not_flagged(df):
    tick_hi = df.support_tickets_last_90d.quantile(0.75)
    def why(r):
        seasonal_ind = any(x in str(r.industry) for x in SEASONAL_INDUSTRIES)
        if r.tier.startswith('3P'):
            return 'Possible decline', 'Smaller drop, no bounce-back yet: automated digital check-in, not a call.'
        if r.tier.startswith('4'):
            if seasonal_ind:
                return 'Seasonal dip', f'Dipped, then came back by month 6. Seasonal industry ({r.industry}).'
            return 'Bounced-back dip', 'Dipped, then came back by month 6.'
        if r.tier.startswith('5'):
            return 'Watch list', 'Volume steady, but behaviour looks like merchants who later declined. Experimental watch, no outreach.'
        if r.growing:
            if r.support_tickets_last_90d >= tick_hi and r.support_tickets_last_90d > 0:
                return 'Busy and growing', f'Volume up {r.tpv_change:.0%} with {int(r.support_tickets_last_90d)} tickets in 90 days: busy, not at risk.'
            return 'Growing', f'Volume up {r.tpv_change:.0%}.'
        if r.segment == 'Small':
            return 'Small and steady', 'Small merchant: low volume is normal, and it has not fallen.'
        return 'Steady', 'Volume within normal month-to-month movement.'
    out = df[~df.declining].apply(lambda r: pd.Series(why(r), index=['why_group', 'why_not_flagged']), axis=1)
    df['why_group'] = out.why_group
    df['why_not_flagged'] = out.why_not_flagged
    sig = df.signal_to_watch.fillna('') != ''
    df.loc[sig & ~df.declining, 'why_not_flagged'] += ' Watch: ' + df.loc[sig & ~df.declining, 'signal_to_watch'] + '.'


# ---------------------------------------------------------------- run everything
def run(merchants: pd.DataFrame, pu_rounds: int = 200) -> dict:
    checks = check_data(merchants)
    if (checks.status == 'FAIL').any():
        return {'ok': False, 'checks': checks}
    df = merchants.copy().reset_index(drop=True)
    vol = _volume(df)
    adjust, size_table = _size_check(df)
    df['score_pu'] = _watch_scores(df, adjust, pu_rounds)
    _drivers(df, adjust)
    _tiers_playbooks(df)
    dec, versions_table = _ranking(df, vol['T'])
    _why_not_flagged(df)
    at_risk = df[df.declining]
    mix = at_risk.driver_main.value_counts().rename_axis('driver').reset_index(name='merchants')
    mix['yearly_value_rs'] = mix.driver.map(at_risk.groupby('driver_main').yearly_value_rs.sum())
    summary = {'merchants': len(df), 'at_risk': int(df.declining.sum()), 'imminent': int(df.imminent.sum()),
               'possible_decline': int(df.possible_decline.sum()), 'bounced': int(df.bounced_dip.sum()),
               'review': int((df.declining & (df.driver_main == NC)).sum()), 'am_alerts': int(df.am_alert.sum()),
               'call_this_week': int(df.call_this_week.sum()), 'typical_wobble': vol['typical_wobble'],
               'decline_line': vol['decline_line']}
    return {'ok': True, 'checks': checks, 'df': df, 'summary': summary, 'value': _value_cost(df), 'driver_mix': mix,
            'versions': versions_table, 'size_check': size_table, 'fix_ideas': FIX_IDEAS}
