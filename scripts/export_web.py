"""Export data for the web (Vercel) version.

1. web/public/data/datapack.json: the data pack scored by the Python engine (same as the Streamlit app), plus the saved
   AI results (reason, drafts with their safety checks, answers to the three preset questions, note categories).
2. web/test/fixtures/*.json: merchant rows and Python engine results for four datasets, so the TypeScript port of the
   engine can be checked decision by decision (npm run parity).
"""
import json, math, pathlib, pickle, sys, warnings
warnings.filterwarnings('ignore')
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd
from engine import core
from ai import llm, pipeline as P

PRESETS = ['Why is this merchant on the call list?', 'What should I focus on in the call?', 'What do we not know about this merchant?']
KEEP = ['merchant_id', 'business_name', 'segment', 'industry', 'city', 'tenure_months'] + core.TPV + [
    'success_rate_m1_m3_pct', 'success_rate_m4_m6_pct', 'support_tickets_last_90d', 'unresolved_tickets', 'last_csat_1_to_5',
    'products_used', 'logins_last_30d', 'days_to_contract_renewal', 'account_manager_note',
    'tpv_change', 'projected_loss_m9', 'declining', 'possible_decline', 'bounced_dip', 'growing', 'imminent', 'tier',
    'driver_main', 'driver_second', 'review_likely_bucket', 'signal_to_watch', 'fam_performance', 'fam_service', 'fam_engagement',
    'playbook_id', 'needs_person', 'delivery', 'yearly_value_rs', 'p_leave', 'save_chance', 'expected_saving_rs', 'revenue_rank',
    'csm_call_list', 'call_list_status', 'call_this_week', 'automated_track', 'am_alert', 'why_group', 'why_not_flagged']


def clean(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    if hasattr(v, 'item'): v = v.item()
    if isinstance(v, float) and math.isnan(v): return None
    return v


def rows(df, cols):
    return [{c: clean(r[c]) for c in cols if c in df} for _, r in df.iterrows()]


def main():
    res = pickle.load(open(ROOT / 'data/datapack_results.pkl', 'rb'))
    df = res['df']
    ai = {}
    with llm.stored_only():
        for _, r in df.iterrows():
            m = r.merchant_id; a = {}
            n = r.get('account_manager_note')
            nr = P.read_note(n) if isinstance(n, str) else None
            if nr and nr.get('category'): a['note_category'] = nr['category']
            if r.declining or r.possible_decline:
                t, _ = P.write_reason(r)
                if t: a['reason'] = t
            if r.csm_call_list:
                d = P.draft_with_retry(r)
                if d:
                    a['draft'] = {'main_language': d['main_language'], 'regional_language': d.get('regional_language'),
                                  'english': d['english'], 'regional': d.get('regional'), 'passed': d['passed'],
                                  'checks': [[c[0], bool(c[1]), c[2]] for c in d['checks']]}
                qa = {}
                for q in PRESETS:
                    ans, chk = P.answer_question(r, q)
                    if ans and not (chk and not chk[1]): qa[q] = ans
                if qa: a['qa'] = qa
            if a: ai[m] = a
    out = {'file': 'Buildathon_Student_Data_Pack.xlsx', 'summary': {k: clean(v) for k, v in res['summary'].items()},
           'value': {k: clean(v) for k, v in res['value'].items()}, 'playbooks': res['playbooks'],
           'merchants': rows(df, KEEP), 'ai': ai}
    dst = ROOT / 'web/public/data'; dst.mkdir(parents=True, exist_ok=True)
    (dst / 'datapack.json').write_text(json.dumps(out, ensure_ascii=False))
    print('datapack.json:', len(out['merchants']), 'merchants,', len(ai), 'with AI results,',
          sum('draft' in v for v in ai.values()), 'drafts,', sum(len(v.get('qa', {})) for v in ai.values()), 'answers')

    fx = ROOT / 'web/test/fixtures'; fx.mkdir(parents=True, exist_ok=True)
    nb = pathlib.Path('/tmp/claude-0/-home-claude/4b95b620-3019-57ea-89dc-dbe5aa122075/scratchpad/nb/run_final')
    sets = {'datapack': ROOT / 'data/Buildathon_Student_Data_Pack.xlsx', 'seed13': nb / 'Synthetic_P2_dataset_seed20261113.xlsx',
            'seed14': nb / 'Synthetic_P2_dataset_seed20261114_nosizegap.xlsx', 'seed15': nb / 'Synthetic_P2_dataset_seed20261115.xlsx'}
    for name, path in sets.items():
        raw = core.load_merchants(path)
        r = core.run(raw, watch=False)
        (fx / f'{name}.json').write_text(json.dumps({'input': rows(core.tidy(raw), list(core.tidy(raw).columns)),
                                                     'expected': rows(r['df'], [c for c in KEEP if c in r['df']]),
                                                     'summary': {k: clean(v) for k, v in r['summary'].items()},
                                                     'value': {k: clean(v) for k, v in r['value'].items()}}, ensure_ascii=False))
        print('fixture', name, len(r['df']))


if __name__ == '__main__':
    main()
