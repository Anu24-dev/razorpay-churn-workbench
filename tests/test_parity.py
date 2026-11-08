"""The app's engine must give the same decisions as the notebook (P2_churn_engine_v3) on Razorpay's file."""
import sys, time, pathlib
import pandas as pd, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from engine import core

def compare(nb_results, data_file):
    t = time.time(); res = core.run(core.load_merchants(data_file)); secs = time.time() - t
    app = res['df'].set_index('merchant_id'); nb = pd.read_excel(nb_results, sheet_name='All merchants (ranked)').set_index('merchant_id')
    cols = ['tier', 'driver_main', 'driver_second', 'review_likely_bucket', 'playbook', 'playbook_also', 'needs_person', 'delivery',
            'revenue_rank', 'csm_call_list', 'call_this_week', 'automated_track', 'am_alert', 'p_leave', 'save_chance', 'expected_saving_rs']
    bad = {}
    for c in cols:
        a = app.loc[nb.index, c]; b = nb[c]
        if a.dtype.kind in 'fi' or b.dtype.kind in 'fi':
            diff = ~np.isclose(pd.to_numeric(a, errors='coerce').fillna(-1), pd.to_numeric(b, errors='coerce').fillna(-1))
        else:
            diff = a.fillna('').replace('None', '').astype(str) != b.fillna('').astype(str)
        if diff.any(): bad[c] = list(nb.index[diff])[:8]
    st_nb = nb.call_list_status.fillna('').str.split(':').str[0]; st_app = app.loc[nb.index, 'call_list_status'].fillna('')
    if (st_nb != st_app).any(): bad['call_list_status'] = list(nb.index[st_nb != st_app])
    return secs, bad, res

if __name__ == '__main__':
    nb_dir = pathlib.Path(sys.argv[1])
    for nbres, data in [('P2_churn_engine_v3_results.xlsx', 'Buildathon_Student_Data_Pack.xlsx'),
                        ('P2_engine_v3_results_Synthetic_P2_dataset_seed20261113.xlsx', 'Synthetic_P2_dataset_seed20261113.xlsx'),
                        ('P2_engine_v3_results_Synthetic_P2_dataset_seed20261114_nosizegap.xlsx', 'Synthetic_P2_dataset_seed20261114_nosizegap.xlsx')]:
        secs, bad, res = compare(nb_dir / nbres, nb_dir / data)
        print(f'{data}: {secs:.1f}s, mismatches: {bad if bad else "none"}')
