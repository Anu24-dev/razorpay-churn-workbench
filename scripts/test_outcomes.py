"""Leave rates by risk tier on the test datasets with known outcomes (the source of the 'likely to leave' step in the money figures).

Usage: python scripts/test_outcomes.py <folder with Synthetic_P2_dataset*.xlsx and matching Synthetic_P2_truth*.xlsx>
Each truth file says whether each merchant left within 90 days (true_churn_90d).
"""
import glob, pathlib, sys, warnings
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import core
warnings.filterwarnings('ignore')

folder = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(folder.glob('Synthetic_P2_dataset*.xlsx')):
    tag = f.stem.replace('Synthetic_P2_dataset', '')
    truth = pd.read_excel(folder / f'Synthetic_P2_truth{tag}.xlsx')[['merchant_id', 'true_churn_90d', 'case']]
    df = core.run(core.load_merchants(f), watch=False)['df'].merge(truth, on='merchant_id')
    df['dataset'] = tag or 'base'
    rows.append(df)
A = pd.concat(rows, ignore_index=True)
A['risk'] = A.tier.map(lambda t: {'1': 'Critical', '2': 'High'}.get(t[0], 'Moderate' if t == '3 Declining' else 'Not flagged'))
A['left'] = A.true_churn_90d.astype(bool)
A['value_left'] = A.yearly_value_rs * A.left
print(f"{A.dataset.nunique()} datasets, {len(A)} merchants, {A.left.sum()} left within 90 days\n")
t = A.groupby('risk').agg(merchants=('left', 'size'), left=('left', 'sum'))
t['stayed'] = t.merchants - t.left
print(t.to_string(), '\n')
calls = A[A.csm_call_list.astype(bool)]
print(f"Calls: {len(calls)}, of which {calls.left.sum()} left. Yearly value reached Rs {calls.yearly_value_rs.sum() / 1e5:.1f} lakh, "
      f"of which Rs {calls.value_left.sum() / 1e5:.1f} lakh left.")
missed = A[~A.declining & A.left]
print(f"Leavers not flagged: {len(missed)}\n{missed.case.value_counts().to_string()}")
