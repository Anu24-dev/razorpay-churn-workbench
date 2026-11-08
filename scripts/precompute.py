"""Run the AI layer once on a data file and store every result in data/ai_cache.json, so the app can show
them with no API key. Usage: python scripts/precompute.py data/Buildathon_Student_Data_Pack.xlsx"""
import sys, pathlib, json
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from engine import core
from ai import pipeline as P

SUGGESTED = ['Why is this merchant on the call list?', 'What should I focus on in the call?', 'What do we not know about this merchant?']

def main(path):
    res = core.run(core.load_merchants(path)); df = res['df']
    notes = sorted(set(df.account_manager_note.dropna()))
    with ThreadPoolExecutor(6) as ex: list(ex.map(P.read_note, notes))
    flagged = [r for _, r in df[df.declining | df.possible_decline].iterrows()]
    calls = [r for _, r in df[df.csm_call_list].iterrows()]
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(P.write_reason, flagged))
        drafts = list(ex.map(P.draft_with_retry, calls))
        list(ex.map(lambda rq: P.answer_question(*rq), [(r, q) for r in calls for q in SUGGESTED]))
    for d in drafts:
        fails = [n for n, ok, _ in d['checks'] if not ok]
        print(d['merchant_id'], d['main_language'], 'PASS' if d['passed'] else f'FAIL {fails}', '(retried)' if d.get('retried') else '')

if __name__ == '__main__':
    main(sys.argv[1])
    from ai.llm import USAGE
    print('Live calls this run:', USAGE)
