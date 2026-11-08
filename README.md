# Merchant Retention Workbench

Razorpay Campus AI Build-a-thon, Problem 2: merchant churn early-warning and proactive outreach.

A CSM team can make 15 proactive calls a month. This app reads the month's merchant file, finds merchants whose payment volume is falling and not coming back, says why, picks the playbook, ranks the 15 calls by expected saving, and drafts a first message for each one in the merchant's language for a person to approve.

## Input and output

- **Input:** the data pack (`P2_Merchants` tab) or any `.xlsx` / `.csv` with the same columns. The app loads `data/Buildathon_Student_Data_Pack.xlsx` by default.
- **Odd files:** a file the app cannot use gets a plain reason instead of a crash (not a spreadsheet, empty, missing columns, fewer than 30 merchants, repeated IDs, text in a number column, blanks other than CSAT, zero volume). Harmless differences are accepted: another tab name if it has a `merchant_id` column, numbers stored as text, extra columns, segment names in any case. A segment other than Enterprise, Mid or Small is handled like Mid, with a warning. `tests/test_edge_cases.py` runs 26 of these cases.
- **Output:** the 15 calls with risk, reason, playbook and drafts; the full ranked list of all merchants; merchants needing a person (unclear cause); account-manager alerts; a "why not flagged" view for seasonal, small and busy merchants; exports of approved messages, the ranked list and the approval log.

## Who does what

| Layer | Does | Why |
|---|---|---|
| Engine (`engine/core.py`, rules + statistics) | Who is declining, how fast, the driver, the tier, the playbook (always one of the six in the data pack), the ranking | Must be reproducible and testable |
| AI (`ai/`, Claude) | Reads account-manager notes, writes the CSM's reason line, drafts the message (regional + English), checks the translation, answers questions about one merchant | Language work |
| Guardrails (`ai/guardrails.py`, code) | Blocks drafts with numbers, promises (refunds, waivers, discounts, price changes, timelines), internal information (volume, leaving, risk, notes, competitors), claims not backed by the merchant's data, or a translation that changes the meaning | The brief's safety rules, enforced in code |
| People | Approve, edit or reject every draft (an edited message is checked again for promises, deadlines and internal information before it can be approved); confirm the cause when it is unclear; account managers own pricing | Nothing reaches a merchant without a human check |

The app never sends anything. Approved messages are exported for the CSM to send.

**Playbooks:** only the six in the data pack (PB-1 to PB-6), with their text shown word for word. Performance gets PB-1, service PB-2, engagement PB-4; a second problem adds its playbook (for example PB-1 + PB-2). When the cause is unclear, the playbook follows the likely bucket (PB-3 when the renewal is within 60 days) and a person confirms it before it runs. Merchant size sets who runs it (account manager with a senior sponsor, CSM call, or CSM via WhatsApp/email); that is the delivery channel, not an extra playbook.

**AI switch:** the sidebar toggle turns AI off completely (no calls to Claude). Stored AI results are still shown; anything else falls back to rule-based text.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

AI results for the data pack are stored in `data/ai_cache.json`, so the app works fully with no API key. With a key, it can draft for a new file and answer new questions.

To use a key locally, put it in `anthropic_key.txt` (ignored by git) or set `ANTHROPIC_API_KEY`. On Streamlit Community Cloud, add it under the app's **Settings > Secrets**:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

`MAX_LIVE_CALLS` (default 400) caps new API calls per running app.

## Files

- `app.py`: the Streamlit app
- `engine/core.py`: the scoring engine (same decisions as the tested notebook `P2_churn_engine_v3.ipynb`; `tests/test_parity.py` checks this)
- `ai/facts.py`: the only facts the AI may use for each merchant (built by code)
- `ai/pipeline.py`: prompts and the AI steps
- `ai/guardrails.py`: checks every draft
- `ai/llm.py`: Claude API wrapper with stored results
- `scripts/precompute.py`: generates and stores the AI results for a file

## Known limits

- The data has no outcomes, so "declining now" stands in for "will leave", and the chance of leaving and the chance of saving are assumptions. The 15 calls are re-ranked under 11 versions of these assumptions; merchants that drop out under some are marked Borderline.
- Pricing cannot be seen in the numbers, only in notes or a conversation, so those merchants go to a person.
- Draft language is guessed from the merchant's city and must be confirmed with the merchant.
- All data is synthetic, from the build-a-thon data pack.
