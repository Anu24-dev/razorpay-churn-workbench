"""
The AI layer. AI does language work only:
  1. read the account-manager note (category + exact quote)
  2. write the CSM's reason line (internal, may use numbers from the data)
  3. draft the merchant message (English, then the regional version)
  4. check the translation (translate back, compare meaning)
  5. answer CSM questions about one merchant, from that merchant's record only
It never decides who is flagged, the tier, the driver, the playbook or the ranking (engine/core.py does).
Every output passes the code guardrails in ai/guardrails.py before a person sees it.
"""
from __future__ import annotations

import json

from ai import guardrails as G
from ai.facts import CHANNEL, internal_facts, languages, merchant_facts
from ai.llm import MODEL_READ, MODEL_WRITE, ask, ask_json
from engine.core import NC

NOTE_CATEGORIES = ['pricing', 'performance', 'service', 'engagement', 'seasonal', 'positive', 'neutral']

SYS_NOTE = ("You classify one short note written by a Razorpay account manager about a merchant. "
            f"Pick exactly one category from {NOTE_CATEGORIES}: pricing = price, competitor, rate card, exploring other options; "
            "performance = failed payments, success rates; service = tickets, slow or unresolved support, escalations; "
            "engagement = not using the dashboard, unresponsive, contact or owner changed; seasonal = seasonal or festive dips; "
            "positive = growth or interest in more products; neutral = anything else. "
            'Return JSON only: {"category": "...", "evidence": "<exact words copied from the note>"}')

SYS_REASON = ("You write a one or two sentence reason for a Razorpay customer success manager (internal, never sent to the merchant), "
              "explaining why this merchant is on the list and what the main problem is. Use only the facts in the JSON record. "
              "Numbers are allowed but must be copied exactly from the record (round percentages to whole numbers if you like). "
              "If driver_main is 'no operational cause found', say the numbers show no operational problem, mention the renewal timing, "
              "and say what the account manager note suggests if there is one. Plain words, no jargon, no em dashes. Return only the reason.")

SYS_DRAFT = ("You write a short first message from a Razorpay team member to a merchant, for a human to review before sending. Rules you must follow:\n"
             "1. Use ONLY the numbered facts provided. Do not add any other claim about the merchant.\n"
             "2. No numbers or digits of any kind, and no percentages.\n"
             "3. Never mention payment volume, sales, business falling, the merchant leaving or switching, risk, scores, internal notes or competitors.\n"
             "4. Never promise refunds, waivers, discounts, credits, price or fee changes, and never give a timeline or deadline "
             "(no 'within', no days or hours, no 'by' a date).\n"
             "5. Open with the merchant's business name. Say plainly that their account matters to Razorpay. Lead with the problem Razorpay can fix "
             "(the first fact listed), then at most one more fact. If the cause is unclear, write a warm, neutral check-in: ask how things are going and offer "
             "to review whether their current plan still fits their needs, with no diagnosis of any problem.\n"
             "6. End with one easy ask: a short call, offering two slots written exactly as [option 1] and [option 2]. Sign off as [CSM name] from Razorpay.\n"
             "7. Warm, direct, respectful. Write mostly as 'we' (the Razorpay team). No em dashes. {length}\n"
             'Return JSON only: {{"subject": "<email subject, or empty for WhatsApp>", "message": "...", "facts_used": ["F1", ...]}}')

SYS_TRANSLATE = ("Translate this message from a Razorpay team member to a merchant into natural, polite, business {language}, the way a local "
                 "customer success manager would actually write it. Keep the meaning exactly. Write both the subject and the message in {language} script "
                 "(not in Roman letters). Prefer 'we' over 'I' so the text reads naturally whatever the sender's gender, and never use slashes for alternative word forms. Keep the business name and the placeholders [CSM name], [option 1], [option 2] unchanged, in English. "
                 "Do not add numbers, promises or new claims. Return exactly one JSON object and nothing else: "
                 '{{"subject": "...", "message": "..."}}')

SYS_BACK = ("Translate this {language} text into plain English, word for word as far as possible. Do not improve or correct it. "
            "Return only the English translation.")

SYS_SAME = ("Compare two English messages. The second is a back-translation of a translated version of the first. Do they say the same thing: "
            "same facts, same ask, no added or missing promises, claims or numbers? Small wording differences are fine. "
            'Return JSON only: {"same_meaning": true/false, "differences": ["..."]}')

SYS_VERIFY = ("You check a message against a list of allowed facts. List every statement in the message about the merchant's situation that is NOT "
              "supported by the allowed facts. Courtesy phrases, the request for a call, and the offer to help or to review their plan are fine. "
              'Return JSON only: {"unsupported_claims": ["..."]}')

SYS_QA = ("You answer a Razorpay customer success manager's question about one merchant. Use ONLY the JSON record and the method notes given. "
          "If the answer is not in them, say that the data does not show it. Copy numbers exactly. Never promise anything to the merchant, "
          "never invent facts, keep it to a few sentences, plain words, no em dashes.")

METHOD_NOTES = ("Method notes: A merchant is 'declining' if its average volume in months 4-6 fell more than 3 typical monthly wobbles below months 1-3 "
                "(or month 6 alone fell sharply) and it did not bounce back. 'Imminent fade' = projected to lose more than half its healthy volume by month 9. "
                "The driver is the strongest problem family (performance, service, engagement) more than 1 standard deviation above average; if none, the "
                "cause is unclear, the playbook follows the likely bucket, and a person confirms it before acting. Every playbook is one of the six in the data pack (PB-1 to PB-6). The 15 calls are ranked by chance of leaving x chance of saving x yearly value minus Rs 1,500; "
                "'Borderline' means the merchant drops out of the 15 under some reasonable assumptions. Chance of leaving and chance of saving are assumptions, "
                "not measured.")


def _length(channel):
    return 'Keep the message under 70 words.' if 'WhatsApp' in channel else 'Keep the message under 120 words, plus a short subject line.'


def _record_numbers(rec):
    import math
    nums = []
    def add(x):
        if isinstance(x, bool) or x is None: return
        if isinstance(x, (int, float)) and not math.isnan(float(x)): nums.extend([float(x), abs(float(x)), float(round(abs(float(x))))])
    for v in rec.values():
        if isinstance(v, list): [add(x) for x in v]
        elif isinstance(v, dict): [add(x) for x in v.values()]
        else: add(v)
    return nums + list(range(1, 10))          # month numbers 1-9 ("months 1 to 3", "month 9")


def read_note(note):
    if not isinstance(note, str) or not note.strip():
        return None
    r = ask_json(SYS_NOTE, note, model=MODEL_READ, max_tokens=400)
    if not r or r.get('category') not in NOTE_CATEGORIES:
        return r
    r['evidence_is_quote'] = bool(r.get('evidence')) and r['evidence'].strip('"\' ').lower() in note.lower()
    return r


def write_reason(row):
    rec = internal_facts(row)
    text = ask(SYS_REASON, json.dumps(rec, ensure_ascii=False, default=str), max_tokens=200)
    if text is None:
        return None, None
    return text, G.check_numbers_against(text, _record_numbers(rec))


def _fact_text(facts):
    return '\n'.join(f"{f['id']} ({f['topic']}): {f['text']}" for f in facts) or '(no specific facts: write a neutral check-in)'


def draft_for(row, retry_feedback=None, progress=None):
    """English draft, regional version, translation check, independent claim check and code guardrails."""
    facts = merchant_facts(row)
    main_lang, alt_lang = languages(row)
    regional = main_lang if main_lang != 'English' else alt_lang
    channel = CHANNEL[row['segment']]
    unclear = row.get('driver_main') == NC
    user = (f"Business name: {row.get('business_name', row['merchant_id'])}\nChannel: {channel}\n"
            f"Cause: {'unclear (neutral check-in only)' if unclear else 'known'}\nFacts you may use:\n{_fact_text(facts)}")
    if retry_feedback:
        user += f"\n\nYour previous draft broke these rules, fix them: {retry_feedback}"
    say = progress or (lambda m: None)
    say('Writing the English message' + (' again, fixing what the checks found' if retry_feedback else ''))
    en = ask_json(SYS_DRAFT.format(length=_length(channel)), user, max_tokens=500)
    if en is None:
        return None
    result = {'merchant_id': row['merchant_id'], 'channel': channel, 'main_language': main_lang, 'alt_language': alt_lang,
              'facts': facts, 'english': en}
    reg = back = same = None
    if regional:
        say(f'Writing the {regional} version')
        reg = ask_json(SYS_TRANSLATE.format(language=regional), json.dumps({'subject': en.get('subject', ''), 'message': en.get('message', '')}, ensure_ascii=False), max_tokens=900)
        if reg and 'message' in reg:
            say(f'Translating the {regional} version back to English to check it')
            back = ask(SYS_BACK.format(language=regional), (reg.get('subject', '') + '\n' + reg['message']).strip(), max_tokens=600)
            same = ask_json(SYS_SAME, f"FIRST:\n{en.get('subject', '')}\n{en.get('message', '')}\n\nSECOND:\n{back}", max_tokens=300)
    result.update({'regional_language': regional, 'regional': reg, 'back_translation': back, 'same_meaning': same})
    say('Checking every claim against the merchant\'s data')
    ver = ask_json(SYS_VERIFY, f"Allowed facts:\n{_fact_text(facts)}\n\nMessage:\n{en.get('message', '')}", max_tokens=300)
    result['verifier'] = ver
    checks = G.check_draft(en.get('message', ''), facts, unclear, str(row.get('business_name', '')), channel,
                           subject=en.get('subject', '') or '', back_translation=back,
                           translation_ok=None if same is None else same.get('same_meaning'),
                           unsupported=None if ver is None else ver.get('unsupported_claims', []))
    if regional:
        checks.append((f'{regional} draft produced and translated back for checking', bool(reg and 'message' in reg and back), ''))
    if reg and 'message' in reg:
        checks.append(('No numbers in the regional draft itself', not G.has_digits(reg['message'].replace('[option 1]', '').replace('[option 2]', '')), ''))
    result['checks'] = checks
    result['passed'] = G.passed(checks)
    return result


def draft_with_retry(row, progress=None):
    """One regeneration if a guardrail fails; if it fails again the draft is blocked and the CSM writes it."""
    r = draft_for(row, progress=progress)
    if r is None or r['passed']:
        return r
    feedback = '; '.join(f'{n}: {d}' if d else n for n, ok, d in r['checks'] if not ok)
    r2 = draft_for(row, retry_feedback=feedback, progress=progress)
    if r2 is None:
        return r
    r2['retried'] = True
    r2['blocked'] = not r2['passed']
    return r2


def answer_question(row, question):
    rec = internal_facts(row)
    text = ask(SYS_QA, f"{METHOD_NOTES}\n\nRecord:\n{json.dumps(rec, ensure_ascii=False, default=str)}\n\nQuestion: {question}", max_tokens=400, use_cache=True)
    if text is None:
        return None, None
    return text, G.check_numbers_against(text, _record_numbers(rec) + [1500, 15, 50, 90, 60, 30, 12, 100])


def template_draft(row):
    """No-AI fallback (English only) so the app still works without a key on a new file. Built from the same facts."""
    facts = merchant_facts(row); name = row.get('business_name', row['merchant_id']); topics = [f['topic'] for f in facts]
    if row.get('driver_main') == NC or not ({'performance', 'service', 'engagement'} & set(topics)):
        body = (f"Hi {name} team, your account matters to us at Razorpay, and I'd like to check in on how things are going and whether your "
                "current plan still fits your needs.")
    elif topics[0] == 'performance':
        body = (f"Hi {name} team, your account matters to us at Razorpay. We've noticed more of your customers' payments failing recently, "
                "and we'd like to fix this with you alongside our payments team.")
    elif topics[0] == 'service':
        body = (f"Hi {name} team, your account matters to us at Razorpay. You have support requests with us that are still open, and I'd like "
                "to take ownership of them personally and be your single point of contact.")
    else:
        body = (f"Hi {name} team, your account matters to us at Razorpay. I'd like to make sure you're getting full value from your account "
                "and that we have the right contact on your side.")
    return body + " Could we set up a short call at [option 1] or [option 2]? [CSM name], Razorpay"
