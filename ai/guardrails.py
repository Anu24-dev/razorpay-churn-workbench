"""
Guardrails, enforced in code (not by asking the AI nicely).

The brief: never invent facts, never promise refunds, dates or waivers the data does not support,
and never take an irreversible action without a human check. Our own rules on top: no numbers in
merchant messages, no mention of volume or leaving, and a neutral check-in when the cause is unknown.
Each check returns (name, passed, detail). A draft is usable only if every check passes.
"""
from __future__ import annotations

import re
import unicodedata

# Topics a draft may only raise if a fact with that topic is in the merchant's fact sheet
TOPIC_WORDS = {
    'performance': [r'fail', r'success rate', r'declin\w* (payment|transaction)', r'payments? (are|were|have been) not going through', r'drop-?offs?'],
    'service': [r'ticket', r'support request', r'open request', r'complain', r'feedback on (our |razorpay )?support', r'unresolved'],
    'engagement': [r'dashboard', r'log ?in', r'products?\b', r'features?\b', r'workshop', r'walkthrough'],
    'renewal': [r'renewal', r'renew', r'contract'],
}
PROMISES = [r'refund', r'waiv', r'discount', r'\bfree\b', r'cash ?back', r'compensat', r'guarantee', r'\bpromise', r'rebate',
            r'\bcredit(?! card)', r'lower (your )?(rate|price|fee|mdr)', r'reduc\w* (your )?(rate|price|fee|charges|mdr)', r'\bmdr\b',
            r'better (rate|price|deal|pricing)', r'special (rate|price|offer|pricing)']
TIMELINES = [r'\bwithin\b', r'\b(by|before) (monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight|end of|the end)',
             r'\b(a|an|one|two|three|four|five|six|seven|a few|few|couple of) (business |working )?(hours?|days?|weeks?)\b', r'\bdeadline',
             r'\bimmediately\b', r'\bright away\b', r'\bsame day\b', r'\bovernight\b', r'\bturnaround',
             r'will (be )?(fixed|resolved|closed|sorted|done) (by|in|within)', r'\bclosure date']
INTERNAL = [r'churn', r'\brisk', r'at-risk', r'\bscore', r'\btier', r'\bflag', r'\bleav(e|ing)\b', r'\bswitch', r'moving (your )?(business|volume|payments)',
            r'\bvolume', r'\btpv\b', r'drop in (your )?(business|sales|revenue)', r'\bsales\b', r'\brevenue', r'competitor', r'other provider',
            r'(account manager|internal|our|account) notes?\b', r'\bnotes? (say|says|show|shows|suggest|mention)', r'our (records|systems?) show', r'\bmodel\b', r'\banalytics?\b',
            r'payu', r'cashfree', r'paytm', r'phonepe', r'stripe', r'ccavenue', r'juspay', r'instamojo', r'billdesk', r'easebuzz', r'pine ?labs']
DIAGNOSIS = TOPIC_WORDS['performance'] + TOPIC_WORDS['service'] + [r'dashboard', r'log ?in']
ALLOWED_PLACEHOLDERS = {'[option 1]', '[option 2]'}


def _find(patterns, text):
    t = text.lower()
    return sorted({m.group(0) for p in patterns for m in re.finditer(p, t)})


def has_digits(text):
    return any(unicodedata.category(ch) == 'Nd' for ch in text)   # catches 0-9 and Indian-script digits


def check_draft(english: str, facts: list, unclear_cause: bool, business_name: str, channel: str,
                subject: str = '', back_translation: str | None = None, translation_ok: bool | None = None,
                unsupported: list | None = None):
    """Run every check on the English draft (and on the back-translation of the regional draft)."""
    out = []
    texts = {'English draft': (subject + '\n' + english).strip()}
    if back_translation:
        texts['regional draft (translated back)'] = back_translation
    topics = {f['topic'] for f in facts}
    for label, t in texts.items():
        out.append((f'No numbers ({label})', not has_digits(re.sub(r'\[option [12]\]', '', t)), ''))
        hits = _find(PROMISES, t); out.append((f'No promises of refunds, waivers, discounts or price changes ({label})', not hits, ', '.join(hits)))
        hits = _find(TIMELINES, t); out.append((f'No timelines or deadlines ({label})', not hits, ', '.join(hits)))
        hits = _find(INTERNAL, t); out.append((f'No internal information: volume, leaving, risk, scores, notes, competitors ({label})', not hits, ', '.join(hits)))
        raised = [tp for tp, pats in TOPIC_WORDS.items() if _find(pats, t)]
        stray = [tp for tp in raised if tp not in topics]
        out.append((f'Every topic raised is backed by the data ({label})', not stray, ('not in fact sheet: ' + ', '.join(stray)) if stray else ''))
        if unclear_cause:
            hits = _find(DIAGNOSIS, t)
            out.append((f'Cause unknown, so neutral check-in only ({label})', not hits, ', '.join(hits)))
    brackets = set(re.findall(r'\[[^\]]+\]', english))
    out.append(('Only allowed placeholders left for the CSM to fill', brackets <= ALLOWED_PLACEHOLDERS, ', '.join(sorted(brackets - ALLOWED_PLACEHOLDERS))))
    out.append(('Addresses the right merchant by name', business_name.lower() in english.lower(), ''))
    limit = 80 if 'WhatsApp' in channel else 140
    n = len(english.split())
    out.append((f'Short enough for the channel (max {limit} words)', n <= limit, f'{n} words'))
    if translation_ok is not None:
        out.append(('Regional and English versions say the same thing', bool(translation_ok), ''))
    if unsupported is not None:
        out.append(('Independent check found no claim outside the fact sheet', not unsupported, '; '.join(unsupported)))
    return out


def passed(checks):
    return all(ok for _, ok, _ in checks)


def check_numbers_against(text: str, allowed_numbers: list):
    """For internal text (reason line, Q&A answer): every number must match a number in the merchant's record."""
    found = [float(x.replace(',', '').rstrip('.')) for x in re.findall(r'(?<![A-Za-z\d])\d[\d,]*\.?\d*', text)]   # skips IDs like C2090
    bad = [x for x in found if not any(abs(x - a) <= max(0.6, abs(a) * 0.01) for a in allowed_numbers)]
    return ('Every number matches the merchant\'s data', not bad, ', '.join(str(b) for b in bad))
