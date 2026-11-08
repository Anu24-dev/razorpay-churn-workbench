"""
Thin wrapper around the Claude API with a disk cache.

- The key is read from the ANTHROPIC_API_KEY environment variable, Streamlit secrets, or a local
  anthropic_key.txt (never committed; see .gitignore).
- Every call is cached by its exact inputs, so the stored results in data/ai_cache.json let the app
  run with no key at all (the demo works with the API switched off).
- Results are stored, so the app shows the same text every time it runs on the same file.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / 'data' / 'ai_cache.json'
MODEL_WRITE = 'claude-sonnet-5-5'     # drafts, translation, reasons, checks, Q&A
MODEL_READ = 'claude-haiku-5-5'       # short classification of account-manager notes
_lock = threading.Lock()
_cache = None
_client = None
USAGE = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}   # live calls made in this process


def _load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_FILE.read_text(encoding='utf-8')) if CACHE_FILE.exists() else {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(_cache, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    tmp.replace(CACHE_FILE)


def api_key():
    key = os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get('ANTHROPIC_API_KEY')
        except Exception:
            key = None
    if not key:
        f = ROOT / 'anthropic_key.txt'
        if f.exists():
            key = f.read_text().strip()
    return key or None


MAX_LIVE_CALLS = int(os.environ.get('MAX_LIVE_CALLS', 400))   # hard cap on new API calls per running app, to protect the budget
COUNTER_FILE = ROOT / 'data' / 'live_calls.json'


def _live_calls_used():
    try:
        return json.loads(COUNTER_FILE.read_text()).get('calls', 0)
    except Exception:
        return 0


def _count_live_call():
    try:
        COUNTER_FILE.write_text(json.dumps({'calls': _live_calls_used() + 1}))
    except Exception:
        pass


def live():
    return api_key() is not None and _live_calls_used() < MAX_LIVE_CALLS


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key())
    return _client


def _key(model, system, user, max_tokens):
    return hashlib.sha256(json.dumps([model, system, user, max_tokens], ensure_ascii=False).encode()).hexdigest()


def ask(system: str, user: str, model: str = MODEL_WRITE, max_tokens: int = 800, use_cache: bool = True):
    """Return the model's text, from cache if available. Returns None if there is no cache entry and no key."""
    cache = _load_cache(); k = _key(model, system, user, max_tokens)
    if use_cache and k in cache:
        return cache[k]
    if not live():
        return None
    kw = {'thinking': {'type': 'between_tools'}} if model == MODEL_WRITE else {}   # no hidden reasoning: answer directly within max_tokens
    resp = _get_client().messages.create(model=model, max_tokens=max_tokens, system=system,
                                         messages=[{'role': 'user', 'content': user}], **kw)
    if resp.stop_reason == 'max_tokens':           # cut off: try once more with room to finish
        resp = _get_client().messages.create(model=model, max_tokens=max_tokens * 4, system=system,
                                             messages=[{'role': 'user', 'content': user}], **kw)
    _count_live_call(); USAGE['calls'] += 1; USAGE['input_tokens'] += resp.usage.input_tokens; USAGE['output_tokens'] += resp.usage.output_tokens
    text = ''.join(getattr(b, 'text', '') for b in resp.content).strip()
    with _lock:
        cache[k] = text; _save_cache()
    return text


def parse_json(text):
    """Take the last complete JSON object in the reply (models sometimes correct themselves mid-answer)."""
    dec = json.JSONDecoder(); found = None; i = 0
    while True:
        i = text.find('{', i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                found = obj
            i = end
        except ValueError:
            i += 1
    return found if found is not None else {'_unparsed': text}


def ask_json(system: str, user: str, model: str = MODEL_WRITE, max_tokens: int = 800, use_cache: bool = True):
    text = ask(system, user, model, max_tokens, use_cache)
    return None if text is None else parse_json(text)
