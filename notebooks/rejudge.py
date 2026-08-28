#!/usr/bin/env python3
"""Re-judge only the records a judge left unparseable, in place.

Targets records flagged unreadable (a Yes/No rubric field came back empty) and
re-runs the SAME judge on just those, up to MAX_ATTEMPTS times. A record is
accepted once every Yes/No field validates; on success its `unreadable`/`error`
flags are cleared. The original file is never touched -- output goes to
`<name>.rejudged.jsonl` plus `rejudge_report.txt`.

Wire judge_record() to your harness (see the ADAPT block). Everything else is
model-agnostic: the Yes/No field set is derived from the corpus itself.

Usage:  python3 rejudge.py mistral-small-2603.jsonl
"""
import json, os, sys, time, shutil, tempfile

MAX_ATTEMPTS = 3
RETRY_SLEEP  = 1.5  # seconds between attempts


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_yesno_fields(recs):
    """A field is Yes/No if every value it takes across the corpus is in {Yes,No,''}."""
    fields = {k for r in recs for k in r}
    out = []
    for k in sorted(fields):
        vals = {json.dumps(r.get(k)) for r in recs}
        if vals <= {'"Yes"', '"No"', '""'} and (vals & {'"Yes"', '"No"'}):
            out.append(k)
    return out


def missing_fields(rec, yesno):
    return [k for k in yesno if rec.get(k, "") not in ("Yes", "No")]


def is_flagged(rec, yesno):
    return bool(str(rec.get("unreadable") or "")) or bool(missing_fields(rec, yesno))


# ============================= ADAPT THIS =============================
# Return a dict of rubric fields for one record, by calling the SAME judge
# that produced the corpus (rec["judge"], e.g. gpt-oss:120b-cloud). Returned
# keys overwrite the record, so at minimum return the Yes/No fields. Raise on
# any transport error so the retry loop can catch and retry it.
#
# Easiest path: import the judge call your harness already uses --
#     from harness.judging import judge_answer
#     def judge_record(rec):
#         return judge_answer(prompt=PROMPTS[rec["prompt_id"]], answer=rec["answer"])
#
# Reference path: call Ollama directly with your real rubric prompt.

def judge_record(rec):
    import ollama  # pip install ollama; needs your Ollama auth/env
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},          # your rubric
        {"role": "user",   "content": build_judge_user_prompt(rec)}, # your (prompt, answer) template
    ]
    resp = ollama.chat(model=rec.get("judge", "gpt-oss:120b-cloud"),
                       messages=messages, options={"temperature": 0})
    return parse_judge_output(resp["message"]["content"])            # -> {field: "Yes"/"No"/...}


JUDGE_SYSTEM_PROMPT = "PASTE YOUR RUBRIC PROMPT HERE"

def build_judge_user_prompt(rec):
    raise NotImplementedError("wire this to your harness")

def parse_judge_output(raw):
    raise NotImplementedError("wire this to your harness")
# =====================================================================


def rejudge(src):
    recs  = load(src)
    yesno = derive_yesno_fields(recs)
    flagged = [(i, r) for i, r in enumerate(recs) if is_flagged(r, yesno)]

    print(f"corpus: {len(recs)} records")
    print(f"Yes/No fields ({len(yesno)}): {', '.join(yesno)}")
    print(f"flagged for re-judge: {len(flagged)}\n")

    report = []
    for i, rec in flagged:
        pid, rep = rec.get("prompt_id"), rec.get("replicate")
        was = missing_fields(rec, yesno) or ["<unreadable flag only>"]
        outcome = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                fields = judge_record(rec)
            except Exception as e:
                print(f"[{pid} r{rep}] attempt {attempt}: judge error: {e}")
                time.sleep(RETRY_SLEEP); continue
            merged = {**rec, **fields}
            still  = missing_fields(merged, yesno)
            if not still:
                merged["unreadable"] = ""
                if "error" in merged:
                    merged["error"] = ""
                recs[i] = merged
                outcome = f"fixed on attempt {attempt}"
                break
            print(f"[{pid} r{rep}] attempt {attempt}: still blank -> {still}")
            time.sleep(RETRY_SLEEP)
        if outcome is None:
            outcome = f"STILL BLANK -> {missing_fields(rec, yesno)}"
        report.append((pid, rep, was, outcome))
        print(f"[{pid} r{rep}] {outcome}\n")

    # atomic write of the full corpus with only the flagged rows changed
    out = os.path.splitext(src)[0] + ".rejudged.jsonl"
    d = os.path.dirname(os.path.abspath(out)) or "."
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False, suffix=".tmp") as tf:
        for r in recs:
            tf.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp = tf.name
    shutil.move(tmp, out)

    with open("rejudge_report.txt", "w") as f:
        for pid, rep, was, outcome in report:
            f.write(f"{pid}\tr{rep}\twas_missing={was}\t{outcome}\n")

    fixed = sum(1 for *_, o in report if o.startswith("fixed"))
    print(f"wrote {out}")
    print(f"re-judged {len(report)}: {fixed} fixed, {len(report) - fixed} still blank")
    if fixed < len(report):
        print("STILL BLANK records need a fallback judge or a hand-labelled field.")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "mistral-small-2603.jsonl"
    rejudge(src)
