#!/usr/bin/env python3
"""Batch-1 decode/prefill benchmark following LocalMaxxing's verified-run rules.

Canonical prompts (reasoning-v1, code-v1) from localmaxxing.com/api/agent-context, temperature 0,
batch 1, streamed, warmups discarded. Decode tok/s = (completion_tokens - 1) / (t_last - t_first),
prefill tok/s = prompt_tokens / TTFT. Spec acceptance comes from the server's Prometheus counters
(vLLM) as deltas around each request. Also runs Fraser's long prompt shape (2,678 in / 2,048 out).

usage: lmx_bench.py --url http://127.0.0.1:8000 --model NAME --label X --out results.jsonl
"""
import argparse, hashlib, json, re, secrets, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="canonical_prompts.json")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--fraser-reps", type=int, default=10)
ap.add_argument("--only", default="")
ap.add_argument("--depth", default="", help="extra long-prompt cases NAME:REPS,... (fraser-unit x REPS)")
ap.add_argument("--no-think", action="store_true", help="disable reasoning via chat_template_kwargs")
ap.add_argument("--think-effort", default="", help="enable thinking with this reasoning_effort (DeepSeek: low|high|xhigh|max)")
ap.add_argument("--sglang-spec-details", action="store_true",
                help="SGLang: request return_spec_tokens_details and take proposed/accepted drafts and verify steps from the "
                     "server's per-request sglext.spec_tokens_details (what LocalMaxxing verification needs)")
a = ap.parse_args()

SPEC = re.compile(r"^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))(?:_total)?(?:\{[^}]*\})?\s+([0-9.e+]+)", re.M)


def spec_counters():
    try:
        txt = urllib.request.urlopen(a.url + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for name, val in SPEC.findall(txt):
        out[name] = out.get(name, 0.0) + float(val)
    for val in SGL_VERIFY.findall(txt):
        out["sglang_verify"] = out.get("sglang_verify", 0.0) + float(val)
    return out


# SGLang (2026-11-04, Ripple DFlash2): one counter of verify steps. Each step emits its accepted drafts plus one
# target token, and the first token comes from prefill, so tokens/step = (completion_tokens - 1) / verify_steps and
# accepted drafts = (completion_tokens - 1) - verify_steps. SGLang exposes no proposed-token count, so no draft_tokens.
SGL_VERIFY = re.compile(r"^sglang:spec_verify_calls_total(?:\{[^}]*\})?\s+([0-9.e+]+)", re.M)


def sglang_spec(before, after, completion_tokens):
    steps = after.get("sglang_verify", 0.0) - before.get("sglang_verify", 0.0)
    if steps <= 0:
        return {}
    return {"drafts": steps, "accepted_tokens": max(completion_tokens - 1 - steps, 0.0)}


def run(prompt_text):
    nonce = f"[LocalMaxxing cache-bust nonce: {secrets.token_hex(16)}]\n"
    body = {"model": a.model, "messages": [{"role": "user", "content": nonce + prompt_text}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.think_effort:
        body["chat_template_kwargs"] = {"thinking": True}
        body["reasoning_effort"] = a.think_effort
    if a.no_think:
        body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
    if a.sglang_spec_details:
        body["return_spec_tokens_details"] = True
    before = spec_counters()
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; usage = None; text = []; sgl_spec = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            if (ev.get("sglext") or {}).get("spec_tokens_details"):
                sgl_spec = ev["sglext"]["spec_tokens_details"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    text.append(piece)
    after = spec_counters()
    ct, pt = usage["completion_tokens"], usage["prompt_tokens"]
    if sgl_spec and sgl_spec.get("spec_verify_ct"):
        # server-reported, per request: verify steps, proposed draft tokens, accepted draft tokens
        spec = {"drafts": float(sgl_spec["spec_verify_ct"]), "draft_tokens": float(sgl_spec["spec_num_proposed_drafts"]),
                "accepted_tokens": float(sgl_spec["spec_num_correct_drafts"])}
    elif "sglang_verify" in after:
        spec = sglang_spec(before, after, ct)
    else:
        spec = {k.split("num_")[1]: after.get(k, 0) - before.get(k, 0) for k in after} if after else {}
    ttft = t_first - t0
    rec = {"label": a.label, "prompt_tokens": pt, "completion_tokens": ct,
           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "ttft_ms": ttft * 1000, "tok_s_out": (ct - 1) / (t_last - t_first),
           "tok_s_prefill": pt / ttft, "tok_s_total": (pt + ct) / (t_last - t0), "spec": spec,
           "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
           "output_sample": "".join(text)[:4000], "prompt_sample": (nonce + prompt_text)[:2000]}
    if spec.get("drafts"):
        rec["mean_accept_len"] = 1 + spec["accepted_tokens"] / spec["drafts"]
        if spec.get("draft_tokens"):  # SGLang reports no proposed-token count
            rec["accept_rate"] = spec["accepted_tokens"] / max(spec["draft_tokens"], 1)
    return rec


prompts = json.load(open(a.prompts))
# Fraser's shape: the code prompt doubled up to ~2.7k tokens of input.
long_text = (prompts["code-v1"] + "\n\n") * 10
cases = [("reasoning-v1", prompts["reasoning-v1"]), ("code-v1", prompts["code-v1"]), ("long-2.7k", long_text)]
if "fraser-unit" in prompts:
    # Fraser's submitted prompt: the webhook task x10, stripped (sha256 5a6b7bc1..., matches his LocalMaxxing runs).
    cases.append(("fraser-webhook", (prompts["fraser-unit"] * int(a.fraser_reps)).strip()))
for spec in filter(None, a.depth.split(",")):
    nm, reps = spec.split(":")
    cases.append((nm, (prompts["fraser-unit"] * int(reps)).strip()))
with open(a.out, "a") as f:
    for name, text in cases:
        if a.only and name not in a.only.split(","):
            continue
        for _ in range(a.warmup):
            run(text)
        recs = []
        for i in range(a.runs):
            r = run(text); r.update(case=name, i=i, prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
            recs.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        med = lambda k: statistics.median(x[k] for x in recs if x.get(k) is not None) if any(x.get(k) is not None for x in recs) else None
        print(f"{a.label:24s} {name:13s} in={recs[0]['prompt_tokens']:5d} out={med('completion_tokens'):6.0f} "
              f"decode={med('tok_s_out'):7.1f} prefill={med('tok_s_prefill'):8.1f} ttft={med('ttft_ms'):7.1f}ms "
              f"accept_len={med('mean_accept_len') or 0:.2f} "
              f"min/max decode={min(x['tok_s_out'] for x in recs):.1f}/{max(x['tok_s_out'] for x in recs):.1f}", flush=True)
