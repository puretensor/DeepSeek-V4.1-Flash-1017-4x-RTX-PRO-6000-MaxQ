#!/usr/bin/env python3
"""stat.py RESULTS.jsonl LABEL -> 'median_tok_s median_step_ms n' over every run with that label."""
import json, statistics as st, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v = [r for r in rows if r["label"] == sys.argv[2] and r.get("mean_accept_len")]
if not v: print("0 999 0"); sys.exit(0)
print(f'{st.median(r["tok_s_out"] for r in v):.1f} {st.median(1000 * r["mean_accept_len"] / r["tok_s_out"] for r in v):.3f} {len(v)}')
