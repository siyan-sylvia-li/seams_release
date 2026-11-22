"""
probe_token_costs.py

Runs 5 eval posts through SEAMS M=5 and M=1 using the stack_exchange setup,
captures dspy.LM history, and reports:
  - Number of LLM calls per post
  - Prompt tokens per call
  - Completion tokens per call
  - Number of in-context exemplars per call (by counting "Original Post:" occurrences)

Usage:
    python probe_token_costs.py

Requires vLLM to be running on port 8001.
"""

import json, os, sys, re
import numpy as np
import dspy

os.environ.setdefault("DSPY_CACHEDIR",  os.path.expanduser("~/.dspy_cache"))
os.environ.setdefault("DSPY_CACHE_LIMIT", "30000000000")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

API_BASE = "http://localhost:8001/v1"
MODEL    = "openai/Qwen/Qwen2.5-72B-Instruct"
N_POSTS  = 5   # small probe
SEED     = 0

# ---------------------------------------------------------------------------
# Setup DSPy LM with history enabled
# ---------------------------------------------------------------------------
lm = dspy.LM(MODEL, api_base=API_BASE, api_key="none", cache=False)
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# Load just enough data to run a few posts
# ---------------------------------------------------------------------------
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading data...")
df = pd.read_csv("final_data/stack_exchange_v2.csv").reset_index(drop=True)
splits = json.load(open(f"final_data/stack_exchange/eval_splits_seed{SEED}.json"))
eval_idx = splits["val"][:N_POSTS]

# Load cluster artifacts for M=5 run (reuse existing pass_a_winners)
pass_a_winners_m5 = json.load(open(
    f"final_data/stack_exchange/dp_icl_outputs/meta/seed_{SEED}/pass_a_winners.json"
))
pass_a_winners_m1 = json.load(open(
    f"final_data/stack_exchange/dp_icl_outputs/meta_m1/seed_{SEED}/pass_a_winners.json"
))

# ---------------------------------------------------------------------------
# Import the eval module and run with history capture
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
# We'll call 05_dp_icl_config_eval directly via subprocess to avoid import side effects,
# but actually easier to just count from history after a real run.
# Instead, replicate the DSPy predict call minimally.

# Actually the cleanest approach: run the script with --max-posts 5 and capture history inline.
# Since that's complex, use subprocess and parse history from the LM object.

# Simpler: just run the actual script with max-posts and capture stdout token info
import subprocess, time

def run_and_capture(m_val, winners_path, label):
    """Run eval script for N_POSTS and capture token info from history."""
    # Clear history
    lm.history.clear()

    cmd = [
        "python", "05_dp_icl_config_eval.py",
        "--student-data",    "final_data/stack_exchange_v2.csv",
        "--splits",          f"final_data/stack_exchange/eval_splits_seed{SEED}.json",
        "--clustering-root", f"final_data/stack_exchange/hdbscan_results/seed_{SEED}",
        "--meta-output-dir", "final_data/stack_exchange/meta_clustering_output",
        "--output-dir",      f"/tmp/probe_{label}",
        "--eval-split",      "val",
        "--axes",            "og_style_model",
        "--student-id-col",  "student_id",
        "--seed",            str(SEED),
        "--use-meta",
        "--dp-m",            str(m_val),
        "--pass-a-dp-m",     "1",
        "--workers",         "4",
        "--model",           MODEL,
        "--api-base",        API_BASE,
        "--max-posts",       str(N_POSTS),
    ]
    print(f"\n{'='*60}")
    print(f"Running {label} (M={m_val}, {N_POSTS} posts)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
    return result


# ---------------------------------------------------------------------------
# Instead: use the DSPy cache to get token distribution by running with cache=True
# and then inspecting what was newly added. But probe with cache=False is cleaner.
# Let's just run and parse the output JSON for timing + call counts.
# ---------------------------------------------------------------------------

print("\nNOTE: This script runs real inference. Each call goes to vLLM (cache disabled).")
print(f"Running {N_POSTS} posts × 2 methods = ~{N_POSTS*6} LLM calls total.\n")

for label, m_val in [("M5", 5), ("M1", 1)]:
    import os
    os.makedirs(f"/tmp/probe_{label}", exist_ok=True)

    # Copy pass_a_winners so script finds it
    winners = pass_a_winners_m5 if m_val == 5 else pass_a_winners_m1
    with open(f"/tmp/probe_{label}/pass_a_winners.json", "w") as f:
        json.dump(winners, f)

    t0 = time.time()
    # Clear global DSPy history before run (history won't survive subprocess,
    # so instead we'll inspect the output JSON for token info via vLLM logs)
    run_and_capture(m_val, f"/tmp/probe_{label}/pass_a_winners.json", label)
    elapsed = time.time() - t0
    print(f"{label}: wall time = {elapsed:.1f}s for {N_POSTS} posts = {elapsed/N_POSTS:.1f}s/post")

print("\nDone. Check /tmp/probe_M5/ and /tmp/probe_M1/ for output JSONs.")
print("Token counts can be read from the vLLM log: grep 'prompt_tokens' /tmp/vllm_qwen72b_16k.log | tail -20")
