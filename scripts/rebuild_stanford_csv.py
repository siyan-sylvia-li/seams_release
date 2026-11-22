"""Rebuild pipeline/final_data/stanford_mooc.csv from your own copy of Stanford MOOCPosts.

We cannot redistribute the post text, so the release ships only IDs and labels
(data/stanford_mooc/stanford_mooc_posts.csv). Given the original spreadsheet
(stanfordMOOCForumPostsSet.xlsx), this script restores `original_post` in the exact
row order used in the paper (row_index), verifying every discussion_post_id.

Then add bullet summaries and structural templates to get the *_v2.csv the pipeline
expects (both are regenerated, so they will not be byte-identical to ours):

  python scripts/rebuild_stanford_csv.py --xlsx /path/to/stanfordMOOCForumPostsSet.xlsx
  cd preprocessing/bullet_summaries && python summarize_student_posts.py \
      --student-data ../../pipeline/final_data/stanford_mooc.csv \
      --output ../../pipeline/final_data/stanford_mooc_v1.csv
  cd ../structural_templates && python rewrite_posts.py \
      --student-data ../../pipeline/final_data/stanford_mooc_v1.csv \
      --output ../../pipeline/final_data/stanford_mooc_v2.csv

For evaluating the released rewrites (scripts/materialize_release.py), stanford_mooc.csv
renamed to stanford_mooc_v2.csv is enough: only original_post and labels are read.
"""
import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True, help="stanfordMOOCForumPostsSet.xlsx")
ap.add_argument("--posts", default=os.path.join(ROOT, "data/stanford_mooc/stanford_mooc_posts.csv"))
ap.add_argument("--output", default=os.path.join(ROOT, "pipeline/final_data/stanford_mooc.csv"))
args = ap.parse_args()

raw = pd.read_excel(args.xlsx)
rel = pd.read_csv(args.posts, dtype={"discussion_post_id": str, "student_id": str})
if len(raw) != len(rel):
    raise SystemExit(f"row count mismatch: xlsx has {len(raw)}, release has {len(rel)}")
raw_ids = raw["forum_post_id"].astype("string").fillna("")
rel_ids = rel["discussion_post_id"].fillna("")  # 14 posts have no forum_post_id in MOOCPosts
mismatch = int((raw_ids.values != rel_ids.values).sum())
if mismatch:
    raise SystemExit(f"{mismatch} rows have a different forum_post_id; is this the same MOOCPosts release?")

out = pd.DataFrame({
    "discussion_post_id": rel["discussion_post_id"],
    "student_id": rel["student_id"],
    "department": rel["department"],
    "original_post": raw["Text"],
    "summary": None,
    "classification_1": rel["classification_1"],
    "classification_2": rel["classification_2"],
    "rewrites": None,
})
os.makedirs(os.path.dirname(args.output), exist_ok=True)
out.to_csv(args.output, index=False)
print(f"wrote {args.output} ({len(out)} rows)")
