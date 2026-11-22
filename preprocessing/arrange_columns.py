"""Map a raw dataset export onto the common SEAMS column schema.

Output columns: discussion_post_id, original_post, department, student_id,
classification_1, classification_2. Run summarize_student_posts.py (adds `summary`)
and then rewrite_posts.py (adds `rewrites`, the structural template) on the result
to get the *_v2.csv file that the pipeline expects.

Usage:
  python arrange_columns.py --dataset stanford_mooc --input stanfordMOOCForumPostsSet.xlsx \
      --output ../pipeline/final_data/stanford_mooc.csv
"""
from argparse import ArgumentParser

import pandas

COLUMN_LOOKUPS = {
    "stack_exchange": {
        "discussion_post_id": "question_id",
        "original_post": "body",
        "department": "site",
        "student_id": "owner_user_id",
        "classification_1": None,  # the paper's SE label is `department` (subject)
        "classification_2": None,
    },
    "stanford_mooc": {
        "discussion_post_id": "forum_post_id",
        "original_post": "Text",
        "department": "CourseType",
        "student_id": "forum_uid",
        "classification_1": "Urgency(1-7)",
        "classification_2": "Confusion(1-7)",
    },
}

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(COLUMN_LOOKUPS))
    ap.add_argument("--input", required=True, help="raw .csv or .xlsx export")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.input.endswith("xlsx"):
        curr_df = pandas.read_excel(args.input)
    else:
        curr_df = pandas.read_csv(args.input)

    fin_df = pandas.DataFrame([])
    for col, src in COLUMN_LOOKUPS[args.dataset].items():
        fin_df[col] = curr_df[src] if src else [None] * len(curr_df)
    fin_df.to_csv(args.output, index=False)
