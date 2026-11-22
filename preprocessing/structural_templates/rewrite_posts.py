from gepa_rewrites_char_acc import Rewriter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dspy
import dotenv
import pandas
from argparse import ArgumentParser

dotenv.load_dotenv("../../.env")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--student-data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    task_lm = dspy.LM("openai/gpt-5-mini")
    dspy.configure(lm=task_lm)

    rewriter = Rewriter()
    rewriter.load("gepa_rewriter_char_acc.json")

    student_df = pandas.read_csv(args.student_data)

    def rewrite_post(og_post):
        return rewriter(post=og_post).rewrite

    total = len(student_df)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rewrites = list(tqdm(executor.map(rewrite_post, student_df["original_post"]), total=total, desc="Rewriting posts"))

    student_df["rewrites"] = rewrites
    student_df.to_csv(args.output, index=False)




