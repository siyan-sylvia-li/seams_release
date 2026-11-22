# SEAMS data release

This release contains rewrites of discussion-forum posts produced by SEAMS and every
baseline in *Reducing Authorship Attribution Risk in Student Texts Through
Style-Conditioned Rewriting* (referred to below as "the paper"). It covers the paper's two
public datasets: Stack Exchange and Stanford MOOCPosts. The paper's third dataset,
proprietary university forum posts, cannot be released under its data agreement.

Following the paper (§Reconstruction Quality), every experiment is repeated over five seeds
(0, 7, 42, 99, 123). Each seed samples 1,000 validation and 1,000 test posts, stratified by
course department or discussion topic. The released rewrites are the 1,000 **validation**
posts of each seed, the set the paper's generation-quality, downstream and authorship
tables are computed on.

```
data/
├── release_stats.json                 # row / withheld counts per dataset, method, seed
├── stack_exchange/
│   ├── stack_exchange_v2.csv           # all 10,000 posts: text, summary, template, labels
│   ├── splits/eval_splits_seed{S}.json # {"train","val","test","meta"}: row indices
│   └── rewrites/{method}/seed_{S}.jsonl
└── stanford_mooc/
    ├── stanford_mooc_posts.csv         # 29,604 posts: IDs and labels only (no text)
    ├── splits/eval_splits_seed{S}.json
    └── rewrites/{method}/seed_{S}.jsonl
```

## Methods

All rewrites are generated with `Qwen2.5-72B-Instruct` (paper §Experiments). The methods
are described in the paper's §SEAMS and §Baselines.

| Directory | Paper name | Description |
|-----------|-----------|-------------|
| `seams` | SEAMS | K = 20 style exemplars retrieved from the post's style meta-cluster, split into M = 5 disjoint partitions. One candidate is generated per partition. The candidates are embedded with StyleDistance, ℓ2-clipped (τ = 1), averaged with Gaussian noise (σ = 0.1), and the candidate closest to the noisy mean is released (Appendix: Aggregation Hyperparameters). |
| `seams_1` | SEAMS-1 | M = 1: all K = 20 exemplars in a single call (standard retrieval-based in-context learning). |
| `cs` | CS | Cluster-summary-centric: conditions on a contrastive summary of the post's cluster instead of individual exemplars (Appendix: Cluster Summary Generation). |
| `esa` | ESA | Embedding Space Aggregation (Wu et al., 2024). The SEAMS aggregation step is applied to ZS-CS candidates generated without private demonstrations. |
| `dp_prompt` | DP-Prompt | Utpala et al. (2023) without logit clipping, temperature t = 1.0 (best reconstruction quality in our sweep). |
| `zs_cs` | ZS-CS | Zero-shot from bullet summary + structural template. |
| `zs_c` | ZS-C | Zero-shot from bullet summary only. |

The bullet summaries are produced with `Qwen2.5-7B-Instruct` and a GEPA-optimized prompt
(paper Appendix: Bullet Summarizer Details). The structural templates are produced with
GPT-5-mini and a GEPA-optimized prompt, which replaces domain content with four markers
(`###`, `***`, `%%%`, `^^^`; Appendix: Academic-Content-Removed Template Details). These
are the three conditioning channels; the baselines use subsets of them.

## Rewrite records (`rewrites/{method}/seed_{S}.jsonl`)

| Field | Meaning |
|-------|---------|
| `row_index` | Row in `stack_exchange_v2.csv` / `stanford_mooc_posts.csv` (0-based); also the index used in `splits/` |
| `discussion_post_id` | Source post ID: the Stack Exchange question ID, or the MOOCPosts `forum_post_id` (empty for the 14 MOOCPosts posts that have none) |
| `seed`, `method` | As in the file path |
| `rewrite` | The released rewrite (`null` if withheld; Stanford only) |
| `recon_semantic_sim` | Cosine similarity of the rewrite to the original in `all-mpnet-base-v2`: the paper's **Sem** |
| `recon_style_sim` | Cosine similarity of the rewrite to the original in `StyleDistance/styledistance`: the paper's **Sty** |
| `withheld`, `withheld_reason` | Stanford only; see below |

For both datasets, averaging `recon_semantic_sim` / `recon_style_sim` per seed and then
taking the mean ± std over seeds (×100) reproduces the Sem and Sty columns of the paper's
generation-quality table (Appendix: Generation Quality: Full Table). The other columns
(Weg, R-1, MATTR, Self-BLEU) are recomputed from the text by
`pipeline/15_audit_paper_tables.py`.

**Stack Exchange SEAMS-1 coverage.** This method has 757–867 rows per seed instead of
1,000 (4,013 in total); no SEAMS-1 output exists for the remaining posts. The missing posts
are long: in seed 0 their median length is 2,015 characters, against 907 for the posts
that were rewritten. This is consistent with the single 20-exemplar SEAMS-1 prompt
exceeding the 16k-token context window, though our logs do not record the cause. The
paper's SEAMS-1 Stack Exchange numbers are computed over the posts released here.

## Stack Exchange

Following the paper's §Stack Exchange, there are 10,000 questions sampled evenly from
10 sites: biology, chemistry, cs, cstheory, economics, linguistics, math, philosophy,
physics and stats. `stack_exchange_v2.csv` has these columns:

| Column | Meaning |
|--------|---------|
| `discussion_post_id` | Question ID |
| `student_id` | Asker's Stack Exchange user ID (the "author" in the authorship attacks) |
| `department` | Site. This is the paper's downstream label: 10-way subject classification, reported as macro-F1. |
| `original_post` | Question body (HTML) |
| `summary` | Bullet summary (content channel) |
| `rewrites` | Structural template (structure channel) |

Stack Exchange content is licensed under **CC BY-SA 4.0**. This release, including the
rewrites as derivative works, is distributed under the same license. Each post is available
at `https://<site>.stackexchange.com/q/<discussion_post_id>`. Because Stack Exchange
authors are public users, the authorship-attack results on this dataset measure how much
individual style survives rewriting, not re-identification risk (paper §Authorship
Attacks).

## Stanford MOOCPosts

MOOCPosts (Agrawal et al., 2015) contains 29,604 discussion-forum posts from Stanford's
online courses (paper §Stanford MOOCPosts). It is distributed by Stanford under terms that
do not allow redistribution, so we release **no original text, bullet summaries or
structural templates**; the template is often near-verbatim. `stanford_mooc_posts.csv` has
these columns:

| Column | Meaning |
|--------|---------|
| `row_index` | 0-based row; matches the row order of `stanfordMOOCForumPostsSet.xlsx` |
| `discussion_post_id` | `forum_post_id` |
| `student_id` | `forum_uid`, MOOCPosts' anonymized author hash |
| `department` | `CourseType` |
| `classification_1` | Urgency (1–7). Not used in the paper. |
| `classification_2` | Confusion (1–7). This is the paper's downstream label: linear regression, reported as R². |

`scripts/rebuild_stanford_csv.py` restores the post text from your own copy of MOOCPosts,
verifying every `forum_post_id`.

**Withheld rewrites.** Some Stanford rewrites reproduce Stanford text. This mostly happens
for short posts whose template had nothing to mask, so the model copies the post back. Of
SEAMS's Stanford rewrites, 10.6% are word-for-word copies of the original, against 0.7% on
the proprietary data and 0.2% on Stack Exchange. A Stanford rewrite is withheld
(`rewrite: null`) when either of these holds:

- `copies_source_post`: its token sequence equals the original, or it contains at least
  50% of the original's k-grams, where k = min(8, original length in tokens). A short
  original therefore counts as copied when it appears whole inside the rewrite.
- `copies_other_post`: it contains at least 50% of the 8-grams of some other Stanford post
  with 8 or more tokens.

| Method | Withheld / 5,000 |
|--------|-----------------|
| SEAMS | 1,460 (29.2%) |
| SEAMS-1 | 1,523 (30.5%) |
| ESA | 1,331 (26.6%) |
| ZS-CS | 1,261 (25.2%) |
| CS | 878 (17.6%) |
| ZS-C | 51 (1.0%) |
| DP-Prompt | 39 (0.8%) |

Withheld rows keep their IDs and Sem/Sty scores, so the paper's Sem and Sty numbers are
reproducible from the release. Text-based metrics, downstream results and attack results
recomputed from this release cover only the released subset, so they will not match the
paper's Stanford numbers exactly.

## Notes

- **No PII scrubbing.** SEAMS does not redact PII (paper, Ethical Considerations). Rewrites carry only the
  anonymization the source datasets already apply: MOOCPosts replaces screen names with
  tokens such as `<anon_screen_name_redacted>`, and Stack Exchange is public.
- **Intended use.** This release is for research on privacy-motivated rewriting and
  authorship risk. Per the paper, institutions that use SEAMS to release their own data
  should redact or abstract PII before release.
