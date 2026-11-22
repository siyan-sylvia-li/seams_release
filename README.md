# SEAMS: Style-Conditioned Rewriting of Student Forum Posts

Code and data for **"Reducing Authorship Attribution Risk in Student Texts Through
Style-Conditioned Rewriting"**.

SEAMS rewrites each student discussion-forum post from three conditioning channels:

1. a **bullet summary** (the content, with style removed),
2. a **structural template** (the post with domain content masked),
3. **style exemplars** retrieved from the post's style cluster.

Each post gets M = 5 parallel generations, each conditioned on a disjoint partition of
K = 20 exemplars. The candidates are embedded, ℓ2-clipped, averaged with Gaussian noise,
and the candidate closest to the noisy mean is released.

This repository contains:

| Path | Contents |
|------|----------|
| [`data/`](data/) | Rewrites from every method in the paper, for **Stack Exchange** and **Stanford MOOCPosts** (5 seeds). See [`data/README.md`](data/README.md). |
| [`preprocessing/`](preprocessing/) | Bullet-summary and structural-template generation (GEPA-optimized DSPy programs). |
| [`pipeline/`](pipeline/) | Clustering, SEAMS and all baselines, quality metrics, downstream and authorship evaluations, table and figure scripts. |
| [`scripts/`](scripts/) | Helpers that unpack the released data into the pipeline layout and rebuild the Stanford CSV from your own copy. |
| [`human_eval/`](human_eval/) | Streamlit app used for the naturalness study (the study data is not released). |

The proprietary university dataset in the paper (`og_students` in the code) cannot be
released under its data agreement. The code paths for it are kept so that the pipeline
matches the paper, but every command below runs on the public datasets.

## Setup

```bash
conda create -n seams python=3.10 && conda activate seams
pip install -r requirements.txt
```

All generation in the paper uses `Qwen/Qwen2.5-72B-Instruct`, served locally with vLLM
(4× 48 GB GPUs, TP = 4):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash pipeline/start_vllm.sh   # serves on :8001
```

The scripts talk to it via DSPy, using `MODEL=openai/Qwen/Qwen2.5-72B-Instruct` and
`API_BASE=http://localhost:8001/v1`; both can be overridden through environment
variables. Keep the `Qwen/` org prefix in the DSPy model string. Set `DSPY_CACHEDIR` to
a disk with room: generations are cached, and a full run uses tens of GB.

Preprocessing uses other models:

- **Bullet summaries:** `Qwen2.5-7B-Instruct`.
- **Structural templates:** `gpt-5-mini`, via the OpenAI API. Put `OPENAI_API_KEY` in a
  `.env` file at the repo root.

## Evaluating the released rewrites (no generation needed)

```bash
python scripts/materialize_release.py --datasets stack_exchange
cd pipeline
python 06_aggregate_quality_metrics.py --help          # Sem / Sty / Weg / R-1 / MATTR / Self-BLEU
python 18_downstream_finetune.py --dataset stack_exchange   # subject labels (department)
python 23_global_attack.py --dataset stack_exchange     # global embedding attackers
python 24_global_stylo.py  --dataset stack_exchange     # global stylometric attacker
python 19_luar_attack.py   --dataset stack_exchange     # LUAR
```

For Stanford MOOC you need your own copy of MOOCPosts (`stanfordMOOCForumPostsSet.xlsx`,
available from Stanford on request):

```bash
python scripts/rebuild_stanford_csv.py --xlsx /path/to/stanfordMOOCForumPostsSet.xlsx
mv pipeline/final_data/stanford_mooc.csv pipeline/final_data/stanford_mooc_v2.csv
python scripts/materialize_release.py --datasets stanford_mooc
```

About 29% of the Stanford SEAMS rows are withheld (see [`data/README.md`](data/README.md)),
so Stanford numbers recomputed from this release cover only the released subset. They will
not match the paper's tables exactly. Stack Exchange is released in full and matches.

## Running the full pipeline

Everything runs from `pipeline/`. Working files go to `pipeline/final_data/`, which is
gitignored. Shown for Stack Exchange; each `*_stack_exchange.sh` script has a
`*_stanford_mooc.sh` counterpart.

```bash
python scripts/materialize_release.py --datasets stack_exchange   # provides stack_exchange_v2.csv + splits
cd pipeline
bash run_clustering_stack_exchange.sh    # 01 splits, 02 HDBSCAN sweep, 03 pick best, 04 meta-clusters
bash run_baseline_stack_exchange.sh      # ZS-C, ZS-CS                  (05_baseline.py)
bash run_cluster_eval_stack_exchange.sh  # CS, cluster-summary baseline (05_cluster_based_approach_eval.py)
bash run_dp_icl_eval_stack_exchange.sh   # SEAMS, M = 5                 (05_dp_icl_config_eval.py)
bash run_m1_dpprompt_stack_exchange.sh   # SEAMS-1 and DP-Prompt        (05_dp_icl_config_eval.py, 05_dp_prompt_eval.py)
bash run_esa_dataset.sh stack_exchange   # ESA                          (05_dp_icl_esa_selection.py)
bash run_downstream_subject_se.sh        # downstream utility: subject labels, embedding probes (09) + DistilBERT (18)
                                         # (Stanford: run_downstream_all.sh, then 18_downstream_finetune.py --dataset stanford_mooc)
DATASET=stack_exchange bash run_authorship_v2b_dataset.sh   # within-cluster attacks (10)
```

`run_clustering_*.sh` regenerates the splits with `01_prepare_eval_splits.py`. For the
paper seeds (0, 7, 42, 99, 123), this reproduces the released `splits/` exactly; we
checked Stack Exchange seeds 0 and 123.

Ablations (Appendix: unconstrained retrieval, deterministic selection, random selection)
are in `05_dp_icl_{unconstrained_retrieval,deterministic_selection,random_selection}.py`.
`run_ablations_og.sh` shows the exact settings, which were run on the proprietary data.
The MIPROv2 instruction search for the rewriter is `00_mipro_propose.py`; the winning
instruction is already in the signatures of `05_*.py`. Tables and figures come from
`15_audit_paper_tables.py`, `16_tradeoff_figure.py`, `17_downstream_table.py`,
`20_audit_attack_tables.py`, `21_authorship_bars.py` and `22_luar_bars.py`. Token and
compute costs come from `probe_token_costs.py`.

### Preprocessing (bullet summaries and templates)

`preprocessing/arrange_columns.py` maps a raw export onto the common schema.
`preprocessing/bullet_summaries/summarize_student_posts.py` adds the `summary` column, and
`preprocessing/structural_templates/rewrite_posts.py` adds the `rewrites` column (the
structural template). The optimized DSPy programs are the `gepa_*.json` files next to them.

## Differences from the code used for the paper

To avoid redistributing student text, the release differs from our internal code in three
places. None of them affects the released rewrites, which were generated with the original
prompts.

- **Template-rewriter in-context demos**
  (`preprocessing/structural_templates/gepa_rewrites_char_acc.py`). The paper used six
  posts from the proprietary dataset. They are replaced by synthetic demos in
  `in_context_demos.synthetic.json`; set `SEAMS_TEMPLATE_DEMOS` to use your own. The GEPA
  optimization data (50 manually templated proprietary posts) is not released either; set
  `SEAMS_MANUAL_REWRITES` to point at your own data.
- **GEPA-optimized instructions** (`gepa_rewriter_char_acc.json`,
  `gepa_bullet_summarizer_stanford_mooc.json`). Concrete example phrases that came from
  student posts were replaced with bracketed placeholders such as `[DrugName]`.
- **Bullet-summarizer demo 5** (`gepa_bullet_summarizer.py`). This was a Stanford MOOCPosts
  post; it is replaced by a synthetic post of the same type.

Templates or summaries you regenerate with this code will therefore differ slightly from
the `summary` / `rewrites` columns in `stack_exchange_v2.csv`.

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Stack Exchange data:** CC BY-SA 4.0, following the Stack Exchange content license.
- **Stanford MOOC rewrites:** see [`data/README.md`](data/README.md).

## Citation

```bibtex
@inproceedings{seams2026,
  title     = {Reducing Authorship Attribution Risk in Student Texts Through Style-Conditioned Rewriting},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```
