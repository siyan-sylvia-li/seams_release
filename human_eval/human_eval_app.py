import streamlit as st
import pandas as pd
import hashlib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "human_eval_data.csv")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "annotations")
os.makedirs(SAVE_DIR, exist_ok=True)

LABELS = {
    "post1": "Post 1 is more naturalistic",
    "post2": "Post 2 is more naturalistic",
    "equal": "Equally naturalistic",
}


def save_path(name):
    return os.path.join(SAVE_DIR, f"annotations_{name}.csv")


def load_data(annotator_name):
    df = pd.read_csv(DATA_PATH, index_col=0).reset_index(drop=True)
    seed = int(hashlib.md5(annotator_name.encode()).hexdigest(), 16) % (2**32)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_existing(name):
    path = save_path(name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["pair_index", "post_1", "post_2", "post_key", "label"])


def save_response(name, row, label):
    path = save_path(name)
    existing = load_existing(name)
    new_row = pd.DataFrame([{
        "pair_index": row.name,
        "post_1": row["post_1"],
        "post_2": row["post_2"],
        "post_key": row["post_key"],
        "label": label,
    }])
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(path, index=False)


# ── Login screen ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Human Eval", layout="wide")

if "annotator" not in st.session_state:
    st.title("Naturalism Evaluation")
    name = st.text_input("Enter your name to begin:")
    if st.button("Start") and name.strip():
        st.session_state.annotator = name.strip()
        st.rerun()
    st.stop()

annotator = st.session_state.annotator
df = load_data(annotator)
done = load_existing(annotator)
done_indices = set(done["pair_index"].tolist())

remaining = [i for i in range(len(df)) if i not in done_indices]
n_total = len(df)
n_done = len(done_indices)

# ── Sidebar progress ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**Annotator:** {annotator}")
    st.progress(n_done / n_total)
    st.markdown(f"{n_done} / {n_total} done")
    if st.button("Switch annotator"):
        del st.session_state["annotator"]
        st.rerun()

# ── Done screen ───────────────────────────────────────────────────────────────

if not remaining:
    st.success(f"All {n_total} pairs complete! Results saved to `annotations/{annotator}.csv`.")
    st.stop()

# ── Annotation screen ─────────────────────────────────────────────────────────

curr_idx = remaining[0]
row = df.iloc[curr_idx]

st.markdown(f"### Pair {n_done + 1} of {n_total}")
st.markdown("**Which post sounds more like something a real person would naturally write?**")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Post 1")
    st.markdown(row["post_1"])
with col2:
    st.markdown("#### Post 2")
    st.markdown(row["post_2"])

st.markdown("---")

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    if st.button("Post 1 is more naturalistic", use_container_width=True):
        save_response(annotator, row, "post1")
        st.rerun()
with bcol2:
    if st.button("Equally naturalistic", use_container_width=True):
        save_response(annotator, row, "equal")
        st.rerun()
with bcol3:
    if st.button("Post 2 is more naturalistic", use_container_width=True):
        save_response(annotator, row, "post2")
        st.rerun()
