import streamlit as st
import os
import json
import io
import math
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# FILE PATHS
# ==========================================
CSV_PATH   = "all_institutions_faculty_data.csv"
EMB_PATH   = "faculty_embeddings.npy"
INDEX_PATH = "faculty_embeddings_index.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

# ==========================================
# APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Jury Matching — Faculty Database",
    page_icon="🎓",
    layout="wide",
)

# ==========================================
# CACHED RESOURCES
# ==========================================
@st.cache_resource(show_spinner="Loading embedding model…")
def get_hf_model():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

@st.cache_data(show_spinner="Loading faculty database…")
def load_faculty_df():
    df = pd.read_csv(CSV_PATH)
    USED_COLS = ["name", "college", "department", "research_area", "research_area_details"]
    df = df[USED_COLS].copy()
    df = df.dropna(subset=["name"])
    df = df.fillna("")
    df = df.reset_index(drop=True)
    return df

def build_faculty_text(row):
    parts = [
        str(row["department"]).strip(),
        str(row["research_area"]).strip(),
        str(row["research_area_details"]).strip(),
    ]
    return " ".join(p for p in parts if p).strip()

@st.cache_data(show_spinner="Loading pre-computed embeddings…")
def load_faculty_embeddings(n_rows: int):
    """
    Load pre-computed embeddings from disk if available.
    Falls back to computing on-the-fly (first run on cloud or missing file).
    n_rows is passed so the cache key is tied to the dataset size.
    """
    if os.path.exists(EMB_PATH) and os.path.exists(INDEX_PATH):
        emb_array = np.load(EMB_PATH)
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
        return emb_array, index
    # Fallback: compute embeddings now
    model = get_hf_model()
    df = load_faculty_df()
    texts = df.apply(build_faculty_text, axis=1).tolist()
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embs.extend(model.embed_documents(batch))
    emb_array = np.array(all_embs, dtype=np.float32)
    np.save(EMB_PATH, emb_array)
    index = df[["name", "college"]].to_dict(orient="records")
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    return emb_array, index

# ==========================================
# HELPERS
# ==========================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def get_project_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(384, dtype=np.float32)
    cache = st.session_state.setdefault("proj_emb_cache", {})
    if text not in cache:
        model = get_hf_model()
        cache[text] = np.array(model.embed_query(text), dtype=np.float32)
    return cache[text]

# ==========================================
# STATE
# ==========================================
if "projects" not in st.session_state:
    st.session_state.projects = []
if "selected_indices" not in st.session_state:
    st.session_state.selected_indices = []

# ==========================================
# LOAD DATA
# ==========================================
hf_model     = get_hf_model()          # warm up model
faculty_df   = load_faculty_df()
all_embs, _  = load_faculty_embeddings(len(faculty_df))

# ==========================================
# TITLE
# ==========================================
st.title("Jury Matching System 🎓")
st.markdown(
    "Faculty-database edition — load faculty from the institutional database "
    "and match them to your projects using semantic similarity."
)

# ==========================================
# SECTION 1 — BROWSE & SELECT FACULTY
# ==========================================
st.header("Section 1 — Browse & Select Faculty")
st.caption(
    f"Database contains **{len(faculty_df):,}** faculty members. "
    "Use the filters below to narrow the pool, then click **Add to Match Pool**."
)

with st.expander("🔍 Filter Faculty", expanded=True):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        q_name = st.text_input("Search by Name", placeholder="e.g. Sharma")
    with fc2:
        q_college = st.text_input("Search by College / Institution", placeholder="e.g. NITT")
    with fc3:
        q_area = st.text_input("Search by Department or Research Area", placeholder="e.g. Machine Learning")

    # Apply filters
    mask = pd.Series([True] * len(faculty_df))
    if q_name.strip():
        mask &= faculty_df["name"].str.contains(q_name.strip(), case=False, na=False)
    if q_college.strip():
        mask &= faculty_df["college"].str.contains(q_college.strip(), case=False, na=False)
    if q_area.strip():
        mask &= (
            faculty_df["department"].str.contains(q_area.strip(), case=False, na=False) |
            faculty_df["research_area"].str.contains(q_area.strip(), case=False, na=False) |
            faculty_df["research_area_details"].str.contains(q_area.strip(), case=False, na=False)
        )

    filtered_df = faculty_df[mask].copy()
    st.caption(f"Showing **{len(filtered_df):,}** results.")

    if len(filtered_df) > 0:
        display_df = filtered_df[["name", "college", "department", "research_area"]].rename(
            columns={
                "name": "Name",
                "college": "College",
                "department": "Department",
                "research_area": "Research Areas",
            }
        )
        st.dataframe(display_df, use_container_width=True, height=300)

        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            if st.button("✅ Add filtered results to match pool", type="primary"):
                new_indices = filtered_df.index.tolist()
                current = set(st.session_state.selected_indices)
                current.update(new_indices)
                st.session_state.selected_indices = sorted(current)
                st.success(
                    f"Added {len(new_indices)} faculty. "
                    f"Pool now has {len(st.session_state.selected_indices)} members."
                )
        with sel_col2:
            if st.button("🗑️ Clear match pool"):
                st.session_state.selected_indices = []
                st.rerun()
    else:
        st.info("No faculty match the current filters.")

# Show current pool
if st.session_state.selected_indices:
    pool_df = faculty_df.loc[st.session_state.selected_indices, ["name", "college", "department", "research_area"]].rename(
        columns={
            "name": "Name",
            "college": "College",
            "department": "Department",
            "research_area": "Research Areas",
        }
    )
    st.subheader(f"Current Match Pool — {len(pool_df)} Faculty")
    st.dataframe(pool_df, use_container_width=True, height=250)
else:
    st.info("Match pool is empty. Use the filters above to add faculty.")

st.divider()

# ==========================================
# SECTION 2 — ADD PROJECT
# ==========================================
st.header("Section 2 — Add Project")
with st.container():
    with st.form("add_project_form"):
        p_title = st.text_input("Project Title")
        p_desc  = st.text_area("Project Description (~200 words)", height=150)
        submit_proj = st.form_submit_button("Add Project")

        if submit_proj:
            if len(st.session_state.projects) >= 15:
                st.error("Maximum of 15 projects allowed.")
            elif not all([p_title, p_desc]):
                st.error("All project fields are required!")
            else:
                st.session_state.projects.append({
                    "title": p_title.strip(),
                    "description": p_desc.strip(),
                })
                st.success(f"Added Project: {p_title}")

if st.session_state.projects:
    st.subheader(f"Project List ({len(st.session_state.projects)}/15)")
    for idx, p in enumerate(st.session_state.projects):
        with st.expander(f"{p['title']}", expanded=False):
            st.markdown(f"**Description:** {p['description']}")
            if st.button("Delete Project", key=f"del_proj_{idx}"):
                st.session_state.projects.pop(idx)
                st.rerun()

st.divider()

# ==========================================
# SECTION 3 — RUN MATCHING
# ==========================================
st.header("Section 3 — Run Jury Matching")

ready = (
    len(st.session_state.selected_indices) > 0
    and len(st.session_state.projects) > 0
)

if not ready:
    st.info("Add at least one faculty to the match pool (Section 1) and one project (Section 2) to enable matching.")

if st.button("Run Matching", type="primary", disabled=not ready):
    selected_idx = st.session_state.selected_indices
    sel_df = faculty_df.loc[selected_idx].copy()

    # Build per-faculty text embeddings from pre-computed array
    faculty_embs = all_embs[selected_idx]  # shape (n_selected, 384)

    with st.spinner("Computing project embeddings and scoring…"):
        jury_total_scores = {row["name"]: 0.0 for _, row in sel_df.iterrows()}
        project_results = []

        for p in st.session_state.projects:
            p_text = f"{p['title']} {p['description']}"
            p_emb  = get_project_embedding(p_text)
            if p_emb is None or np.linalg.norm(p_emb) == 0:
                continue

            scores = []
            for i, (df_idx, row) in enumerate(sel_df.iterrows()):
                j_emb = faculty_embs[i]
                sim   = cosine_similarity(j_emb, p_emb)
                scores.append({
                    "name":        row["name"],
                    "college":     row["college"],
                    "department":  row["department"],
                    "similarity":  round(sim, 4),
                })
                jury_total_scores[row["name"]] += sim

            scores.sort(key=lambda x: x["similarity"], reverse=True)
            project_results.append({
                "project":  p["title"],
                "rankings": scores,
            })

    # ---- RESULTS ----
    st.divider()
    st.header("🏆 Matching Results")
    for pr in project_results:
        st.subheader(f"Project: {pr['project']}")
        for rank_idx, r in enumerate(pr["rankings"]):
            st.markdown(
                f"**Rank {rank_idx + 1} — {r['name']} — {r['similarity']:.4f}**"
            )
            st.caption(
                f"College: {r['college']} | Department: {r['department']} | "
                f"Cosine Similarity: {r['similarity']:.4f}"
            )
        st.write("")

    # ---- OVERALL RANKING ----
    st.divider()
    st.header("🌟 Overall Faculty Ranking")
    num_projects = len(st.session_state.projects)
    avg_scores   = [
        (name, total / num_projects)
        for name, total in jury_total_scores.items()
    ]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    for rank_idx, (name, avg) in enumerate(avg_scores[:5]):
        st.markdown(f"**{rank_idx + 1}. {name}** — Avg Similarity: {avg:.4f}")

    # ---- EXPORT ----
    st.divider()
    st.header("📥 Export Results")

    # Build score lookup: {faculty_name: {project_title: similarity}}
    score_lookup = {}
    for pr in project_results:
        for r in pr["rankings"]:
            score_lookup.setdefault(r["name"], {})[pr["project"]] = r["similarity"]

    project_titles = [pr["project"] for pr in project_results]

    # Faculty metadata lookup
    fac_meta = sel_df.set_index("name")[["college", "department", "research_area", "research_area_details"]].to_dict(orient="index")

    export_rows = []
    for rank_idx, (faculty_name, avg_score) in enumerate(avg_scores, start=1):
        meta = fac_meta.get(faculty_name, {})
        row  = {
            "Rank":           rank_idx,
            "Faculty Name":   faculty_name,
            "College":        meta.get("college", ""),
            "Department":     meta.get("department", ""),
            "Research Details": meta.get("research_area_details", ""),
        }
        for pt in project_titles:
            row[pt] = score_lookup.get(faculty_name, {}).get(pt, 0.0)
        row["Overall Avg Similarity"] = round(avg_score, 4)
        export_rows.append(row)

    export_df = pd.DataFrame(export_rows)
    st.dataframe(export_df, use_container_width=True)

    # Excel download
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Faculty Matching")
    excel_buf.seek(0)
    st.download_button(
        label="⬇️ Download as Excel (.xlsx)",
        data=excel_buf,
        file_name="faculty_matching_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # CSV download
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV (.csv)",
        data=csv_data,
        file_name="faculty_matching_results.csv",
        mime="text/csv",
    )

# ==========================================
# DEPLOYMENT NOTES
# ==========================================
# How to run locally:
#   streamlit run app_faculty.py
#
# For fast startup on Streamlit Cloud:
#   1. Run:  python precompute_embeddings.py
#   2. Commit faculty_embeddings.npy and faculty_embeddings_index.json
#      to the faculty-preloaded branch alongside this file.
#   3. Deploy a new Streamlit Cloud app pointing to that branch
#      with app_faculty.py as the main file.
