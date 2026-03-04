import streamlit as st
import os
import json
import io
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
TOP_K      = 5          # top results to keep per project

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
    # Include all columns we display or use for filtering/matching
    USED_COLS = ["url", "name", "designation", "college", "department",
                 "research_area", "research_area_details"]
    # Only keep columns that actually exist in the CSV
    USED_COLS = [c for c in USED_COLS if c in df.columns]
    df = df[USED_COLS].copy()
    df = df.dropna(subset=["name"])
    df = df.fillna("")
    df = df.reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def get_designation_options(df: pd.DataFrame):
    """Sorted unique designation values for the dropdown filter."""
    if "designation" not in df.columns:
        return []
    vals = df["designation"].dropna().unique().tolist()
    vals = sorted([v for v in vals if str(v).strip()])
    return vals

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
    Returns L2-normalised float32 matrix so cosine similarity == dot product.
    Falls back to computing on-the-fly when files are missing.
    n_rows is passed so the cache key is tied to the dataset size.
    """
    if os.path.exists(EMB_PATH) and os.path.exists(INDEX_PATH):
        emb_array = np.load(EMB_PATH).astype(np.float32)
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        # Fallback: compute embeddings now (slow – run precompute_embeddings.py instead)
        model = get_hf_model()
        df    = load_faculty_df()
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

    # L2-normalise once so dot product == cosine similarity at query time
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
    emb_array = emb_array / norms
    return emb_array, index

# ==========================================
# HELPERS
# ==========================================
def get_project_embedding(text: str) -> np.ndarray:
    """Embed a project text, L2-normalised, with session-level caching."""
    if not text.strip():
        return np.zeros(384, dtype=np.float32)
    cache = st.session_state.setdefault("proj_emb_cache", {})
    if text not in cache:
        model = get_hf_model()
        vec = np.array(model.embed_query(text), dtype=np.float32)
        n   = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        cache[text] = vec
    return cache[text]


def top_k_scores(emb_matrix: np.ndarray, query_vec: np.ndarray, k: int):
    """
    Vectorised top-k cosine similarity.
    emb_matrix : (N, D) normalised float32
    query_vec  : (D,)   normalised float32
    Returns arrays (indices, scores) of the top-k rows, sorted descending.
    """
    # One BLAS matrix-vector multiply → (N,) similarities
    sims = emb_matrix @ query_vec          # shape (N,)
    k    = min(k, len(sims))
    # argpartition is O(N) – much faster than a full sort for large N
    part = np.argpartition(sims, -k)[-k:]
    part = part[np.argsort(sims[part])[::-1]]   # sort just the k candidates
    return part, sims[part]

# ==========================================
# STATE
# ==========================================
if "projects" not in st.session_state:
    st.session_state.projects = []
if "selected_indices" not in st.session_state:
    st.session_state.selected_indices = []
if "proj_form_key" not in st.session_state:
    st.session_state.proj_form_key = 0

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
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        q_name = st.text_input("Search by Name", placeholder="e.g. Sharma")
    with fc2:
        q_college = st.text_input("Search by College / Institution", placeholder="e.g. NITT")
    with fc3:
        q_area = st.text_input("Search by Department or Research Area", placeholder="e.g. Machine Learning")
    with fc4:
        desig_options = get_designation_options(faculty_df)
        if desig_options:
            q_desig = st.selectbox(
                "Filter by Designation",
                options=["— All Designations —"] + desig_options,
            )
        else:
            q_desig = "— All Designations —"

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
    if q_desig and q_desig != "— All Designations —":
        mask &= faculty_df["designation"].str.fullmatch(q_desig, na=False)

    filtered_df = faculty_df[mask].copy()
    st.caption(f"Showing **{len(filtered_df):,}** results.")

    if len(filtered_df) > 0:
        # Build a flexible display DataFrame — include every useful column
        display_cols_map = {
            "name":                "Name",
            "designation":         "Designation",
            "college":             "College",
            "department":          "Department",
            "research_area":       "Research Areas",
            "research_area_details": "Research Details",
        }
        disp_cols = [c for c in display_cols_map if c in filtered_df.columns]
        display_df = filtered_df[disp_cols].rename(columns=display_cols_map)

        # Add a clickable Profile URL column if available
        if "url" in filtered_df.columns:
            display_df["Profile URL"] = filtered_df["url"]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
            column_config={
                "Profile URL": st.column_config.LinkColumn(
                    label="Profile URL",
                    display_text="Open ↗",
                ),
            },
        )

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
    pool_cols_map = {
        "name":        "Name",
        "designation": "Designation",
        "college":     "College",
        "department":  "Department",
        "research_area": "Research Areas",
    }
    pool_src_cols = [c for c in pool_cols_map if c in faculty_df.columns]
    pool_df = faculty_df.loc[st.session_state.selected_indices, pool_src_cols].rename(
        columns=pool_cols_map
    )
    if "url" in faculty_df.columns:
        pool_df["Profile URL"] = faculty_df.loc[st.session_state.selected_indices, "url"]
    st.subheader(f"Current Match Pool — {len(pool_df)} Faculty")
    st.dataframe(
        pool_df,
        use_container_width=True,
        height=250,
        column_config={
            "Profile URL": st.column_config.LinkColumn(
                label="Profile URL",
                display_text="Open ↗",
            ),
        },
    )
else:
    st.info("Match pool is empty. Use the filters above to add faculty.")

st.divider()

# ==========================================
# SECTION 2 — ADD PROJECT
# ==========================================
st.header("Section 2 — Add Project")
with st.container():
    # A unique form key forces Streamlit to re-render blank inputs after each add
    with st.form(key=f"add_project_form_{st.session_state.proj_form_key}"):
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
                # Increment key → next render gets a fresh blank form
                st.session_state.proj_form_key += 1
                st.rerun()

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
    sel_df       = faculty_df.loc[selected_idx].reset_index(drop=False)  # keep orig idx in col

    # Slice only the selected rows from the pre-normalised matrix
    faculty_embs = all_embs[selected_idx]   # shape (n_selected, 384)

    with st.spinner("Scoring faculty against projects (vectorised)…"):
        # Track cumulative similarity per faculty name for aggregation
        jury_total_scores = {row["name"]: 0.0 for _, row in sel_df.iterrows()}
        project_results   = []

        for p in st.session_state.projects:
            p_text = f"{p['title']} {p['description']}"
            p_emb  = get_project_embedding(p_text)
            if p_emb is None or np.linalg.norm(p_emb) == 0:
                continue

            # ⚡ Vectorised: one matrix multiply for all faculty at once
            top_indices, top_sims = top_k_scores(faculty_embs, p_emb, k=TOP_K)

            rankings = []
            for local_i, sim in zip(top_indices, top_sims):
                row = sel_df.iloc[local_i]
                rankings.append({
                    "name":        row["name"],
                    "url":         row.get("url", ""),
                    "designation": row.get("designation", ""),
                    "college":     row["college"],
                    "department":  row["department"],
                    "similarity":  round(float(sim), 4),
                })
                jury_total_scores[row["name"]] += float(sim)

            project_results.append({
                "project":  p["title"],
                "rankings": rankings,   # already sorted top-K
            })

    # Build a lookup: name → {url, designation} for quick access in results
    fac_url_lookup = {row["name"]: {"url": row.get("url", ""), "designation": row.get("designation", "")}
                      for _, row in sel_df.iterrows()}

    # ---- RESULTS ----
    st.divider()
    st.header("🏆 Matching Results")
    for pr in project_results:
        st.subheader(f"Project: {pr['project']}")
        for rank_idx, r in enumerate(pr["rankings"]):
            url   = r.get("url", "").strip()
            desig = r.get("designation", "").strip()
            # Render name as a hyperlink if a URL is available
            name_md = f"[{r['name']}]({url})" if url else r['name']
            desig_md = f" · *{desig}*" if desig else ""
            st.markdown(
                f"**#{rank_idx + 1} — {name_md}{desig_md} — {r['similarity']:.4f}**"
            )
            st.caption(
                f"College: {r['college']} | Dept: {r['department']} | "
                f"Cosine Similarity: {r['similarity']:.4f}"
            )
        st.write("")

    # ---- OVERALL RANKING (aggregate across projects) ----
    st.divider()
    st.header("🌟 Overall Faculty Ranking (Top 5)")
    num_projects = len(project_results)
    avg_scores   = sorted(
        [(name, total / num_projects) for name, total in jury_total_scores.items() if total > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    for rank_idx, (name, avg) in enumerate(avg_scores[:TOP_K]):
        meta  = fac_url_lookup.get(name, {})
        url   = meta.get("url", "").strip()
        desig = meta.get("designation", "").strip()
        name_md  = f"[{name}]({url})" if url else name
        desig_md = f" · *{desig}*" if desig else ""
        st.markdown(f"**{rank_idx + 1}. {name_md}{desig_md}** — Avg Similarity: {avg:.4f}")

    # ---- EXPORT ----
    st.divider()
    st.header("📥 Export Results")

    # Build score lookup: {faculty_name: {project_title: similarity}}
    score_lookup = {}
    for pr in project_results:
        for r in pr["rankings"]:
            score_lookup.setdefault(r["name"], {})[pr["project"]] = r["similarity"]

    project_titles = [pr["project"] for pr in project_results]

    # Faculty metadata lookup — include all available fields for the export
    # NOTE: use iterrows instead of set_index().to_dict() to safely handle
    # duplicate faculty names (set_index raises if the index is not unique).
    export_meta_cols = [c for c in ["college", "designation", "department",
                                     "research_area", "research_area_details", "url"]
                        if c in sel_df.columns]
    fac_meta = {}
    for _, row in sel_df[["name"] + export_meta_cols].iterrows():
        fac_meta[row["name"]] = row[export_meta_cols].to_dict()

    export_rows = []
    for rank_idx, (faculty_name, avg_score) in enumerate(avg_scores, start=1):
        meta = fac_meta.get(faculty_name, {})
        row  = {
            "Rank":             rank_idx,
            "Faculty Name":     faculty_name,
            "Designation":      meta.get("designation", ""),
            "College":          meta.get("college", ""),
            "Department":       meta.get("department", ""),
            "Research Details": meta.get("research_area_details", ""),
            "Profile URL":      meta.get("url", ""),
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
