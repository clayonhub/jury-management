import streamlit as st
import os
import numpy as np
import math
import re
import json
import io
import pandas as pd
# No external APIs are used, so no dotenv is needed.

DATA_FILE = "data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"juries": [], "projects": []}

def save_data():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "juries": st.session_state.juries,
            "projects": st.session_state.projects
        }, f)

# ==========================================
# APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Jury Matching System", page_icon="⚖️", layout="wide")

# ==========================================
# STATE MANAGEMENT
# ==========================================
loaded_data = load_data()
if 'juries' not in st.session_state:
    st.session_state.juries = loaded_data.get("juries", [])
if 'projects' not in st.session_state:
    st.session_state.projects = loaded_data.get("projects", [])
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = {}

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the model once globally to save RAM and time
@st.cache_resource
def get_hf_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

hf_model = get_hf_model()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_embedding(text):
    """Fetch embeddings using local HuggingFace model and cache them."""
    if not text.strip():
        return np.zeros(384)
    if text in st.session_state.embeddings:
        return st.session_state.embeddings[text]
        
    try:
        emb = np.array(hf_model.embed_query(text))
        st.session_state.embeddings[text] = emb
        return emb
    except Exception as e:
        st.error(f"Error fetching embedding via HuggingFace: {e}")
        return None

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors manually."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def extract_words(text):
    """Extract words for lexical matching."""
    return re.findall(r'\b\w+\b', text.lower())

# ==========================================
# UI LAYOUT
# ==========================================
st.title("Jury Matching System ⚖️")
st.markdown("A fairness-aware matching system computing optimal jury-project alignment.")

# --- SECTION 1: ADD JURY ---
st.header("Section 1 — Add Jury")
with st.container():
    with st.form("add_jury_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            degree = st.text_input("Degree")
        with col2:
            designation = st.text_input("Designation")
            research_areas = st.text_input("Research Areas (comma-separated)")
        
        research_details = st.text_area("Research Details", height=100)
        submit_jury = st.form_submit_button("Add Jury")
        
        if submit_jury:
            if not all([name, degree, designation, research_areas, research_details]):
                st.error("All fields are required!")
            else:
                st.session_state.juries.append({
                    "name": name.strip(),
                    "degree": degree.strip(),
                    "designation": designation.strip(),
                    "research_areas": research_areas.strip(),
                    "research_details": research_details.strip()
                })
                save_data()
                st.success(f"Added Jury: {name}")

if st.session_state.juries:
    st.subheader(f"Added Juries ({len(st.session_state.juries)})")
    for idx, j in enumerate(st.session_state.juries):
        with st.expander(f"{j['name']} — {j['designation']}", expanded=False):
            st.markdown(f"**Degree:** {j['degree']}")
            st.markdown(f"**Research Areas:** {j['research_areas']}")
            st.markdown(f"**Research Details:** {j['research_details']}")
            if st.button(f"Delete {j['name']}", key=f"del_jury_{idx}"):
                st.session_state.juries.pop(idx)
                save_data()
                st.rerun()

st.divider()

# --- SECTION 2: ADD PROJECT ---
st.header("Section 2 — Add Project")
with st.container():
    with st.form("add_project_form"):
        p_title = st.text_input("Project Title")
        p_desc = st.text_area("Project Description (~200 words)", height=150)
        submit_proj = st.form_submit_button("Add Project")
        
        if submit_proj:
            if len(st.session_state.projects) >= 15:
                st.error("Maximum of 15 projects allowed.")
            elif not all([p_title, p_desc]):
                st.error("All project fields are required!")
            else:
                st.session_state.projects.append({
                    "title": p_title.strip(),
                    "description": p_desc.strip()
                })
                save_data()
                st.success(f"Added Project: {p_title}")

if st.session_state.projects:
    st.subheader(f"Project List ({len(st.session_state.projects)}/15)")
    for idx, p in enumerate(st.session_state.projects):
        with st.expander(f"{p['title']}", expanded=False):
            st.markdown(f"**Description:** {p['description']}")
            if st.button(f"Delete Project", key=f"del_proj_{idx}"):
                st.session_state.projects.pop(idx)
                save_data()
                st.rerun()

st.divider()

# --- SECTION 3: RUN MATCHING ---
st.header("Section 3 — Run Jury Matching")
if st.button("Run Matching", type="primary"):
    if not st.session_state.juries:
        st.warning("Please add at least one jury.")
        st.stop()
    if not st.session_state.projects:
        st.warning("Please add at least one project.")
        st.stop()
        
    with st.spinner("Computing embeddings and evaluating alignment scores..."):
        # Pre-process juries
        jury_metrics = []
        for j in st.session_state.juries:
            j_text = f"{j['degree']} {j['designation']} {j['research_areas']} {j['research_details']}"
            emb = get_embedding(j_text)
            
            # Prepare dimensions for formulas
            areas = [x.strip().lower() for x in j['research_areas'].split(',')]
            areas = [x for x in areas if x]
            
            j_details_words = extract_words(j['research_details'])
            j_details_set = set(j_details_words)
            
            j_len = len(extract_words(j_text))
            c_score = 0.9 if len(j_details_words) < 30 else 1.0
            
            jury_metrics.append({
                "name": j["name"],
                "emb": emb,
                "areas": areas,
                "j_details_set": j_details_set,
                "j_len": j_len,
                "c_score": c_score
            })
            
        jury_total_scores = {j["name"]: 0.0 for j in jury_metrics}
        project_results = []
        
        # Match each project
        for p in st.session_state.projects:
            p_text = f"{p['title']} {p['description']}"
            p_emb = get_embedding(p_text)
            if p_emb is None:
                continue
                
            p_text_lower = p_text.lower()
            p_words = extract_words(p_text)
            p_words_len = len(p_words)
            
            scores = []
            for jm in jury_metrics:
                # 1. Semantic Similarity (S)
                s_sem = cosine_similarity(jm['emb'], p_emb) if jm['emb'] is not None else 0.0
                    
                # 2. Specialization Ratio (R)
                matched_kws = sum(1 for a in jm['areas'] if a in p_text_lower)
                r_score = (matched_kws / len(jm['areas'])) if len(jm['areas']) > 0 else 0.0
                
                # 3. Keyword Score (B)
                overlap = sum(1 for w in p_words if w in jm['j_details_set'])
                b_score = (overlap / p_words_len) if p_words_len > 0 else 0.0
                
                # 4. Length Normalization (L)
                l_score = 1.0 / math.log(1 + jm['j_len']) if jm['j_len'] > 0 else 0.0
                
                # 5. Confidence (C)
                c_score = jm['c_score']
                
                # FINAL FORMULA
                final_score = (0.55 * s_sem) + (0.20 * r_score) + (0.15 * b_score) + (0.05 * l_score) + (0.05 * c_score)
                final_score = round(final_score, 3)
                
                scores.append({
                    "name": jm["name"],
                    "s_sem": s_sem,
                    "r_score": r_score,
                    "b_score": b_score,
                    "l_score": l_score,
                    "c_score": c_score,
                    "final_score": final_score
                })
                jury_total_scores[jm["name"]] += final_score

            scores.sort(key=lambda x: x["final_score"], reverse=True)
            project_results.append({
                "project": p["title"],
                "rankings": scores
            })
            
    # Display Breakdown
    st.divider()
    st.header("🏆 Matching Results")
    for pr in project_results:
        st.subheader(f"Project: {pr['project']}")
        for idx, r in enumerate(pr["rankings"]):
            with st.container():
                st.markdown(f"**Rank {idx+1} — {r['name']} — {r['final_score']:.3f}**")
                st.caption(
                    f"Breakdown: Ssemantic: {r['s_sem']:.3f} | "
                    f"R: {r['r_score']:.3f} | "
                    f"B: {r['b_score']:.3f} | "
                    f"L: {r['l_score']:.3f} | "
                    f"C: {r['c_score']:.3f}"
                )
        st.write("")
        
    st.divider()
    st.header("🌟 Overall Jury Ranking")
    num_projects = len(st.session_state.projects)
    avg_scores = [
        (name, total / num_projects if num_projects > 0 else 0.0)
        for name, total in jury_total_scores.items()
    ]
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    
    for idx, (name, avg) in enumerate(avg_scores[:5]):
        st.markdown(f"**{idx+1}. {name}** — Avg Score: {avg:.3f}")

    # ==========================================
    # EXPORT RESULTS
    # ==========================================
    st.divider()
    st.header("📥 Export Results")

    # Build a lookup: {jury_name: {project_title: final_score}}
    score_lookup = {}
    for pr in project_results:
        proj_title = pr["project"]
        for r in pr["rankings"]:
            score_lookup.setdefault(r["name"], {})[proj_title] = r["final_score"]

    # Project column names (in order)
    project_titles = [pr["project"] for pr in project_results]

    # Build rows: one per jury, sorted by overall rank
    export_rows = []
    for rank_idx, (jury_name, avg_score) in enumerate(avg_scores, start=1):
        row = {
            "Rank": rank_idx,
            "Jury Name": jury_name,
        }
        for pt in project_titles:
            row[pt] = round(score_lookup.get(jury_name, {}).get(pt, 0.0), 4)
        row["Overall Avg Score"] = round(avg_score, 4)
        export_rows.append(row)

    export_df = pd.DataFrame(export_rows)

    st.dataframe(export_df, use_container_width=True)

    # --- Excel Download ---
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Jury Results")
    excel_buffer.seek(0)
    st.download_button(
        label="⬇️ Download as Excel (.xlsx)",
        data=excel_buffer,
        file_name="jury_matching_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # --- CSV Download ---
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV (.csv)",
        data=csv_data,
        file_name="jury_matching_results.csv",
        mime="text/csv",
    )

# ==========================================
# DEPLOYMENT INSTRUCTIONS
# ==========================================
# How to run locally:
# 1. Install dependencies:
#    pip install -r requirements.txt
# 2. Start the application:
#    streamlit run app.py
#
# How to deploy to Streamlit Cloud (Free & Easy):
# 1. Create a GitHub account and install Git on your computer if you haven't already.
# 2. Push this folder (containing app.py and requirements.txt) to a new public GitHub repository.
# 3. Go to https://share.streamlit.io/, log in with GitHub, and click "New app".
# 4. Select your repository, the main branch, and type "app.py" as your Main file path.
# 5. Click "Deploy". Streamlit Cloud will automatically install the libraries from requirements.txt and host your app!
# Note: Because the app uses a local model, the first time it boots on the cloud it will spend a few seconds downloading the 80MB model before the app appears.
# ==========================================
