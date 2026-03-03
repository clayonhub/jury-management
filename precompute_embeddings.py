"""
precompute_embeddings.py
========================
Run this ONCE locally before deploying app_faculty.py to Streamlit Cloud.
It reads all_institutions_faculty_data.csv, computes sentence embeddings for
every faculty row, and saves:
  - faculty_embeddings.npy      : float32 array of shape (N, 384)
  - faculty_embeddings_index.json : list of {name, college} for each row (same order)

Usage:
    python precompute_embeddings.py
"""

import json
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

CSV_PATH   = "all_institutions_faculty_data.csv"
EMB_PATH   = "faculty_embeddings.npy"
INDEX_PATH = "faculty_embeddings_index.json"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Keep only the columns we care about; drop rows with missing name or research data
USED_COLS = ["name", "college", "department", "research_area", "research_area_details"]
df = df[USED_COLS].copy()
df = df.dropna(subset=["name"])
df = df.fillna("")

print(f"  {len(df)} faculty rows loaded.")

# Build the text that will be embedded per faculty
def build_faculty_text(row):
    parts = [
        str(row["department"]).strip(),
        str(row["research_area"]).strip(),
        str(row["research_area_details"]).strip(),
    ]
    return " ".join(p for p in parts if p).strip()

texts = df.apply(build_faculty_text, axis=1).tolist()
print(f"  Sample text[0]: {texts[0][:120]!r}")

print("Loading HuggingFace model...")
model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

print(f"Computing embeddings in batches of {BATCH_SIZE}...")
all_embeddings = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    embs  = model.embed_documents(batch)
    all_embeddings.extend(embs)
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Processed {min(i + BATCH_SIZE, len(texts))} / {len(texts)}")

emb_array = np.array(all_embeddings, dtype=np.float32)

# L2-normalise now so the deployed app can use fast dot products (no per-query norm)
norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)
emb_array = emb_array / norms

np.save(EMB_PATH, emb_array)
print(f"Saved normalised embeddings → {EMB_PATH}  shape={emb_array.shape}")

# Save the index (name + college per row, same order as array)
index = df[["name", "college"]].to_dict(orient="records")
with open(INDEX_PATH, "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False)
print(f"Saved index     → {INDEX_PATH}  ({len(index)} entries)")

print("\n✅ Done! You can now deploy app_faculty.py.")
