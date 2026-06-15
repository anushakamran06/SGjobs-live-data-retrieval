import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import CrossEncoder
 
load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
 
#bi-encoder then cross-encoder.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
 
HTML = r"<[^>]+>"   
 
#KNN, top-20 by cosine similarity
def retrieve(job_id, k=20):
    df = pd.read_sql(text("""
        WITH target AS (SELECT embedding AS vec FROM job_embeddings WHERE job_id = :id)
        SELECT j.job_id, j.job_title, j.field, j.job_description,
               1 - (e.embedding <=> target.vec) AS cosine_sim
        FROM job_embeddings e
        JOIN jobs j USING (job_id)
        CROSS JOIN target
        WHERE e.job_id <> :id
        ORDER BY e.embedding <=> target.vec
        LIMIT :k
    """), engine, params={"id": job_id, "k": k})
    df["clean_text"] = (df.job_title.fillna("") + " " + df.job_description.fillna("")) \
        .str.replace(HTML, " ", regex=True)
    return df

def query_text(job_id):
    row = pd.read_sql(
        text("SELECT job_title, job_description FROM jobs WHERE job_id = :id"),
        engine, params={"id": job_id},
    ).iloc[0]
    return clean(row.job_title) + " " + clean(row.job_description)

 #cross encoder reranker
def recommend(job_id, k_retrieve=20, k_final=5):
    cands = retrieve(job_id, k_retrieve)
    qtext = query_text(job_id)
    cands["ce_score"] = reranker.predict([(qtext, c) for c in cands.clean_text])  # rerank
    out = cands.sort_values("ce_score", ascending=False).head(k_final)
    return out[["job_id", "job_title", "field", "cosine_sim", "ce_score"]]
#Personal note: u could add a filter before doing reranking, getting rid of jobs that have 0 overlapping skills but there r alot of edge cases so put on hold. 
