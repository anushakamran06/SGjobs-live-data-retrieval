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
    df = pd.read_sql(
        text("SELECT job_title, job_description FROM jobs WHERE job_id = :id"),
        engine, params={"id": job_id},
    )
    return (df.job_title.fillna("") + " " + df.job_description.fillna("")) \
        .str.replace(HTML, " ", regex=True).iloc[0]
 
 #cross encoder reranker
def recommend(job_id, k_retrieve=20, k_final=5):
    cands = retrieve(job_id, k_retrieve)
    qtext = query_text(job_id)
    cands["ce_score"] = reranker.predict([(qtext, c) for c in cands.clean_text])  # rerank
    out = cands.sort_values("ce_score", ascending=False).head(k_final)
    return out[["job_id", "job_title", "field", "cosine_sim", "ce_score"]]

#testing if recommender works
def precision_at_k(job_id, k=5):
    qfield = pd.read_sql(text("SELECT field FROM jobs WHERE job_id = :id"),
                         engine, params={"id": job_id}).field.iloc[0]
    if pd.isna(qfield):
        return None
    cands = retrieve(job_id, k=20)
    qtext = query_text(job_id)
    cands["ce_score"] = reranker.predict([(qtext, c) for c in cands.clean_text])
    bi = cands.head(k)                                             
    ce = cands.sort_values("ce_score", ascending=False).head(k)   # reranked order
    return (bi.field == qfield).mean(), (ce.field == qfield).mean()
 
 
if __name__ == "__main__":
    sample = pd.read_sql(text("""
        SELECT job_id FROM jobs
        WHERE field IS NOT NULL AND job_id IN (SELECT job_id FROM job_embeddings)
        ORDER BY random() LIMIT 50
    """), engine).job_id
    if sample.empty:
        raise SystemExit("no jobs with embeddings + field — run 03_embeddings.py "
                         "and confirm job_embeddings is populated")
 
    demo = sample.iloc[0]

    cands = retrieve(demo)
    print(f"[retrieve] bi-encoder top candidates for {demo} (by cosine):")
    print(cands[["job_id", "job_title", "cosine_sim"]].head().to_string(index=False))
 
    print(f"\n[query_text] raw text of the query job fed to cross-encoder (first 150 chars):")
    print(query_text(demo)[:150])
 
    print(f"\n[recommend] final reranked top 5:")
    print(recommend_reranked(demo).to_string(index=False))
 
#TEST comparison between CE and cosine_sim values (using precision@k as metric)
# PROXY relevance: candidate shares the query job's `field`. NOT ground truth
# (field is coarse + partly circular) -> sanity signal. Point: does rerank lift it?
    scores = [s for s in (precision_at_k(j, k=5) for j in sample) if s]
    bi_mean = np.mean([s[0] for s in scores])
    ce_mean = np.mean([s[1] for s in scores])
    print(f"\nproxy precision@5 (same-field) over {len(scores)} jobs:")
    print(f"  bi-encoder only : {bi_mean:.3f}")
    print(f"  + cross-encoder : {ce_mean:.3f}")
 
# RESULT proxy precision@5 (same-field) over 50 jobs:
#   bi-encoder only : 0.500
#   + cross-encoder : 0.484 all that work for ce to worsen the results game over



