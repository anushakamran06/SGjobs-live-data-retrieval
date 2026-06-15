#using job_description to run miniLM model

import os 
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # downloads ~80MB once, then cached locally

load_dotenv()
engine = create_engine(os.environ['DATABASE_URL'])

df = pd.read_sql('SELECT job_id, job_title, job_description FROM jobs WHERE job_description IS NOT NULL', engine)
print(df.shape)
print(df.head())

#remove tags
df["job_description"] = df["job_description"].str.replace(r"<[^>]+>", " ", regex=True)

#converting text in description into vectors. enabled pgvector extension back in 00_schema.sql. hopefully stays consistent with upsert logic.
df['text'] = (df['job_title'].fillna("") + '' + df['job_description'].fillna(''))

emb = model.encode(
    df["text"].tolist(),
    normalize_embeddings=True,   
    batch_size=32,
    show_progress_bar=True,
)
print(emb.shape)


df["embedding"] = ["[" + ",".join(map(str, v)) + "]" for v in emb]

with engine.begin() as conn:
    conn.execute(
    text("""
        INSERT INTO job_embeddings (job_id, embedding)
        VALUES (:job_id, :embedding)
        ON CONFLICT (job_id) DO UPDATE SET embedding = EXCLUDED.embedding
    """),
    df[["job_id", "embedding"]].to_dict("records"),
)
    
#testing 
q = pd.read_sql(text("""
    WITH target AS (SELECT embedding FROM job_embeddings WHERE job_id = :id)
    SELECT j.job_id, j.job_title,
           1 - (e.embedding <=> (SELECT embedding FROM target)) AS cosine_sim
    FROM job_embeddings e
    JOIN jobs j USING (job_id), target
    WHERE e.job_id <> :id
    ORDER BY e.embedding <=> (SELECT embedding FROM target)
    LIMIT 5
"""), engine, params={"id": "PUT_A_REAL_JOB_ID"})
print(q)
