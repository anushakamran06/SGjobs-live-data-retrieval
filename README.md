# job-listings-live-data: live ETL & semantic retrieval

A live, incremental ETL pipeline over a daily-mutating feed of ~2.6k Singapore
government job postings
([opengovsg/careersgovsg-jobs-data](https://github.com/opengovsg/careersgovsg-jobs-data)),
landing in PostgreSQL / Supabase and auto-refreshed via GitHub Actions built around
idempotent upserts and disappearance detection, not a one-off Kaggle-style load.

On the same store, a two-stage matching layer pairs exact (lexical) match using spaCy
lexicon-based skill extraction for a first tier. Second tier carries out semantic match: sentence-transformers
embeddings indexed in pgvector for job-to-job retrieval. The vector-search half is the
same retrieval primitive that underpins RAG, used here standalone for recommendation
(no generation step), kept deliberately lightweight at this scale.

**Techniques:** EDA · lexicon-based NLP (spaCy PhraseMatcher) · sentence-transformers
(MiniLM-L6, 384-d) · pgvector cosine KNN retrieval · exact-match vs semantic-match
skill matching · bi-encoder vs cross-encoder reranking, evaluated against a
field-overlap precision@k proxy → reranking dropped (net-negative on precision and
added per-query latency view test.py for more detail), retaining a lightweight bi-encoder retrieval system.

## Repository Layout

```
.
├── .github/workflows/refresh.yml   # daily cron: ETL + snapshot commit
├── scripts/refresh.sh              # download CSV, run SQL
├── sql/
│   ├── 00_schema.sql               # DDL, indexes, pgvector, view
│   ├── 01_load.sql                 # raw CSV
│   └── 02_clean.sql                # coerce, dedup, upsert, disappearance → jobs_latest
├── analysis/
│   ├── 01_eda.py                   # EDA, coverage audit, distributions
│   ├── 02_skill_extraction.py      # spaCy PhraseMatcher → job_skills
│   ├── 03_embeddings.py            # MiniLM → job_embeddings
│   └── 04_semantic_search.py       # pgvector KNN retrieval
├── skills/skills_dict.txt          # skill lexicon
├── data/jobs_latest.csv            # committed daily snapshot
├── requirements.txt                # dependencies (new)
├── tests.py                        # testing ideas
└── LICENSE
```

## Run the ETL (build + load + clean)

```bash
bash scripts/refresh.sh
```

Or run the SQL manually in order:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/00_schema.sql
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/01_load.sql   # expects db/job-listings.csv
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/02_clean.sql
```

## Build the semantic layer

```bash
python analysis/01_eda.py               # optional: EDA + figures
python analysis/02_skill_extraction.py  # → job_skills
python analysis/03_embeddings.py        # → job_embeddings
```

## Query recommendations

`04_semantic_search.py` is functions-only and the filename has a numeric prefix, so it
can't be imported directly. Run it interactively:

```bash
python -i analysis/04_semantic_search.py
>>> recommend("<job_id>")     # top-k by cosine (bi-encoder)
>>> retrieve("<job_id>", 20)  # raw cosine KNN
```

## Future Plans

`analysis/05_survival_analysis.py`. Planned to do a Kaplan-Meier + Cox
on `removed_at − first_seen` (time-on-platform). Blocked until enough `removed_at` events
accumulate. Also included rationale behind the scrap in tests.py.
