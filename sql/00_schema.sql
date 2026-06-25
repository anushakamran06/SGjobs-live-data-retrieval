--CREATE TABLE not CTAS because we want to preserve indexes and constraints across refreshes. Dont use a query.
CREATE TABLE IF NOT EXISTS jobs (
    job_id               text PRIMARY KEY,
    job_title            text,
    agency               text,
    agency_description   text,
    platform             text,
    start_date           timestamptz,
    closing_date         timestamptz,
    employment_type      text,
    field                text,
    functional_area      text,
    industry             text,
    category             text,
    education_code       text,
    exp_years_min        int,
    exp_years_max        int,
    is_new               boolean,
    job_description      text,
    job_responsibilities text,
    job_requirements     text
);

ALTER TABLE jobs
ADD COLUMN IF NOT EXISTS first_seen timestamptz, 
ADD COLUMN IF NOT EXISTS last_seen timestamptz, 
ADD COLUMN IF NOT EXISTS removed_at timestamptz;

-- Creating job_skills. Populated by 02_skill_extraction.py.
CREATE TABLE IF NOT EXISTS job_skills (
    job_id text REFERENCES jobs(job_id) ON DELETE CASCADE,  
    skill  text NOT NULL,
    PRIMARY KEY (job_id, skill)   
);

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id    text PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_skills_skill ON job_skills (skill);  

CREATE INDEX IF NOT EXISTS idx_jobs_field    ON jobs (field);
CREATE INDEX IF NOT EXISTS idx_jobs_agency   ON jobs (agency);
CREATE INDEX IF NOT EXISTS idx_jobs_platform ON jobs (platform);
CREATE INDEX IF NOT EXISTS idx_jobs_closing  ON jobs (closing_date);
CREATE INDEX IF NOT EXISTS idx_jobs_desc_fts ON jobs
    USING gin (to_tsvector('english', coalesce(job_description, '')));

DROP VIEW IF EXISTS v_jobs_clean CASCADE;

CREATE VIEW v_jobs_clean AS
SELECT
    job_id, job_title, agency, platform, field, industry, employment_type,
    exp_years_min, exp_years_max, (exp_years_min + exp_years_max) / 2.0 AS exp_years_mid,
    start_date, closing_date, (closing_date IS NULL OR closing_date >= now()) AS is_open,
    first_seen, last_seen, removed_at,
    job_description
FROM jobs;
