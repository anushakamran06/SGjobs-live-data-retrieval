#!/usr/bin/env bash
set -euo pipefail   
: "${DATABASE_URL:?DATABASE_URL is not set}"   

CSV_URL="https://raw.githubusercontent.com/opengovsg/careersgovsg-jobs-data/main/data/job-listings.csv"

mkdir -p db
curl -fsSL "$CSV_URL" -o db/job-listings.csv   # -f fail

if [ ! -s db/job-listings.csv ]; then
  echo "ERROR: CSV fetch failed or invalid"
  exit 1
fi

for f in sql/00_schema.sql sql/01_load.sql sql/02_clean.sql; do
  psql -v ON_ERROR_STOP=1 -f "$f" "$DATABASE_URL"   # ON_ERROR_STOP aborts on first SQL error
done
