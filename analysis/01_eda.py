import os
import pandas as pd
from sqlalchemy import create_engine 
from dotenv import load_dotenv
import seaborn as sns 
import matplotlib.pyplot as plt
 
load_dotenv()  
engine = create_engine(os.environ["DATABASE_URL"]) 
 
df = pd.read_sql("SELECT * FROM v_jobs_clean", engine)
print(df.shape)
print(df.head())
print(df.info())
 
 
#field coverage audit (field is a grouping dim for the trends dashboard, so we want to know how complete it is)
field_total_sum = len(df)
field_labelled_sum = df['field'].notna().sum()
field_null_sum   = df['field'].isna().sum()
print(f"total: {field_total_sum}, labeled: {field_labelled_sum}, null: {field_null_sum})")
 
#hiring volume by field
import matplotlib.pyplot as plt
field_labelled = df[df['field'].notna()]
 
field_labelled['field'].value_counts(dropna=False).plot.barh( #want to see null in plot
    figsize=(10, 6),
    title='Postings by Field',
    xlabel='Number of postings',
    ylabel='Field',
    color='steelblue'
)
plt.tight_layout()
 
#histogram and kde
desc = df[df['job_description'].notna()].copy() #drop null descriptions in filtering and make it a completely diff df dont mess with main df
desc['desc_len']  = desc['job_description'].str.len()          # characters
desc['desc_words'] = desc['job_description'].str.split().str.len()  # tokens: embeddings + skill matching care about words, not chars
 
print(desc['desc_len'].describe())     # min/max/median (chars)
print(desc['desc_words'].describe())   # min/max/median (words)
 
plt.figure(figsize=(10, 6))
sns.histplot(desc['desc_len'], kde=True, color='steelblue')
plt.title('Job Description Length Distribution')
plt.xlabel('Characters per description')
plt.ylabel('Number of postings')
plt.xlim(0, desc['desc_len'].quantile(0.99)) #or hardcode
 
 
#checking html tags
desc_has_html = desc['job_description'].str.contains(r'<[^>]+>',regex=True)
print(f"Percentage rows with html: {desc_has_html.sum()} / {len(desc)} // {100}")
#since its pretty significant, we gotta strip away before skill extraction (02) + embeddings (03). markup is noise for both.
 
# TODO: Once 02_skill_extraction.py writes job_skills, report what fraction of jobs matched >=1 skill. low coverage => dictionary too small.
