# Data Setup

This directory requires OULAD data files. They are not included in the repo due to size.

## Download Instructions

1. Go to https://analyse.kmi.open.ac.uk/open_dataset
2. Download the dataset
3. Run the preprocessing to generate the cohort CSVs, or copy them from your local Kappa-Education repo:
   - `data_AAA_2014J_pass.csv`
   - `data_AAA_2014J_fail.csv`
   - `data_AAA_2014J_distinction.csv`
   - `data_AAA_2014J_withdrawn.csv`

Each CSV should have columns: `date`, plus the 9 activity channels:
`clicks_dataplus`, `clicks_forumng`, `clicks_glossary`, `clicks_homepage`,
`clicks_oucollaborate`, `clicks_oucontent`, `clicks_resource`, `clicks_subpage`, `clicks_url`
