# TO DO
# - Connect to google big query via command line entries (Project ID)
# - Partial antibiotic cohort query (vitals, sepsis indicator, labs, all cause mortality)
# - SQL query to dataframe
# - dataframe to pytorch dataset
# - (optional) dataframe to csv files / loader function

#import os
from google.cloud import bigquery
import pandas as pd
from pandas.io import gbq

# Connect to Google Big Query

# Query sepsis cohort
project = "mimic-380019"

query = """
    SELECT *
    FROM physionet-data.mimiciv_derived.vitalsign
    LIMIT 50 
    """

data = gbq.read_gbq(query, project_id=project)
print(data.head(10))

# Optional save to data directory

# Create PyTorch dataset from queried data
