# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1: Prepare Environment
# MAGIC
# MAGIC Import libraries and set logging

# COMMAND ----------

import logging
import mlflow
import warnings
from utility import bencmarkCPU, generate_1M_data 

# Adjust logging levels
warnings.filterwarnings("ignore", message="Imported version of grpc is 1.51.0")
logging.getLogger('py4j.clientserver').setLevel(logging.CRITICAL)
logging.getLogger('py4j').setLevel(logging.CRITICAL)
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load public data
# MAGIC
# MAGIC In this notebook we will explore a dataset of fine food reviews

# COMMAND ----------

df_max = generate_1M_data(spark, "wasbs://publicwasb@mmlspark.blob.core.windows.net/fine_food_reviews_1k.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Set your OpenAI credentials
# MAGIC
# MAGIC In this notebook we will get embeddings from OpenAI service

# COMMAND ----------

# Fill in the following lines with your service information
# Learn more about selecting which embedding model to choose: https://openai.com/blog/new-and-improved-embedding-model
service_name = "ams-oai"
deployment_name_embeddings = "text-embedding-ada-002"
key = "fa4f18225ee54a7f8c86645a7b5bbcd4" 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Benchmark different scale of input data
# MAGIC
# MAGIC We will print duration of different stages of the experiment (OpenAI embeddings with Ball Tree KNN)
# MAGIC
# MAGIC For example: [100, 1000, 10000, 100000] rows of text data

# COMMAND ----------

specified_values = [100, 1000, 10000, 100000]

print(f"********  Test OAI + BT KNN  ************")
print()

for lim in specified_values:
  bencmarkCPU(df_max, lim, key, deployment_name_embeddings, service_name)