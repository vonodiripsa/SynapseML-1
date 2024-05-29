# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1: Prepare Environment
# MAGIC
# MAGIC Import libraries and set logging

# COMMAND ----------

import logging

import mlflow
import model_navigator as nav
import torch
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer

from stnavigator import SentenceTransformerNavigator
from utility import benchmarkGPU, generate_1M_data

# Adjust logging levels
logging.getLogger('py4j').setLevel(logging.ERROR)
logging.getLogger("sentence_transformers")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
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
# MAGIC ## Step 3: Set prediction model
# MAGIC
# MAGIC Set encoding using NVIDIA SentenceTransformerNavigator with TRT acceleration using Model Navigator

# COMMAND ----------

def predict_batch_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformerNavigator("intfloat/e5-large-v2").eval()
    model = nav.Module(model, name="e5-large-v2")
    model = model.to(device)
    nav.load_optimized()

    def predict(inputs):
        with torch.no_grad():
            output = model.encode(inputs.tolist(), convert_to_tensor=False, show_progress_bar=True)
        return output
    return predict

encode = predict_batch_udf(predict_batch_fn, return_type=ArrayType(FloatType()), batch_size=10) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Benchmark different scale of input data
# MAGIC
# MAGIC We will print duration of different stages of the experiment (SentanseTransformer NVIDIA TensorRT embedding with Rapids KNN)
# MAGIC
# MAGIC For example: [100, 1000, 10000, 100000] rows of text data

# COMMAND ----------

specified_values = [100, 1000, 10000, 100000]

print(f"********  Test TRT E5 with IVFlat KNN  ************")
print()

for lim in specified_values:
  benchmarkGPU(df_max, lim, encode)
