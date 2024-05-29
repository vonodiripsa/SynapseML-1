# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Union

import model_navigator as nav
import mlflow
import numpy as np
import pyspark.sql.functions as F
import torch
from databricks.sdk.runtime import *
from numpy import ndarray
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, expr, lit, monotonically_increasing_id,
                                   struct, trim)
from pyspark.sql.types import (ArrayType, FloatType, IntegerType, StringType,
                               StructField, StructType)
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from spark_rapids_ml.knn import (ApproximateNearestNeighbors,
                                 ApproximateNearestNeighborsModel,
                                 NearestNeighbors)
from stnavigator import SentenceTransformerNavigator
from synapse.ml.core.platform import find_secret
from synapse.ml.nn import *
from synapse.ml.services.openai import OpenAIEmbedding
from torch import Tensor
from tqdm import trange

# Define the schema for query DataFrame
schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("query", StringType(), nullable=False),
    StructField("embeddings", ArrayType(FloatType(), containsNull=False), nullable=False)
])

# Convert seconds to hours:minutes:seconds
def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    if hours == 0:
        if minutes == 0:
            return f"{seconds}"
        else:
            return f"{int(minutes)}m : {int(sec)}s ({seconds})"
    else:
        return f"{int(hours)}h : {int(minutes)} : {int(sec)} ({seconds})"    

# Read data from file_path and generate syntetic data to have 1 Million rows 
def generate_1M_data(spark, file_path):
    # Load the CSV file with specified options
    df = spark.read.options(inferSchema="True", delimiter=",", header=True).csv(file_path)
    df = df.withColumn("combined", F.format_string("Title: %s; Content: %s", F.trim(df.Summary), F.trim(df.Text)))

    # Cross-join the DataFrame with itself to create n x n pairs for string concatenation (syntetic data)
    cross_joined_df = df.crossJoin(df.withColumnRenamed("combined", "combined_2"))

    # Create a new column 'result_vector' by concatenating the two source vectors
    tmp_df = cross_joined_df.withColumn("result_vector", F.concat(F.col("combined"), F.lit(". \n"), F.col("combined_2")))

    # Select only the necessary columns and show the result
    tmp_df = tmp_df.select("result_vector")
    df = tmp_df.withColumnRenamed("result_vector", "combined").withColumn("id", monotonically_increasing_id())

    return df   

# Print benchmark results for OpenAI embeddings and Ball Tree KNN running on CPU
def bencmarkCPU(df, lim, key, deployment_name_embeddings, service_name):
    df = df.limit(lim).repartition(10).cache()
    print(f"Scale {lim} rows")
    # Embeddings
    start_time_emb = time.time()

    embedding = (
            OpenAIEmbedding()
            .setSubscriptionKey(key)
            .setDeploymentName(deployment_name_embeddings)
            .setCustomServiceName(service_name)
            .setTextCol("combined")
            .setErrorCol("error")
            .setOutputCol("embeddings")
        )

    all_embeddings = embedding.transform(df).cache()
    all_embeddings.collect()

    # Record end time
    end_time_emb = time.time()

    # Calculate and print the duration
    duration_emb = end_time_emb - start_time_emb  
    duration_emb_decimal = float(Decimal(duration_emb).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))

    # KNN
    start_time_knn = time.time()

    knn = (
        KNN()
        .setFeaturesCol("embeddings")
        .setValuesCol("id")
        .setOutputCol("output")
        .setK(10)
    )

    knn_index = knn.fit(all_embeddings)

    query_df = (
        spark.createDataFrame(
            [
                (
                    0,
                    "desserts",
                ),
                (
                    1,
                    "disgusting",
                ),
            ]
        )
        .toDF("id", "query")
        .withColumn("id", F.col("id").cast("long"))
    )

    embedding_query = (
        OpenAIEmbedding()
        .setSubscriptionKey(key)
        .setDeploymentName(deployment_name_embeddings)
        .setCustomServiceName(service_name)
        .setTextCol("query")
        .setErrorCol("error")
        .setOutputCol("embeddings")
    )

    query_embeddings = embedding_query.transform(query_df).cache()

    df_matches = knn_index.transform(query_embeddings)

    df_matches.collect()

    end_time_knn = time.time()

    # Calculate and print the duration
    duration_knn = end_time_knn - start_time_knn
    duration_knn_decimal = float(Decimal(duration_knn).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))
    duration = duration_emb_decimal + duration_knn_decimal
    duration_decimal = float(Decimal(duration).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))    
    formatted_time = convert_seconds(duration_decimal)
        
    # Create output dataFrame
    data = [("Embeddings", duration_emb_decimal), ("KNN", duration_knn_decimal), ("All stages", formatted_time)]
    df_out = spark.createDataFrame(data, ["Benchmark Name ", "Duration (sec)"])
    df_out.show(truncate=False, vertical=False)

# Print benchmark results for TRT E5 embeddings and Rapids ANN running on GPU
def benchmarkGPU(df, lim, encode):
    df = df.limit(lim).repartition(10).cache()
    print(f"Scale {lim} rows")

    # Embeddings
    start_time_emb = time.time()

    all_embeddings = df.withColumn("embeddings", encode(struct("combined"))).cache()   
    all_embeddings.collect()

    end_time_emb = time.time()

    # Calculate and print the duration
    duration_emb = end_time_emb - start_time_emb
    duration_emb_decimal = float(Decimal(duration_emb).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))    

    # KNN
    start_time_knn = time.time()

    if 'model' not in globals():
        model = SentenceTransformer("intfloat/e5-large-v2").eval()

    # Generate embeddings
    with torch.no_grad():
        query = ["desserts", "disgusting"]
        embeddings = [embedding.tolist() for embedding in model.encode(query)]

    # Prepare data including IDs
    data_with_ids = [(1, query[0], embeddings[0]), (2, query[1], embeddings[1])]

    # Define the schema for the DataFrame
    schema = StructType([
        StructField("id", IntegerType(), nullable=False),
        StructField("query", StringType(), nullable=False),
        StructField("embeddings", ArrayType(FloatType(), containsNull=False), nullable=False)
    ])

    # Create a DataFrame using the data with IDs and the schema
    query_embeddings = spark.createDataFrame(data=data_with_ids, schema=schema).cache()        

    rapids_knn = ApproximateNearestNeighbors(k=10)
    rapids_knn.setInputCol("embeddings").setIdCol("id")

    rapids_knn_model = rapids_knn.fit(all_embeddings.select("id", "embeddings"))
    (_, _, knn_df) = rapids_knn_model.kneighbors(query_embeddings.select("id", "embeddings"))

    knn_df.collect()

    end_time_knn = time.time()

    # Calculate and print the duration
    duration_knn = end_time_knn - start_time_knn
    duration_knn_decimal = float(Decimal(duration_knn).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))
    duration = duration_emb_decimal + duration_knn_decimal
    duration_decimal = float(Decimal(duration).quantize(Decimal('1.00'), rounding=ROUND_HALF_UP))    
    formatted_time = convert_seconds(duration_decimal)
    
    # Create output dataFrame
    data = [("Embeddings", duration_emb_decimal), ("KNN", duration_knn_decimal), ("All stages", formatted_time)]
    df_out = spark.createDataFrame(data, ["Benchmark Name ", "Duration (sec)"])
    df_out.show(truncate=False, vertical=False)
