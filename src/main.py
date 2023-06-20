import argparse

from pyspark.sql import SparkSession

import clickhouse
from kmeans import KmeansParams, PySparkKMeans
from preprocessing import DataTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_name", default="datasets.openfood")
    parser.add_argument("--columns_json_path", default="config/columns.json")
    parser.add_argument("--save_path", default="models/kmeans")
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--max_iter", default=8, type=int)
    parser.add_argument("--distance_measure", default="euclidean")
    parser.add_argument("--tol", default=1e-4, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--driver_cores", default=4, type=int),
    parser.add_argument("--driver_memory", default="8g"),
    parser.add_argument("--executor_memory", default="16g"),
    parser.add_argument("--clickhouse_jdbc_path", default="jars/clickhouse-jdbc-0.4.6-all.jar")
    args = parser.parse_args()

    print("Initializing SparkSession...")
    app = SparkSession.builder \
        .appName("openfood-kmeans") \
        .master("local[*]") \
        .config("spark.driver.cores", args.driver_cores) \
        .config("spark.driver.memory", args.driver_memory) \
        .config("spark.executor.memory", args.executor_memory) \
        .config("spark.driver.extraClassPath", args.clickhouse_jdbc_path) \
        .getOrCreate()

    print("Loading data...")
    df = DataTransformer(
        table_name=args.table_name,
        columns_json_path=args.columns_json_path
    ).load(app).transform()

    params = KmeansParams(
        k=args.k,
        max_iter=args.max_iter,
        distance_measure=args.distance_measure,
        tol=args.tol,
        seed=args.seed
    )

    print("Training model...")
    model = PySparkKMeans(params).fit(df)
    model.save(args.save_path)

    predictions = model.predict(df).select("prediction")
    clickhouse.write(predictions, "datasets.openfood_predictions")
