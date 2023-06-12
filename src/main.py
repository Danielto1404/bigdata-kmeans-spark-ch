import argparse

from pyspark.sql import SparkSession

from kmeans import KmeansParams, PySparkKMeans
from preprocessing import DataTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--columns_json_path", default="../config/columns.json")
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--max_iter", default=5, type=int)
    parser.add_argument("--distance_measure", default="euclidean")
    parser.add_argument("--tol", default=1e-4, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--driver_cores", default=2, type=int),
    parser.add_argument("--driver_memory", default="4g"),
    parser.add_argument("--executor_memory", default="10g"),
    args = parser.parse_args()

    print("Initializing SparkSession...")
    app = SparkSession.builder \
        .appName("openfood-kmeans") \
        .master("local[*]") \
        .config("spark.driver.cores", args.driver_cores) \
        .config("spark.driver.memory", args.driver_memory) \
        .config("spark.executor.memory", args.executor_memory) \
        .getOrCreate()

    print("Loading data...")
    df = DataTransformer(
        df_path=args.data_path,
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
