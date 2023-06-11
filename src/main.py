import argparse
from pyspark.sql import SparkSession

from kmeans import PySparkKMeans
from preprocessing import DataTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--columns_json_path", default="../config/columns.json")
    parser.add_argument("--k", default=2)
    parser.add_argument("--max_iter", default=5)
    parser.add_argument("--driver_cores", default=2),
    parser.add_argument("--driver_memory", default="4g"),
    parser.add_argument("--executor_memory", default="10g"),
    args = parser.parse_args()

    app = SparkSession.builder \
        .appName("openfood-kmeans") \
        .master("local[*]") \
        .config("spark.driver.cores", args.driver_cores) \
        .config("spark.driver.memory", args.driver_memory) \
        .config("spark.executor.memory", args.executor_memory) \
        .getOrCreate()

    model_params = {
        "k": args.k,
        "max_iter": args.max_iter
    }

    df = DataTransformer(df_path=args.data_path, columns_json_path=args.columns_json_path) \
        .load(app) \
        .transform()

    model = PySparkKMeans(model_params).fit(df)
    model.save(args.save_path)
