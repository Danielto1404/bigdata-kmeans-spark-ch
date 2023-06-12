from pyspark.sql import SparkSession

CLICKHOUSE_URL = "jdbc:clickhouse://clickhouse"
USER = "default"
PASSWORD = ""
DRIVER = "com.clickhouse.jdbc.ClickHouseDriver"


def read(spark: SparkSession, table_name: str):
    return spark.read \
        .format("jdbc") \
        .option("url", CLICKHOUSE_URL) \
        .option("user", USER) \
        .option("password", PASSWORD) \
        .option("driver", DRIVER) \
        .option("dbtable", table_name) \
        .load()


def write(data, table_name: str):
    return data.write \
        .format("jdbc") \
        .mode("append") \
        .option("driver", DRIVER) \
        .option("url", CLICKHOUSE_URL) \
        .option("user", USER) \
        .option("password", PASSWORD) \
        .option("dbtable", table_name) \
        .save()


__all__ = ["read", "write"]