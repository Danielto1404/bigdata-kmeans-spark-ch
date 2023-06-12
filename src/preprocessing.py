import json

import pyspark.ml.feature as ml
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import clickhouse


class DataTransformer:
    def __init__(
            self,
            table_name: str,
            columns_json_path: str
    ):
        self.table_name = table_name
        self.columns_json_path = columns_json_path
        self.df = None

    def load(self, spark: SparkSession) -> "DataTransformer":
        self.df = clickhouse.read(spark, self.table_name)
        return self

    def transform(self):
        self._to_float()
        self._fill_nans()
        self._vectorize()
        self._standardize()
        return self.df

    def _read_columns(self) -> list:
        with open(self.columns_json_path, "r") as f:
            return json.load(f)
        
    def _to_float(self) -> "DataTransformer":
        columns = self._read_columns()
        columns = [F.col(c).cast("float").alias(c) for c in columns]
        self.df = self.df.select(*columns)
        return self

    def _fill_nans(self) -> "DataTransformer":
        self.df = self.df.na.fill(0.0)
        return self

    def _vectorize(self) -> "DataTransformer":
        assembler = ml.VectorAssembler(inputCols=self.df.columns, outputCol="vector_features")
        self.df = assembler.transform(self.df)
        return self

    def _standardize(self) -> "DataTransformer":
        scaler = ml.MinMaxScaler(inputCol="vector_features", outputCol="features")
        self.df = scaler.fit(self.df).transform(self.df)
        return self
