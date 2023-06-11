import json

from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.functions import col, count, isnull, lit, when


class DataTransformer:
    def __init__(
            self,
            df_path: str,
            columns_json_path: str,
            filter_null_threshold: int = 0.5
    ):
        self.data_path = df_path
        self.columns_json_path = columns_json_path
        self.filter_null_threshold = filter_null_threshold
        self.df = None

    def load(self, spark) -> "DataTransformer":
        self.df = spark.read.option("sep", "\t").option("header", True).csv(self.data_path)
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
        float_cols = self._read_columns()
        columns = [col(x).cast("float") for x in float_cols]
        self.df = self.df.select(columns)
        return self

    def _fill_nans(self) -> "DataTransformer":
        expressions = [(count(when(isnull(c), c)) / count("*")).alias(c) for c in self.df.columns]
        nulls_stats = self.df.select(expressions).collect()[0].asDict()

        columns_to_save = [k for k, v in nulls_stats.items() if v < 1 - self.filter_null_threshold]
        self.df = self.df.select(columns_to_save)
        self.df.na.fill(0.0).na.fill("unknown")

        return self

    def _vectorize(self) -> "DataTransformer":
        assembler = VectorAssembler(inputCols=self.df.columns, outputCol="vector_features").setHandleInvalid("error")
        self.df = assembler.transform(self.df)
        return self

    def _standardize(self) -> "DataTransformer":
        scaler = StandardScaler(inputCol="vector_features", outputCol="features", withStd=True, withMean=True)
        self.df = scaler.fit(self.df).transform(self.df)
        return self
