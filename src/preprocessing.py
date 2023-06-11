import json


class DataTransformer:
    def __init__(self, df_path: str, columns_json_path: str):
        self.data_path = df_path
        self.columns_json_path = columns_json_path
        self.df = None

    def load(self, spark) -> "DataTransformer":
        self.df = spark.read.option("sep", "\t").option("header", True).csv(self.data_path)
        return self

    def transform(self) -> "DataTransformer":
        self.df = self.df.select(self._read_columns())
        return self

    def _read_columns(self) -> list:
        with open(self.columns_json_path, "r") as f:
            return json.load(f)
