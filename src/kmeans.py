from pyspark.ml.clustering import KMeans
from pyspark.sql import DataFrame


class PySparkKMeans:
    def __init__(self, model_params: dict):
        self.model_params = model_params
        self.model = None

    def fit(self, df: DataFrame) -> "PySparkKMeans":
        assert not self.is_fitted(), "Model is already trained."

        kmeans = KMeans() \
            .setK(self.model_params.get("k", 2)) \
            .setSeed(1) \
            .setMaxIter(self.model_params.get("max_iter", 5))

        self.model = kmeans.fit(df)
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        assert self.is_fitted(), "Model is not trained yet."
        return self.model.transform(df)

    def is_fitted(self) -> bool:
        return self.model is not None

    def save(self, path: str):
        self.model.save(path)


__all__ = ["PySparkKMeans"]
