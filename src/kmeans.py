import dataclasses

from pyspark.ml.clustering import KMeans
from pyspark.sql import DataFrame


@dataclasses.dataclass
class KmeansParams:
    k: int = 2
    max_iter: int = 5
    distance_measure: str = "euclidean"
    tol: float = 1e-4
    seed: int = 1


class PySparkKMeans:
    def __init__(self, params: KmeansParams):
        self.params = params
        self.model = None

    def fit(self, df: DataFrame) -> "PySparkKMeans":
        assert not self.is_fitted(), "Model is already trained."

        kmeans = KMeans() \
            .setK(self.params.k) \
            .setMaxIter(self.params.max_iter) \
            .setDistanceMeasure(self.params.distance_measure) \
            .setTol(self.params.tol) \
            .setSeed(self.params.seed)

        self.model = kmeans.fit(df)
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        assert self.is_fitted(), "Model is not trained yet."
        return self.model.transform(df)

    def is_fitted(self) -> bool:
        return self.model is not None

    def save(self, path: str):
        self.model.save(path)


__all__ = [
    "PySparkKMeans",
    "KmeansParams"
]
