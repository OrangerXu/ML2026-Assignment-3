"""作业三训练与推理接口。"""
import numpy as np
import pandas as pd

from model import Model


class Solution:
    def __init__(self):
        self.model = Model()

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """使用评测器提供的训练数据训练模型。"""
        self.model.fit(X_train, y_train)

    def forward(self, sample: dict) -> dict:
        """返回分类标签和正类概率。"""
        probability = self.model.predict_probability(sample)
        return {"prediction": int(probability >= 0.5), "probability": probability}
