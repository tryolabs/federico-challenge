from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class DelayModel:

    THRESHOLD_IN_MINS = 15
    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    def __init__(self):
        self._model = None  # TODO: can I instanciate it without n_y0 and n_y1 ?
        self.n_y0 = 0
        self.n_y1 = 0
        self.target_column = "target"

    def _get_min_diff(self, data):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        features = features[self.FEATURES_COLS]

        if target_column:
            self.target_column = target_column
            data["min_diff"] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data["min_diff"] > self.THRESHOLD_IN_MINS, 1, 0)
            return features, data[[target_column]]

        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        self.n_y0 = len(target[target[self.target_column] == 0])
        self.n_y1 = len(target[target[self.target_column] == 1])

        self._model = LogisticRegression(
            class_weight={1: self.n_y0 / len(target), 0: self.n_y1 / len(target)}
        )
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        features = features[self.FEATURES_COLS]

        return list(self._model.predict(features))
