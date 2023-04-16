import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class Modeltrainerconfig:
    trained_model_file = os.path.join("artifacts", "model.pkl")


class modeltrainer:
    def __init__(self):
        self.model_trainer_config = Modeltrainerconfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split train & test data")
            X_train, X_test, Y_train, Y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, models=models)

            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("f2 scores received")

            save_object(
                file_path=self.model_trainer_config.trained_model_file,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_scores = r2_score(Y_test, predicted)

            return r2_scores, best_model



        except Exception as e:
            raise Exception(e, sys)
