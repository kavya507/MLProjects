import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        # print(dir_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, Y_train,X_test,Y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)

            #model.fit(X_train, y_train)  # Train model

            Y_train_pred = model.predict(X_train)

            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)

            test_model_score = r2_score(Y_test, Y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
