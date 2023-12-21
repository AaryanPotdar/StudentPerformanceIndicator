import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# for EDA
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# StandardScaler is used to bring all the values in a range and OneHotEncoder is used for forming new column for categorical vaules

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTrasformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            logging.info("creating numerical and categorical pipelines")

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), # handling missing values
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
# StandardScaler is subtracting the mean from each feature, which can result in some features having negative values. However, StandardScaler() alone assumes that the features have positive values, which can cause issues when working with features that have negative values.
# By setting with_mean=False, the StandardScaler does not subtract the mean from each feature, and instead scales the features based on their variance. This can help preserve the positive values of the features and avoid issues.

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline( # need to do OneHotEcnoding
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")), #replacing missing value with mode
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("creating preprocessor obj")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training df and testing df")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomExcpetion(e, sys)