"""
Download real public datasets from UCI ML Repository and other sources.
"""
import logging
import requests
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("ml_tree_system")


class DatasetDownloader:
    """
    Download and prepare real public datasets.
    """

    DATASETS = {
        "heart_disease": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "columns": [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
            ],
            "target": "target",
            "description": "UCI Heart Disease (Cleveland) - Predict heart disease presence"
        },
        "adult_income": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "columns": [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
            ],
            "target": "income",
            "description": "UCI Adult Income - Predict if income >50K"
        },
        "bank_marketing": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
            "target": "y",
            "description": "UCI Bank Marketing - Predict term deposit subscription"
        },
        "wine_quality": {
            "url_red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "url_white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
            "target": "quality",
            "description": "UCI Wine Quality - Predict wine quality score"
        }
    }

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_heart_disease(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download UCI Heart Disease dataset (Cleveland).

        Features: 13 clinical attributes
        Target: 0 = no disease, 1 = disease present
        Samples: 303
        """
        logger.info("Downloading UCI Heart Disease dataset...")

        url = self.DATASETS["heart_disease"]["url"]
        columns = self.DATASETS["heart_disease"]["columns"]

        filepath = self.data_dir / "heart_disease.csv"

        # Download if not exists
        if not filepath.exists():
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "w") as f:
                f.write(response.text)
            logger.info(f"Downloaded to {filepath}")

        # Load data
        df = pd.read_csv(filepath, names=columns, na_values="?")

        # Clean data
        df = df.dropna()

        # Convert target to binary (0 = no disease, 1-4 = disease)
        df["target"] = (df["target"] > 0).astype(int)

        X = df.drop("target", axis=1)
        y = df["target"]

        logger.info(f"Loaded Heart Disease: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def download_adult_income(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download UCI Adult Income dataset.

        Features: 14 attributes (mixed numeric/categorical)
        Target: <=50K or >50K
        Samples: ~32,561
        """
        logger.info("Downloading UCI Adult Income dataset...")

        url = self.DATASETS["adult_income"]["url"]
        columns = self.DATASETS["adult_income"]["columns"]

        filepath = self.data_dir / "adult_income.csv"

        if not filepath.exists():
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "w") as f:
                f.write(response.text)
            logger.info(f"Downloaded to {filepath}")

        # Load data
        df = pd.read_csv(
            filepath,
            names=columns,
            skipinitialspace=True,
            na_values="?"
        )

        # Clean
        df = df.dropna()

        # Encode target
        df["income"] = (df["income"].str.strip() == ">50K").astype(int)

        # Encode categorical features
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != "income"]

        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]

        X = df.drop("income", axis=1)
        y = df["income"]

        logger.info(f"Loaded Adult Income: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def download_wine_quality(self, wine_type: str = "red") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download UCI Wine Quality dataset.

        Features: 11 physicochemical properties
        Target: Quality score 0-10 (converted to binary: good/bad)
        Samples: ~1599 (red) or ~4898 (white)
        """
        logger.info(f"Downloading UCI Wine Quality ({wine_type}) dataset...")

        url_key = f"url_{wine_type}"
        url = self.DATASETS["wine_quality"][url_key]

        filepath = self.data_dir / f"wine_quality_{wine_type}.csv"

        if not filepath.exists():
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "w") as f:
                f.write(response.text)
            logger.info(f"Downloaded to {filepath}")

        # Load data
        df = pd.read_csv(filepath, sep=";")

        # Convert to binary classification (quality >= 6 is good)
        df["quality"] = (df["quality"] >= 6).astype(int)

        X = df.drop("quality", axis=1)
        y = df["quality"]

        logger.info(f"Loaded Wine Quality: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def download_diabetes(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download Pima Indians Diabetes dataset.

        Features: 8 medical attributes
        Target: 0 = no diabetes, 1 = diabetes
        Samples: 768
        """
        logger.info("Downloading Pima Indians Diabetes dataset...")

        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

        columns = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
        ]

        filepath = self.data_dir / "diabetes.csv"

        if not filepath.exists():
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "w") as f:
                f.write(response.text)
            logger.info(f"Downloaded to {filepath}")

        df = pd.read_csv(filepath, names=columns)

        X = df.drop("outcome", axis=1)
        y = df["outcome"]

        logger.info(f"Loaded Diabetes: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y

    def get_dataset(self, name: str) -> Tuple[pd.DataFrame, pd.Series, dict]:
        """
        Get a dataset by name.

        Args:
            name: Dataset name (heart_disease, adult_income, wine_quality, diabetes)

        Returns:
            Tuple of (X, y, metadata)
        """
        metadata = {"name": name}

        if name == "heart_disease":
            X, y = self.download_heart_disease()
            metadata["target_names"] = ["no_disease", "disease"]
            metadata["description"] = "Predict presence of heart disease"

        elif name == "adult_income":
            X, y = self.download_adult_income()
            metadata["target_names"] = ["<=50K", ">50K"]
            metadata["description"] = "Predict if income exceeds $50K"

        elif name == "wine_quality":
            X, y = self.download_wine_quality("red")
            metadata["target_names"] = ["bad", "good"]
            metadata["description"] = "Predict wine quality (good/bad)"

        elif name == "diabetes":
            X, y = self.download_diabetes()
            metadata["target_names"] = ["no_diabetes", "diabetes"]
            metadata["description"] = "Predict diabetes diagnosis"

        else:
            raise ValueError(f"Unknown dataset: {name}")

        metadata["n_samples"] = len(X)
        metadata["n_features"] = len(X.columns)
        metadata["feature_names"] = list(X.columns)

        return X, y, metadata
