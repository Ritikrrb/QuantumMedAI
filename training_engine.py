import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from disease_registry import get_disease


# =========================================
# GENERIC TRAINING ENGINE
# =========================================

def train_disease(disease_id):

    # -------------------------------------
    # 1. Get disease configuration
    # -------------------------------------

    disease = get_disease(disease_id)

    if disease is None:
        raise ValueError(
            f"Disease '{disease_id}' not found."
        )

    print(
        f"\nTraining model for: "
        f"{disease['name']}"
    )


    # -------------------------------------
    # 2. Load dataset
    # -------------------------------------

    path = disease["path"]

    df = pd.read_csv(path)

    print(
        f"Dataset loaded: "
        f"{len(df)} rows, "
        f"{len(df.columns)} columns"
    )


    # -------------------------------------
    # 3. Remove rows with missing target
    # -------------------------------------

    target_column = disease["target_column"]

    df = df.dropna(
        subset=[target_column]
    )


    # -------------------------------------
    # 4. Apply target mapping
    # -------------------------------------

    if "target_mapping" in disease:

        mapping = disease["target_mapping"]

        df[target_column] = (
            df[target_column]
            .astype(str)
            .map(mapping)
        )

        df = df.dropna(
            subset=[target_column]
        )


    # -------------------------------------
    # 5. Separate features and target
    # -------------------------------------

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]


    # -------------------------------------
    # 6. Identify column types
    # -------------------------------------

    numeric_columns = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()


    # -------------------------------------
    # 7. Numeric preprocessing
    # -------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # -------------------------------------
    # 8. Categorical preprocessing
    # -------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    # -------------------------------------
    # 9. Combined preprocessing
    # -------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )


    # -------------------------------------
    # 10. Train/test split
    # -------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # -------------------------------------
    # 11. Define models
    # -------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced"
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ),

        "SVM":
            SVC(
                class_weight="balanced",
                random_state=42
            )
    }


    # -------------------------------------
    # 12. Train and compare models
    # -------------------------------------

    results = {}

    trained_models = {}

    for model_name, classifier in models.items():

        print(
            f"Training {model_name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[model_name] = round(
            accuracy * 100,
            2
        )

        trained_models[
            model_name
        ] = pipeline


    # -------------------------------------
    # 13. Select best model
    # -------------------------------------

    best_model_name = max(
        results,
        key=results.get
    )

    best_model = trained_models[
        best_model_name
    ]


    # -------------------------------------
    # 14. Save best model
    # -------------------------------------

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    model_path = os.path.join(
        "saved_models",
        f"{disease_id}_best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"Best model saved to: "
        f"{model_path}"
    )


    # -------------------------------------
    # 15. Return results
    # -------------------------------------

    return {

        "disease":
            disease["name"],

        "samples":
            len(df),

        "features":
            len(X.columns),

        "model_results":
            results,

        "best_model":
            best_model_name,

        "best_accuracy":
            results[best_model_name],

        "model_path":
            model_path,

        "model":
            best_model
    }


# =========================================
# TEST ENGINE
# =========================================

if __name__ == "__main__":

    result = train_disease(
        "breast_cancer"
    )

    print("\nTraining Results")
    print("================")

    print(
        "Disease:",
        result["disease"]
    )

    print(
        "Samples:",
        result["samples"]
    )

    print(
        "Features:",
        result["features"]
    )

    print("\nModel Comparison:")

    for model_name, accuracy in result[
        "model_results"
    ].items():

        print(
            f"{model_name}: "
            f"{accuracy}%"
        )

    print(
        "\nBest Model:",
        result["best_model"]
    )

    print(
        "Best Accuracy:",
        result["best_accuracy"],
        "%"
    )

    print(
        "Saved Model:",
        result["model_path"]
    )