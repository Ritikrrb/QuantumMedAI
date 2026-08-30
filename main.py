from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from disease_registry import get_diseases, get_disease
from training_engine import train_disease
from prediction_engine import predict_disease


# ============================================================
# APP
# ============================================================

app = FastAPI(title="QuantumMedAI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"


# ============================================================
# BREAST CANCER DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train,
    y_train
)

logistic_predictions = model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


# ============================================================
# SVM
# ============================================================

svm = SVC(
    probability=True,
    random_state=42
)

svm.fit(
    X_train,
    y_train
)

svm_predictions = svm.predict(
    X_test
)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)


# ============================================================
# MEDICAL METRICS
# ============================================================

precision = precision_score(
    y_test,
    logistic_predictions,
    pos_label=0
)

recall = recall_score(
    y_test,
    logistic_predictions,
    pos_label=0
)

f1 = f1_score(
    y_test,
    logistic_predictions,
    pos_label=0
)


cm = confusion_matrix(
    y_test,
    logistic_predictions
)


# Class 0 = malignant
# Class 1 = benign

TN = cm[1, 1]
FP = cm[1, 0]

specificity = TN / (TN + FP)


# ============================================================
# PATIENT MODEL
# ============================================================

class PatientData(BaseModel):
    features: list[float]


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        path=INDEX_FILE,
        media_type="text/html"
    )


# ============================================================
# API STATUS
# ============================================================

@app.get("/api-status")
def api_status():
    return {
        "message": "QuantumMedAI Backend is Running!"
    }


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.get("/dataset-info")
def dataset_info():

    return {
        "dataset": "Breast Cancer Wisconsin Dataset",
        "total_samples": len(data.data),
        "total_features": len(data.feature_names),
        "features": list(data.feature_names),
        "target_names": list(data.target_names)
    }


# ============================================================
# BREAST CANCER TEST SAMPLE
# ============================================================

@app.get("/breast-cancer/test-sample")
def breast_cancer_test_sample():

    sample = data.data[0]
    target = int(data.target[0])

    return {
        "sample_number": 1,
        "features": [
            float(value)
            for value in sample
        ],
        "target": target,
        "target_name": data.target_names[target],
        "feature_names": list(data.feature_names)
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/model-performance")
def model_performance():

    return {
        "Logistic Regression Accuracy (%)":
            round(
                float(logistic_accuracy) * 100,
                2
            ),

        "Random Forest Accuracy (%)":
            round(
                float(rf_accuracy) * 100,
                2
            ),

        "SVM Accuracy (%)":
            round(
                float(svm_accuracy) * 100,
                2
            )
    }


# ============================================================
# BREAST CANCER PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    patient: PatientData
):

    if len(patient.features) != 30:

        return {
            "error":
            "Exactly 30 medical features are required"
        }


    patient_data = scaler.transform(
        [patient.features]
    )


    prediction = model.predict(
        patient_data
    )[0]


    probabilities = model.predict_proba(
        patient_data
    )[0]


    return {

        "prediction":
            data.target_names[
                int(prediction)
            ],

        "malignant_probability":
            round(
                float(probabilities[0]) * 100,
                2
            ),

        "benign_probability":
            round(
                float(probabilities[1]) * 100,
                2
            )
    }


# ============================================================
# TEST PREDICTION
# ============================================================

@app.get("/test-prediction")
def test_prediction():

    patient = data.data[0]


    patient_scaled = scaler.transform(
        [patient]
    )


    prediction = model.predict(
        patient_scaled
    )[0]


    probabilities = model.predict_proba(
        patient_scaled
    )[0]


    return {

        "actual_result":
            data.target_names[
                int(data.target[0])
            ],

        "predicted_result":
            data.target_names[
                int(prediction)
            ],

        "malignant_probability":
            round(
                float(probabilities[0]) * 100,
                2
            ),

        "benign_probability":
            round(
                float(probabilities[1]) * 100,
                2
            )
    }


# ============================================================
# MEDICAL METRICS
# ============================================================

@app.get("/medical-metrics")
def medical_metrics():

    return {

        "model":
            "Logistic Regression",

        "accuracy":
            round(
                float(logistic_accuracy) * 100,
                2
            ),

        "precision":
            round(
                float(precision) * 100,
                2
            ),

        "recall_sensitivity":
            round(
                float(recall) * 100,
                2
            ),

        "f1_score":
            round(
                float(f1) * 100,
                2
            ),

        "specificity":
            round(
                float(specificity) * 100,
                2
            )
    }


# ============================================================
# DISEASE REGISTRY
# ============================================================

@app.get("/diseases")
def list_diseases():

    return {

        "total_diseases":
            len(get_diseases()),

        "diseases":
            get_diseases()
    }


@app.get("/diseases/{disease_id}")
def disease_details(
    disease_id: str
):

    disease = get_disease(
        disease_id
    )


    if disease is None:

        return {
            "error":
            "Disease not found"
        }


    return disease


# ============================================================
# TRAIN DISEASE
# ============================================================

@app.get("/train/{disease_id}")
def train_selected_disease(
    disease_id: str
):

    try:

        result = train_disease(
            disease_id
        )


        return {

            "disease":
                result["disease"],

            "samples":
                result["samples"],

            "features":
                result["features"],

            "model_results":
                result["model_results"],

            "best_model":
                result["best_model"],

            "best_accuracy":
                result["best_accuracy"]
        }


    except Exception as error:

        return {
            "error":
            str(error)
        }


# ============================================================
# DISEASE PREDICTION
# ============================================================

@app.post("/predict-disease/{disease_id}")
def predict_selected_disease(
    disease_id: str,
    patient_data: dict
):

    try:

        result = predict_disease(
            disease_id,
            patient_data
        )


        return result


    except Exception as error:

        return {

            "error":
            str(error)

        }
