import os
import joblib
import pandas as pd

from disease_registry import get_disease


# =========================================
# GENERIC PREDICTION ENGINE
# =========================================

def load_saved_model(disease_id):

    path = os.path.join(
        "saved_models",
        f"{disease_id}_best_model.joblib"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved model not found for "
            f"'{disease_id}'. Train the disease first."
        )

    return joblib.load(path)


def predict_disease(
    disease_id,
    patient_data
):

    disease = get_disease(
        disease_id
    )

    if disease is None:
        raise ValueError(
            f"Disease '{disease_id}' "
            f"not found."
        )

    model = load_saved_model(
        disease_id
    )

    # Convert patient data to DataFrame
    patient_df = pd.DataFrame(
        [patient_data]
    )

    # Make prediction
    prediction = model.predict(
        patient_df
    )[0]

    result = {
        "disease": disease["name"],
        "prediction": int(prediction)
    }

    # Get probabilities if available
    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            patient_df
        )[0]

        result["probabilities"] = [
            round(
                float(p) * 100,
                2
            )
            for p in probabilities
        ]

    return result


# =========================================
# TEST: BREAST CANCER
# =========================================

if __name__ == "__main__":

    # Example Breast Cancer patient
    patient = {
        "mean radius": 17.99,
        "mean texture": 10.38,
        "mean perimeter": 122.80,
        "mean area": 1001.0,
        "mean smoothness": 0.1184,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "mean symmetry": 0.2419,
        "mean fractal dimension": 0.07871,

        "radius error": 1.095,
        "texture error": 0.9053,
        "perimeter error": 8.589,
        "area error": 153.4,
        "smoothness error": 0.006399,
        "compactness error": 0.04904,
        "concavity error": 0.05373,
        "concave points error": 0.01587,
        "symmetry error": 0.03003,
        "fractal dimension error": 0.006193,

        "worst radius": 25.38,
        "worst texture": 17.33,
        "worst perimeter": 184.60,
        "worst area": 2019.0,
        "worst smoothness": 0.1622,
        "worst compactness": 0.6656,
        "worst concavity": 0.7119,
        "worst concave points": 0.2654,
        "worst symmetry": 0.4601,
        "worst fractal dimension": 0.1189
    }

    result = predict_disease(
        "breast_cancer",
        patient
    )

    print("\nPrediction Result")
    print("=================")

    print(result)