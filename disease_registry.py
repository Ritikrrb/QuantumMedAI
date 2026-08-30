# =========================================
# QuantumMedAI Disease Registry
# =========================================

DISEASES = {

    "breast_cancer": {
        "name": "Breast Cancer",
        "category": "Cancer",
        "task": "classification",
        "dataset": "Wisconsin Breast Cancer Dataset",
        "path": "data/breast_cancer/breast_cancer.csv",
        "target_column": "target",
        "positive_class": 0
    },

    "heart_disease": {
        "name": "Heart Disease",
        "category": "Cardiovascular",
        "task": "binary_classification",
        "dataset": "UCI Heart Disease Dataset",
        "path": "data/heart/heart.csv",
        "target_column": "target",
        "positive_class": 1,
        "target_mapping": {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1
        }
    },

    "diabetes": {
        "name": "Diabetes",
        "category": "Metabolic",
        "task": "binary_classification",
        "dataset": "Pima Indians Diabetes Dataset",
        "path": "data/diabetes/diabetes.csv",
        "target_column": "Outcome",
        "positive_class": 1
    },

    "chronic_kidney_disease": {
        "name": "Chronic Kidney Disease",
        "category": "Kidney",
        "task": "binary_classification",
        "dataset": "UCI Chronic Kidney Disease Dataset",
        "path": "data/chronic_kidney_disease/kidney.csv",
        "target_column": "classification",
        "positive_class": 1
    },

    "liver_disease": {
        "name": "Liver Disease",
        "category": "Liver",
        "task": "binary_classification",
        "dataset": "Indian Liver Patient Dataset",
        "path": "data/liver/liver.csv",
        "target_column": "Dataset",
        "positive_class": 1
    },

    "hepatitis": {
        "name": "Hepatitis",
        "category": "Liver",
        "task": "binary_classification",
        "dataset": "Hepatitis Dataset",
        "path": "data/hepatitis/hepatitis.csv",
        "target_column": "target",
        "positive_class": 1
    },

    "parkinsons": {
        "name": "Parkinson's Disease",
        "category": "Neurological",
        "task": "binary_classification",
        "dataset": "Parkinson's Dataset",
        "path": "data/parkinsons/parkinsons.csv",
        "target_column": "status",
        "positive_class": 1
    },

    "lung_cancer": {
        "name": "Lung Cancer",
        "category": "Cancer",
        "task": "binary_classification",
        "dataset": "Lung Cancer Dataset",
        "path": "data/lung_cancer/lung_cancer.csv",
        "target_column": "target",
        "positive_class": 1
    },

    "thyroid_disease": {
        "name": "Thyroid Disease",
        "category": "Endocrine",
        "task": "classification",
        "dataset": "Thyroid Disease Dataset",
        "path": "data/thyroid/thyroid.csv",
        "target_column": "target",
        "positive_class": 1
    },

    "cervical_cancer": {
        "name": "Cervical Cancer",
        "category": "Cancer",
        "task": "classification",
        "dataset": "Cervical Cancer Dataset",
        "path": "data/cervical_cancer/cervical_cancer.csv",
        "target_column": "target",
        "positive_class": 1
    }
}


def get_diseases():
    return DISEASES


def get_disease(disease_id):
    return DISEASES.get(disease_id)


if __name__ == "__main__":

    print("QuantumMedAI Disease Registry")
    print("============================")

    for disease_id, disease in DISEASES.items():

        print(
            f"{disease_id} -> "
            f"{disease['name']} | "
            f"{disease['category']} | "
            f"{disease['task']}"
        )