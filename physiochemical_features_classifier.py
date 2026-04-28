#----------------IMPORTS----------------#
import numpy as np
import pandas as pd

# Biopython for reading FASTA files
from Bio import SeqIO

# ML Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Model selection and evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score



#----------------FASTA AND FEATURE PROCESSING----------------#

# Reads FASTA file and converts to datafram with columns: label, sequence, length
def fastaConverter(filename):
    dataList = []

    with open(filename, "r") as handle:
        for entry in SeqIO.parse(handle, "fasta"):
            dataList.append({
                "label": int(entry.id[8]),
                "sequence": str(entry.seq),
                "length": len(entry.seq)
            })

    return pd.DataFrame(dataList)


# reads amino acid features from an excel_file and returns dictionary mapping amino acid to its features
def amino_acid_features(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name, skiprows=1)
    df = df.set_index(df.columns[1])
    df = df.drop(columns=[df.columns[0]])
    df = df.T

    aa_dict = {}

    for aa in df.index:
        aa_dict[aa] = df.loc[aa].values

    return aa_dict



# converts a sequence to a matrix: each amino acid -> feature vector
def sequence_to_matrix(sequence, aa_dict):
    return np.array([aa_dict[aa] for aa in sequence])


# converts a matrix to a mean feature vector for a sequence
def matrix_to_mean(matrix):
    return matrix.mean(axis=0)


# converts a matrix to a feature vector containing mean, std, min, and max for each feature across the sequence 
def matrix_to_mean_std_min_max(matrix):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    min_val = matrix.min(axis=0)
    max_val = matrix.max(axis=0)

    return np.concatenate([mean, std, min_val, max_val])



# adds physiochemical feature vectors to the dataframe 
def add_physiochemical_features(dataframe, aa_dict, method="mean"):

    # convert each sequence into a matrix
    dataframe["matrix"] = dataframe["sequence"].apply(
        lambda seq: sequence_to_matrix(seq, aa_dict)
    )

    # different methods
    if method == "mean":
        dataframe["features"] = dataframe["matrix"].apply(matrix_to_mean)

    elif method == "mean_std_min_max":
        dataframe["features"] = dataframe["matrix"].apply(matrix_to_mean_std_min_max)

    elif method == "mean_std_min_max_length":
        dataframe["features"] = dataframe["matrix"].apply(matrix_to_mean_std_min_max)

        dataframe["features"] = dataframe.apply(
            lambda row: np.append(row["features"], row["length"]),
            axis=1
        )

    else:
        raise ValueError("Invalid method")

    return dataframe



# does grid search with cross validation for SVM
def tune_svm_model(X_train, y_train):
    # stratified ensures class balance in fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)


    # pipeline to avoid data leakage 
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=44))
    ])

    param_grid = {
        "svm__C": [0.1, 1, 5, 10],
        "svm__gamma": [0.001, 0.01, 0.05, 0.1],
        "svm__class_weight": [None, "balanced"]
    }


    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy", # metric used to pick best model
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid




# grid search for random forest
def tune_random_forest_model(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

    model = RandomForestClassifier(random_state=44)

    param_grid = {
        "n_estimators": [10, 50, 100, 200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid




# grid search for adaboost with decision tree base learner
def tune_adaboost_model(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)

    base_tree = DecisionTreeClassifier(random_state=44)

    model = AdaBoostClassifier(
        estimator=base_tree,
        random_state=44
    )

    param_grid = {
        "n_estimators": [10, 50, 100, 200, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "estimator__max_depth": [1, 2, 3],
        "estimator__min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid



# model evaluation: train model on entire training set and evaluate on test set
def evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    mcc = matthews_corrcoef(y_test, y_pred)

    return cm, accuracy, sensitivity, specificity, mcc




#-----------------------------------------#
#               MAIN-CODE                 #
#-----------------------------------------#

#---------------LOAD-DATA-----------------#

# load amino acid feature dictionary
aa_dict = amino_acid_features("physiochemical_attributes.xlsx", "Sheet2")

# load training and test data
trainingData = fastaConverter("Train.fasta")
testingData = fastaConverter("test.fasta")


# convert sequences to features
trainingData = add_physiochemical_features(
    trainingData,
    aa_dict,
    method="mean_std_min_max_length"
)
testingData = add_physiochemical_features(
    testingData,
    aa_dict,
    method="mean_std_min_max_length"
)

# convert to numpy arrays
X_train = np.vstack(trainingData["features"].values)
y_train = trainingData["label"].values

X_test = np.vstack(testingData["features"].values)
y_test = testingData["label"].values



#----------------GRID-SEARCH----------------#

grid_svm = tune_svm_model(X_train, y_train)
grid_rf = tune_random_forest_model(X_train, y_train)
grid_adaboost = tune_adaboost_model(X_train, y_train)


# print best parameters and cv performance
print("Best SVM Params:")
print(grid_svm.best_params_)
print(f"Best SVM CV Score: {grid_svm.best_score_:.4f}")

print("\nBest Random Forest Params:")
print(grid_rf.best_params_)
print(f"Best Random Forest CV Score: {grid_rf.best_score_:.4f}")

print("\nBest AdaBoost Params:")
print(grid_adaboost.best_params_)
print(f"Best AdaBoost CV Score: {grid_adaboost.best_score_:.4f}")



#----------------FINAL-EVALUATION----------------#
best_svm_model = grid_svm.best_estimator_
best_rf_model = grid_rf.best_estimator_
best_adaboost_model = grid_adaboost.best_estimator_

# evaluate all models
cm_svm, acc_svm, sens_svm, spec_svm, mcc_svm = evaluate_model(
    X_train, y_train, X_test, y_test, best_svm_model)

cm_rf, acc_rf, sens_rf, spec_rf, mcc_rf = evaluate_model(
    X_train, y_train, X_test, y_test, best_rf_model)

cm_adaboost, acc_adaboost, sens_adaboost, spec_adaboost, mcc_adaboost = evaluate_model(
    X_train, y_train, X_test, y_test, best_adaboost_model)




print("\nFinal SVM Test Results")
print("Confusion Matrix:")
print(cm_svm)
print(f"Accuracy: {acc_svm:.4f}")
print(f"Sensitivity: {sens_svm:.4f}")
print(f"Specificity: {spec_svm:.4f}")
print(f"MCC: {mcc_svm:.4f}")


print("\nFinal Random Forest Test Results")
print("Confusion Matrix:")
print(cm_rf)
print(f"Accuracy: {acc_rf:.4f}")
print(f"Sensitivity: {sens_rf:.4f}")
print(f"Specificity: {spec_rf:.4f}")
print(f"MCC: {mcc_rf:.4f}")


print("\nFinal AdaBoost Test Results")
print("Confusion Matrix:")
print(cm_adaboost)
print(f"Accuracy: {acc_adaboost:.4f}")
print(f"Sensitivity: {sens_adaboost:.4f}")
print(f"Specificity: {spec_adaboost:.4f}")
print(f"MCC: {mcc_adaboost:.4f}")