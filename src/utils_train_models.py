"""
For training and evaluating the MLPClassifier
"""
import sys
sys.path.append("..")

from src.window_features import compute_windowed_dataset
from src.preprocessing import (select_measurement_cols, 
                               build_single_source_pipeline)
from src.utils_eval import performance_eval
from src.config import GROUP_COLS, SORT_COLS, path_list

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import time
import os

Z = [5000,10000,20000]
RANDOM_STATE = 42
TEST_SIZE = 0.25

ACTIVATION = "logistic"
SOLVER = "adam"
NUM_ITER= 200
NEURONS = 5
LAYERS = 2

Z_WINDOW_EVAL_OUTCOME = "outcome_windowed_features.txt"

def train_NN_classifier(X_tr, y_tr, activ, neur, lyrs, solv_name, num_iter):
    print("Training a NN...")

    size = (neur,)*lyrs 

    start_training = time.time()
    model= MLPClassifier(
                        hidden_layer_sizes=size, 
                        activation=activ,
                        solver=solv_name,
                        max_iter=num_iter)
    model.fit(X_tr, y_tr)
    end_training = time.time()
    training_time = end_training - start_training
    
    return model, training_time


def load_preprocessed_dataset():
    # To create the ../data/outcome_preprocess directory
    if not os.path.exists(path_list["FEATURES_DATASET_DIR"]):
        os.makedirs(path_list["FEATURES_DATASET_DIR"])

    filepath = os.path.join(path_list["FEATURES_DATASET_DIR"],path_list["FEATURES_DATASET_FILEPATH"])

    dataset = pd.DataFrame()
    
    # If there is no precomputed dataset, it calls the build_single_source_pipeline() function
    # and generates a dataset obtained from the columns concatenation of the original .csv files
    if not os.path.exists(filepath):
        dataset = build_single_source_pipeline()

        dataset.to_csv(filepath, index=False)
        print("Preprocessed dataset stored at: {}\n".format(filepath))
    else:
        print("Reading the content of {}".format(filepath))
        dataset = pd.read_csv(filepath)

    return dataset

def compute_classification_report(X_test, y_test, model, target, labels):
    y_pred = model.predict(X_test)
    
    classification_analysis = classification_report(y_test, y_pred, target_names=target, labels=labels)

    return classification_analysis

def generate_grouped_dataset(dataset):
    LIST_COL = [col for col in dataset if any(value in col for value in SORT_COLS)]
    dataset = dataset.sort_values(by=LIST_COL, axis=1, ascending=True)
    groups = dataset.groupby(GROUP_COLS)

    group_list = pd.DataFrame()
    for _, index in groups:
        group_list = pd.concat([group_list, index], ignore_index=True)
    encoded_groups = [i for i in range(len(group_list))]
    group_list["encode"] = encoded_groups

    group_encoded = pd.merge(dataset, group_list, how="inner")["encode"].to_numpy()
    return group_encoded

def generate_outcome_classification(windowed_dataset, split_outcome, z, labels, target):
    features_set = windowed_dataset.copy()

    model, training_time = train_NN_classifier(split_outcome["X_train"], split_outcome["y_train"], ACTIVATION, NEURONS, LAYERS, SOLVER, NUM_ITER)
    y_pred = model.predict(split_outcome["X_test"])

    classification_report = compute_classification_report(split_outcome["X_test"], split_outcome["y_test"], model, target, labels)

    outcome_training = {
                "z_value": [z],
                "features_set": [features_set],
                "pretrained_model": [model],
                "training_time": [training_time],
                "y_test": [split_outcome["y_test"]],
                "y_pred": [y_pred],
                "classification_report": [classification_report],
                "extracted_labels": [labels],
            }

    return outcome_training

"""
This function has the main purpose of extracting for each z value chosen the corresponding windowed
dataset and computing statistical measurements and evaluations on the obtained dataset.
We proposed two distinct ways to generate the windowed dataset:
- compute_window_features(): is feasible for smaller datasets or sampling data. 
                            The complete dataset counts over 5000000 rows and 21 measurements columns.
                            The rolling process for each column will instantly result in a Kernel crash.
- compute_windowed_dataset(): is an alternative and robust solution for larger datasets.
                              For each size z, it manually generates and shifts the window towards the dataset
                              until it reaches its end.
                              Some tests computed on the original dataset (without splitting the dataset) for z = 100
                              required nearly 1h for the generation of the windowed dataset and the model training.
                              With the introduction of the GroupShuffleSplit() we can divide the original dataset in two
                              distinct sections while preserving the content of each group.
                              Additional tests can be performed to evaluate how different ways of managing the input dataset during the 
                              windowed dataset generation will affect the system performance.
"""

def generate_X_features__models_per_z(dataset: pd.DataFrame, target: list, model: str = "MLPClassifier"):
    # Select measurement columns
    measurement_cols = select_measurement_cols(dataset)
    print(f"Measurement columns ({len(measurement_cols)}): {measurement_cols}")

    outcome_training_per_z = pd.DataFrame()
    
    group_encoded = generate_grouped_dataset(dataset)

    for z in Z:
        windowed_dataset = pd.DataFrame()
        
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for i, (train_index, test_index) in enumerate(gss.split(dataset, None, group_encoded)):
            print("Fold={}\n".format(i))
            print("Train group={}\n".format(group_encoded[train_index]))
            print("Test group={}\n".format(group_encoded[test_index]))

            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]

            print("Train dataset length: {}, test dataset length: {} obtained for Z: {}".format(len(train_set),len(test_set), z))
            windowed_dataset_train = compute_windowed_dataset(train_set, z, measurement_cols)
            windowed_dataset_test = compute_windowed_dataset(test_set, z, measurement_cols)
            windowed_dataset = pd.concat([windowed_dataset_train,windowed_dataset_test], axis=0, ignore_index=True)
            
            print("Windowed dataset computed for z = {}\n".format(z))
            print("Length dataset: {}\n".format(len(windowed_dataset)))
            
            X = windowed_dataset.copy()
            y = X["rat"].copy()
            X.drop("rat", axis=1, inplace=True)

            labels = np.unique(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit(X_test)

            split_outcome = {
                "X_train": X_train.copy(),
                "X_test": X_test.copy(),
                "y_train": y_train.copy(),
                "y_test": y_test.copy(),
            }

            outcome_training = generate_outcome_classification(windowed_dataset, split_outcome, z, labels, target)
            outcome_training_per_z = pd.concat([outcome_training_per_z, pd.DataFrame(outcome_training)], ignore_index=True)

    return outcome_training_per_z

def compute_statistics(outcome_training_per_z, RAT_NAME):
    accuracy_scores = []

    for _, outcome_training in outcome_training_per_z.iterrows():
        accuracy, global_precision, global_recall, global_f1score  = performance_eval(outcome_training["features_set"], outcome_training["y_test"], outcome_training["y_pred"], outcome_training["extracted_labels"], RAT_NAME)
        accuracy_scores.append(accuracy)

        print("------------------------------------\n")
        print("Results for Z: {}\n".format(outcome_training["z_value"]))
        print("Training time[s]: {}\n".format(outcome_training["training_time"]))
        print("Accuracy: {}\n".format(accuracy))
        print("Global precision: {}\n".format(global_precision))
        print("Global recall: {}\n".format(global_recall))
        print("Global f1score: {}\n".format(global_f1score))
        print("------------------------------------\n")
    return accuracy_scores

def store_windowed_training_analysis(outcome_windowed_dataset_eval):
    EXPORTED_DIR = "results/exported"
    MODEL_NAME = "MLPClassifier"
    EXPORTED_DIR = "results/exported"
    MODEL_NAME = "MLPClassifier"

    with open(os.path.join(path_list["ANALYSIS_DIR"], Z_WINDOW_EVAL_OUTCOME),"w") as f:
        f.write("--------------------------------------------------------------------------\n")
        f.write("MOST ACCURATE WINDOWED MODEL\n")
        f.write("Window size:{}\n".format(outcome_windowed_dataset_eval["window_size"]))
        f.write("Pretrained model:{}\n".format(outcome_windowed_dataset_eval["selected_model"]))
        f.write("Windowed dataset length:{}\n".format(len(outcome_windowed_dataset_eval["windowed_dataset"])))
        f.write("--------------------------------------------------------------------------\n")

    if not os.path.exists(path_list["EXPORTED_DIR"]):
        os.makedirs(path_list["EXPORTED_DIR"])
    
    # To store the selected windowed dataset
    outcome_windowed_dataset_eval["windowed_dataset"].to_csv(path_list["EXPORTED_FEATURES_PATH"], index=False)
    
    # To store the selected model in pkl format
    joblib.dump(outcome_windowed_dataset_eval["selected_model"], os.path.join(EXPORTED_DIR,MODEL_NAME)+".pkl")