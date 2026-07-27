"""
For training and evaluating the RandomForest and MPLClassifier
"""
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import time
import os

from src.window_features import compute_windowed_dataset
from src.preprocessing import (select_measurement_cols, 
                               build_single_source_pipeline)
from src.utils_eval import compute_classification_report
from src.utils import store_json_content
from src.config import (GROUP_COLS, SORT_COLS, path_main_folders, path_files)

# According to the project specifications, RAT classification must be performed with two distinct ML models:
#- *Random Forest classifier*, relies on a *decision tree structure*, takes the outcome produced by each node
#  of a given level and merges it with results obtained in corresponding nodes of subsequent levels. 
#  *All final predictions* are stored in leaf nodes and are summed up to obtain the *last prediction*. 
#- *Multi-Layer Perceptron*, is a *feed-forward Neural Network* with an architecture slightly more complex 
#  than the previous model. 
# The MLP classifier is a network composed of neurons called *perceptrons*, whose number and size can be 
# setted while creating the model.

# Training a ML model usually implies the following steps:
# - evaluation of the dataset in input to determine the best way for extracting training and testing sets.
#   Our purpose is still *reducing* as much as possible the *training time* while preserving the *model stability*
#   and *accuracy* and *reaching convergence*;
# - splitting the *target class* (Radio Access Technology code) from the windowed dataset;
# - extraction of the training and testset by splitting the original dataset in two distinct subsets.
#   The shape of the windowed dataset in input should suggest the size of the training set to be extracted.
#   Usually, it is a good practice to consider a smaller percentage of the overall content of the windowed dataset 
#   as training set and the remaining 60%/80% of data as the testing set;
# - training of the classifier and performance evaluation;
# - comparison of the accuracy of both models through *MSE*, *MAE*, *Mean Absolute Percentage Error*, *Root Mean 
#   Square Error* and *R2* Prediction Accuracy;

# We evaluated two distinct ways for splitting the dataset in training and testing set.
# The first one takes in input the feature set, the target class and a *percentage value* to compute the size of
# the testset from the size of the overall feature set.
# The second method relies on *group-aware splitting* and reflects the group distinction previously computed in the 
# windowed dataset. 
# An important difference between these methods is noticeable in the fact that the second one:
# -  extracts *groups from the windowed dataset* using grouped attributes previously defined;
# -  splits the feature set and the corresponding labelled targets according to the group IDs found and the percentage 
#    of learning rate specified as testset size;
# - preserves the amount of aggregated data for each group;
# In other words, all elements belonging to a given group, which was assigned to a specific set (either training or 
# testing set), must be located in the same set.

# This can be useful to *preserve the amount of meaningful data* within the training and testing sets compared to the 
# informational content of the original windowed one. 
# A metric to quantify the *feature importance* within a dataset is known as *correlation*. 
# We should ensure that, for each feature, the percentage of correlated data won't be affected heavily by the dataset 
# splitting process, otherwise they cannot be used for training the ML model.

# We introduced the *early_stopping* parameter in the MLPClassifier model to ensure *convergence even in case of overfitting*.
# This will reduce the *training time* when the model is stuck training, but no improvements are recorded.
 
# The final step of the ML training process consists on *performance evaluation*.
# We decided to introduce the following parameters:
# - *accuracy*, which quantifies the number of correct predictions with respect to the total number of predictions made;
# - *precision*, fraction of the number of true positives predictions with respect to all predictions (both correctly and
#    uncorrectly) classified as positives;
# - *f1 score*, is the harmoning mean of precision and recall;
#   In our scenario is the most accurate metric to evaluate the system performance due to the highly unbalanced dataset.
# - *recall*, fraction of true positive predictions with respect to all actual predictions made that should have been 
#   classified as positives;

Z = [1000,5000,10000,50000,100000,500000]
RANDOM_STATE = 42
TEST_SIZE = 0.25

ACTIVATION = "logistic"
SOLVER = "adam"
NUM_ITER= 200
NEURONS = 5
LAYERS = 2

NUM_ESTIMATORS = 100

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

def train_RandomForest(X_train, y_train):
    print("Training a RandomForest classifier...")

    start_training = time.time()
    clf = RandomForestClassifier(
                                n_estimators=NUM_ESTIMATORS, 
                                random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    end_training = time.time()

    training_time = end_training - start_training

    return clf, training_time

def load_preprocessed_dataset():
    # To create the ../data/outcome_preprocess directory
    if not os.path.exists(path_main_folders["FEATURES_DATASET_DIR"]):
        os.makedirs(path_main_folders["FEATURES_DATASET_DIR"])

    filepath = path_files["FEATURES_DATASET"]

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

def generate_grouped_dataset(dataset):
    LIST_COL = [col for col in dataset if any(value in col for value in SORT_COLS)]
    dataset = dataset.sort_values(by=LIST_COL, ascending=True)
    print("Dataset was sorted by timestamps: {}\n".format(LIST_COL))
    groups = dataset.groupby(GROUP_COLS)
    encoded_groups = [i for i in range(len(groups))]

    group_list = pd.DataFrame()
    for index, (group_key, group_content) in enumerate(groups):
        new_value = np.full(len(group_content),encoded_groups[index])
        group_content["encoded"] = new_value
        group_list = pd.concat([group_list, group_content], ignore_index=True)

    print('Number of encoded groups: {}\n'.format(len(groups)))
    return group_list["encoded"]

def generate_outcome_classification(windowed_dataset, split_outcome, z, labels, target, model_name):
    features_set = windowed_dataset.copy()
    model, training_time = None, 0

    if model_name == "MLPClassifier":
        model, training_time = train_NN_classifier(split_outcome["X_train"], split_outcome["y_train"], ACTIVATION, NEURONS, LAYERS, SOLVER, NUM_ITER)
    elif model_name == "RandomForest":
        model, training_time = train_RandomForest(split_outcome["X_train"], split_outcome["y_train"])
    y_pred = model.predict(split_outcome["X_test"])

    classification_report = compute_classification_report(split_outcome["X_test"], split_outcome["y_test"], model, target, labels)

    outcome_training = {
                "z_value": [z],
                "model_name": [model_name],
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

def generate_outcome_training_per_z(dataset: pd.DataFrame, target: list, model: str):
    # Select measurement columns
    measurement_cols = select_measurement_cols(dataset)
    print(f"Measurement columns ({len(measurement_cols)}): {measurement_cols}")

    outcome_training_per_z = pd.DataFrame()
        
    group_encoded = generate_grouped_dataset(dataset)
    print('Generation of the windowed dataset was ultimated')

    for z in Z:
        windowed_dataset = pd.DataFrame()
            
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for i, (train_index, test_index) in enumerate(gss.split(dataset, None, group_encoded)):
            
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]

            print("Train dataset length: {}, test dataset length: {} obtained for Z: {}".format(len(train_set),len(test_set), z))
            X_train = compute_windowed_dataset(train_set, z, measurement_cols)
            X_test = compute_windowed_dataset(test_set, z, measurement_cols)
            windowed_dataset = pd.concat([X_train, X_test], axis=0, ignore_index=True)
                
            print("Windowed dataset computed for z = {}\n".format(z))
            print("Length dataset: {}\n".format(len(windowed_dataset)))
            
            labels = np.unique(windowed_dataset["rat"])

            y_train = X_train["rat"]
            y_test = X_test["rat"]
            X_train.drop("rat", axis=1, inplace=True)
            X_test.drop("rat", axis=1, inplace=True)

            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

            split_outcome = {
                "X_train": pd.DataFrame(X_train_norm),
                "X_test": pd.DataFrame(X_test_norm),
                "y_train": pd.DataFrame(y_train),
                "y_test": pd.DataFrame(y_test),
            }
            outcome_training = generate_outcome_classification(windowed_dataset, split_outcome, z, labels, target, model)
            outcome_training_per_z = pd.concat([outcome_training_per_z, pd.DataFrame(outcome_training)], ignore_index=True)

    return outcome_training_per_z


def store_windowed_training_analysis(outcome_windowed_dataset_eval, model_name):
    os.makedirs(path_main_folders["ANALYSIS_DIR"], exist_ok=True)
    os.makedirs(path_main_folders["EXPORTED_DIR"], exist_ok=True)
    
    filename = path_files["Z_WINDOW_EVAL_NN"] if model_name == "MLPClassifier" else path_files["Z_WINDOW_EVAL_RF"]

    new_data = {
        "window_size": [int(outcome_windowed_dataset_eval["window_size"])],
        "windowed_dataset_size": [int(outcome_windowed_dataset_eval["windowed_dataset"].shape[1])],
        "windowed_dataset_length": [int(outcome_windowed_dataset_eval["windowed_dataset"].shape[0])],
        "windowed_dataset_columns": [outcome_windowed_dataset_eval["windowed_dataset"].columns.to_list()]
    }

    store_json_content(filename, new_data)
    
    label = "_"+("NN" if model_name == "MLPClassifier" else "RF")

    # To store the selected windowed dataset
    filename = os.path.join(path_main_folders["EXPORTED_DIR"],"aggregated_features{}.csv".format(label))

    outcome_windowed_dataset_eval["windowed_dataset"].to_csv(filename, index=False)
    
    # To store the selected model in pkl format
    joblib.dump(outcome_windowed_dataset_eval["selected_model"], os.path.join(path_main_folders["EXPORTED_DIR"],model_name)+".pkl")