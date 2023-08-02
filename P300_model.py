# Load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from collections import Counter
# from psychopy import visual
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from mne import concatenate_epochs
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_relevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i in indexes]


def remove_irrelevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i not in indexes]


class P300_model:
    def __init__(self, data_path=None):
        self.shape_of_sad = None
        self.shape_of_happy = None
        self.X_Train = None
        self.y_Train = None
        self.X_Test_happy = None
        self.X_Test_sad = None
        self.clf = None
        self.num_channels = None

    def create_x_y(self, data, blocks, train=True, new=False):
        """
        Transforming the results from the records into input fot the model
        :param blocks: List of the relevant blocks
        :param new: Old helmet or new helmet
        :param train: boolean var to check if to preprocess the date as input for train or test
        :param data: List of MNE epochs objects
        """
        if train:
            train = get_relevant_blocks(data, blocks)
            train = concatenate_epochs(train)
            target_train = train['target']._get_data()
            non_target_train = train['non-target']._get_data()
            if new:
                target_train = target_train[:, :9, :]
                non_target_train = non_target_train[:, :9, :]
            else:
                target_train = target_train[:, :13, :]
                non_target_train = non_target_train[:, :13, :]
            target_train = np.reshape(target_train,
                                      (target_train.shape[1], target_train.shape[0],
                                       target_train.shape[2]))
            non_target_train = np.reshape(non_target_train,
                                          (non_target_train.shape[1], non_target_train.shape[0],
                                           non_target_train.shape[2]))
            self.X_Train = np.concatenate((target_train, non_target_train), axis=1)
            self.y_Train = np.hstack((np.ones(target_train.shape[1]), np.zeros(non_target_train.shape[1])))

        else:
            train = get_relevant_blocks(data, blocks)
            target_test = train['target']._get_data()
            non_target_test = train['non-target']._get_data()
            if new:
                target_test = target_test[:, :9, :]
                non_target_test = non_target_test[:, :9, :]
            else:
                target_test = target_test[:, :13, :]
                non_target_test = non_target_test[:, :13, :]
            target_test = np.reshape(target_test,
                                     (target_test.shape[1], target_test.shape[0],
                                      target_test.shape[2]))
            non_target_test = np.reshape(non_target_test,
                                         (non_target_test.shape[1], non_target_test.shape[0],
                                          non_target_test.shape[2]))
            self.X_Test_happy = target_test
            self.X_Test_sad = non_target_test

    def train_modelCV(self, relevant_channels=None, min_channels=1):
        """
        :param min_channels: minimal number of channels to use for grid search (odd number)
        :param relevant_channels: relevant channels
        Training the model anf finding the optimal hyperparameters,
        the results of each combination of channels are saved in a DataFrame and in a csv file
        """
        warnings.filterwarnings('ignore')

        # Define your hyperparameters grid
        param_grid = {'n_estimators': [1000, 1500, 2000],
                      'criterion': ['gini', 'entropy', 'log_loss'],
                      'max_depth': [15, 20],
                      'max_features': ['sqrt', 'log2', None]
                      }

        # Create a DataFrame to save the results
        results_df = pd.DataFrame(columns=['Channels', 'Hyperparameters', 'Train Accuracy', 'Validation Accuracy'])
        X_Train = self.X_Train.copy()
        if relevant_channels is not None:
            X_Train = X_Train[relevant_channels, :, :]

        n_channels, n_trials, n_samples = X_Train.shape

        # Total number of odd-sized combinations of channels
        total_combinations = sum(1 for r in range(1, n_channels + 1, 2) for _ in combinations(range(n_channels), r))

        # Iterate through odd-sized combinations of channels
        for r in range(min_channels, n_channels + 1, 2):
            for channels in tqdm(combinations(range(n_channels), r), total=total_combinations,
                                 desc="Processing combinations"):
                print(f"Running for channels: {channels}")

                if relevant_channels is not None:
                    # Select the data for the current combination of channels
                    X_Train = X_Train[channels, :, :].reshape(-1, n_samples)

                # Repeat the labels for each channel in the combination
                y_Train = np.repeat(self.y_Train, len(channels))

                # Split the selected data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train, test_size=0.2,
                                                                    random_state=42)

                # Create the Random Forest model
                rf = RandomForestClassifier()

                # Run the grid search with parallelization
                grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Get the best hyperparameters
                best_hyperparams = grid_search.best_params_
                print(f"Best hyperparameters: {best_hyperparams}")

                # Get the accuracy scores
                train_accuracy = grid_search.best_estimator_.score(X_train, y_train)
                validation_accuracy = grid_search.best_estimator_.score(X_test, y_test)

                print(f"Train Accuracy: {train_accuracy}")
                print(f"Validation Accuracy: {validation_accuracy}")

                # Save the results to the DataFrame
                results_to_add = pd.DataFrame({
                    'Channels': [channels],
                    'Hyperparameters': [best_hyperparams],
                    'Train Accuracy': [train_accuracy],
                    'Validation Accuracy': [validation_accuracy]
                })
                results_df = pd.concat([results_df, results_to_add], ignore_index=True)

                # Save the DataFrame to a CSV file after each combination
                results_df.to_csv('grid_search_results.csv', index=False)

    def train_final_model(self, hyperparameters, relevant_channels=None):

        # Create the Random Forest model
        self.clf = RandomForestClassifier(**hyperparameters)
        self.num_channels = len(relevant_channels)

        if relevant_channels is not None:
            # Select the data for the current combination of channels
            self.X_Train = self.X_Train[relevant_channels, :, :].reshape(-1, self.X_Train.shape[2])
            self.y_Train = np.repeat(self.y_Train, len(relevant_channels))
        else:
            self.X_Train = self.X_Train.reshape(-1, self.X_Train.shape[2])
            self.y_Train = np.repeat(self.y_Train, len(self.X_Train.shape[0]))

        print('Training the model')
        self.clf.fit(self.X_Train, self.y_Train)

    def test_model(self, threshold=0.5):
        happy_predicted_classes = self.clf.predict(np.array(self.X_Test_happy))
        happy_predicted_classes = self.reshape_test(happy_predicted_classes)
        happy_trials = []
        for i in range(int(len(happy_predicted_classes) / self.num_channels)):
            trail_proba = np.mean(happy_predicted_classes[i * self.num_channels:(i + 1) * self.num_channels])
            if trail_proba > threshold:
                happy_trials.append(1)
            else:
                happy_trials.append(0)
        happy_chance = sum(happy_trials) / len(happy_trials)

        sad_predicted_classes = self.clf.predict(np.array(self.X_Test_sad))
        sad_predicted_classes = self.reshape_test(sad_predicted_classes, happy=False)
        sad_trials = []
        for i in range(int(len(sad_predicted_classes) / self.num_channels)):
            trail_proba = np.mean(sad_predicted_classes[i * self.num_channels:(i + 1) * self.num_channels])
            if trail_proba > threshold:
                sad_trials.append(1)
            else:
                sad_trials.append(0)
        sad_chance = sum(sad_trials) / len(sad_trials)

        return happy_predicted_classes, happy_trials, happy_chance, sad_predicted_classes, sad_trials, sad_chance

        # if happy_chance > sad_chance:
        #     print('Yes')
        #     return 1
        # elif happy_chance < sad_chance:
        #     print('No')
        #     return 0
        # else:
        #     return 2

    def reshape_test(self, lst, happy=True):
        if happy:
            idx = 0
            new_pred = []
            for i in range(len(self.shape_of_happy)):
                if self.shape_of_happy[i] == 1:
                    new_pred.append(lst[idx])
                    idx += 1
                else:
                    new_pred.append(0.5)
            return new_pred
        else:
            idx = 0
            new_pred = []
            for i in range(len(self.shape_of_sad)):
                if self.shape_of_sad[i] == 1:
                    new_pred.append(lst[idx])
                    idx += 1
                else:
                    new_pred.append(0.5)
            return new_pred
