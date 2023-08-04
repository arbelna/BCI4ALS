# Load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from itertools import combinations
from mne import concatenate_epochs
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_relevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i in indexes]


def remove_irrelevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i not in indexes]


class P300_model:
    def __init__(self):
        self.shape_of_sad = None
        self.shape_of_happy = None
        self.X_Train = None
        self.y_Train = None
        self.X_Test_happy = None
        self.X_Test_sad = None
        self.clf = None
        self.num_channels = None
        self.relevant_channels = None

    def create_x_y(self, data, blocks, train=True, new=False):
        """
        Transforming the results from the records into input fot the model
        :param blocks: List of the relevant blocks
        :param new: Old helmet or new helmet
        :param train: boolean var to check if to preprocess the date as input for train or test
        :param data: List of MNE epochs objects
        """
        relevant_data = get_relevant_blocks(data, blocks)
        relevant_data = concatenate_epochs(relevant_data)
        target = relevant_data['target']._get_data()
        non_target = relevant_data['non-target']._get_data()
        if new:
            target = target[:, :9, :]
            non_target = non_target[:, :9, :]
        else:
            target = target[:, :13, :]
            non_target = non_target[:, :13, :]
        target = np.reshape(target,
                            (target.shape[1], target.shape[0],
                             target.shape[2]))
        non_target = np.reshape(non_target,
                                (non_target.shape[1], non_target.shape[0],
                                 non_target.shape[2]))
        if train:
            self.X_Train = np.concatenate((target, non_target), axis=1)
            self.y_Train = np.hstack((np.ones(target.shape[1]), np.zeros(non_target.shape[1])))

        else:
            self.X_Test_happy = target
            self.X_Test_sad = non_target

    def train_modelCV(self, clf, param_grid, relevant_channels=None, min_channels=1):
        """
        :param param_grid: hyperparameters for thr grid search
        :param clf: model to train
        :param min_channels: minimal number of channels to use for grid search (odd number)
        :param relevant_channels: relevant channels
        Training the model anf finding the optimal hyperparameters,
        the results of each combination of channels are saved in a DataFrame and in a csv file
        """
        warnings.filterwarnings('ignore')

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

                # Run the grid search with parallelization
                grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
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

    def train_final_model(self, clf, hyperparameters, relevant_channels=None):
        """
           Trains the final model using the given classifier, hyperparameters, and relevant channels.

           :param clf: Classifier object (e.g., RandomForestClassifier).
           :param hyperparameters: Dictionary containing hyperparameters to be set for the classifier.
           :param relevant_channels: List of relevant channels to be considered in training, default is None.
                                     If provided, the data will be reshaped accordingly.

           :return: None. The method updates the classifier (self.clf) in place with the trained model.
           """
        self.clf = clf.set_params(**hyperparameters)
        self.relevant_channels = relevant_channels
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

    def test_model(self):
        """
        Tests the model using the test data for two categories: "happy" and "sad".

        The method first reshapes the test data according to the relevant channels,
        then predicts the classes for both categories and computes the average across channels.
        It then compares the chances for "happy" and "sad" categories and prints and returns the result.

        :return: 1 if "happy" chance is greater, 0 if "sad" chance is greater, or None if they are equal.
        """

        # Reshaping and predicting for the answer "Yes"
        self.X_Test_happy = self.X_Test_happy[self.relevant_channels, :, :].reshape(-1, self.X_Test_happy.shape[2])
        happy_predicted_classes = self.clf.predict(np.array(self.X_Test_happy))
        num_columns = len(happy_predicted_classes) // len(self.relevant_channels)
        happy_matrix = happy_predicted_classes.reshape(num_columns, (len(self.relevant_channels))).T
        happy_trials = np.round(np.mean(happy_matrix, axis=0))
        happy_matrix = np.vstack((happy_matrix, happy_trials))
        happy_chance = np.mean(happy_matrix[-1, :])

        # Reshaping and predicting for the answer "No"
        self.X_Test_sad = self.X_Test_sad[self.relevant_channels, :, :].reshape(-1, self.X_Test_sad.shape[2])
        sad_predicted_classes = self.clf.predict(np.array(self.X_Test_sad))
        num_columns = len(sad_predicted_classes) // len(self.relevant_channels)
        sad_matrix = sad_predicted_classes.reshape(num_columns, (len(self.relevant_channels))).T
        sad_trials = np.round(np.mean(sad_matrix, axis=0))
        sad_matrix = np.vstack((sad_matrix, sad_trials))
        sad_chance = np.mean(sad_matrix[-1, :])

        # return happy_predicted_classes, happy_trials, happy_chance, sad_predicted_classes, sad_trials, sad_chance

        # Comparing chances and returning result
        if happy_chance > sad_chance:
            print('Yes')
            return 1
        elif happy_chance < sad_chance:
            print('No')
            return 0
        else:
            return None

    # def reshape_test(self, lst, happy=True):
    #     if happy:
    #         idx = 0
    #         new_pred = []
    #         for i in range(len(self.shape_of_happy)):
    #             if self.shape_of_happy[i] == 1:
    #                 new_pred.append(lst[idx])
    #                 idx += 1
    #             else:
    #                 new_pred.append(0.5)
    #         return new_pred
    #     else:
    #         idx = 0
    #         new_pred = []
    #         for i in range(len(self.shape_of_sad)):
    #             if self.shape_of_sad[i] == 1:
    #                 new_pred.append(lst[idx])
    #                 idx += 1
    #             else:
    #                 new_pred.append(0.5)
    #         return new_pred
