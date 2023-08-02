import pandas as pd
import numpy as np
from mne import concatenate_epochs
import pickle
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

with open('new_helmets_epochs.pkl', 'rb') as f:
    new_helmets_rec = pickle.load(f)


def get_relevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i in indexes]


def remove_irrelevant_blocks(records, indexes):
    return [ele for i, ele in enumerate(records) if i not in indexes]


with open('old_helmets_epochs.pkl', 'rb') as f:
    old_helmet_rec = pickle.load(f)

# # remove bad records
# indexes_to_remove = [0, 2, 7, 14]  # indexes of bad records [1,3,8,15]
# new_helmets_rec = remove_irrelevant_blocks(new_helmets_rec, indexes_to_remove)
#
# test_new_helmet_indexes = [4, 6, 9, 10]
# train_new_helmet_indexes = [0, 1, 2, 3, 5, 7, 8, 11]
# train_new_helmet = get_relevant_blocks(new_helmets_rec, train_new_helmet_indexes)
# test_new_helmet = get_relevant_blocks(new_helmets_rec, test_new_helmet_indexes)


train_old_helmet_indexes = [0, 1, 3, 4, 6, 7]
test_old_helmet_indexes = [2, 5, 8]

train_old_helmet = get_relevant_blocks(old_helmet_rec, train_old_helmet_indexes)
test_old_helmet = get_relevant_blocks(old_helmet_rec, test_old_helmet_indexes)

train_old_helmet = concatenate_epochs(train_old_helmet)
test_old_helmet = concatenate_epochs(test_old_helmet)

target_old_train = train_old_helmet['target']._get_data()
non_target_old_train = train_old_helmet['non-target']._get_data()

target_new_train = target_old_train[:, :13, :]
non_target_new_train = non_target_old_train[:, :13, :]

target_new_train = np.reshape(target_new_train,
                              (target_new_train.shape[1], target_new_train.shape[0], target_new_train.shape[2]))
non_target_new_train = np.reshape(non_target_new_train, (
    non_target_new_train.shape[1], non_target_new_train.shape[0], non_target_new_train.shape[2]))

labels = np.hstack((np.ones(target_new_train.shape[1]), np.zeros(non_target_new_train.shape[1])))

# Concatenate them along the second axis
X = np.concatenate((target_new_train, non_target_new_train), axis=1)

# Define your hyperparameters grid
param_grid = {'n_estimators': [1000, 1500, 2000],
              'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': [15, 20],
              'max_features': ['sqrt', 'log2', None]
              }

# Create a DataFrame to save the results
results_df = pd.DataFrame(columns=['Channels', 'Hyperparameters', 'Train Accuracy', 'Validation Accuracy'])

n_channels, n_trials, n_samples = X.shape

# Total number of odd-sized combinations of channels
total_combinations = sum(1 for r in range(1, n_channels + 1, 2) for _ in combinations(range(n_channels), r))

# Iterate through odd-sized combinations of channels
for r in range(1, n_channels + 1, 2):
    for channels in tqdm(combinations(range(n_channels), r), total=total_combinations, desc="Processing combinations"):
        print(f"Running for channels: {channels}")

        # Select the data for the current combination of channels
        selected_data = X[channels, :, :].reshape(-1, n_samples)

        # Repeat the labels for each channel in the combination
        selected_labels = np.repeat(labels, len(channels))

        # Split the selected data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_labels, test_size=0.2, random_state=42)

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


