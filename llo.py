from sklearn.model_selection import LeaveOneOut
import pandas as pd
import pickle
from P300_model import P300_model

with open('new_helmets_epochs.pkl', 'rb') as f:
    new_helmets_rec = pickle.load(f)

# Splitting indexes into train and test using Leave One Out method
loo = LeaveOneOut()
indexes = [i for i in range(12)]  # Assuming that new_helmets_rec is a list or something with len

hyperparameters = {'n_estimators': 2000,
                   'criterion': 'log_loss',
                   'max_depth': 20,
                   'max_features': None
                   }
relevant_channels = [1, 2, 4]

new_helmet_model = P300_model()

# DataFrame to hold the results
results_df = pd.DataFrame(
    columns=["test_index", "happy_predicted_classes", "happy_trials", "happy_chance", "sad_predicted_classes", "sad_trials",
             "sad_chance"])

for train_indexes, test_indexes in loo.split(indexes):
    train_new_helmet_indexes = [indexes[i] for i in train_indexes]
    test_new_helmet_indexes = [indexes[i] for i in test_indexes]

    print("Test indexes: ", test_new_helmet_indexes)

    new_helmet_model.create_x_y(new_helmets_rec, train_new_helmet_indexes, new=True)
    new_helmet_model.train_final_model(hyperparameters, relevant_channels)

    new_helmet_model.create_x_y(new_helmets_rec, test_new_helmet_indexes, train=False, new=True)
    happy_predicted_classes, happy_trials, happy_chance, sad_predicted_classes, sad_trials, sad_chance = new_helmet_model.test_model()

    # Store the results
    results_df = results_df.append({
        "test_index": test_new_helmet_indexes[0],
        "happy_predicted_classes": happy_predicted_classes,
        "happy_trials": happy_trials,
        "happy_chance": happy_chance,
        "sad_predicted_classes": sad_predicted_classes,
        "sad_trials": sad_trials,
        "sad_chance": sad_chance
    }, ignore_index=True)

    # Save the DataFrame to a CSV file after each combination
    results_df.to_csv('loo_results.csv', index=False)

# Save the results as a pickle file
with open('loo_results.pkl', 'wb') as f:
    pickle.dump(results_df, f)
