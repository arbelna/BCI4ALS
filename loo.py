from sklearn.model_selection import LeaveOneOut
import pandas as pd
import pickle
from P300_model import P300_model
from sklearn.ensemble import RandomForestClassifier

with open('new_helmets_epochs.pkl', 'rb') as f:
    new_helmets_rec = pickle.load(f)

# Splitting indexes into train and test using Leave One Out method
loo = LeaveOneOut()
indexes = [i for i in range(len(new_helmets_rec))]  # Assuming that new_helmets_rec is a list or something with len

hyperparameters ={'criterion': 'entropy', 'max_depth': 15, 'max_features': 0.2, 'min_samples_leaf': 3, 'n_estimators': 500,'n_jobs': -1}
                   
                   
relevant_channels = [2, 6,7]

new_helmet_model = P300_model()

# DataFrame to hold the results
results_df = pd.DataFrame(
    columns=["test_index", "happy_predicted_classes", "happy_predicted_classes_percent", "happy_trials",
             "happy_chance", "sad_predicted_classes", "sad_predicted_classes_percent", "sad_trials",
             "sad_chance"])

for train_indexes, test_indexes in loo.split(indexes):
    train_new_helmet_indexes = [indexes[i] for i in train_indexes]
    test_new_helmet_indexes = [indexes[i] for i in test_indexes]

    print("Test indexes: ", test_new_helmet_indexes)

    new_helmet_model.create_x_y(new_helmets_rec, train_new_helmet_indexes, new=True)
    new_helmet_model.train_final_model(RandomForestClassifier(), hyperparameters, relevant_channels)

    new_helmet_model.create_x_y(new_helmets_rec, test_new_helmet_indexes, train=False, new=True)
    happy_predicted_classes, happy_trials, happy_chance, sad_predicted_classes, sad_trials, sad_chance = new_helmet_model.test_model()

    # Store the results
    results_df = results_df.append({
        "test_index": test_new_helmet_indexes[0],
        "happy_predicted_classes": happy_predicted_classes,
        "happy_predicted_classes_percent": sum(happy_predicted_classes)/len(happy_predicted_classes),
        "happy_trials": happy_trials,
        "happy_chance": happy_chance,
        "sad_predicted_classes": sad_predicted_classes,
        "sad_predicted_classes_percent": sum(sad_predicted_classes) / len(sad_predicted_classes),
        "sad_trials": sad_trials,
        "sad_chance": sad_chance
    }, ignore_index=True)

    # Save the DataFrame to a CSV file after each combination
    results_df.to_csv('loo1_results.csv', index=False)

# Save the results as a pickle file
with open('loo_results.pkl', 'wb') as f:
    pickle.dump(results_df, f)
