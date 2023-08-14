from P300_model import P300_model
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

with open('new_helmets_epochs.pkl', 'rb') as f:
    new_helmets_rec = pickle.load(f)

test_new_helmet_indexes = [4, 6, 9, 10]
train_new_helmet_indexes = [0, 1, 2, 3, 5, 7, 8, 11]

train  = True
test = False
# Define your hyperparameters grid
param_grid_randomforest = {
    'n_estimators': [500, 1000, 1500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [7, 10, 15],
    'max_features': ['sqrt', 0.2],
    'min_samples_leaf': [3, 7, 10]
}

param_grid_xgb = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 15],
    'subsample': [0.75, 0.9],
    'max_features': ['sqrt', 0.2],
    'min_samples_leaf': [3, 7, 10]
}


hyperparameters = {'criterion': 'entropy', 'max_depth': 15, 
                    'max_features': 0.2, 'min_samples_leaf': 3
                    ,'n_estimators': 500,'n_jobs': -1
                    }
                   
                   
relevant_channels = [2, 6,7]
new_helmet_model = P300_model()
new_helmet_model.create_x_y(new_helmets_rec, train_new_helmet_indexes, new=True)


if train:
 model_list   = new_helmet_model.train_modelCV(GradientBoostingClassifier(), param_grid_xgb,min_channels=1, max_channels=2)

 
if test:
    new_helmet_model.train_final_model(RandomForestClassifier(), hyperparameters, relevant_channels)
    for block in test_new_helmet_indexes:
        new_helmet_model.create_x_y(new_helmets_rec, [block], train=False, new=True)
        happy_predicted_classes, happy_trials, happy_chance, sad_predicted_classes, sad_trials, sad_chance =  new_helmet_model.test_model()
        print(happy_chance>sad_chance)
print("goodbye")
