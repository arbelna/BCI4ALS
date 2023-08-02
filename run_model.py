from P300_model import P300_model
import pickle
from sklearn.ensemble import RandomForestClassifier

with open('new_helmets_epochs.pkl', 'rb') as f:
    new_helmets_rec = pickle.load(f)

test_new_helmet_indexes = [4, 6, 9, 10]
train_new_helmet_indexes = [0, 1, 2, 3, 5, 7, 8, 11]
hyperparameters = {'n_estimators': 2000,
                   'criterion': 'log_loss',
                   'max_depth': 20,
                   'max_features': None,
                   'n_jobs': -1
                   }
relevant_channels = [1, 2, 4]

new_helmet_model = P300_model()
new_helmet_model.create_x_y(new_helmets_rec, train_new_helmet_indexes, new=True)
new_helmet_model.train_final_model(RandomForestClassifier(), hyperparameters, relevant_channels)

for block in test_new_helmet_indexes:
    new_helmet_model.create_x_y(new_helmets_rec, [block], train=False, new=True)
    new_helmet_model.test_model()

