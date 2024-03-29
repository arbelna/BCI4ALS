# %% imports
import mne
import os
import mne
import numpy as np
from eeg import Eeg as eeg
import pandas as pd
from mne import concatenate_epochs
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import warnings
from mne_preprocessing import  mne_preprocessing
from mne_icalabel import label_components
# %% function that helps us load the data into lists of that data.
def load_npy_series(path, base_filename, start=1, end=10, data_list=None, from_folder=False, folder_name=None):
    """
    Load a series of .npy files with similar names and different trailing numbers.

    Args:
    path (str): Directory containing the .npy files.
    base_filename (str): Base name of the files to load.
    start (int): Starting number for the file sequence.
    end (int): Ending number for the file sequence.

    Returns:
    list: A list of NumPy arrays loaded from the files.
    """
    if not data_list:
        data_list = []
    for i in range(start, end + 1):
        if from_folder:
            file_path = os.path.join(path, f"{folder_name}{i}", f"{base_filename}{i}.npy")
        else:
            file_path = os.path.join(path, f"{base_filename}{i}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)
        else:
            print(f"File {file_path} not found.")
    return data_list

def load_csv_series(path, base_filename, start=1, end=10, df_list=None, from_folder=False, folder_name=None):
    """
    Load a series of .csv files with similar names and different trailing numbers.

    Args:
    path (str): Directory containing the .csv files.
    base_filename (str): Base name of the files to load.
    start (int): Starting number for the file sequence.
    end (int): Ending number for the file sequence.

    Returns:
    list: A list of pandas DataFrames loaded from the files.
    """
    if not df_list:
        df_list = []
    for i in range(start, end + 1):
        if from_folder:
            file_path = os.path.join(path, f"{folder_name}{i}", f"{base_filename}{i}.csv")
        else:
            file_path = os.path.join(path, f"{base_filename}{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
        else:
            print(f"File {file_path} not found.")
    return df_list

# %% define rather you want to create plot, or save a pickle of
# all the epochs list ,if to remove bad records from  the new helmet records.
create_plots = True
save_pickle = False
remove_new_bad_records = False
trial_rejection = False
save_res = False # save results of each bad trial individually
also_old = False # also run creat epochs for the old helmet
indexes_to_remove = [2,3,7,14,20,21,26,27] #indexes of bad records [3,4,8,15,21,22,27,28]

# %% Load your data
# define path for the plots
plot_path_new = "C:\\Users\\Cheif\\Desktop\\bci4als\\BCI4ALS\\plots\\Michael_new_helmet"
plot_path_old = "C:\\Users\\Cheif\\Desktop\\bci4als\\BCI4ALS\\plots\\Michael_old_helmet"
# define basic names for the folders and files
folder_name = 'exp_num_'
base_filename = 'records_'
# %% load the data %%#
# load the records data from the new helmet : all records are 1 block and 200 trials
path_new = 'C:\\Users\\Cheif\\Desktop\\bci4als\\records\\Michael_new_helmet'
data_list_new = load_npy_series(path_new, base_filename, start=1, end=28, data_list=None,from_folder = True,folder_name = folder_name) 
event_table_new = load_csv_series(path_new, base_filename, start=1, end=28, df_list=None,from_folder = True,folder_name = folder_name)
eeg_data_list_new = []
epoched_data_list_new = []

# load the records data from the new helmet : all records are 1 block and 200 trials
path_old = 'C:\\Users\\Cheif\\Desktop\\bci4als\\records\\Michael_old_helmet'
data_list_old = load_npy_series(path_old, base_filename, start=17, end=25, data_list=None, from_folder=True,
                                folder_name=folder_name)
event_table_old = load_csv_series(path_old, base_filename, start=17, end=25, df_list=None, from_folder=True,
                                  folder_name=folder_name)
eeg_data_list_old = []
epoched_data_list_old = []
bad_trials_new = []
sum_channels_bad_new = []
bad_trials_old = []
sum_channels_bad_old = []

# %%##### pre proceessing the data - creating plots, and epochs %%%%
# set the reject trials critertia
reject = 250
# remove all warnings from this runs - to make it more readable
print("I solemnly swear that i am up to no good")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
   
    # %%loop over all the files in the new helmet
    for block, data in tqdm(enumerate(data_list_new)):
        # create list of our preproccsing object using mne objects of mne, filtered already by defult of the class:
        #   Sfreq = 125, notch filter = 50 , band pass filter = min :0.5, max :40
        eeg_data_list_new.append(mne_preprocessing(data, event_table_new[block], new=True,re_refrence=True))
        eeg_data_list_new[block].epoch_it()
        if create_plots:
            # create plots and save them - no show!
            # define the directory 
            directory_name = f"{plot_path_new}\\exp_num_{block + 1}"
            # created the plots for each expiriment already so not relevant now.
            eeg_data_list_new[block].all_plots(dir=directory_name, exp_num=block + 1)
        # replace bad trial in a componnet to NaN and document them
        if trial_rejection:
            _, bad_trials_df, ch_trial_rejected_df = eeg_data_list_new[block].trial_rejections(
                rejection_critrerion_amp=reject,
                block=block, save_res=save_res)
            if save_res == False:
                bad_trials_new.append(bad_trials_df)
                sum_channels_bad_new.append(ch_trial_rejected_df)

                # create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
        epoched_data_list_new.append(eeg_data_list_new[block].epochs)
        if trial_rejection and save_res == False:
            pd.concat(bad_trials_new,ignore_index=True).to_csv(f"bad_trials_new.csv")
            pd.concat(sum_channels_bad_new,ignore_index=True).to_csv(f"sum_channels_bad_new.csv")
 
    
    # remove bad records manually from the new helmet    
    if remove_new_bad_records:
        # Sort indexes in descending order to avoid shifting
        epoched_data_list_new = [ele for i, ele in enumerate(epoched_data_list_new)
                                 if i not in indexes_to_remove]
        eeg_data_list_new = [ele for i, ele in enumerate(eeg_data_list_new)
                                 if i not in indexes_to_remove]

    if also_old :
        for block ,data in tqdm(enumerate(data_list_old)):    
            #create list of our preproccsing object using mne objects of mne, filtered already by defult of the class:
            #Sfreq = 125, notch filter = 50 , band pass filter = min :0.5, max :40
            eeg_data_list_old.append(mne_preprocessing(data,event_table_new[block],new = False))
            eeg_data_list_old[block].epoch_it()
            if create_plots:
                ##create plots and save them - no show!
                ## define the directory 
                directory_name = f"{plot_path_old}\\exp_num_{block + 1}"
                # create the plots and save them using a function the the class
                eeg_data_list_old[block].all_plots(dir=directory_name, exp_num=block + 1)
            ##replace bad trial in a componnet to NaN and document them
            if trial_rejection:
                _, bad_trials_df, ch_trial_rejected_df = eeg_data_list_old[block].trial_rejections(
                    rejection_critrerion_amp=reject,
                    block=block, save_res=save_res)
                if save_res == False:
                    bad_trials_old.append(bad_trials_df)
                sum_channels_bad_old.append(ch_trial_rejected_df)
                # create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
            epoched_data_list_old.append(eeg_data_list_new[block].epochs)
            if trial_rejection and save_res == False:
                pd.concat(bad_trials_old, ignore_index=True).to_csv(f"bad_trials_old.csv")
                pd.concat(sum_channels_bad_old, ignore_index=True).to_csv(f"sum_channels_bad_old.csv")
    
    # %% save the lists for later use in the classification part
    if save_pickle:
        if also_old:
             with open('old_helmets_epochs.pkl', 'wb') as f:
                 pickle.dump(epoched_data_list_old, f)

        with open('new_helmets_epochs.pkl', 'wb') as f:
            pickle.dump(epoched_data_list_new, f)
print("mischief managed")
