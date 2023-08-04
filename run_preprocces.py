#%% imports
import mne 
import os
import mne 
import numpy as np
from eeg import Eeg as eeg
import pandas as pd
from mne_preproccessing import mne_preprocessing
from mne import concatenate_epochs
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
#%% function that helps us load the data into lists of that data.
def load_npy_series(path, base_filename, start=1, end=10, data_list=None,from_folder = False,folder_name = None):
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
            file_path = os.path.join(path,f"{folder_name}{i}", f"{base_filename}{i}.npy")
        else:
            file_path = os.path.join(path, f"{base_filename}{i}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)
        else:
            print(f"File {file_path} not found.")
    return data_list

def load_csv_series(path, base_filename, start=1, end=10, df_list=None,from_folder = False,folder_name = None):
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
                file_path = os.path.join(path,f"{folder_name}{i}", f"{base_filename}{i}.csv")
        else:
            file_path = os.path.join(path, f"{base_filename}{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
        else:
            print(f"File {file_path} not found.")
    return df_list
#%% define rather you want to create plot, or save a pickle of  
# all the epochs list ,if to remove bad records from  the new helmet records.
create_plots = False
save_pickle = True
remove_new_bad_records = True
trial_rejection = True
indexes_to_remove = [0,2,7,14] #indexes of bad records [1,3,8,15]

#%% Load your data
#define path for the plots
plot_path_new = "C:\\Users\\Cheif\\Desktop\\bci4als\\BCI4ALS\\plots\\Michael_new_helmet"
plot_path_old = "C:\\Users\\Cheif\\Desktop\\bci4als\\BCI4ALS\\plots\\Michael_old_helmet"
#define basic names for the folders and files 
folder_name  = 'exp_num_'
base_filename = 'records_'
#%% load the data %%#
#load the records data from the new helmet : all records are 1 block and 200 trials
path_new = 'C:\\Users\\Cheif\\Desktop\\bci4als\\records\\Michael_new_helmet'
data_list_new = load_npy_series(path_new, base_filename, start=1, end=16, data_list=None,from_folder = True,folder_name = folder_name) 
event_table_new = load_csv_series(path_new, base_filename, start=1, end=16, df_list=None,from_folder = True,folder_name = folder_name)
eeg_data_list_new = []
epoched_data_list_new = []

#load the records data from the new helmet : all records are 1 block and 200 trials
path_old = 'C:\\Users\\Cheif\\Desktop\\bci4als\\records\\Michael_old_helmet'
data_list_old = load_npy_series(path_old, base_filename, start=17, end=25, data_list=None,from_folder = True,folder_name = folder_name)
event_table_old = load_csv_series(path_old, base_filename, start=17, end=25, df_list=None,from_folder = True,folder_name = folder_name)
eeg_data_list_old = []
epoched_data_list_old = []
#%%##### pre proceessing the data - creating plots, and epochs %%%%
#set the reject trials critertia 
reject =  250

#%%loop over all the files in the new helmet 
for block,data in tqdm(enumerate(data_list_new)):    
    #create list of our preproccsing object using mne objects of mne, filtered already by defult of the class:
    #   Sfreq = 125, notch filter = 50 , band pass filter = min :0.5, max :40
    eeg_data_list_new.append(mne_preprocessing(data,event_table_new[block],new = True))
    eeg_data_list_new[block].epoch_it()
    if create_plots:
        #create plots and save them - no show! 
        # define the directory 
        directory_name = f"{plot_path_new}\\exp_num_{block+1}"
        # created the plots for each expiriment already so not relevant now.
        epochs.all_plots(dir = directory_name , exp_num = i+1,epochs= epoched_data_list_new[i]) #dont have to use the epochs argument  
    #replace bad trial in a componnet to NaN and document them
    if trial_rejection:
        eeg_data_list_new[block].trial_rejections(rejection_critrerion_amp = reject,
                            block = block , save_res = True)
        #create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
    epoched_data_list_new.append(eeg_data_list_new[block].epochs)
    

for block ,data in tqdm(enumerate(data_list_old)):    
    #create list of our preproccsing object using mne objects of mne, filtered already by defult of the class:
    #Sfreq = 125, notch filter = 50 , band pass filter = min :0.5, max :40
    eeg_data_list_old.append(mne_preprocessing(data,event_table_new[block],new = False))
    epochs = eeg_data_list_old[block].epoch_it()
    if create_plots:
            ##create plots and save them - no show! 
        ## define the directory 
        directory_name = f"{plot_path_old}\\exp_num_{i+1}"
        # create the plots and save them using a function the the class
        epochs[i].all_plots(dir = directory_name , exp_num = i+1,epochs= epoched_data_list_old[i]) #dont have to use the epochs argument  
   
    ##replace bad trial in a componnet to NaN and document them
    if trial_rejection:
        eeg_data_list_old[block].trial_rejections(rejection_critrerion_amp = reject,
                            block = block , save_res = True)
    #create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
    epoched_data_list_old.append(eeg_data_list_new[block].epochs)
    
#remove bad records manually from the new helmet
if remove_new_bad_records:
    # Sort indexes in descending order to avoid shifting
    epoched_data_list_new = [ele for i, ele in enumerate(epoched_data_list_new)
                             if i not in indexes_to_remove]

#%% create a concatenedted object of all the epoched data for both new and old helmet    
all_epoched_new= concatenate_epochs(epoched_data_list_new)
all_epoched_old = concatenate_epochs(epoched_data_list_old)


#%%create plots over all the epochs together for both helmets
#new helmet:
#define directory name
if create_plots:
    directory_name_new = f"{plot_path_new}\\exp_num_"
    #create the plots and save them using a function the the class
    eeg_data_list_new[0].all_plots(dir = directory_name_new + "All" ,exp_num = 'All',epochs= all_epoched_new)
    #old helmet 
    #define directory name
    directory_name_old = f"{plot_path_old}\\exp_num_"
    #create the plots and save them using a function the the class
    eeg_data_list_new[0].all_plots(dir = directory_name_old + "All" ,exp_num = 'All',epochs= all_epoched_old)

#%% save the lists for later use in the classification part
if save_pickle:
    with open('old_helmets_epochs.pkl', 'wb') as f:
        pickle.dump(epoched_data_list_old, f)

    with open('new_helmets_epochs.pkl', 'wb') as f:
        pickle.dump(epoched_data_list_new, f)
print("Done")



"""
some tests        
"""

# for i in range(len(epoched_data_list_new)):
#     mean_list_new.append(epoched_data_list_new[i].average()._data.transpose().mean(axis = 0))
# mean_new_df = pd.DataFrame(mean_list_new, columns = epoched_data_list_new[0].ch_names[:-1])

# for i in range(len(epoched_data_list_old)):
#     mean_list_old.append(epoched_data_list_old[i].average()._data.transpose().mean(axis = 0))
# mean_old_df = pd.DataFrame(mean_list_old, columns = epoched_data_list_old[0].ch_names[:-1])
