import warnings 
save_res = False # save results of each bad trial individually
bad_trials_new = []
sum_channels_bad_new = []
bad_trials_old = []
sum_channels_bad_old = []

#remove all warnings from this runs - to make it more readable
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
            eeg_data_list_new[block].epochs.all_plots(dir = directory_name , exp_num = block+1
                                                      ,epochs= epoched_data_list_new[block]) #dont have to use the epochs argument  
        #replace bad trial in a componnet to NaN and document them
        if trial_rejection:
            _ ,bad_trials_df ,ch_trial_rejected_df = eeg_data_list_new[block].trial_rejections(rejection_critrerion_amp = reject,
                                block = block , save_res = True)
            if save_res == False:
                bad_trials_new.append(bad_trials_df)
                sum_channels_bad_new.append(ch_trial_rejected_df)    
        
            #create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
        epoched_data_list_new.append(eeg_data_list_new[block].epochs)
        pd.concat(bad_trials_old,ignore_index=True).to_csv(f"bad_trials_new.csv")
        pd.concat(sum_channels_bad_old,ignore_index=True).to_csv(f"sum_channels_bad_new.csv")

    for block ,data in tqdm(enumerate(data_list_old)):    
        #create list of our preproccsing object using mne objects of mne, filtered already by defult of the class:
        #Sfreq = 125, notch filter = 50 , band pass filter = min :0.5, max :40
        eeg_data_list_old.append(mne_preprocessing(data,event_table_new[block],new = False))
        epochs = eeg_data_list_old[block].epoch_it()
        if create_plots:
                ##create plots and save them - no show! 
            ## define the directory 
            directory_name = f"{plot_path_old}\\exp_num_{block+1}"
            # create the plots and save them using a function the the class
            epochs[block].all_plots(dir = directory_name , exp_num = block+1,epochs= epoched_data_list_old[block]) #dont have to use the epochs argument  
        ##replace bad trial in a componnet to NaN and document them
        if trial_rejection:
            _,bad_trials_df, ch_trial_rejected_df  = eeg_data_list_old[block].trial_rejections(rejection_critrerion_amp = reject,
                                                                                               block = block , save_res = save_res)
        if save_res == False:
            bad_trials_old.append(bad_trials_df)
            sum_channels_bad_old.append(ch_trial_rejected_df)    
        #create list of epoched data - segmented and divted into trials : Idle, Target, Non Target
        epoched_data_list_old.append(eeg_data_list_new[block].epochs)
        pd.concat(bad_trials_old,ignore_index=True).to_csv(f"bad_trials_old.csv")
        pd.concat(sum_channels_bad_old,ignore_index=True).to_csv(f"sum_channels_bad_old.csv")
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
