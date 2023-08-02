import mne 
import os
import mne 
import numpy as np
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from eeg import Eeg as eeg
import pandas as pd

class mne_preprocessing():
    """
    this class is used to preproccess the data using mne library and our eeg class
    for now it can only handle several records with only 1 block in them each time.
    params:
    data - the eeg records from the helmet (numpy array)
    event_table - the event table of the data (pandas dataframe)
    new - a boolean that tells us if it was record using old or new helmet (bool; default = False)
        new helmet - 9 channels
        old helmet - 13 channels
    sfreq - the sampling frequency of the data (int; default = 125)
    notch - the notch frequency for notch filter  (int; default = 50)
    highcut - the highcut frequency for band pass filterng  (int; default = 40)
    lowcut - the lowcut frequency for band pass filterng  (int; default = 0.5)
    
    attributes:
    self.info - the info of the data for creating raw.mne object
    self.raw_data - the raw data with no filtering (raw.mne object)
    self.filterd_data - raw data after filtering (raw.mne object)
    slef.markers - markers of the data from the markers channel (numpy array) - use for segmentation
    self.event_table - the event table of the data created from the markers (pandas dataframe)
    self.annotations - the annotations of the data created from the event table (mne annotations)   
    """
    def __init__(self,data,event_table,new,sfreq = 125,notch =50 ,highcut = 40 ,lowcut = 0.5):
        data,ch_names,ch_types,markers = self.prepare_info(data,new)
        self.info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        self.raw_data = mne.io.RawArray(data, self.info)
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw_data.set_montage(montage)
        self.markers = markers
        self.event_table = self.create_event_table(markers, sfreq, event_table)
        self.set_annotations_from_event_table()
        filtered_data = self.raw_data.copy().notch_filter(freqs = notch, verbose=False)
        self.filterd_data = filtered_data.filter(l_freq = lowcut, h_freq = highcut, verbose=False)

    def create_event_table(self,markers, sfreq, target_table):
        """
        this functions use the markers and the target table to create the event table
        params:
        markers - the markers of the data (numpy array)
        sfreq - the sampling frequency of the data (int)
        target_table - the target table of the data (pandas dataframe)
        return:
        event_table - the event table of the data (pandas dataframe)
        """
        # Initialize lists to store the columns
        event_numbers = []
        labels = []
        times = []
        targets = []
        start_stop = []

        # Iterate through the markers and decode the non-zero ones
        for index, marker_value in enumerate(markers):
            if marker_value != 0:
                status, label, event_number = eeg.decode_marker(marker_value)
                time = index / sfreq
                target = target_table['Target'][event_number]  

                # Append the values to the lists
                event_numbers.append(event_number)
                labels.append(label)
                times.append(time)
                targets.append(target)
                start_stop.append(status)

        # Create a DataFrame with the columns
        event_table = pd.DataFrame({
            'event_number': event_numbers,
            'Label': labels,
            'Time': times,
            'Target': targets,
            'start_stop': start_stop
        })

        return event_table        

    def set_annotations_from_event_table(self,duration = 1):
        """
        this function sets the annotations of the data using the event table
        params:
        duration - the duration of the segment (float; default = 1)
        """
        #duration is the duration of the event:
        # -0.2 ml (time before the event) + 0.8 ml (time after the event)
        event_table = self.event_table
        # a function we aplly to define the labels of the events
        def label_description(row):
            if row['Label'] == 0:
                return 'Idle'
            elif row['Label'] == row['Target']:
                return 'target'
            else:
                return 'non-target'
            return None
        
       #Add a description column to the DataFrame with the labels
        event_table['Description'] = event_table.apply(label_description, axis=1)
        # Subtract the offset from all of the timestamps
        event_times = event_table['Time']  # get the timestamp column
        # convert the timestamp column to a list of event times
        event_descriptions = event_table['Description'].tolist()
        # create an mne.Annotations object - check duration
        annotations = mne.Annotations(onset=event_times, duration=[duration] * len(event_times), description=event_descriptions)
        # set the annotations for the Raw object
        self.raw_data.set_annotations(annotations)
            
    def prepare_info(self,data,new):
        """
        this function prepares the info for creating the raw.mne object
        params:
            data - the eeg records from the helmet (numpy array)
            new - a boolean that tells us if it was record using old or new helmet (bool)
        return:
            relevant_data - the relevant data for creating the raw.mne object (numpy array)
            ch_names - the names of the channels (list)
            ch_types - the types of the channels (list)
            markers - a coded markers of the data - define where a stimuli showed and what stimuli  (numpy array)
        """
        if new:
            markers = data[31]
            relevant_data = data[[1,2,3,4,5,7,8,9,11,31]]
            ch_names = ["Pz","Fz","Cz","CP1","FC1","AF3","CP2","FC2","AF4","Markers"]
            ch_types = ["eeg","eeg","eeg","eeg","eeg","eeg","eeg","eeg","eeg","stim"]
        else:
            markers = data[31]
            relevant_data = data[[1,2,3,4,5,6,7,8,9,10,11,12,13,31]]
            ch_names = ["C3", "C4", "Cz", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "O1", "O2","Markers"]
            ch_types = ["eeg","eeg","eeg","eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg","eeg" ,"stim"]
        return relevant_data , ch_names , ch_types,markers
    
    def epoch_it(self,tmin = -0.2 ,tmax = 0.8 ,baseline = (-0.2, 0),preload = True):
        """
        this function creates an epochs object from the filtered data and the event table
        as defult returns and object deviding the data to Idle, Target and Non-Target epochs'
        in with samples of 1 secound in tthe duartion of -0.2 ml (time before the event) + 0.8 ml (time after the event)
        params:
            tmin - the start of the epoch (float; default = -0.2 before the showing)
            tmax - the end of the epoch (float; default = 0.8 after the showing)
            baseline - the baseline of the epoch (tuple; default = (-0.2, 0))
            preload - if to preload the data (bool; default = True)
        return:
            epochs - the epochs object (mne object)
        """
        #get evenets and event_id from the annotations
        events, event_id = mne.events_from_annotations(self.raw_data) 
        #create the epochs object
        epochs = mne.Epochs(self.filterd_data, events, event_id, tmin = tmin, tmax = tmax ,baseline = baseline, preload = preload)
        #set the epochs object and return it, so we can use it as a function or as an attribute
        self.epochs = epochs
        return epochs

    def all_plots(self,dir,exp_num,epochs = None):
        """
        :params 
            dir - the directory to save the plots
            exp_num - the experiment number
            epochs - the epochs to plot we want to use this function for other epochs as well
        this function plots the following plots:       
        1) general eeg activity for each channel
        2) the average of the target and the idle epochs
        3) the difference between the target and the idle epochs
        4) the topomap of the difference between the target and the idle epochs
        5) the time-voltage plot of the difference between the target and the idle epochs
        6) the activity of each chanel for the target,non-target and the idle epochs
        7) the time-voltage plot of the difference between the target and the idle epochs for each channel
        8)  topographical map 
        9) power spectrum density 
        10) the power spectrum density for topomap
        """
        if not os.path.exists(dir):
            # If not, create the directory
            os.makedirs(dir)
            print(f"Directory {dir} created.")
        
        if epochs == None:
            epochs = self.epochs
        
        #1)
        fig = epochs.plot(show=False)
        fig.savefig(f"{dir}\\general_idea.png")
        plt.close(fig)
        
        #2)
        IdleAVG = epochs["Idle"].average()
        fig = IdleAVG.plot(show = False)
        fig.savefig(f"{dir}\\exp_{exp_num}_Idle_avg.png")
        plt.close(fig) 
        
        #3)
        targetAVG = epochs["target"].average()
        fig = targetAVG.plot(show = False)
        fig.savefig(f"{dir}\\exp_{exp_num}_target_avg.png")
        plt.close(fig)
        
        #4)
        diff = mne.combine_evoked((targetAVG,-IdleAVG), weights='equal')
        fig = diff.plot_joint(times=0.35, show=False)
        fig.savefig(f"{dir}\\exp_{exp_num}_diff_p300.png")
        plt.close(fig)
        
        #5)
        fig = diff.plot_image(show = False)
        fig.savefig(f"{dir}\\exp_{exp_num}_diff_time_Volts.png")
        plt.close(fig)
        
        #8)
        fig = targetAVG.plot_topomap(show = False)
        fig.savefig(f"{dir}\\exp_{exp_num}_target_topomap_avg.png")
        plt.close(fig)   
        fig = IdleAVG.plot_topomap(show = False)
        fig.savefig(f"{dir}\\exp_{exp_num}_Idle_topomap_avg.png")
        plt.close(fig)   
        
        #9)
        fig = epochs.plot_psd(show=False)
        fig.savefig(f"{dir}\\epochs_psd.png")
        plt.close(fig)
        #10)
        fig = epochs.plot_psd_topomap(show=False)
        fig.savefig(f"{dir}\\epochs_psd_topomap.png")
        plt.close(fig)
        
        
        
        
        for pick in range(0,len(epochs.ch_names)-1):
            #6)
            fig  = epochs['target'].plot_image(pick, show=False)[0]
            fig.savefig(f"{dir}\\exp_{exp_num}_target_electrode_erp_{epochs.ch_names[pick]}.png")
            plt.close(fig)
            
            fig = epochs['non-target'].plot_image(pick,show=False)[0]
            fig.savefig(f"{dir}\\exp_{exp_num}_non_target_electrode_erp_{epochs.ch_names[pick]}.png")
            plt.close(fig)
            
            fig = epochs['Idle'].plot_image(pick,show=False)[0]
            fig.savefig(f"{dir}\\exp_{exp_num}_Idle_electrode_erp_{epochs.ch_names[pick]}.png")
            plt.close(fig)
            
            #7)
            fig = mne.viz.plot_compare_evokeds({"IdleAVG": IdleAVG, "targetAVG": targetAVG}, picks=[pick],show=False)[0]
            fig.savefig(f"{dir}\\exp_{exp_num}_avg_compare_elctorde_{epochs.ch_names[pick]}.png")
            plt.close(fig)
            