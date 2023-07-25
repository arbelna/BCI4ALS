from eeg import Eeg
import experiment as ex
import numpy as np
import pickle
import os
from tkinter import Tk, Entry, Label, Button
from random import randrange
from psychopy import visual, core, logging, sound, event
import psychtoolbox as ptb
import random
import pandas as pd
import time
import brainflow
import numpy as np
import pickle
import os


# intialize the Eeg class

#set all the parmeter for the experiment
eeg = Eeg(new = True)
exp = ex.Experiment(eeg,michael = False)
exp.run_experiment()  # Run the experiment
#exp.save_results() #save the results



# ## --- Initializations ---
# exp = ex.Experiment()  # Create an experiment object
# eeg = Eeg()
# eeg.stream_on()  # Start to record data from the electrodes
# eeg.clear_board()  # Clear the board data
# exp = ex.Experiment(eeg)
# exp.run_experiment(eeg)  # Run the experiment
# data = eeg.get_stream_data()  # Save the data as numpy array
# eeg.stream_off()  # Stop recording
# exp.save_results()  # Save the results
# num_record = 5  # Experiment number
# subject = exp.subject_name  # Subject name

"""
## --- Save the data to the PC ---
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))  # The location of the script folder
if not os.path.isdir(f'{__location__}/records'):
    os.mkdir(f'{__location__}/records')  # Creates a folder name records

if not os.path.isdir(f'{__location__}/records/{subject}'):
    os.mkdir(f'{__location__}/records/{subject}')  # Create a folder with the subject name

if not os.path.isdir(f'{__location__}/records/{subject}/exp num {num_record}'):
    os.mkdir(f'{__location__}/records/{subject}/exp num {num_record}')  # Create a folder of the experiment number


with open(f'{__location__}/records/{subject}/exp num {num_record}/records.npy', 'wb') as f:
    np.save(f, data, allow_pickle=True)  # Save the numpy array datate

file = open(f'{__location__}/records/{subject}/exp num {num_record}/records', 'wb')
# --- Dump information to that file
pickle.dump(exp, file)  # Save the pickle data
file.close()  # Close the file

df = exp.results
df.to_csv(f'{__location__}/records/{subject}/exp num {num_record}/records.csv', index=False)  # Save the csv data

"""
