# importing relevant libraries
from tkinter import Tk, Entry, Label, Button
from random import randrange
from psychopy import visual, core, logging, sound, event
import psychtoolbox as ptb
import random
import pandas as pd
from eeg import Eeg
import time
import numpy as np
import pickle
import os


class Experiment:
    def __init__(self, eeg, michael=False):
        """
        This is the constructor method that is called when an object of this class is created.
        It initializes several instance variables
        """
        
        self.num_blocks = None
        self.num_trials = None
        self.subject_name = None
        if michael: 
            self.num_blocks = 1
            self.num_trials = 200
            self.subject_name = 'Michael'
        else:
            self.ask_subject()
            self.ask_num_blocks()
            self.ask_num_trials()
        self.temp = None
        self.eeg = eeg
        self.results = []
        self.enum_image = {0: 'Idle', 1: 'No', 2: 'Yes'}
        self.image_discription = {1: 'Sad Bugs Bunny', 2: 'Happy Bugs Bunny'}
    
        #labels
        self.labels = []
        self._init_labels()
    
    def log_init(self):
        """
        this method initializes the log file
        the log file is a csv file that contains the following columns:
            exp_num : the experiment number
            blocks_number : the number of blocks in the experiment
            num_trials : the number of trials in the experiment
            Block_answer: the answer of the subject to the target question
            Counted_Target : the amount of times the subject counted the target in the block
            True_target_amount : the true amount of targets in the bloc
        
        """
        columns=['exp_num', 'block_number', 'num_trials',"Block_answer","Counted_Target","True_target_amount"]
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))  # The location of the script folder
        if not os.path.isdir(f'{__location__}\\records\\{self.subject_name}'):
            os.mkdir(f'{__location__}\\records\\{self.subject_name}')  # Create a folder with the subject name
            self.log = pd.DataFrame(columns = columns)
            return
        else:
            def search_file(directory =f"{ __location__}\\records\\{self.subject_name}", filename = 'exp_log.csv'):
                for root, dirs, files in os.walk(directory):
                    print(files)
                    if filename in files:
                        return os.path.join(root, filename)
                return None
        # Search for the file in the directory
        file_path = search_file()
        # Check if the file was found
        if file_path is not None:
            self.log = pd.read_csv(file_path)
        else:
             self.log = pd.DataFrame(columns = columns)
            
    def ask_subject(self):
                # Define a function to return the Input data
        """
        This method prompts the user to enter the number of blocks they want in their experiment.
         If the input is not a valid number, it displays an error message.
        :return: the number of desired blocks
        """
        def get_name_ent(event):
            return get_name(entry.get())

        def get_name(input=None):
            if input is None:
                input1 = entry.get()
            else:
                input1 = input
            try:
                self.subject_name = input1
            except:
                self.subject_name = None
            win.destroy()
     
        self.subject_name = None
        while True:
            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text="Insert the subject name", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_name).place(relx=.5, rely=.3)
            win.bind("<Return>", get_name_ent)
            win.mainloop()

            if self.subject_name is not None:
                break
                
    def _init_labels(self):
        """
        This method creates dict containing a stimulus vector
        :return: the stimulus in each trial (list)
        """
        for i in range(self.num_blocks):
            self.labels.append([])
            while len(self.labels[i]) < self.num_trials: 
                Idle_sequence_length = random.randint(2, 4)
                self.labels[i] += [0] * Idle_sequence_length
                target_non_target = random.choice([1, 2])
                if target_non_target == 2:
                    self.labels[i].append(2)
                elif target_non_target == 1:
                    self.labels[i].append(1)
            self.labels[i] = self.labels[i][:self.num_trials]
    
    def ask_num_trials(self):
        # Define a function to return the Input data
        """
        This method prompts the user to enter the number of trials they want in their experiment.
         If the input is not a valid number, it displays an error message.
         :param: input: none
        :return:  the number of trials
        """

        def get_num_trials_ent(event):
            return get_num_trials(entry.get())

        def get_num_trials(input=None):
            if input is None:
                input1 = entry.get()
            else:
                input1 = input
            try:
                self.num_trials = int(input1)
            except:
                self.num_trials = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.bind("<Return>", cont)
            err.after(1, lambda: err.focus_force())
            err.mainloop()

        def cont():
            err.destroy()
            pass

        self.num_trials = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text="Enter the number of trials you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_trials).place(relx=.5, rely=.3)
            win.bind("<Return>", get_num_trials_ent)
            win.mainloop()

            if self.num_trials is not None:
                break
            err = Tk()
            error("You should enter a number!")

    def ask_num_blocks(self):
        # Define a function to return the Input data
        """
        This method prompts the user to enter the number of blocks they want in their experiment.
         If the input is not a valid number, it displays an error message.
        :return: the number of desired blocks
        """

        def get_num_block_ent(event):
            return get_num_block(entry.get())

        def get_num_block(input=None):
            if input is None:
                input1 = entry.get()
            else:
                input1 = input
            try:
                self.num_blocks = int(input1)
            except:
                self.num_blocks = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.bind("<Return>", cont)
            err.after(1, lambda: err.focus_force())
            err.mainloop()

        def cont():
            err.destroy()
            pass

        self.num_blocks = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text="Enter the number of blocks you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_block).place(relx=.5, rely=.3)
            win.bind("<Return>", get_num_block_ent)
            win.mainloop()

            if self.num_blocks is not None:
                break
            err = Tk()
            error("You should enter a number!")

    def choose_targets(self):
            blocks_targets = np.empty((self.num_blocks), dtype=object)
            for i in range(self.num_blocks):
                # Create a window
                win = visual.Window(size=(800, 600), monitor="testMonitor", units="pix")
                rect1 = visual.Rect(win, width=200, height=100, fillColor='blue', pos=(-200, 0))
                rect2 = visual.Rect(win, width=200, height=100, fillColor='red', pos=(200, 0))

                # Create some visual stimuli
                Title = visual.TextStim(win, text= f"Hey {self.subject_name}!, For block number {(i+1)}, What is your answer ", pos=(0, 200))
                stimulus1 = visual.TextStim(win, text="Yes", pos=(-200, 0),bold= True)
                stimulus2 = visual.TextStim(win, text="No", pos=(200, 0),bold= True)
                
                # Draw the stimuli
                rect1.draw()
                rect2.draw()
                stimulus1.draw()
                stimulus2.draw()
                Title.draw()
                win.flip()

                # Wait for a mouse click
                mouse = event.Mouse(win=win)
                while True:
                    if mouse.getPressed()[0]:
                        # Check if the mouse is within the bounding box of stimulus1
                        if rect1.contains(mouse):
                            blocks_targets[i] ={"Target":"Yes","Number":2}
                            break
                        # Check if the mouse is within the bounding box of stimulus2
                        elif rect2.contains(mouse):
                            blocks_targets[i] ={"Target":"No","Number":1}
                            break

                # Close the window
                win.close()
            return blocks_targets
    
    def try_again(self):
        # Create a window
        win = visual.Window(size=(800, 600), monitor="testMonitor", units="pix")
        rect1 = visual.Rect(win, width=200, height=100, fillColor='blue', pos=(-200, 0))
        rect2 = visual.Rect(win, width=200, height=100, fillColor='red', pos=(200, 0))

        # Create some visual stimuli
        Title = visual.TextStim(win, text= f"Hey {self.subject_name}!, would you like to do the experiment again? ", pos=(0, 200))
        stimulus1 = visual.TextStim(win, text="Yes", pos=(-200, 0),bold = True)
        stimulus2 = visual.TextStim(win, text="No", pos=(200, 0),bold = True)
        
        # Draw the stimuli
        rect1.draw()
        rect2.draw()
        stimulus1.draw()
        stimulus2.draw()
        Title.draw()
        win.flip()

        # Wait for a mouse click
        mouse = event.Mouse(win=win)
        while True:
            if mouse.getPressed()[0]:
                # Check if the mouse is within the bounding box of stimulus1
                if rect1.contains(mouse):
                    answer = True
                    break
                # Check if the mouse is within the bounding box of stimulus2
                elif rect2.contains(mouse):
                    answer = False
                    break
        # Close the window
        win.close()
        return answer

    def run_experiment(self):
        """
        This method runs the experiment by displaying images to the user and collecting their responses.
         It stores the results in the results instance variable.
        :param eeg:
        :return: csv file with expermient results
        Target: 1 =     sad, 2 = happy
        image: 0 = distractor/furious,1 = sad, 2 = happy, when 1 or 2 are target or non target
        """
        
        self.targets = self.choose_targets()
        self.eeg.stream_on()  # Start to record data from the electrodes
        self.eeg.clear_board()  # Clear the board data
        #overwrite (filemode='w') a detailed log of the last run in this dir
        
        lastLog = logging.LogFile("lastRun.log", level=logging.CRITICAL, filemode='w')

        for i in range(self.num_blocks):
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
            #the choice of the target look  : (i % 2) + 1
            look  = self.targets[i]["Number"]           
            shout = self.targets[i]["Target"]
            start_block_win = visual.TextStim(mywin, f'Block number {i + 1} \n\n Target Image:{self.image_discription[look]} \n\n Target Sound:{shout}',
                                              color=(1, 1, 1),
                                              colorSpace='rgb')
            start_block_win.draw()
            mywin.logOnFlip(level=logging.CRITICAL, msg=f'+{i + 1}')
            mywin.flip(clearBuffer=True)
            core.wait(10.0)
            mywin.close()
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg", fullscr=True)
            for j in range(self.num_trials):
                wait = random.uniform(0.6, 1)
                core.wait(wait)
                start_block_win = visual.ImageStim(win=mywin,
                                                   image=f'Pictures/{self.enum_image[self.labels[i][j]]}.png')
                mySound = sound.Sound(f'Pictures/{self.enum_image[self.labels[i][j]]}.mp3')
                nextFlip = mywin.getFutureFlipTime(clock='ptb')
                start_block_win.draw()
                mySound.play(when=nextFlip)
                # status: str, label: int, index: int
                mywin.logOnFlip(level=logging.CRITICAL, msg=f'{self.labels[i][j]} {time.time()} {look}')
                mywin.flip(clearBuffer=True)
                self.eeg.insert_marker(status='start', label=self.labels[i][j], index=j)
                core.wait(0.7)
                start_block_win = visual.ImageStim(win=mywin)
                start_block_win.draw()
                mywin.flip()
                wait = 2 - 0.7 - wait
                core.wait(wait)
            mywin.close()

        with open('lastRun.log') as file:
            file = [line.rstrip('\n').split('\t') for line in file]
        pre_dataframe = []
        curr_block = 0

        for line in file:
            temp = line[2].split(" ")
            if len(temp) == 1:
                curr_block += 1
                curr_trial = 0
                continue
            curr_trial += 1
            pre_dataframe.append([curr_block, curr_trial, temp[0], temp[2], line[0], temp[1]])
        self.results = np.array([np.array(x) for x in pre_dataframe])
        self.results = pd.DataFrame(self.results)
        self.results = self.results.set_axis(['Block', 'Trial', 'Label', 'Target', 'Time', 'Unix time'], axis=1)
        self.data = self.eeg.get_stream_data()  # Save the eeg data as numpy array
        self.eeg.stream_off()  # Stop recording
        
        counted_targets = np.empty((self.num_blocks), dtype=object)
        for b in range(self.num_blocks):
            counted_targets[b] = self.ask_counted_target(b)
            if counted_targets[b] == self.labels[b].count(self.targets[b]['Number']):
                self.success_screen()
            else:
                self.fail_screen(counted_targets[b],self.labels[b].count(self.targets[b]['Number']))
            
        self.counted_targets = counted_targets
        self.save_results()
        answer = self.try_again()
        if answer:
            if self.subject_name == 'Michael':
                self.run_experiment()
            else:                 
                self.ask_num_blocks()
                self.ask_num_trials()
                self.run_experiment()
          
    def success_screen(self):
        # Create a window
        win = visual.Window(size=[1000, 800],monitor="testMonitor")
        # Create a text stimulus
        text_1 = visual.TextStim(win, text=f'{self.subject_name} You are the best !', pos=(0,0.6), color='red',bold=True,height=0.15)
        text_2 = visual.TextStim(win, text='You counted succcesfully', pos=(0,0), color='red',bold=True,height=0.1)
        # Create an image stimulus
        image = visual.ImageStim(win, image='Pictures/success.png', size=[2, 2])
        image.contrast = 0.5
        # Draw the image and text to the window buffer
        image.draw()
        text_1.draw()
        text_2.draw()   
        # Flip the window to show the buffer
        win.flip()
        # Wait for 5 seconds
        core.wait(5)
        # Close the window
        win.close()

    def fail_screen(self,your_count,real_count):
                # Create a window
        win = visual.Window(size=[1000, 800], monitor="testMonitor")
        # Create a text stimulus
        text_1 = visual.TextStim(win, text=f'{self.subject_name}, Not exactly!', pos=(0,0.6), color='red',bold=True,height=0.2)
        text_2 = visual.TextStim(win, text=f'You counted {your_count} and there was {real_count} targets, good job anyway!', pos=(0,0), color='red',bold=True,height=0.1)
        # Create an image stimulus
        image = visual.ImageStim(win, image='Pictures/fail.png', size=[2, 2])
        image.contrast = 0.5
        # Draw the image and text to the window buffer
        image.draw()
        text_1.draw()
        text_2.draw()   
        # Flip the window to show the buffer
        win.flip()
        # Wait for 5 seconds
        core.wait(5)
        # Close the window
        win.close()

    def tar_block(self):
        target = []
        target_num = []
        num = self.df['Trial'].nunique() + 1
        for idx, row in self.df.iterrows():
            if idx % num == 0:
                target_num.append(int(row[3]))
        for j in range(len(target_num)):
            target.append(self.enum_image[target_num[j]])
        return target
    
    def ask_counted_target(self,i): 
        """
        this method asks the user to count the number of targets in a block
        :param i: block number
        
        """
        def get_num_target_ent():
            return get_num_target(entry.get())

        def get_num_target(num_target_input=None):
    
            if num_target_input is None:
                input1 = entry.get()
            else:
                input1 = num_target_input
            try:
                self.temp = int(input1)
            except:
                self.temp = None
            win.destroy()
            
        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.bind("<Return>", cont)
            err.after(1, lambda: err.focus_force())
            err.mainloop()

        def cont():
            err.destroy()
            pass

        self.temp = None
        while True:
            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text=f"Enter the number of targets you counted for block {i+1}.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_target).place(relx=.5, rely=.3)
            win.bind("<Return>", get_num_target_ent)
            win.mainloop()
            if self.temp is not None:
                break
            err = Tk()
            error("You should enter a number!")
        return self.temp  
    
    def save_results(self):
        """
        this method saves the results of the experiment:
        - the eeg data as a numpy array
        - the metadata as csv file
        - the updated log file for the subject as csv file
        """
        
        data = self.data
        self.log_init()
        log = self.log
        subject = self.subject_name
        
        if log['exp_num'].shape[0] == 0:
            #if no log file exists this is the first experiment
            num_record = 1
        else:
            ##--define num_record from log file-##
            num_record = log['exp_num'].iloc[-1] + 1
        
        for i in range(self.num_blocks):   
            df_block = self.results[self.results['Block'] == str(i+1)].copy()
            True_counted_target = df_block[df_block['Label']==str(self.targets[i]['Number'])]['Label'].value_counts()
            if True_counted_target.shape[0] == 0:
                True_counted_target = 0 
            else:
                True_counted_target = int(True_counted_target)
            
            new_block = {'exp_num': num_record,
                            'block_number': i+1,
                            'num_trials': self.num_trials,
                            'Block_answer': self.targets[i]['Target'],
                            'Counted_Target': self.counted_targets[i],
                            'True_target_amount': True_counted_target}
            
            log = log.append(new_block, ignore_index=True)
            
            ## --- Save the data to the PC ---
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))  # The location of the script folder
        if not os.path.isdir(f'{__location__}\\records'):
            os.mkdir(f'{__location__}\\records')  # Creates a folder name records

        if not os.path.isdir(f'{__location__}\\records\\{subject}'):
            os.mkdir(f'{__location__}\\records/{subject}')  # Create a folder with the subject name

        if not os.path.isdir(f'{__location__}\\records\\{subject}\\exp_num_{num_record}'):
            os.mkdir(f'{__location__}\\records\\{subject}\\exp_num_{num_record}')  # Create a folder of the experiment number


        with open(f'{__location__}\\records\\{subject}\\exp_num_{num_record}\\records_{num_record}.npy', 'wb') as f:
            np.save(f, data, allow_pickle=True)  # Save the numpy array data

        file = open(f'{__location__}\\records\\{subject}\\exp_num_{num_record}\\records_{num_record}', 'wb')
        # --- Dump information to that file
        pickle.dump(self, file)  # Save the pickle data
        file.close()  # Close the file

        df = self.results
        df.to_csv(f'{__location__}\\records\\{subject}\\exp_num_{num_record}\\records_{num_record}.csv', index=False)  # Save the csv data
        log.to_csv(f'{__location__}\\records\\{subject}\\exp_log.csv', index=False)  # Save the csv data