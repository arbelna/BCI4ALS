from eeg import Eeg
import experiment as ex

# set all the parameters for the experiment

eeg = Eeg(new=False)
exp = ex.Experiment(eeg, michael=True)
exp.run_experiment()  # Run the experiment
