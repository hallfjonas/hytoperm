
# external imports
import os
import sys

# internal imports
from src.Experiment import Experiment
from experiment_small import plotResults

if __name__ == '__main__':
    
    file = sys.argv[1]
    if os.path.isfile(file):
        ex : Experiment = Experiment.deserialize(file)
        plotResults(ex)
    else:
        print(f"File {file} not found.")
