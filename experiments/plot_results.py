
import os
import sys
from Experiment import Experiment
import matplotlib.pyplot as plt
from unittests.Tests import plot_results

if __name__ == '__main__':
    
    file = sys.argv[1]
    if os.path.isfile(file):
        ex : Experiment = Experiment.deserialize(file)
        plot_results(ex)
    else:
        print(f"File {file} not found.")
