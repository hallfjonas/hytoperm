
# external imports
import unittest

# internal imports
from hytoperm import *

pass_settings = {
    "n_sets": [1, 5, 10, 100],
    "fraction": [0, 0.5, 1],
    "seed": [None, 234],
    "min_dist": [0.0]
}

fail_settings = {
    "n_sets": [ 10],
    "fraction": [-1, 1],
    "min_dist": [2]
}


class TesttestWorldGeneration(unittest.TestCase):
    def testWorldGeneration(self):

        # pass settings: expected to generate an Experiment instance
        for n_sets in pass_settings["n_sets"]:
            for fraction in pass_settings["fraction"]:
                for seed in pass_settings["seed"]:
                    for min_dist in pass_settings["min_dist"]:
                        ex = Experiment.generate(
                            n_sets=n_sets, 
                            fraction=fraction, 
                            seed=seed, 
                            min_dist=min_dist
                        )
                        self.assertIsInstance(ex, Experiment)

        # fail settings: expected Experiment generation to fail and return None
        for n_sets in fail_settings["n_sets"]:
            for fraction in fail_settings["fraction"]:
                for min_dist in fail_settings["min_dist"]:
                    with self.assertRaises(AssertionError):
                        ex = Experiment.generate(
                            n_sets=n_sets, 
                            fraction=fraction, 
                            seed=None, 
                            min_dist=min_dist
                        )
                        # self.assertIsNone(ex)

    
if __name__ == "__main__":
    unittest.main()
