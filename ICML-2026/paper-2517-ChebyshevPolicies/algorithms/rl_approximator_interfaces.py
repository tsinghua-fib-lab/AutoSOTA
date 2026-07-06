"""rl_approximator_interfaces.py: Interfaces serving as blueprints for approximators compatible to the TrainableMountainCarMRPWrapper class.
"""
__author__ = "Hannes Unger"
__version__ = "1.0.0"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


class Approximator():
    def evaluate_point(self, x):
        raise Exception('Not implemented.')

    def evaluate(self, *data):
        raise Exception('Not implemented.')