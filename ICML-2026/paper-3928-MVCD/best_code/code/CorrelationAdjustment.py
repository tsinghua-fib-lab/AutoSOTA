import numpy as np
import pandas as pd


def correlation_matrix_single_training_type1(state):




def correlation_matrix_combination_one(state_matrix):
    all_area = []

    for i in range(state_matrix.shape[0]):
        for j in range(state_matrix.shape[1]):
            target_matrix = correlation_matrix_single_training_type1(state_matrix[i,j])
            all_area += target_matrix

    return np.asarray(all_area)


def correlation_matrix_single_training_type2(state):


def correlation_matrix_combination(state_matrix):
    all_area = []

    for i in range(state_matrix.shape[0]):
        row_area=[]
        for j in range(state_matrix.shape[1]):
            target_area = correlation_matrix_single_training_type2(state_matrix[i, j])
            row_area.append(target_area)
        all_area.append(row_area)

    return np.asarray(all_area)