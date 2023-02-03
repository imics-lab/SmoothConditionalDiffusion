#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         03 Feb, 2023            #
#########################################

import numpy as np
import math
import torch


# cylinder bell funnel based on "Learning comprehensible descriptions of multivariate time series"
def generate_bell(length, amplitude, default_variance):
    bell = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)/length
    return bell

def generate_funnel(length, amplitude, default_variance):
    funnel = np.random.normal(0, default_variance, length) + amplitude * np.arange(length)[::-1]/length
    return funnel

def generate_cylinder(length, amplitude, default_variance):
    cylinder = np.random.normal(0, default_variance, length) + amplitude
    return cylinder

std_generators = [generate_bell, generate_funnel, generate_cylinder]



def generate_signal_as_tensor(length=100, avg_pattern_length=5, avg_amplitude=1,
                          default_variance=1, variance_pattern_length=10, variance_amplitude=2,
                          generators=std_generators, include_negatives=True):
    data = np.random.normal(0, default_variance, length)
    current_start = np.random.randint(0, avg_pattern_length)
    current_length = current_length = max(1, math.ceil(np.random.gauss(avg_pattern_length, variance_pattern_length)))

    while current_start + current_length < length:
        generator = np.random.choice(generators)
        current_amplitude = np.random.gauss(avg_amplitude, variance_amplitude)

        while current_length <= 0:
            current_length = -(current_length-1)
        pattern = generator(current_length, current_amplitude, default_variance)

        if include_negatives and np.random.random() > 0.5:
            pattern = -1 * pattern

        data[current_start : current_start + current_length] = pattern

        current_start = current_start + current_length + np.random.randint(0, avg_pattern_length)
        current_length = max(1, math.ceil(np.random.gauss(avg_pattern_length, variance_pattern_length)))

    return torch.Tensor(data)

