import numpy as np
from roi_graphs import roiFreq
def accuracy(freq_x_real, freq_y_real, freq_x_amp, freq_y_amp):
    
    return (1/len(freq_x_real))*(np.linalg.norm(freq_x_real - freq_x_amp)**2 + np.linalg.norm(freq_y_real - freq_y_amp)**2)**0.5