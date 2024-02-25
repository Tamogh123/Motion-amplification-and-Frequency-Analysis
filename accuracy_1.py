import numpy as np
from roi_graphs import roiFreq



freq_x_real,freq_y_real = roiFreq('output_video.mp4')
freq_x_amp,freq_y_amp = roiFreq('result.mp4')

def accuracy(freq_x_real, freq_y_real, freq_x_amp, freq_y_amp):
    a = np.linalg.norm(freq_x_real - freq_x_amp)/len(freq_x_real) 
    b = np.linalg.norm(freq_y_real - freq_y_amp)/len(freq_y_real) 
    return 1 - ((a+b)/len(freq_x_real))**0.5



a = accuracy(freq_x_real,freq_y_real,freq_x_amp,freq_y_amp)

print(a)