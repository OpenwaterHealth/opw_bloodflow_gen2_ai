
from ReadGen2Data import ReadGen2Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import signal
import statsmodels.api as sm
import tkinter as tk
from tkinter import filedialog


analysisFolder = None
scanType = 2 #Scan type 1 is the sequential near-far camera pairing from the initial unit. Scan type 2 in the simultaneous scan.

if analysisFolder is None:
    root = tk.Tk()
    root.withdraw()

    analysisFolder = filedialog.askdirectory()

scanData = ReadGen2Data(analysisFolder, scanTypeIn=scanType-1)
scanData.ReadDataAndComputeContrast()

for channel in scanData.channelData:
    ft = np.fft.fft(channel.contrast-np.mean(channel.contrast))
    ft = np.fft.fftshift(ft)
    freq = np.fft.fftfreq(len(ft),0.025)
    freq = np.fft.fftshift(freq)
    plt.subplot(2, 1, 1)
    plt.plot(freq, np.abs(ft))
    plt.subplot(2, 1, 2)
    plt.plot(channel.contrast)
    plt.show()