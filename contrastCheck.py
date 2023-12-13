#%% Simple script to run the stats on all scans in a given folder (delete the testscan first!)
import math, os, re, tkinter as tk, numpy as np, pandas as pd
from ReadGen2Data import ReadGen2Data, ConvenienceFunctions
import matplotlib.pyplot as plt
from tkinter import filedialog

plotContrast = True
plotImageMean = True
plotContrastFreq = False
plotGoldenPulses = False

savepath = '.'
#Enter either the root folder to all the scans or a specific scan folder here
scanRoot = ''

#Ask for directory if not specified
if scanRoot == '':
    root = tk.Tk()
    root.withdraw()
    scanRoot = filedialog.askdirectory()

def list_non_folder_dirs(relative_path):
    non_folder_dirs = []
    absolute_path = os.path.abspath(relative_path)
    for dirpath, dirnames, filenames in os.walk(absolute_path):
        if not dirnames:
            if ("FULLSCAN" in dirpath or "LONGSCAN" in dirpath):
                non_folder_dirs.append(os.path.abspath(dirpath))

    return non_folder_dirs

scans = []
non_folder_dirs = list_non_folder_dirs(scanRoot)
for path in non_folder_dirs:
    scanData = ReadGen2Data(path)
    scanData.ReadDataAndComputeContrast()
    scanData.DetectPulseAndFindFit(nofilter=True)
    scanData.ComputeGoldenPulseFeatures()
    scanData.ComputePulseSegmentFeatures()
    scans.append(scanData)
#%%
for scanInd,scan in enumerate(scans):
    savename = re.split('/', non_folder_dirs[scanInd])[-2] + '_' + re.split('/', non_folder_dirs[scanInd])[-1]
    # scan.PlotInvertedAndCenteredContrastAndMean( scan.path )
    
    scan.PlotContrastMeanAndFrequency(scan.path, plotContrast=True, plotMean=True, plotGoldenPulses=False, plotFreq=False,  plotUnfilteredContrast=False, plotTemp = False)
    scan.PlotCompare4Channels(titleStr=savename, saveFig=savepath, plotGoldenPulses=False)
    # scan.PrintCV()
    #Code to get feature arrays and time points in long scans
    pulseAmplitudes = []
    pulseStarts = []
    pulseRaw = []
    
    if 1: #scan.scanType==2 or scan.scanType==4:
        for index, ch in enumerate(scan.channelData):
            if ch.pulseSegmentsFeatures:
                pulseAmplitudes.append(ch.pulseSegmentsFeatures.amplitude)
                pulseStarts.append(ch.onsets)
                pulseRaw.append(ch.pulseSegments)
                print('Amplitudes',ch.pulseSegmentsFeatures.amplitude)
                print('Onsets',ch.onsets)
# %%
