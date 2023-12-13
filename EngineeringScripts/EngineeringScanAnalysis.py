import math, sys
sys.path.append('..')
from ReadGen2Data import ReadGen2Data, PrettyFloat3, ConvenienceFunctions, PulseFeatures
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re
from tkinter.filedialog import askdirectory


# Intended purpose: easy to use tool for engineering personnel to quickly analyze experimental data at Front st
# Input: file path to a folder containing scan data, saved on a local drive
# Toggle plots of different types on and off
# Generate array of optical features which can be copied into google sheets or excel for additional plotting

#%% Plotting Options
plotContrast = False
plotImageMean = False
plotContrastFreq = False
plotGoldenPulses = False
dt = 1/40

#%% Input Scan Folder Path
searchpath = r"C:/Users/ahouc/Documents/Headset Testing and Analysis/optode contact/2023_04_07_151806_G"

###############################################################################################################################################

scanname = askdirectory(title='Select Scan', initialdir=searchpath)
scanPath = scanname
    
print("Scan Folder: " + scanPath)
scan = ReadGen2Data(scanPath)#, deviceID=row['device'], scanTypeIn=row['scanType']-1)
scan.ReadDataAndComputeContrast()
scan.DetectPulseAndFindFit()
scan.ComputeGoldenPulseFeatures()

#%% Generate plots if needed
if plotContrast or plotImageMean or plotContrastFreq or plotGoldenPulses:
    scan.PlotContrastMeanAndFrequency( titleStr="Image Mean and Contrast",
                                       plotContrast=plotContrast, plotMean=plotImageMean, plotGoldenPulses=True,
                                       plotFreq=plotContrastFreq, plotUnfilteredContrast=True )

# %%
#The features are implemented for golden pulses
#We keep the list as is to access the feature by name
tempPulseFeatures = PulseFeatures()
opticalFeatureNamesGP = tempPulseFeatures.featureList + ['histWidthLsrOff','imageMean','contrastMean']

channels = range(8)

goldenPulseFeaturesForChannels = np.full((len(channels), len(opticalFeatureNamesGP)), np.nan)
#Push golden pulse features features into a single array
for j,ch in enumerate(channels):
    for k in range(len(opticalFeatureNamesGP)-3):
        goldenPulseFeaturesForChannels[j,k] = getattr(scan.channelData[ch].goldenPulseFeatures, opticalFeatureNamesGP[k])
    goldenPulseFeaturesForChannels[j,-3] = scan.channelData[ch].imageLaserOffHistWidth
    goldenPulseFeaturesForChannels[j,-2] = np.mean(scan.channelData[ch].correctedMean)
    goldenPulseFeaturesForChannels[j,-1] = np.mean(scan.channelData[ch].contrast)


#Create data frame with all features
allOpticalFeaturesDf = pd.DataFrame(goldenPulseFeaturesForChannels, columns=opticalFeatureNamesGP)



