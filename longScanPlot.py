#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 08:40:02 2023

@author: brad
"""

#%%
import math
from ReadGen2Data import ReadGen2Data
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re
from tkinter.filedialog import askdirectory

#%% Input Scan Folder Path
savepath = '/Users/brad/Desktop/Test/'
scanRoot = '/Users/brad/Desktop/gen2 data/TCD/2023_05_25_142850_CVRhBH017'

def list_non_folder_dirs(relative_path):
    non_folder_dirs = []
    absolute_path = os.path.abspath(relative_path)
    for dirpath, dirnames, filenames in os.walk(absolute_path):
        if not dirnames:
            if ("testscan" not in dirpath.lower()) and ("fullscan" not in dirpath.lower()):
                non_folder_dirs.append(os.path.abspath(dirpath))

    return non_folder_dirs

scans = []
non_folder_dirs = list_non_folder_dirs(scanRoot)
for path in non_folder_dirs:
    print(path)
    #scanData = ReadGen2Data(path, correctionTypeIn=1, scanTypeIn = 3, enablePlotsIn=False )
    scanData = ReadGen2Data(path,scanTypeIn=4, correctionTypeIn=2)
    scanData.ReadDataAndComputeContrast()
    scanData.DetectPulseAndFindFit()
    scanData.ComputeGoldenPulseFeatures()
    scanData.ComputePulseSegmentFeatures()
    scans.append(scanData)

#%%
for scanInd,scan in enumerate(scans):
    savename = re.split('/', non_folder_dirs[scanInd])[-2] + '_' + re.split('/', non_folder_dirs[scanInd])[-1]
    scan.PlotLongScan(titleStr=savename, saveFig=savepath, cropTime=[])
    scan.PlotLongScanCompare4Channels(titleStr=savename, saveFig=savepath, cropTime=[0,20])
