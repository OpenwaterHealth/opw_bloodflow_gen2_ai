#%% Simple script to run the stats on all scans in a given folder
import math, os, re, sys
import matplotlib.pyplot as plt
import miscFcns_BH as fcns
import numpy as np
import pandas as pd
sys.path.append('..')
from ReadGen2Data import ReadGen2Data, ConvenienceFunctions

plotContrast = True
plotImageMean = True
plotContrastFreq = False
plotGoldenPulses = False

#Enter either the root folder to all the scans or a specific scan folder here
scanRoot = '/Users/brad/Desktop/gen2 data/2023_11_09_190310_Brad/FULLSCAN_4C_2023_11_09_190412/' # 100-1000us
scanRoot = '/Users/brad/Desktop/gen2 data/2023_11_13_165153_Brad/' # 200-1000-200-1000us
 
# long scans: RH*15 LN*15 LH*15 RN*15
# full scans: ['RH','RH','LN','RH',
            #  'LN','RV','RV','LN',
            #  'LH','LH','RN','LH',
            #  'RN','LV','LV','RN',], #4 channel scan

#Get the folder via a tk dialog if no folder is specified above
if (scanRoot == ''):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    scanRoot = filedialog.askdirectory()

scans = []
non_folder_dirs = fcns.list_non_folder_dirs(scanRoot)
non_folder_dirs = fcns.natural_sort(non_folder_dirs)
for path in non_folder_dirs:
    print(path)
    # to skip wavelet filter include filterNoiseIn=False, to skip high pass filter include filterDriftIn=False
    scanData = ReadGen2Data(path,filterNoiseIn=False,filterDriftIn=False,correctionTypeIn=1)
    scanData.ReadDataAndComputeContrast()
    scanData.DetectPulseAndFindFit(nofilter=True)
    scanData.ComputeGoldenPulseFeatures()
    scanData.ComputePulseSegmentFeatures()
    scans.append(scanData)

for scan in scans:
    # scan.PlotInvertedAndCenteredContrastAndMean( scan.path )
    scan.PlotContrastMeanAndFrequency(scan.path, plotContrast, plotImageMean, plotGoldenPulses, plotContrastFreq,  plotUnfilteredContrast=False, plotTemp=False)
    #Code to get feature arrays and time points in long scans
    pulseAmplitudes = []
    pulseStarts = []
    pulseRaw = []
    pulseNoiseMetric = []
    if scan.scanType==2 or scan.scanType==4:
        for index, ch in enumerate(scan.channelData):
            if ch.pulseSegmentsFeatures:
                pulseAmplitudes.append(ch.pulseSegmentsFeatures.unbiasedAmp)
                pulseStarts.append(ch.onsets)
                pulseRaw.append(ch.pulseSegments)  
                pulseNoiseMetric.append(ch.pulseSegmentsFeatures.noiseMetric)  
                print('Amplitudes',ch.pulseSegmentsFeatures.unbiasedAmp)
                print('Onsets',ch.onsets)

#%% Displays laser on/off images from first module
scanInd = 0
numCam = 4
numMod = int(len(scans[scanInd].channelData)/numCam)

for chInd in [0,4,8,12]:
    scans[scanInd].displayRawImage(chInd,0)
    scans[scanInd].displayRawImage(chInd,1)
 
#%% Converts vector/list format of data into arrays
scanInd = 0
chInd = 0
numCam = 4
numMod = int(len(scans[scanInd].channelData)/numCam)
numScans = len(scans)

# Get all data from single scan
avgTemps = np.zeros((numCam,numMod,numScans))
correctedMean = np.zeros((numCam,numMod,numScans,600))
contrastNoFilter = np.zeros((numCam,numMod,numScans,600))
pulseNoiseMetricAvg = np.zeros((numCam,numMod,numScans))
darkHistWid = np.zeros((numCam,numMod,numScans))

imgMainLsrOffMean = np.zeros((numCam,numMod,numScans))
imgMainLsrOffStd = np.zeros((numCam,numMod,numScans))
imgObLsrOffMean = np.zeros((numCam,numMod,numScans))

imgMainLsrOnMean = np.zeros((numCam,numMod,numScans))
imgMainLsrOnStd = np.zeros((numCam,numMod,numScans))
imgObLsrOnMean = np.zeros((numCam,numMod,numScans))

histLsrOnTimeStamp = np.zeros((numCam,numMod,numScans,600))
histLsrOnCamTemps = np.zeros((numCam,numMod,numScans,600))
histMainLsrOnMean = np.zeros((numCam,numMod,numScans,600))
histMainLsrOnStd = np.zeros((numCam,numMod,numScans,600))
histObLsrOnMean = np.zeros((numCam,numMod,numScans,600))
histObLsrOnStd = np.zeros((numCam,numMod,numScans,600))

histLsrOffTimeStamp = np.zeros((numCam,numMod,numScans,2))
histLsrOffCamTemps = np.zeros((numCam,numMod,numScans,2))
histMainLsrOffMean = np.zeros((numCam,numMod,numScans,2))
histMainLsrOffStd = np.zeros((numCam,numMod,numScans,2))
histObLsrOffMean = np.zeros((numCam,numMod,numScans,2))
histObLsrOffStd = np.zeros((numCam,numMod,numScans,2))

for scanInd in range(len(scans)):
    for chInd in range(numMod*numCam):
        # print([chInd//numMod,chInd%numMod])
        
        try:
            avgTemps[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].camTemps.mean()
            contrastNoFilter[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].contrastNoFilter
            correctedMean[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].correctedMean
            pulseNoiseMetricAvg[chInd//numMod,chInd%numMod,scanInd] = np.mean(scans[scanInd].channelData[chInd].pulseSegmentsFeatures.noiseMetric)
            darkHistWid[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imageLaserOffHistWidth #lsrOffObWidth
            
            imgMainLsrOffMean[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgMainLsrOffMean
            imgMainLsrOffStd[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgMainLsrOffStd
            imgObLsrOffMean[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgObLsrOffMean
            
            imgMainLsrOnMean[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgMainLsrOnMean
            imgMainLsrOnStd[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgMainLsrOnStd
            imgObLsrOnMean[chInd//numMod,chInd%numMod,scanInd] = scans[scanInd].channelData[chInd].imgObLsrOnMean
            
            histLsrOnTimeStamp[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histLsrOnTimeStamp
            histLsrOnCamTemps[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histLsrOnCamTemps
            histMainLsrOnMean[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histMainLsrOnMean
            histMainLsrOnStd[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histMainLsrOnStd
            histObLsrOnMean[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histObLsrOnMean
            histObLsrOnStd[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histObLsrOnStd
            
            histLsrOffTimeStamp[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histLsrOffTimeStamp
            histLsrOffCamTemps[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histLsrOffCamTemps
            histMainLsrOffMean[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histMainLsrOffMean
            histMainLsrOffStd[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histMainLsrOffStd
            histObLsrOffMean[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histObLsrOffMean
            histObLsrOffStd[chInd//numMod,chInd%numMod,scanInd,:] = scans[scanInd].channelData[chInd].histObLsrOffStd
            
        except:
            print('No data for: ch. ' + str(chInd) + ' of scan ' + str(scanInd))
    timeZero = histLsrOnTimeStamp[0,0,scanInd,0]
    histLsrOnTimeStamp = (histLsrOnTimeStamp - timeZero)/1e9 
    histLsrOffTimeStamp = (histLsrOffTimeStamp - timeZero)/1e9
    
histMainLsrOnVar = histMainLsrOnStd**2

#???????
# obStd  = np.copy(histogramTemp[:,17])
#???????

test = contrastNoFilter[[0,3],0,:,:].mean(axis=2)
test = correctedMean[[0,3],0,:,:].mean(axis=2)


#%%
camInd = 0

contrastNoFilter_merge = np.reshape(contrastNoFilter,[4,numMod*600,1,1])
correctedMean_merge = np.reshape(correctedMean,[4,numMod*600,1,1])

imgLsrOnTimeStamp_merge = histLsrOnTimeStamp[:,:,:,0]-0.025

histLsrOnTimeStamp_merge = np.reshape(histLsrOnTimeStamp,[4,numMod*600,1,1])
histMainLsrOnMean_merge = np.reshape(histMainLsrOnMean,[4,numMod*600,1,1])
histObLsrOnMean_merge = np.reshape(histObLsrOnMean,[4,numMod*600,1,1])
histLsrOnCamTemps_merge = np.reshape(histLsrOnCamTemps,[4,numMod*600,1,1])
histLsrOnCamTemps_merge[histLsrOnCamTemps_merge > 100] = np.nan
test5 = np.reshape(histMainLsrOnStd,[4,numMod*600,1,1])
test6 = np.reshape(histObLsrOnStd,[4,numMod*600,1,1])

histLsrOffCamTemps_merge = np.reshape(histLsrOffCamTemps,[4,numMod*2,1,1])
histLsrOffCamTemps_merge[histLsrOffCamTemps_merge > 100] = np.nan

histLsrOffTimeStamp_merge = np.reshape(histLsrOffTimeStamp,[4,numMod*2,1,1])
histMainLsrOffMean_merge = np.reshape(histMainLsrOffMean,[4,numMod*2,1,1])


#%%

plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],histMainLsrOnMean_merge[camInd,:,0,0])
plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],histObLsrOnMean_merge[camInd,:,0,0])
ax2 = plt.twinx()
ax2.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],histMainLsrOnMean_merge[camInd,:,0,0]-histObLsrOnMean_merge[camInd,:,0,0],'g')
plt.show()

plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],test5[camInd,:,0,0])
plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],test6[camInd,:,0,0])
ax2 = plt.twinx()
ax2.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],test5[camInd,:,0,0]-test6[camInd,:,0,0],'g')
plt.show()

plt.plot(imgLsrOnTimeStamp_merge[camInd,:,0],imgMainLsrOffMean[camInd,:,0])
ax2 = plt.twinx()
ax2.plot(imgLsrOnTimeStamp_merge[camInd,:,0],imgMainLsrOffStd[camInd,:,0],'g')
plt.show()

plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],histMainLsrOnMean_merge[camInd,:,0,0],'-r')
plt.plot(imgLsrOnTimeStamp_merge[camInd,:,0],imgMainLsrOnMean[camInd,:,0],'-b')

plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],histObLsrOnMean_merge[camInd,:,0,0],'-r')
plt.plot(imgLsrOnTimeStamp_merge[camInd,:,0],imgMainLsrOffMean[camInd,:,0],'-b')
plt.plot(histLsrOffTimeStamp_merge[camInd,:,0,0],histMainLsrOffMean_merge[camInd,:,0,0],'-g')
plt.show()

plt.plot(histLsrOnCamTemps_merge[camInd,:,0,0],histMainLsrOnMean_merge[camInd,:,0,0],'-r')
ax2 = plt.twinx()
ax2.plot(histLsrOffCamTemps_merge[camInd,:,0,0],histMainLsrOffMean_merge[camInd,:,0,0],'g')
plt.show()


plt.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],contrastNoFilter_merge[camInd,:,0,0],'-k')
ax2 = plt.twinx()
ax2.plot(histLsrOnTimeStamp_merge[camInd,:,0,0],correctedMean_merge[camInd,:,0,0],'-r')
plt.show()

# for camInd in
# plt.plot(histLsrOnCamTemps_merge[camInd,:,0,0],contrastNoFilter_merge[camInd,:,0,0],'-k')
# ax2 = plt.twinx()
# ax2.plot(histLsrOnCamTemps_merge[camInd,:,0,0],correctedMean_merge[camInd,:,0,0],'-r')
# plt.show()



# Calculating variance by subtracting all of variance from demod
# test = ((histMainLsrOnVar[:,:,0]-histMainLsrOnVar[:,:,3])**0.5) / avgImgMean[:,:,0]


        
# ambientImgs = imgMainLsrOffMean - imgObLsrOffMean
# ambientImgObHist = imgMainLsrOffMean - histObLsrOnMean
    
# Note that initialCamTemp is calcd wrong (should be first value from test scan)

# 600    frames LsrOn hist 
# 5      frames blank
# 2      frames LsrOff hist
# ~18-20 frames (1 LasOff, 1 LsrOn image)

holder = np.zeros(29)
for ind in range(29):
    holder[ind] = histLsrOffTimeStamp[0,ind,0,1]-histLsrOnTimeStamp[0,ind+1,0,0]

#%% Get all of one module from all scans (long scans)
avgTemps = np.zeros((numCh,numScans))
avgImgMean = np.zeros((numCh,numScans))
avgCont = np.zeros((numCh,numScans))
darkHistWid = np.zeros((numCh,numScans))
for scanInd in range(numScans):
    for chInd in range(numMod):
        # print([chInd,scanInd])
        avgTemps[chInd,scanInd] = scans[scanInd].channelData[chInd*numMod].camTemps.mean()
        avgCont[chInd,scanInd] = scans[scanInd].channelData[chInd*numMod].contrastNoFilter.mean()
        avgImgMean[chInd,scanInd] = scans[scanInd].channelData[chInd*numMod].correctedMean.mean()
        darkHistWid[chInd,scanInd] = scans[scanInd].channelData[chInd*numMod].imageLaserOffHistWidth
        
        histLsrOffObMean = np.zeros((numCh,numMod))
        histObMean = np.zeros((numCh,numMod))
    
#%%
ind=0
t = np.arange(600)/40
for ind1 in range(3):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,14))
    fig.tight_layout()
    for ind2 in range(5):
        data2plot = (scanData.channelData[ind].contrastVertNorm+scanData.channelData[ind+15*2].contrastVertNorm)/2
        data2plot2 = (scanData.channelData[ind+15*1].contrastVertNorm+scanData.channelData[ind+15*3].contrastVertNorm)/2
        ax[ind2].plot(t+(ind)*15,1-data2plot,'b')
        ax[ind2].plot(t+(ind)*15,1-data2plot2,'r')
        # ax[ind2].plot(t+(ind)*15,1-(data2plot+data2plot2)/2,'k')
        ax[ind2].set_title(ind+1)
        ind = ind+1
    ax[0].legend(['Far','Near'])
    fig.savefig('/Users/brad/Desktop/ExhaleInhale_' + str(ind1) + '.png',dpi=300)

ind=0
t = np.arange(600)/40
for ind1 in range(3):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,14))
    fig.tight_layout()
    for ind2 in range(5):
        data2plot = (scanData.channelData[ind].correctedMean+scanData.channelData[ind+15*2].correctedMean)/2
        data2plot2 = (scanData.channelData[ind+15*1].correctedMean+scanData.channelData[ind+15*3].correctedMean)/2
        ax[ind2].plot(t+(ind)*15,data2plot,'b')
        # ax[ind2].plot(t+(ind)*15,data2plot2,'r')
        ax2 = ax[ind2].twinx()
        ax2.plot(t+(ind)*15,data2plot2,'r')
        ax2.set_ylim(207,233)
        ax[ind2].set_ylim(39,44.5)
        ax[ind2].set_title(ind+1)
        ind = ind+1
    # ax[0].legend(['Far','Near'])
    fig.savefig('/Users/brad/Desktop/ExhaleInhale_' + str(ind1) + '.png',dpi=300)


t = np.arange(600)/40

minMax = [1,1,0,0]
for ind in range(15):
    data2plot = (scanData.channelData[ind].contrastNoFilter+scanData.channelData[ind+15*2].contrastNoFilter)/2
    data2plot2 = (scanData.channelData[ind+15*1].contrastNoFilter+scanData.channelData[ind+15*3].contrastNoFilter)/2
    minMax[0] = np.min([minMax[0],data2plot.min()])
    minMax[1] = np.min([minMax[1],data2plot2.min()])
    minMax[2] = np.max([minMax[2],data2plot.max()])
    minMax[3] = np.max([minMax[3],data2plot2.max()])
    
ind=0
for ind1 in range(3):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,14))
    fig.tight_layout()
    for ind2 in range(5):
        data2plot = (scanData.channelData[ind].contrastNoFilter+scanData.channelData[ind+15*2].contrastNoFilter)/2
        data2plot2 = (scanData.channelData[ind+15*1].contrastNoFilter+scanData.channelData[ind+15*3].contrastNoFilter)/2
        ax[ind2].plot(t+(ind)*15,data2plot,'b')
        ax2 = ax[ind2].twinx()
        ax2.plot(t+(ind)*15,data2plot2,'r')
        ax[ind2].set_title(ind+1)
        ax[ind2].set_ylim(minMax[0],minMax[2])
        ax2.set_ylim(minMax[1],minMax[3])
        ind = ind+1
    # ax[0].legend(['Far','Near'])
    fig.savefig('/Users/brad/Desktop/ExhaleInhale_' + str(ind1) + '.png',dpi=300)
    
#%%
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,7))
# ax[0].plot(scans[0].channelData[8].contrastVertNorm)
# ax[1].plot(scans[0].channelData[11].contrastVertNorm)

ind=0
t = np.arange(600)/40
for ind1 in range(3):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,14))
    fig.tight_layout()
    for ind2 in range(5):
        data2plot = scanData.channelData[ind+15*2].contrastVertNorm
        ax[ind2].plot(t+(ind)*15,1-data2plot,'b')
        ax[ind2].set_title(ind+1)
        ind = ind+1
    fig.savefig('/Users/brad/Desktop/ExhaleInhale_' + str(ind1) + '.png',dpi=300)

#%%
if 0:
    for scanInd,scan in enumerate(scans):
        savename = re.split('/', non_folder_dirs[scanInd])[-2] + '_' + re.split('/', non_folder_dirs[scanInd])[-1]
        # scan.PlotInvertedAndCenteredContrastAndMean( scan.path )
        
        scan.PlotContrastMeanAndFrequency(scan.path, plotContrast=True, plotMean=True, plotGoldenPulses=False, plotFreq=False,  plotUnfilteredContrast=False, plotTemp = False)
        scan.PlotCompare4Channels(titleStr=savename, saveFig=savepath, plotGoldenPulses=False)
        # scan.dt = 0.0125
        
        # scan.PrintCV()
        # Code to get feature arrays and time points in long scans
        pulseAmplitudes = []
        pulseStarts = []
        pulseRaw = []
        
        if 1: # scan.scanType==2 or scan.scanType==4:
            for index, ch in enumerate(scan.channelData):
                if ch.pulseSegmentsFeatures:
                    pulseAmplitudes.append(ch.pulseSegmentsFeatures.amplitude)
                    pulseStarts.append(ch.onsets)
                    pulseRaw.append(ch.pulseSegments)
                    # print('Amplitudes',ch.pulseSegmentsFeatures.amplitude)
                    # print('Onsets',ch.onsets)
        
        temps = np.zeros(len(scan.channelData))
        darkHistWid = np.zeros(len(scan.channelData))
        for chInd in range(len(scan.channelData)):
            temps[chInd] = scan.channelData[chInd].camTemps.mean()
            darkHistWid[chInd] = scan.channelData[chInd].imageLaserOffHistWidth
        print(temps[[0,1,2,3,8,9,10,11,12,13,14,15]].mean())
        print(darkHistWid[darkHistWid > 15].mean())
    
#%%
for scanInd,scan in enumerate(scans):
    savename = re.split('/', non_folder_dirs[scanInd])[-2] + '_' + re.split('/', non_folder_dirs[scanInd])[-1]
    scan.PlotLongScan(titleStr=savename, saveFig=savepath, cropTime=[])
    
    # scan.PlotLongScanCompare4Channels(titleStr=savename, saveFig=savepath, cropTime=[0,25])
    
    for tInd in range(10):
        cTime = [tInd*24,(tInd+1)*24]
        cTimeStr = '_diff' + str(cTime[0]).zfill(3) + '-' + str(cTime[1]).zfill(3)
        scan.PlotLongScanCompare4Channels(titleStr=savename+cTimeStr, saveFig=savepath, cropTime=cTime)
        