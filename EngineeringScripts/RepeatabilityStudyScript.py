#%%
import sys
sys.path.append('.'); sys.path.append('..')
from ReadGen2Data import ReadGen2Data, PrettyFloat4, ConvenienceFunctions, PulseFeatures
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re, math
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
searchpath = r""

###############################################################################################################################################

#scanname = askdirectory(title='Select Scan', initialdir=searchpath)
scanname = '/Users/kedar/Desktop/brad-20230807'
scanPath = scanname

print("Subject Folder: " + scanPath)

# %%
#Go to each subfolder with the pattern Fullscan_* in its name and read the data
subfolders = [f.path for f in os.scandir(scanPath) if f.is_dir() and re.search('FULLSCAN*', f.path)]
subfolders.sort()
print("Scan Folders: ", subfolders)

#Create empty data frame to store all features - columns are amplitude, time to P1, contrast average,  Dark frame image mean, Dark frame histogram width, cam temp
allOpticalFeaturesDf = pd.DataFrame(columns=['folder','camera','amplitude', 'timeToP1', 'contrastAvg', 'imageMean', 'histWidthLsrOff', 'imgMeanLsrOff', 'camTemp'])
pltInd = [1,4,2,3]
#%% Loop through each subfolder and read the data
for scanPath in subfolders:
    scan = ReadGen2Data(scanPath)
    scan.ReadDataAndComputeContrast()
    scan.DetectPulseAndFindFit()
    scan.ComputeGoldenPulseFeatures()
    ind = 0
    for cam in range(0,15,4):
        #Create data frame with all features
        folder = scanPath.split('/')[-1]
        camera = scan.channelData[cam].channelPosition
        amplitude = scan.channelData[cam].goldenPulseFeatures.amplitude
        timeToP1 = scan.channelData[cam].goldenPulseFeatures.pulseOnset
        contrastAvg = np.mean(scan.channelData[cam].contrast)
        imageMean = np.mean(scan.channelData[cam].correctedMean)
        histWidthLsrOff = scan.channelData[cam].imageLaserOffHistWidth
        camTemp = scan.channelData[cam].camTemps[-1] #last value in the list is the most recent initialCamTemp has the first data point if need to swap it
        imgMeanLsrOff = scan.channelData[cam].imageLaserOffImgMean
        allOpticalFeaturesDf = allOpticalFeaturesDf.append({'folder':folder, 'camera':camera, 'amplitude':amplitude, 'timeToP1':timeToP1, 'contrastAvg':contrastAvg, 'imageMean':imageMean, 'histWidthLsrOff':histWidthLsrOff, 'imgMeanLsrOff':imgMeanLsrOff, 'camTemp':camTemp}, ignore_index=True)
        #Plot contrast
        plt.subplot(2,2,pltInd[ind])
        plt.plot(np.arange(len(scan.channelData[cam].contrast))*dt, scan.channelData[cam].contrast)
        plt.gca().invert_yaxis()
        plt.title(camera)
        plt.xlabel('Time (s)')
        plt.ylabel('Contrast')
        ind += 1
    plt.suptitle(folder)
    plt.tight_layout()
    #Save plot
    folder = scanPath.split('/')[-1]
    plotOutputPath = os.path.join(scanPath, folder + '_contrast.png')
    plt.savefig(plotOutputPath)
    plt.show()
    ind = 0
    for cam in range(0,15,4):
        camera = scan.channelData[cam].channelPosition
        #Plot golden pulse and pulse segments that compose the golden pulse
        plt.subplot(2,2,pltInd[ind])
        if scan.channelData[cam].goldenPulse is None or scan.channelData[cam].pulseSegments is None:
            continue
        plt.plot(np.arange(len(scan.channelData[cam].pulseSegments))*dt,scan.channelData[cam].pulseSegments, color='0.8')
        plt.plot(np.arange(len(scan.channelData[cam].goldenPulse))*dt,scan.channelData[cam].goldenPulse)
        plt.gca().invert_yaxis()
        plt.title(camera + ' Amplitude: ' + str(PrettyFloat4(scan.channelData[cam].goldenPulseFeatures.amplitude)) )
        plt.xlabel('Time (s)')
        plt.ylabel('Contrast')
        ind += 1
    plt.suptitle(folder)
    plt.tight_layout()
    #Save plot
    folder = scanPath.split('/')[-1]
    plotOutputPath = os.path.join(scanPath, folder + '_contrast.png')
    plt.savefig(plotOutputPath)
    plt.show()
#%% For each camera write data to a csv file with the mean, std and coefficient of variation
cameras = allOpticalFeaturesDf['camera'].unique()
for camera in cameras:
    csvName = scanname + '/' + camera + '.csv'
    data = allOpticalFeaturesDf.loc[ (allOpticalFeaturesDf['camera'] == camera)]
    #Drop camera column
    data = data.drop(columns=['camera'])
    #Drop index column
    data = data.reset_index(drop=True)
    #Calculate the mean, std and coefficient of variation
    mean = data.mean()
    std = data.std()
    cv = std/mean
    #Append the mean vector, std vector and cv to data use mean, std and cv strings as the folder column
    data = data.append({'folder':'mean', 'amplitude':mean['amplitude'], 'timeToP1':mean['timeToP1'], 'contrastAvg':mean['contrastAvg'], 'imageMean':mean['imageMean'], 'histWidthLsrOff':mean['histWidthLsrOff'], 'camTemp':mean['camTemp']}, ignore_index=True)
    data = data.append({'folder':'std', 'amplitude':std['amplitude'], 'timeToP1':std['timeToP1'], 'contrastAvg':std['contrastAvg'], 'imageMean':std['imageMean'], 'histWidthLsrOff':std['histWidthLsrOff'], 'camTemp':std['camTemp']}, ignore_index=True)
    data = data.append({'folder':'cv', 'amplitude':cv['amplitude'], 'timeToP1':cv['timeToP1'], 'contrastAvg':cv['contrastAvg'], 'imageMean':cv['imageMean'], 'histWidthLsrOff':cv['histWidthLsrOff'], 'camTemp':cv['camTemp']}, ignore_index=True)
    
    #Write the data to a csv file
    data.to_csv(csvName)

# %%
camerasTemp = [ camera+'_Temp' for camera in cameras]
camerasHistWidth = [ camera+'_HistWidth' for camera in cameras]
camerasimgMeanLsrOff = [ camera+'_imgMeanLsrOff' for camera in cameras]
camTempAndHistWidthDf = pd.DataFrame(columns=['folder', *camerasTemp, *camerasHistWidth, *camerasimgMeanLsrOff])
#Add a row for each folder
for folder in allOpticalFeaturesDf['folder'].unique():
    camTempAndHistWidthDf = camTempAndHistWidthDf.append({'folder':folder}, ignore_index=True)

#Loop through each folder and add corresponding data
for folder in allOpticalFeaturesDf['folder'].unique():
    data = allOpticalFeaturesDf.loc[ (allOpticalFeaturesDf['folder'] == folder)]
    for camera in cameras:
        camTempAndHistWidthDf.loc[ (camTempAndHistWidthDf['folder'] == folder), camera+'_Temp'] = data.loc[ (data['camera'] == camera)]['camTemp'].values[0]
        camTempAndHistWidthDf.loc[ (camTempAndHistWidthDf['folder'] == folder), camera+'_HistWidth'] = data.loc[ (data['camera'] == camera)]['histWidthLsrOff'].values[0]
        camTempAndHistWidthDf.loc[ (camTempAndHistWidthDf['folder'] == folder), camera+'_imgMeanLsrOff'] = data.loc[ (data['camera'] == camera)]['imgMeanLsrOff'].values[0]

#Write the data to a csv file
camTempAndHistWidthDf.to_csv(scanname + '/camTempAndHistWidth.csv')

# %%
