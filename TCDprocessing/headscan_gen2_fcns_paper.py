
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import re
import copy
import pandas as pd
from scipy.optimize import least_squares
from collections.abc import Sized
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.special import erf  
from scipy.special import comb   #N choose k
#from matplotlib import animation
#import matplotlib as mpl 
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from scipy.stats import mode
from tkinter.filedialog import askdirectory
import biosppy

from shapely.geometry import Polygon 
from scipy.stats import moment
from scipy.stats import describe
import biosppy
from skimage.restoration import denoise_wavelet
from matplotlib.lines import Line2D

import scipy.stats as stats

def processWaveforms(dataIn,freqIn,invertedData,freqOut,dataPeakFindingRaw,plotTitle,plotOutputs):
    # Takes in raw optical or TCD timeseries data and process the waveforms from it

    ### Dummy data loading for debugging
    # S12/8/7/4/3/0 don’t exist
        # modInd = 9 
        # scanInd = 1 # TCD 23 19 17 16
        # dataIn = copy.deepcopy(scans[scanInd]['moduleData_rBFI'][:,1,modInd])
        # dataIn = copy.deepcopy(scans[scanInd]['data_TCD'][:])
        # freqIn = 125
        # invertedData = 0
        # freqOut = 125
        # dataPeakFinding = copy.deepcopy(scans[scanInd]['moduleData_rBFI'][:,:,modInd].mean(axis=1))
        # dataPeakFinding = copy.deepcopy(scans[scanInd]['data_TCD'][:])
        # dataPeakFinding -= np.nanmin(dataPeakFinding)
        # plotTitle = 'Subj23_Cam1_Mod14'
        # plotOutputs = True
        
        # dataIn = copy.deepcopy(envTCD)
        # freqIn = 125
        # invertedData = 0
        # freqOut = 125
        # dataPeakFinding = copy.deepcopy(envTCD)
        # plotTitle = 'Subj23'
        # plotOutputs = True
    
    colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','r','k','m','y','c','g']
    tDataIn = np.arange(dataIn.shape[0])/freqIn
    
    ### Normalize data used for peak finding from 0-1
    if invertedData:
        dataPeakFinding = -dataPeakFindingRaw
    else:
        dataPeakFinding = dataPeakFindingRaw
    # dataPeakFinding = dataPeakFinding - np.nanmin(dataPeakFinding)
    # dataPeakFinding = dataPeakFinding / np.nanmax(dataPeakFinding)
    
    period = getTcdPeriod(dataPeakFinding,freqIn,'')
    
    if 0:
        # Creating box car averaged version of the time series
        winWid = int(np.ceil(period*0.3) // 2 * 2 + 1) # 61 # 45% of period ish?s
        # dataMedFilt = median_filter(dataPeakFinding,size=winWid,mode='nearest')
        # dataConvValid = np.convolve(dataPeakFinding,np.ones(winWid)/winWid,'valid')
        dataCumsum = np.nancumsum(np.insert(dataPeakFinding, 0, 0)) 
        dataConvValid = (dataCumsum[winWid:] - dataCumsum[:-winWid]) / winWid
        # print(np.allclose(dataConvValid,dataConvValid2)) # confirms equivalence to convolution method
        
        dataConv_peaks = find_peaks(dataConvValid, distance=int(period/2))[0]+int(np.floor(winWid/2))
        dataConv_trofs = find_peaks(-dataConvValid, distance=int(period/2))[0]+int(np.floor(winWid/2))
        
        pulsesTrofs_loc = []
        # if first is peak and min-point is less than first point of time series, use as start of pulse
        # Ex. s25 c1 m8
        test1 = np.nanmin(dataPeakFinding[:dataConv_peaks[0]]) - dataPeakFinding[0]
        if dataConv_peaks[0] < dataConv_trofs[0] and test1 < 0:
            pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[:dataConv_peaks[0]]))
        # Ex. s25 c1 m4
        if dataConv_trofs[0] < dataConv_peaks[0]:
            pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_trofs[0]:dataConv_peaks[0]])+dataConv_trofs[0])
        # Ex. s25 c1 m14 needs this
        # elif dataConv_peaks[0] < pulsesPeaks_loc[0]:
        #     pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_peaks[0]:pulsesPeaks_loc[0]])+dataConv_peaks[0])
        
        # Finds troughs between all peaks in dataConv_peaks
        for peakInd in range(len(dataConv_peaks)-1):
            pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_peaks[peakInd]:dataConv_peaks[peakInd+1]])+dataConv_peaks[peakInd])
        
        # if last trough is before last peak, trust that true trough exists at tend
        # May have to set theshold factor to less than 1.0 to ensure no dichrotic notches are detected
        # Ex. s18 c1 m0
        test3 = np.nanmin(dataPeakFinding[dataConv_peaks[-1]:]) < dataPeakFinding[dataConv_trofs[-1]]*1.0 and \
            dataPeakFinding[-1] - np.nanmin(dataPeakFinding[dataConv_peaks[-1]:]) > 0
        if dataConv_trofs[-1] < dataConv_peaks[-1] and test3:
            pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_peaks[-1]:])+dataConv_peaks[-1])
            # pulsesPeaks_loc.append(np.nanargmax(dataPeakFinding[dataConv_trofs[-1]:dataConv_peaks[-1]])+dataConv_trofs[-1])
        test2 = np.nanmax(dataPeakFinding[dataConv_trofs[-1]:]) > dataPeakFinding[dataConv_peaks[-1]]*0.75
        if dataConv_peaks[-1] < dataConv_trofs[-1] and test2:
            pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_trofs[-1]:])+dataConv_trofs[-1])
        
        # Finds peaks between all troughs in conv data
        pulsesPeaks_loc = []
        for trofInd in range(len(pulsesTrofs_loc)-1):
            pulsesPeaks_loc.append(np.nanargmax(dataPeakFinding[pulsesTrofs_loc[trofInd]:pulsesTrofs_loc[trofInd+1]])+pulsesTrofs_loc[trofInd])
    
    # if len(np.argwhere(np.isnan(x))) == 0:
    #     soshp = signal.butter(2,15,'lp',fs=freqIn,output='sos')
    #     x = signal.sosfilt(soshp, x-np.nanmean(x))+np.nanmean(x)
    #     dataIn = signal.sosfilt(soshp, dataIn-np.nanmean(dataIn))+np.nanmean(dataIn)
    #     # x = denoise_wavelet(xUnfilt,method='BayesShrink', mode='soft', wavelet_levels=6, wavelet='sym3', rescale_sigma='True')
    #     # _, filteredCh, onsetsCh, _, hr = biosppy.signals.ppg.ppg(signal=-xUnfilt, sampling_rate=freqIn, show=True)
    # else:
    #     print('Skipping LP filtering due to NaNs')
        # nanlocs = np.argwhere(np.isnan(x))
        # fig, ax = plt.subplots()
        # ax.plot(nanlocs, np.ones(nanlocs.shape),'o')
        # ax.set_ylabel('1 = location of NaN')
        # ax.set_xlabel('Time (TCD pts)')
        # ax.set_title(filename + '\nWarning, ' + str(len(np.argwhere(np.isnan(x)))) + ' Nans removed for FFT calc.')
        # x = x[~np.isnan(x)]
    
    ### Kedar's pulse segmentation methods
    # chPeaks, chUseSegment, goldenPulse, pulseSegments = FindLowNoisePulseAndAverage(-dataIn[:,np.newaxis], period, 'NA', 1/freqIn)
    # if invertedData:
    #     offset = 1
    # else:
    #     offset = 5
    # pulsesStarts = copy.deepcopy(chPeaks[0])-offset
    
    ### Based on Soren's pulse segmentation methods ish
    # minPeriod = int(np.floor(period*0.75))
    # pulsesStarts = find_peaks(-dataPeakFinding, distance=minPeriod)[0]
    # pulsesPeaks_loc = find_peaks(dataPeakFinding, distance=minPeriod)[0]
    #pulsesStarts = copy.deepcopy(onsetsCh)
    
    ### Brad's pulse segmenation method
    winWid = int(np.ceil(period*0.3) // 2 * 2 + 1) # 30% of period ish, to nearest odd number
    pulsesTrofs_loc,pulsesPeaks_loc,dataConv_trofs,dataConv_peaks = findPeaksTrofs(dataPeakFinding,dataIn,period,winWid)
    pulsesStarts = copy.deepcopy(pulsesTrofs_loc)
    
    ### Removes false pulses that are above minPeriod requirement or too close to start/end of dataset
    pulsesToUse = []
    for pkInd in range(len(pulsesStarts)-1):
        pulseMin = np.nanmin(dataPeakFinding[pulsesStarts[pkInd]:pulsesStarts[pkInd+1]])
        pulseMax = np.nanmax(dataPeakFinding[pulsesStarts[pkInd]:pulsesStarts[pkInd+1]])
        if (dataPeakFinding[pulsesStarts[pkInd]]-pulseMin) < (pulseMax-pulseMin)*0.5 and len(dataPeakFinding)-pulsesStarts[pkInd+1] > period/10:
            pulsesToUse.append(True)
        else:
            pulsesToUse.append(False)
    pulsesToUse.append(False)
    
    ### Vertically Normalized Waveforms from containwaveform2
    if 0:
        # use gaussian filter if analyzing IMAGE MEAN DATA
        dataVertNorm = gaussian_filter1d(dataPeakFinding, 1)
    
    # Only uses data from first & last valid pulse
    firstValidPulseStart_ind = np.argmax(np.array(pulsesToUse*1)>0)
    firstValidPulseStart_loc = pulsesStarts[firstValidPulseStart_ind]
    lastValidPulseStart_ind = len(pulsesToUse)-np.argmax(np.array(np.flip(pulsesToUse)*1)>0)-1
    lastValidPulseStart_loc = pulsesStarts[lastValidPulseStart_ind]
    
    # Shouldn't need anymore
    minPeakLoc = firstValidPulseStart_loc - 10
    if minPeakLoc < 0:
        minPeakLoc = 0
    
    trofs_loc = np.array(pulsesStarts[firstValidPulseStart_ind:])
    peaks_loc = np.array([x for x in pulsesPeaks_loc if x <= lastValidPulseStart_loc+period/2])
    
    dataVertNorm = vertNormWaveform(dataIn,trofs_loc,peaks_loc)
    dataVertNorm = vertNormWaveform(dataVertNorm,trofs_loc,peaks_loc)
    # Okay to have negative values actually
    # dataVertNorm[dataVertNorm<0]=0
    # dataVertNorm[dataVertNorm>1]=1
    
    ### Old functions for flattening pulses
    # vertNormWaveform(x,minPeriod,minPeakLoc) flattens bottom & top to 0-1
    # flattenbottom(x,minPeriod)   flattens bottom to 0
    # containwaveform2(x,period)
    #   2 flattenwaveform and then get top peaks (essentially what was here)
    #   wipes anything above/below 1/0
    # platinumPeriod(x,period)
    # platinumPulse(x,period)
    
    ### Creating List of Regular & Vertically Normalized Pulses, including bad pulses
    normLength = 160
    pulses = []
    pulsesVertNorm = []
    pulsesVertHorNorm = []
    pulsesVertHorNormAvg = np.zeros(normLength)
    for pulseInd in range(len(pulsesStarts)):
        if pulseInd == len(pulsesStarts)-1:
            if len(dataIn[pulsesStarts[pulseInd]:]) > 1:
                pulses.append(dataIn[pulsesStarts[pulseInd]:])
                pulsesVertNorm.append(dataVertNorm[pulsesStarts[pulseInd]:])
                
                tPulseVertNorm = np.round(np.linspace(0,len(pulsesVertNorm[pulseInd])*freqOut,num=pulsesVertNorm[pulseInd].shape[0], endpoint=False),7)
                tPulseVertHorNorm = np.round(np.linspace(0,(len(pulsesVertNorm[pulseInd])-1)*freqOut,num=normLength, endpoint=True),7)
                f = interp1d(tPulseVertNorm,pulsesVertNorm[pulseInd],kind='linear')
                pulsesVertHorNorm.append(f(tPulseVertHorNorm))
                if pulsesToUse[pulseInd]:
                    pulsesVertHorNormAvg = pulsesVertHorNormAvg + f(tPulseVertHorNorm)
        else:
            pulses.append(dataIn[pulsesStarts[pulseInd]:(pulsesStarts[pulseInd+1]+1)])
            pulsesVertNorm.append(dataVertNorm[pulsesStarts[pulseInd]:(pulsesStarts[pulseInd+1]+1)])
            
            tPulseVertNorm = np.round(np.linspace(0,len(pulsesVertNorm[pulseInd])*freqOut,num=pulsesVertNorm[pulseInd].shape[0], endpoint=False),7)
            tPulseVertHorNorm = np.round(np.linspace(0,(len(pulsesVertNorm[pulseInd])-1)*freqOut,num=normLength, endpoint=True),7)
            f = interp1d(tPulseVertNorm,pulsesVertNorm[pulseInd],kind='linear')
            pulsesVertHorNorm.append(f(tPulseVertHorNorm))
            if pulsesToUse[pulseInd]:
                pulsesVertHorNormAvg = pulsesVertHorNormAvg + f(tPulseVertHorNorm)
    pulsesVertHorNormAvg -= pulsesVertHorNormAvg.min()
    pulsesVertHorNormAvg /= pulsesVertHorNormAvg.max()
    #%%
            
    ### Calculating Stats of the Pulses
    numStats = 20 # assumes less than or equal to 20 stats calculated
    pulsesStats = np.zeros((len(pulsesVertNorm),numStats))*np.nan 
    dataStats = np.zeros((dataIn.shape[0],numStats))*np.nan
    
    for pulseInd in range(len(pulsesVertNorm)-1):
        
        # Fix for issue with processing crappy TCD data
        if np.isnan(pulsesVertNorm[pulseInd][:5]).all():
            pulsesToUse[pulseInd] = False
            print('Excluding pulse number: ' + str(pulseInd) + ' (startInd: ' + str(pulsesStarts[pulseInd]) + ') due to all NaNs')
        
        if pulsesToUse[pulseInd]:
            pulseVertNorm = copy.deepcopy(pulsesVertNorm[pulseInd])
            
            pulsePeriod = pulsesStarts[pulseInd+1]-pulsesStarts[pulseInd]
            pulseAvg = np.nanmean(pulses[pulseInd])
            # pulseAmp = np.nanmax(pulses[pulseInd])-np.nanmin(pulses[pulseInd])
            pulseAmp = np.nanmax(pulses[pulseInd])-pulses[pulseInd][0]
            
            pulseVertHorNorm_Skew = describe(pulsesVertHorNorm[pulseInd]).skewness
            pulseVertHorNorm_Kurt = describe(pulsesVertHorNorm[pulseInd]).kurtosis
            
            gpDistribution = pulseVertNorm-np.amin(pulseVertNorm)
            gpDistribution = np.amax(gpDistribution)-gpDistribution
            gpDistribution = gpDistribution / np.sum(gpDistribution)
            pulseVertNormAuc = describe(gpDistribution)
            pulseVertNormAuc_Skew = pulseVertNormAuc.skewness
            pulseVertNormAuc_Kurt = pulseVertNormAuc.kurtosis
            
            # pulseVertNormPolygon = Polygon(list(zip(np.arange(len(pulseVertNorm))/len(pulseVertNorm),pulseVertNorm)))
            # pulseVertNorm_Centroid = list(pulseVertNormPolygon.centroid.coords)[0]
            pulseVertNorm_Centroid = [0,0]
            
            pulseRange = np.nanmax(pulseVertNorm)-np.nanmin(pulseVertNorm)
            totalTime  = np.count_nonzero(~np.isnan(pulseVertNorm))
            indsInCanopy = pulseVertNorm < np.nanmax(pulseVertNorm)-pulseRange*0.25
            
            canopy    = np.sum(indsInCanopy)/totalTime
            startInd  = np.nanargmax(pulseVertNorm[:5])
            sysRange  = np.nanmax(pulseVertNorm)-pulseRange*0.9
            sysInds   = pulseVertNorm>sysRange
            endInd    = np.nonzero(~sysInds)[0][0]
            onset     = float(abs(endInd-startInd))*freqOut
            onsetProp = float(abs(endInd-startInd))/totalTime
            secMoment = moment(pulseVertNorm,2)
            
            velCurInd, velCurIndNorm = ComputeVCI(pulseVertNorm,1/freqOut,False)
            velCurIndHann, velCurIndHannNorm = ComputeVCI(pulseVertNorm,1/freqOut,True)
            
            allParams = np.array([
                pulsePeriod,pulseAvg,pulseAmp,
                pulseVertHorNorm_Skew,pulseVertHorNorm_Kurt,pulseVertNormAuc_Skew,pulseVertNormAuc_Kurt,
                pulseVertNorm_Centroid[0],pulseVertNorm_Centroid[1],
                canopy,onset,onsetProp,secMoment,
                velCurInd,velCurIndNorm,velCurIndHann,velCurIndHannNorm,
                ])
            pulsesStats[pulseInd,:allParams.shape[0]] = allParams
            dataStats[pulsesStarts[pulseInd]:pulsesStarts[pulseInd+1],:] = np.tile(pulsesStats[pulseInd,:],(pulsePeriod,1))
    
    ### Plotting Outputs to Assess Accuracy
    if plotOutputs:
        # Comparing biosppy PPG
        mask = np.isnan(dataPeakFinding)*1
        leading_nans = mask.argmin()
        dataPeakFinding_Temp = dataPeakFinding[leading_nans:(leading_nans+np.sum((~np.isnan(dataPeakFinding))*1))]
        # soshp = signal.butter(2,15,'lp',fs=freq,output='sos')
        # dataPeakFinding_Temp = signal.sosfilt(soshp, dataPeakFinding_Temp-np.nanmean(dataPeakFinding_Temp))+np.nanmean(dataPeakFinding_Temp)
        # dataPeakFinding[leading_nans:(leading_nans+np.sum((~np.isnan(dataPeakFinding))*1))] = dataPeakFinding_Temp
        _,_,onsetsPpg,_,_ = biosppy.signals.ppg.ppg(signal=-dataPeakFinding_Temp, sampling_rate=freqIn, show=False)
        onsetsPpg = onsetsPpg + leading_nans
        
        tPulseVertHorNorm = np.arange(tPulseVertHorNorm.shape[0])/tPulseVertHorNorm.shape[0]
        dataConv = np.convolve(dataPeakFinding,np.ones(winWid)/winWid,'same')
        if dataPeakFinding.shape[0] > 1875:
            pulsesVertHorNormAvg_parts =[]
            plotRange = np.arange(np.ceil(dataPeakFinding.shape[0]/1875)).astype(int)
        else:
            pulsesVertHorNormAvg_parts = pulsesVertHorNormAvg
            plotRange = [0]
            
        for figInd in plotRange:
            fig, ax = plt.subplots(nrows=5,ncols=1,gridspec_kw={'height_ratios': [1,1,1,2,1]},figsize=(8,8))
            fig.tight_layout(h_pad=1.5)
            
            ax[0].plot(tDataIn[dataConv_peaks],dataPeakFinding[dataConv_peaks],c='r',linewidth=0.75,marker='o',markersize=3)
            ax[0].plot(tDataIn[dataConv_trofs],dataPeakFinding[dataConv_trofs],c='g',linewidth=0.75,marker='o',markersize=3)
            ax[0].plot(tDataIn,dataPeakFinding,'k',linewidth=0.5)
            ax[0].set_ylabel('dataPeakFinding')
            ax2 = plt.twinx(ax[0])
            ax2.plot(tDataIn,dataConv,color='gray',linewidth=0.5,linestyle='dashed')
            ax2.set_ylabel('Rolling Mean',color='gray')
            
            ax[1].plot(tDataIn,dataVertNorm,'k',linewidth=0.5)
            ax[1].scatter(tDataIn[pulsesPeaks_loc],dataVertNorm[pulsesPeaks_loc]+0.005,marker='v',s=10,c='r')
            ax[1].scatter(tDataIn[peaks_loc],dataVertNorm[peaks_loc]+0.005,marker='v',s=10,c='lime')
            ax[1].plot(tDataIn[onsetsPpg],dataVertNorm[onsetsPpg]-0.1,c='b',linewidth=0.75,marker='|',markersize=4)
            ax[1].scatter(tDataIn[pulsesStarts],dataVertNorm[pulsesStarts]-0.005,marker='^',s=10,c='r')
            ax[1].scatter(tDataIn[pulsesStarts[pulsesToUse]],dataVertNorm[pulsesStarts[pulsesToUse]]-0.005,marker='^',s=10,c='lime')
            for pulseInd in range(len(pulsesStarts)):
                ax[1].annotate(str(pulseInd),(tDataIn[pulsesStarts][pulseInd]+0.2,0))
            ax[1].legend(['data','rmvd','kept','ppg'],loc='upper right',fontsize=6)
            ax[1].set_ylabel('Vert. Norm. (0-1)')
            
            ax[2].plot(tDataIn,dataStats[:,0],'b')
            ax[2].tick_params(axis='y', colors='b')
            ax[2].set_ylabel('Period (samples)')
            ax2 = ax[2].twinx()
            ax2.plot(tDataIn,dataStats[:,1],'r')
            ax2.tick_params(axis='y', colors='r')
            ax2.set_ylabel('Pulse Avg.')
            ax3 = ax[2].twinx()
            ax3.spines.right.set_position(("axes", 1.1))
            ax3.plot(tDataIn,dataStats[:,2],'g')
            ax3.tick_params(axis='y', colors='g')
            ax3.set_ylabel('Pulse Amp.')
            ax[2].set_xlabel('Time (s)')
            ax[2].set_xlim(ax[1].get_xlim())
            
            ax[4].plot(tDataIn,dataStats[:,7],'b')
            ax[4].tick_params(axis='y', colors='b')
            ax[4].set_ylabel('pulseVertNorm_Centroid')
            ax2 = ax[4].twinx()
            ax2.plot(tDataIn,dataStats[:,3],'r')
            ax2.tick_params(axis='y', colors='r')
            ax2.set_ylabel('pulseVertHorNorm_Skew')
            ax3 = ax[4].twinx()
            ax3.spines.right.set_position(("axes", 1.1))
            ax3.plot(tDataIn,dataStats[:,16],'g')
            ax3.tick_params(axis='y', colors='g')
            ax3.set_ylabel('velCurIndHannNorm')
            ax[4].set_xlabel('Time (s)')
            ax[4].set_xlim(ax[1].get_xlim())
            
            pulseIndPerPlot = np.where((tDataIn[pulsesStarts]>=figInd*15-1) & (tDataIn[pulsesStarts]<=figInd*15+16))[0]
            legendNames = []
            # for pulseInd in range(len(pulsesStarts)):
            for pulseInd in pulseIndPerPlot:
                if pulsesStarts[pulseInd] in pulsesStarts[pulsesToUse]:
                    ## Align on trough
                    # tPulse = np.arange(len(pulsesVertNorm[pulseInd]))/freqIn
                    # ax[3].plot(tPulse,pulsesVertNorm[pulseInd],c=colors[pulseInd%len(colors)],linewidth=1.0)
                    ## Aligned on left FWHM
                    pulseVertNormShifted = alignLeftFwhm(pulsesVertNorm[pulseInd],30,freqIn)
                    tPulse = np.arange(len(pulseVertNormShifted))/freqIn
                    ax[3].plot(tPulse,pulseVertNormShifted,c=colors[pulseInd%len(colors)],linewidth=1.0)
                    # ax[3].plot(tPulseVertHorNorm,pulsesVertHorNorm[pulseInd],c=colors[pulseInd%len(colors)],linewidth=1.0)
                    legendNames.append(str(pulseInd))
            ax[3].set_ylabel('Normalized Height')
            ax[3].set_xlabel('Time (s)')
            ax[3].legend(legendNames,loc='upper right',fontsize=6)
            ax[3].grid()
            
            ax[0].set_xlim(figInd*15-1,figInd*15+16)
            ax[1].set_xlim(figInd*15-1,figInd*15+16)
            ax[2].set_xlim(figInd*15-1,figInd*15+16)
            ax[4].set_xlim(figInd*15-1,figInd*15+16)
             
            plotTitle2 = plotTitle + '_Part' + str(figInd)
            ax[0].set_title(plotTitle2)
            fig.savefig('/Users/brad/Desktop/gen2 results/WaveformPlots/'+plotTitle[0:6]+'/'+plotTitle2+'.png',dpi=300,bbox_inches='tight')
            plt.close()
 
    return dataVertNorm,pulsesStarts,pulsesToUse,pulses,pulsesVertNorm,pulsesVertHorNorm,pulsesStats,dataStats #,pulsesVertHorNormAvg_parts

def plotAgPulse(dataIn,plotTitle,manualInput_loc,pulses_parts,freq,peakCutoffs):
    ### Plot Silver Waveform
    agPulse = copy.deepcopy(dataIn)
    
    peaksManual_loc = manualInput_loc[0:3]
    peaksUsed_loc = copy.deepcopy(peaksManual_loc)
    trofsManual_loc = manualInput_loc[3:5]
    trofsUsed_loc = copy.deepcopy(trofsManual_loc)
    
    # Setting up parameters
    results_full = signal.peak_widths(agPulse, [np.nanargmax(agPulse[:int(0.6*freq)])], rel_height=0.5)
    leftHalfMax_ind = int(np.round(results_full[2][0]))
    if freq == []:
        tNormFact = agPulse.shape[0]
        xlabel = 'Normalize Time'
    else:
        tNormFact = freq
        xlabel = 'Time (s)'
    fig,ax = plt.subplots(figsize=(6.4,4.8))
    t = np.arange(agPulse.shape[0])/tNormFact
    
    # Finding starting location of waveform
    agPulse_dt = np.diff(agPulse)*freq
    if np.nanmin(agPulse_dt[:leftHalfMax_ind+1]) > 1:
        # If all values before fwhm point have slopes greater than 1, use first non-NaN point
        startInd = (np.isnan(agPulse)*1).argmin()
    else:
        # Otherwise use first point where slope is greater than 1
        startInd = leftHalfMax_ind - np.nanargmax(np.flip(agPulse_dt[:leftHalfMax_ind+1]) < 1) + 1
    
    # Automatic Peak Finding
    peaksAuto_ind = find_peaks(agPulse[leftHalfMax_ind:], distance=agPulse[leftHalfMax_ind:].shape[0]/10/2)
    peaksAuto_ind = peaksAuto_ind[0]+leftHalfMax_ind
    peaksAuto_ind = np.array([x for x in peaksAuto_ind if x <= 0.7*freq]) # drops any peaks found after 3 major peaks
    if len(peaksAuto_ind) == 0:
        peaksAuto_ind = np.array([0])
    
    # Select either automatic peak based on cutoffs or manual peak if provided
    for peakInd in range(3):
        if peaksManual_loc[peakInd] == 0:
            for autoPeakInd in range(len(peaksAuto_ind)):
                if peaksAuto_ind[autoPeakInd]/freq > peakCutoffs[0,peakInd] and peaksAuto_ind[autoPeakInd]/freq <= peakCutoffs[1,peakInd]:
                    peaksUsed_loc[peakInd] = peaksAuto_ind[autoPeakInd]/freq
    peaksManual_ind = np.round(np.array(peaksManual_loc)*freq).astype(int)
    peaksManual_loc = peaksManual_loc
    peaksManual_val = agPulse[peaksManual_ind]
    peaksAuto_loc = np.array(peaksAuto_ind)/tNormFact
    peaksAuto_val = agPulse[peaksAuto_ind]
    peaksUsed_ind = np.round(np.array(peaksUsed_loc)*freq).astype(int)
    peaksUsed_val = agPulse[peaksUsed_ind]
    
    if ~np.all(peaksUsed_ind): # checking for any zeros
        print('WARNING, peak not manually or automatically detected')
        print('Zero(s) found at ',np.where(peaksUsed_ind == 0)[0],' peak indexes (0-2)')
        
        for peakInd in range(3):
            if peaksUsed_ind[peakInd] == 0:
                peaksUsed_ind[peakInd] = np.mean(peakCutoffs[:,peakInd])*freq
        # peaksUsed_val = np.nan_to_num(peaksUsed_val)
        # peaksUsed_val[peaksUsed_val == 0] = np.nan
    
    trofsAuto_loc = np.zeros(2)
    for trofInd in range(2):
        trofsAuto_loc[trofInd] = (peaksUsed_ind[trofInd]+np.argmin(agPulse[peaksUsed_ind[trofInd]:peaksUsed_ind[trofInd+1]]))/freq
        if trofsUsed_loc[trofInd] == 0:
            trofsUsed_loc[trofInd] = (peaksUsed_ind[trofInd]+np.argmin(agPulse[peaksUsed_ind[trofInd]:peaksUsed_ind[trofInd+1]]))/freq
    trofsManual_ind = np.round(np.array(trofsManual_loc)*freq).astype(int)
    trofsManual_val = agPulse[trofsManual_ind]
    trofsAuto_ind = np.round(np.array(trofsAuto_loc)*freq).astype(int)
    trofsAuto_val = agPulse[trofsAuto_ind]
    trofsUsed_ind = np.round(np.array(trofsUsed_loc)*freq).astype(int)
    trofsUsed_val = agPulse[trofsUsed_ind]
    
    ## Plot Individual pulse that make up golden pulse & automatic peak picking
    if pulses_parts != []:
        for pulInd in range(len(pulses_parts)):
            if freq == []:
                tPart = np.arange(len(pulses_parts[pulInd]))/len(pulses_parts[pulInd])
            else:
                tPart = np.arange(len(pulses_parts[pulInd]))/freq
            ax.plot(tPart,pulses_parts[pulInd],linewidth=1,alpha=0.25,zorder=1)
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(0, np.ceil(xmax*10)/10, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid()
    
    # Plots AgPulse and final peaks used
    ax.plot(t,agPulse,'k',linewidth=1,zorder=2.5)
    ax.scatter(peaksUsed_loc,peaksUsed_val,marker='v',s=10,c='k',zorder=2.5)
    ax.scatter(trofsUsed_loc,trofsUsed_val,marker='^',s=10,color='k',zorder=2.5)
    ax.scatter(peaksManual_loc,peaksManual_val+0.005,marker='v',s=20,c='r',zorder=2.4)
    ax.scatter(trofsManual_loc,trofsManual_val-0.005,marker='^',s=20,c='r',zorder=2.4)
    ax.scatter(peaksAuto_loc,peaksAuto_val+0.005,marker='v',s=20,c='aqua',zorder=2.4)
    ax.scatter(trofsAuto_loc,trofsAuto_val-0.005,marker='^',s=20,c='aqua',zorder=2.4)
    ax.scatter(t[startInd],agPulse[startInd]-0.005,marker='^',s=10,c='lime',zorder=2.4)
    
    #ax.set_xlim(-0.025,1.025)
    ax.set_ylim(-0.025,1.025)
    ax.set_xlabel(xlabel,fontsize=16)
    ax.set_ylabel('Normalized Waveform',fontsize=16)
    # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    # ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title('Silver Pulse: ' + plotTitle + '\nAuto Peaks: ' + str(peaksAuto_loc*1000) + '\nUsed Peaks: ' + str(np.array(peaksUsed_loc)*1000) + 
                                                '\nAuto Troughs: ' + str(trofsAuto_loc*1000) + '\nUsed Troughs: ' + str(np.array(trofsUsed_loc)*1000) + 
                                                '\nStart Time: ' + str(t[startInd]*1000))
    saveDir = '/Users/brad/Desktop/gen2 results/WaveformPlots/'+plotTitle[0:6]+'/silverPulses/'
    saveDir = '/Users/brad/Desktop/silverPulses/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    #fig.savefig(saveDir+plotTitle+'_AgPulse.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    print(plotTitle)
    # print('peaks_loc(x10): ',peaks_loc*10)
    
    return peaksAuto_loc, peaksAuto_val, np.array(peaksUsed_loc), np.array(peaksUsed_val), np.array(trofsUsed_loc), t[startInd]

def alignLeftFwhm(dataIn,pulAlign,freq):
    # pulAlign: index to align pulses to
    
    pulPadLen = freq*2
    dataInPadded = np.zeros(pulPadLen)*np.nan
    dataInPadded[:len(dataIn)] = dataIn
    
    results_full = signal.peak_widths(dataInPadded,[np.nanargmax(dataInPadded)],rel_height=0.5)
    # widths = results_full[0]
    # width_heights = results_full[1]
    # left_ips = results_full[2]
    # right_ips = results_full[3]
    pulsesVertNormShifted = np.roll(dataInPadded,int(pulAlign-np.round(results_full[2][0])),axis=0)
    
    return pulsesVertNormShifted

def plotAgPulseDouble(dataIn,plotTitle,peaksManual2_loc,pulses_parts2,pulsesVertNorm_parts_names2,freq,peakCutoffs):
    ### Plot Silver Waveform
    
    plotIndividualPulses = 1
    
    fig,ax = plt.subplots(ncols=2,figsize=(11.2,4.8))
    titlesLR = ['Baseline','Post Hold']
    
    for pltInd in range(2):
    
        agPulse = copy.deepcopy(dataIn[pltInd])
        pulses_parts = copy.deepcopy(pulses_parts2[pltInd])
        
        # Low Pass filter of the AgPulse (FAILS ON SUB. 14)
        # mask = np.isnan(agPulse)*1
        # leading_nans = mask.argmin()
        # agPulseTemp = agPulse[leading_nans:(leading_nans+np.sum((~np.isnan(agPulse))*1))]
        # soshp = signal.butter(2,15,'lp',fs=freq,output='sos')
        # agPulseTemp = signal.sosfilt(soshp, agPulseTemp-np.nanmean(agPulseTemp))+np.nanmean(agPulseTemp)
        # agPulse[leading_nans:(leading_nans+np.sum((~np.isnan(agPulse))*1))] = agPulseTemp
        
        peaksUsed_loc = peaksManual2_loc[pltInd][0:3]
        trofsManual_loc = peaksManual2_loc[pltInd][3:5]
        peaksManual_loc = peaksManual2_loc[pltInd][0:3]
        trofsUsed_loc = copy.deepcopy(trofsManual_loc)
    
        # Setting up parameters
        results_full = signal.peak_widths(agPulse, [np.nanargmax(agPulse[:int(0.6*freq)])], rel_height=0.5)
        # widths = results_full[0]
        # width_heights = results_full[1]
        # left_ips = results_full[2]
        # right_ips = results_full[3]
        leftHalfMax_ind = int(np.round(results_full[2][0]))
        if freq == []:
            tNormFact = agPulse.shape[0]
            xlabel = 'Normalize Time'
        else:
            tNormFact = freq
            xlabel = 'Time (s)'
        t = np.arange(agPulse.shape[0])/tNormFact
        
        # Finding starting location of waveform
        agPulse_dt = np.diff(agPulse)*freq
        if np.nanmin(agPulse_dt[:leftHalfMax_ind+1]) > 1:
            # If all values before fwhm point have slopes greater than 1, use first non-NaN point
            startInd = (np.isnan(agPulse)*1).argmin()
        else:
            # Otherwise use first point where slope is greater than 1
            startInd = leftHalfMax_ind - np.nanargmax(np.flip(agPulse_dt[:leftHalfMax_ind+1]) < 1) + 1
        
        # Automatic Peak Finding
        peaksAuto_ind = find_peaks(agPulse[leftHalfMax_ind:], distance=agPulse[leftHalfMax_ind:].shape[0]/10/2)
        peaksAuto_ind = peaksAuto_ind[0]+leftHalfMax_ind
        peaksAuto_ind = np.array([x for x in peaksAuto_ind if x <= 0.7*freq]) # drops any peaks found after 3 major peaks
        if len(peaksAuto_ind) == 0:
            peaksAuto_ind = np.array([0])
        
        # Select either automatic peak based on cutoffs or manual peak if provided
        for peakInd in range(3):
            if peaksManual_loc[peakInd] == 0:
                for autoPeakInd in range(len(peaksAuto_ind)):
                    if peaksAuto_ind[autoPeakInd]/freq > peakCutoffs[0,peakInd] and peaksAuto_ind[autoPeakInd]/freq <= peakCutoffs[1,peakInd]:
                        peaksUsed_loc[peakInd] = peaksAuto_ind[autoPeakInd]/freq
        peaksManual_ind = np.round(np.array(peaksManual_loc)*freq).astype(int)
        peaksManual_loc = peaksManual_loc
        peaksManual_val = agPulse[peaksManual_ind]
        peaksAuto_loc = np.array(peaksAuto_ind)/tNormFact
        peaksAuto_val = agPulse[peaksAuto_ind]
        peaksUsed_ind = np.round(np.array(peaksUsed_loc)*freq).astype(int)
        peaksUsed_val = agPulse[peaksUsed_ind]
        
        if ~np.all(peaksUsed_ind): # checking for any zeros
            print('WARNING, peak not manually or automatically detected')
            print('Zero(s) found at ',np.where(peaksUsed_ind == 0)[0],' peak indexes (0-2)')
        
            for peakInd in range(3):
                if peaksUsed_ind[peakInd] == 0:
                    peaksUsed_ind[peakInd] = np.mean(peakCutoffs[:,peakInd])*freq
        
        trofsAuto_loc = np.zeros(2)
        for trofInd in range(2):
            trofsAuto_loc[trofInd] = (peaksUsed_ind[trofInd]+np.argmin(agPulse[peaksUsed_ind[trofInd]:peaksUsed_ind[trofInd+1]]))/freq
            if trofsUsed_loc[trofInd] == 0:
                trofsUsed_loc[trofInd] = (peaksUsed_ind[trofInd]+np.argmin(agPulse[peaksUsed_ind[trofInd]:peaksUsed_ind[trofInd+1]]))/freq
        trofsManual_ind = np.round(np.array(trofsManual_loc)*freq).astype(int)
        trofsManual_val = agPulse[trofsManual_ind]
        trofsAuto_ind = np.round(np.array(trofsAuto_loc)*freq).astype(int)
        trofsAuto_val = agPulse[trofsAuto_ind]
        trofsUsed_ind = np.round(np.array(trofsUsed_loc)*freq).astype(int)
        trofsUsed_val = agPulse[trofsUsed_ind]
        
        ## Plot Individual pulse that make up golden pulse & automatic peak picking
        if plotIndividualPulses and pulses_parts != []:
            for pulInd in range(len(pulses_parts)):
                if freq == []:
                    tPart = np.arange(len(pulses_parts[pulInd]))/len(pulses_parts[pulInd])
                else:
                    tPart = np.arange(len(pulses_parts[pulInd]))/freq
                ax[pltInd].plot(tPart,pulses_parts[pulInd],linewidth=1,alpha=0.25,zorder=1)
        
        # Plots AgPulse and final peaks used
        peaksManual_val[np.array(peaksManual_loc) < 0.1] = np.nan
        trofsManual_val[np.array(trofsManual_loc) < 0.1] = np.nan
        ax[pltInd].plot(t,agPulse,'k',linewidth=1,zorder=2.5)
        ax[pltInd].scatter(peaksUsed_loc,peaksUsed_val+0.005,marker='v',s=10,c='k',zorder=2.5)
        ax[pltInd].scatter(trofsUsed_loc,trofsUsed_val-0.005,marker='^',s=10,color='k',zorder=2.5)
        ax[pltInd].scatter(peaksManual_loc,peaksManual_val+0.010,marker='v',s=20,c='r',zorder=2.4)
        ax[pltInd].scatter(trofsManual_loc,trofsManual_val-0.010,marker='^',s=20,c='r',zorder=2.4)
        ax[pltInd].scatter(peaksAuto_loc,peaksAuto_val+0.010,marker='v',s=20,c='aqua',zorder=2.4)
        ax[pltInd].scatter(trofsAuto_loc,trofsAuto_val-0.010,marker='^',s=20,c='aqua',zorder=2.4)
        ax[pltInd].scatter(t[startInd],agPulse[startInd]-0.005,marker='^',s=10,c='lime',zorder=2.4)
        
        xmin, xmax = ax[pltInd].get_xlim()
        ax[pltInd].set_xticks(np.arange(0, np.ceil(xmax*10)/10, 0.1))
        ax[pltInd].set_yticks(np.arange(0, 1.1, 0.1))
        ax[pltInd].grid()
        
        ax[pltInd].set_ylim(-0.025,1.025)
        ax[pltInd].set_xlabel(xlabel,fontsize=16)
        ax[pltInd].set_ylabel('Normalized Waveform',fontsize=16)
        # ax[pltInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        # ax[pltInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        # ax[pltInd].tick_params(axis='both', which='major', labelsize=16)
        
        ax[pltInd].set_title(titlesLR[pltInd] + '\nAuto Peaks (ms)): ' + str(peaksAuto_loc*1000) + '\nUsed Peaks (ms): ' + str(np.array(peaksUsed_loc)*1000) + 
                                                '\nAuto Troughs (ms): ' + str(trofsAuto_loc*1000) + '\nUsed Troughs (ms): ' + str(np.array(trofsUsed_loc)*1000) + 
                                                '\nStart Time: ' + str(t[startInd]*1000))
    
    # Legend for Chris/others
    markSize = 6
    legend_elements = [#Line2D([0],[0],color='k', lw=1, label='AgPulse'),
                       Line2D([0],[0],marker='v',color='w',label='Used Peak',markerfacecolor='k',markersize=markSize),
                       Line2D([0],[0],marker='^',color='w',label='Used Trough',markerfacecolor='k',markersize=markSize),
                       Line2D([0],[0],marker='v',color='w',label='Auto Peak',markerfacecolor='aqua', markersize=markSize),
                       Line2D([0],[0],marker='^',color='w',label='Auto Trough',markerfacecolor='aqua', markersize=markSize),
                       Line2D([0],[0],marker='v',color='w',label='Manual Peak',markerfacecolor='r', markersize=markSize),
                       Line2D([0],[0],marker='^',color='w',label='Manual Trough',markerfacecolor='r', markersize=markSize),
                       ]
    ax[1].legend(handles=legend_elements, loc='upper right', fontsize=6)
    # Legends listing all pulse indexes
    # ax[0].legend(pulsesVertNorm_parts_names2[0], loc='upper right', fontsize=6)
    # ax[1].legend(pulsesVertNorm_parts_names2[1], loc='upper right', fontsize=6)
    
    fig.suptitle(plotTitle[:11])   
    saveDir = '/Users/brad/Desktop/gen2 results/WaveformPlots/'+plotTitle[0:6]+'/silverPulses/'
    saveDir = '/Users/brad/Desktop/silverPulses/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    fig.savefig(saveDir+plotTitle+'_AgPulse.png',dpi=300,bbox_inches='tight')
    plt.close()
    
    print(plotTitle)
    # print('peaks_loc(x10): ',peaks_loc*10)
    
    return peaksAuto_loc, peaksAuto_val, np.array(peaksUsed_loc), np.array(peaksUsed_val), np.array(trofsUsed_loc), t[startInd]

def findPeaksTrofs(dataPeakFinding,dataIn,period,winWid):
    
    # Creating box car averaged version of the time series
    # dataMedFilt = median_filter(dataPeakFinding,size=winWid,mode='nearest')
    # dataConvValid = np.convolve(dataPeakFinding,np.ones(winWid)/winWid,'valid')
    dataCumsum = np.nancumsum(np.insert(dataPeakFinding, 0, 0)) 
    dataConvValid = (dataCumsum[winWid:] - dataCumsum[:-winWid]) / winWid
    # print(np.allclose(dataConvValid,dataCumsum)) # confirms equivalence to convolution
    
    # dataConv_peaks = find_peaks(dataConvValid, distance=int(period/2))[0]+int(np.floor(winWid/2))
    # dataConv_trofs = find_peaks(-dataConvValid, distance=int(period/2))[0]+int(np.floor(winWid/2))
    dataConv_peaks = find_peaks(dataConvValid, distance=int(period*0.7))[0]+int(np.floor(winWid/2))
    dataConv_trofs = find_peaks(-dataConvValid, distance=int(period*0.7))[0]+int(np.floor(winWid/2))
    
    pulsesTrofs_loc = []
    # if first is peak and min-point is less than first point of time series, use as start of pulse
    # Ex. s25 c1 m8
    test1 = np.nanmin(dataPeakFinding[:dataConv_peaks[0]]) - dataPeakFinding[0]
    if dataConv_peaks[0] < dataConv_trofs[0] and test1 < 0:
        pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[:dataConv_peaks[0]]))
    # Ex. s25 c1 m4
    if dataConv_trofs[0] < dataConv_peaks[0]:
        pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_trofs[0]:dataConv_peaks[0]])+dataConv_trofs[0])
    # Ex. s25 c1 m14 needs this
    # elif dataConv_peaks[0] < pulsesPeaks_loc[0]:
    #     pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_peaks[0]:pulsesPeaks_loc[0]])+dataConv_peaks[0])
    
    # Finds troughs between all peaks in dataConv_peaks
    for peakInd in range(len(dataConv_peaks)-1):
        nextTrofInd = np.nanargmin(dataPeakFinding[(dataConv_peaks[peakInd]+int(period*0)):dataConv_peaks[peakInd+1]])+dataConv_peaks[peakInd]
        if pulsesTrofs_loc == []:
            pulsesTrofs_loc.append(nextTrofInd)
        else:
            # if min is same as one of bounds
            if nextTrofInd - pulsesTrofs_loc[-1]+1 > 5:
                pulsesTrofs_loc.append(nextTrofInd)
            else:
                print(peakInd,pulsesTrofs_loc[-1])
        
    # if last trough is before last peak, trust that true trough exists at tend
    # May have to set theshold factor to less than 1.0 to ensure no dichrotic notches are detected
    # Ex. s18 c1 m0
    test3 = np.nanmin(dataPeakFinding[dataConv_peaks[-1]:]) < dataPeakFinding[dataConv_trofs[-1]]*1.0 and \
        dataPeakFinding[-1] - np.nanmin(dataPeakFinding[dataConv_peaks[-1]:]) > 0
    if dataConv_trofs[-1] < dataConv_peaks[-1] and test3:
        pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_peaks[-1]:])+dataConv_peaks[-1])
        # pulsesPeaks_loc.append(np.nanargmax(dataPeakFinding[dataConv_trofs[-1]:dataConv_peaks[-1]])+dataConv_trofs[-1])
    test2 = np.nanmax(dataPeakFinding[dataConv_trofs[-1]:]) > dataPeakFinding[dataConv_peaks[-1]]*0.75
    if dataConv_peaks[-1] < dataConv_trofs[-1] and test2:
        pulsesTrofs_loc.append(np.nanargmin(dataPeakFinding[dataConv_trofs[-1]:])+dataConv_trofs[-1])
    
    # Finds peaks between all troughs in dataIn (important to use dataIn because of differences in waveforms of channels)
    pulsesPeaks_loc = []
    for trofInd in range(len(pulsesTrofs_loc)-1):
        pulsesPeaks_loc.append(np.nanargmax(dataIn[pulsesTrofs_loc[trofInd]:pulsesTrofs_loc[trofInd+1]])+pulsesTrofs_loc[trofInd])
    
    pulsesTrofs_loc = np.array(pulsesTrofs_loc)
    pulsesPeaks_loc = np.array(pulsesPeaks_loc)
    
    return pulsesTrofs_loc,pulsesPeaks_loc,dataConv_trofs,dataConv_peaks

def resampleData(dataIn,freqIn,freqOut,normLength=[]):
    #%%
    # dataIn = meanRaw[:,camInd,modInd]
    # freqIn = 40
    # freqOut = 125
    # normLength = []
    
    tIn = np.round(np.linspace(0,dataIn.shape[0]/freqIn,num=dataIn.shape[0], endpoint=False),7)
    if normLength == []:
        normLength = int(np.round(dataIn.shape[0]/freqIn*freqOut))
        tOut = np.round(np.linspace(0,(normLength-1)/freqOut,num=normLength, endpoint=True),7)
    else:
        tOut = np.round(np.linspace(0,(dataIn.shape[0]-1)/freqIn,num=normLength, endpoint=True),7)
    
    f = interp1d(tIn,dataIn,kind='linear',fill_value='extrapolate') # linear zero slinear quadratic cubic
    dataOut = f(tOut)
    
    # fig,ax = plt.subplots()
    # ax.plot(tIn,dataIn)
    # ax.plot(tOut,dataOut)
    # ax.legend(['dataIn','dataOut'])
    #%%
    
    return dataOut, tOut

def linRegAndCI(x, y):
    # https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
    
    # x = np.linspace(0,10)
    # y = 3*np.random.randn(50) + x
    # slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment
    x = np.array(x)
    y = np.array(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    y_model = np.polyval([slope, intercept], x)   # modeling...
    n = x.shape[0] # number of samples
    m = 2 # number of parameters
    dof = n - m # degrees of freedom
    t = stats.t.ppf(0.975, dof) # Students statistic of interval confidence
    residual = y - y_model
    std_err = (np.sum(residual**2) / (dof))**0.5 # Standard deviation of the error
    x_mean = np.mean(x)
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = np.polyval([slope, intercept], x_line)
    ci = t * std_err * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**0.5
    
    # Validated against this method to give same 95% CI
    # import statsmodels.api as sm
    # from statsmodels.stats.outliers_influence import summary_table
    # X = sm.add_constant(x)
    # res = sm.OLS(y, X).fit()
    # st, data, ss2 = summary_table(res, alpha=0.05)
    # fittedvalues = data[:,2]
    # predict_mean_se  = data[:,3]
    # predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    # predict_ci_low, predict_ci_upp = data[:,6:8].T
    # see also https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
    
    return slope, intercept, x_line, y_line, r_value, ci

def FindLowNoisePulseAndAverage(y, period, loc, dt):
    '''
        Runs pulse detection and crops, then fits a spline to the pulses in each camera to get the exemplar pulse in each
        camera

        Parameters
        ----------
        y : 2D array
            First dimension is the time the second is the number of channels/cameras

        period : float
            Period(in number of samples) of pulse detected in the signal

        
        Returns
        -------
        fewPulseCount: int
            Count where fewer than three pulses are detected in a channel

        noPulseCount: int
            Count where no pulse is detected in a channel

        fits: list of arrays
            Coefficients for the spline fit
    '''
    noPulseCount  = 0
    fewPulseCount = 0
    ### separate periods ###
    ##Take away all the higher frequency components from the signal to help find the start of systolic
    bpm = 60/dt/period
    cut   = np.median(bpm)/60
    soslp = signal.butter(2,cut,'lp',fs=1/dt,output='sos')
    m = np.mean(y, axis=0)
    ySmoothed = signal.sosfilt(soslp,y-m,axis=0)
    minPeriod = int(np.floor(period))-int(period*0.15) #allow for a 15% variation decrease in pulse width
    chPeaksSm = [signal.find_peaks(ySmoothed[:,ch], distance=minPeriod)[0] for ch in range(ySmoothed.shape[1])]

    #Make the arrays the same size so that we can use numpy tools on the array
    lenPeaks  = max(map(len,chPeaksSm))
    for tmpInd1, pkList in enumerate(chPeaksSm):
        if len(pkList) == lenPeaks:
            mxLenPeakInd  = tmpInd1
            break
    for tmpInd1 in range(len(chPeaksSm)):
        peaks = chPeaksSm[tmpInd1]
        if len(peaks) < lenPeaks:
            #Append to the correct side - when we have even number of cameras, median can split
            #the difference between the lists and yeild incorrect waveforms
            if np.sum(np.abs(peaks-chPeaksSm[mxLenPeakInd][:len(peaks)])) < \
                np.sum(np.abs(peaks-chPeaksSm[mxLenPeakInd][-len(peaks):])):
                for _ in range(lenPeaks-len(peaks)):
                    peaks = np.append(peaks.astype(float),np.nan)
            else:
                for _ in range(lenPeaks-len(peaks)):
                    peaks = np.insert(peaks.astype(float),0,np.nan)
            chPeaksSm[tmpInd1] = peaks
    
    #Heartbeat should be consistent across cameras
    hbPeaks = np.nanmedian(np.asarray(chPeaksSm),axis=0)
    hbPeaks = hbPeaks.astype(int)
    
    #We search for the systolic peak around the start of the 
    yPrime = np.zeros(y.shape)
    for ind in range(y.shape[1]):
        yPrime[:,ind] = np.gradient(y[:,ind])
    #Primary method to find pulse
    probSystolic1 = [ signal.find_peaks(yPrime[:,yP]**2,distance=minPeriod)[0] for yP in range(yPrime.shape[1]) ]
    #Backup method to find pulse
    probSystolic2 = [ signal.find_peaks(yPrime[:,yP]**2)[0] for yP in range(yPrime.shape[1]) ]
    segStarts = np.zeros((ySmoothed.shape[1],len(hbPeaks)),dtype='i')
    for ind in range(len(hbPeaks)):
        inds1 = probSystolic1 - hbPeaks[ind]
        inds2 = probSystolic2 - hbPeaks[ind]
        for ch,chInds1 in enumerate(inds1):
            chInds1[chInds1>0] = -ySmoothed.shape[0]
            if np.abs(np.amax(chInds1))<5:
                segStarts[ch,ind] = probSystolic1[ch][np.argmax(chInds1)]
            else:
                chInds2 = copy.deepcopy(inds2[ch])
                chInds2[chInds2>0] = -ySmoothed.shape[0]
                segStarts[ch,ind] = probSystolic2[ch][np.argmax(chInds2)]

    segStarts -=1 #We want the index where the pulse starts
    chPeaks   = [ segSt for segSt in segStarts ]

    chLengths = [peaks[1:]-peaks[:-1] for peaks in chPeaks]

    ### get segments only pick ones that are close < +/-2 of approximation
    chLengthsLst = [l.tolist() for l in chLengths]
    chLengthsLst = sum(chLengthsLst, [])
    maxLength = np.ceil(np.median(chLengthsLst)+1.5)
    minLength = np.floor(np.median(chLengthsLst)-1.5)

    L = int(maxLength+1)

    chUseSegment = [ ( (minLength<=np.array(lengths)).astype(int) + (maxLength>=np.array(lengths)).astype(int) ) == 2
                    for lengths in chLengths ]

    goldenPulse  = [{}] * len(chUseSegment)
    pulseSegments = [{}] * len(chUseSegment)
    pulseDeteced = np.empty((len(chUseSegment)), dtype=bool)
    pulseDeteced.fill(1)

    fits = [[]] * len(chUseSegment)
    for ch in range(len(chUseSegment)):
        nSegments =int(np.sum(chUseSegment[ch]))
        if nSegments == 0:
            goldenPulse[ch] = np.nan*np.zeros((10,))
            pulseDeteced[ch] = False
            noPulseCount += 1
            continue
        segments = np.zeros((L, nSegments))
        if nSegments<3:
            fewPulseCount += 1
        ind=0
        for j in range(len(chLengths[ch])):
            if not chUseSegment[ch][j]:
                continue
            curSeg = y[chPeaks[ch][j]:chPeaks[ch][j+1]+1, ch]
            if curSeg[0]<curSeg[1]: #We might be off by 1 index from systolic start
                curSeg = np.roll(curSeg, -1)
                chPeaks[ch][j] += 1
            segments[:chLengths[ch][j]+1, ind] = curSeg
            ind += 1
        segments[segments == 0] = np.nan
        
        numNan  = np.sum(np.isnan(segments),axis=1)
        delInd0 = np.where(numNan>3)#np.where(numNan>int((segments.shape[1]+0.5)/2))
        #Remove segments that are too long provided there are enough segments to sample
        if delInd0[0].shape[0]>1:
            numNan  = np.sum(~np.isnan(segments[delInd0]),axis=0)
            delInd1 = np.where(numNan>=2)
            segments = np.delete(segments,delInd1,1)
        segments = np.delete(segments,delInd0,0)

        #xArr = np.indices((segments.shape))[0].flatten()
        #segsFlat = segments.flatten()
        #xArr = xArr[~np.isnan(segsFlat)]
        #segsFlat = segsFlat[~np.isnan(segsFlat)]
        #timePoints = np.arange(segments.shape[0])

        if 1:#segments.shape[1]<=2:
            fitOrAvgSegment = np.nanmean(segments-np.nanmean(segments,axis=0),axis=1)+np.nanmean(segments)
        #else:
        #    transformed_x2 = dmatrix("cr(xArr, df=16)",
        #                            {"xArr": xArr}, return_type='matrix')
        #    fit2 = sm.RLM(segsFlat, transformed_x2).fit()
        #    fitOrAvgSegment = fit2.predict(dmatrix("cr(timePoints, df=16)",
        #                            {"timePoints": timePoints}, return_type='matrix'))
        #    fits[ch] = fit2
        goldenPulse[ch] = fitOrAvgSegment
        pulseSegments[ch] = segments-np.nanmean(segments,axis=0)+np.nanmean(segments)
        '''if ch==0:
            plt.plot(y,label=('C1','C2','C3','C4'));plt.title('Location '+str(loc+1)); plt.legend(); plt.show()
        if ch==0 or ch==3:
            plt.plot(segments);plt.title('Location '+str(loc+1)+' Channel '+str(ch));plt.plot(fitOrAvgSegment,color='k'); plt.show()'''
    return chPeaks, chUseSegment, goldenPulse, pulseSegments

def genScanNames(mypath,keys,layer):
    # mypath = string directory at patient level or one above, if left blank will pop up dialog
    # keys = list of 1 to 3 strings to look for, ['FULLSCAN','TESTSCAN','LONGSCAN']
    # layer = 1 single patient folder, 2 folder of patients, 3 folder of folders of patients
    
    if mypath == []:
        mypath = askdirectory(title='Select folder', initialdir='/Users/brad/Desktop/gen2 data/')
    if mypath[-1] != '/':
        mypath = mypath + '/'
        
    if layer == 1:
        path = os.path.normpath(mypath)
        # path.split(os.sep)[-1]
        
        scanNames = []
        foldNames = natural_sort(os.listdir(mypath))
        foldNames = foldNames[:-1]
        for fldInd in range(len(foldNames)):
            if os.path.isdir(mypath + '/' + foldNames[fldInd]):
                test = foldNames[fldInd]
                for keyInd in range(len(keys)):
                    if bool(re.findall(keys[keyInd], test)):
                        scanNames.append(path.split(os.sep)[-1] + '/' + test)
    
    if layer == 2:
        scanNames = []
        foldNames = natural_sort(os.listdir(mypath))
        foldNames = foldNames[:-1]
        for fldInd in range(len(foldNames)):
            if os.path.isdir(mypath + '/' + foldNames[fldInd]):
                fnames = natural_sort(os.listdir(mypath + '/' + foldNames[fldInd]))
                for filInd in range(len(fnames)):
                    test = fnames[filInd]
                    for keyInd in range(len(keys)):
                        if bool(re.findall(keys[keyInd], test)):
                            scanNames.append(foldNames[fldInd] + '/' + test)
    elif layer == 3:
        scanNames = []
        supfoldNames = natural_sort(os.listdir(mypath))
        supfoldNames = supfoldNames[:-1]
        for supfldInd in range(len(supfoldNames)):
            if os.path.isdir(mypath + supfoldNames[supfldInd]):
                foldNames = natural_sort(os.listdir(mypath + supfoldNames[supfldInd]))
                foldNames = foldNames[:-1]
                for fldInd in range(len(foldNames)):
                    if os.path.isdir(mypath + supfoldNames[supfldInd] + '/' + foldNames[fldInd]):
                        fnames = natural_sort(os.listdir(mypath + supfoldNames[supfldInd] + '/' + foldNames[fldInd]))
                        for filInd in range(len(fnames)):
                            test = fnames[filInd]
                            for keyInd in range(len(keys)):
                                if bool(re.findall(keys[keyInd], test)):
                                    scanNames.append(supfoldNames[supfldInd] + '/' + foldNames[fldInd] + '/' + test)
    print(scanNames)
    return scanNames

def calcTimeDelay(dataTCD,dataOptical,rng,plotAll):
    # assumes TCD and optical data are at same sampling rate
    
    # dataTCD = copy.deepcopy(allRaw_data[i][3]) # TCD data
    # dataOptical = copy.deepcopy(allRaw_data[i][10][:,0]) # Optical data
    
    freqTCD = 125
    # rng = 2000 # optical time points
    nans1 = np.zeros(rng)*np.nan
    errs = np.zeros(rng*2+1)
    tErrs = np.arange(rng*2+1)-rng
    
    minDim = np.min([dataTCD.shape[0],dataOptical.shape[0]])
    dataTCD = dataTCD[:minDim]
    dataOptical = dataOptical[:minDim]
    data2Conc = np.concatenate((nans1,dataTCD,nans1),axis=0)
    
    for ptInd in range(0,rng*2+1):
        nans1 = np.zeros(ptInd)*np.nan
        #data = np.array(dataTCD[rng:-(rng)])
        nans2 = np.zeros(rng*2-ptInd)*np.nan
        data1Conc = np.concatenate((nans1,dataOptical,nans2),axis=0)
        # print(nans1.shape[0],nans2.shape[0],data.shape[0],dataOptical.shape[0])
        errs[ptInd] = np.nanmean((data1Conc-data2Conc)**2)**0.5
    fig, ax = plt.subplots()
    ax.plot(tErrs,errs)
    ax.set_xlabel('Time Point Offset of TCD relative to Optical')
    ax.set_ylabel('RMSE')
    timeDelay = np.argmin(errs)-rng
    ax.set_title('Min. Location: ' + str(np.argmin(errs)-rng) + ' TCD points: ' + str(timeDelay))
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(2*4.8,2*4.8)) # 6.4,4.8 default
    fig.tight_layout(h_pad=1.5)
    
    t = np.arange(dataTCD.shape[0])
    ax[0,0].plot(t,dataTCD)
    ax[0,0].plot(t,dataOptical)
    ax[0,0].legend(['TCD','Optical'])
    ax[0,0].set_xlim(0,1250)
    
    ax[0,1].plot(t,dataTCD)
    ax[0,1].plot(t,dataOptical)
    ax[0,1].legend(['TCD','Optical'])
    ax[0,1].set_xlim(minDim-1250,minDim)
    
    t = np.arange(dataTCD.shape[0])
    ax[1,0].plot(t-timeDelay,dataTCD)
    ax[1,0].plot(t,dataOptical)
    ax[1,0].legend(['TCD','Optical'])
    ax[1,0].set_title('TCD points Offset: ' + str(round(timeDelay)))
    ax[1,0].set_xlim(0,1250)
    
    ax[1,1].plot(t-timeDelay,dataTCD)
    ax[1,1].plot(t,dataOptical)
    ax[1,1].legend(['TCD','Optical'])
    ax[1,1].set_title('TCD points Offset: ' + str(round(timeDelay)))
    ax[1,1].set_xlim(minDim-1250,minDim)
    
    if plotAll:
        for ind in range(6):
            fig, ax = plt.subplots(nrows=2, ncols=1)
            fig.tight_layout(h_pad=1.5)
            
            t = np.arange(dataTCD.shape[0])/freqTCD
            ax[0].plot(t,dataTCD)
            ax[0].plot(t,dataOptical)
            ax[0].legend(['TCD','Optical'])
            ax[0].set_ylabel('HR (bpm)')
            ax[0].set_xlim(ind*40,ind*40+40)
            ax[0].set_title('TCD point Offset: ' + str(0))
            
            ax[1].plot(t-timeDelay/freqTCD,dataTCD)
            ax[1].plot(t,dataOptical)
            ax[1].legend(['TCD','Optical'])
            ax[1].set_ylabel('HR (bpm)')
            ax[1].set_xlim(ind*40,ind*40+40)
            ax[1].set_title('TCD point Offset: ' + str(round(timeDelay)))
    
    return timeDelay

def loadTCDdata(tcdDir,TCDname,TCDmarks):
    
    # file = open('tcdDir + '/' + TCDname)
    # csvreader = csv.reader(file)
    # header = []
    # header = next(csvreader)
    # rows = []
    # for row in csvreader:
    #     rows.append(row)
    # file.close()
    # rows = np.array(rows)
    
    rows = pd.read_csv(tcdDir + '/' + TCDname)
    rows = np.array(rows)
    
    # if one of the weird data sets that are tab delimited, reload
    if rows.shape[1] == 1: 
        rows = pd.read_csv(tcdDir + '/' + TCDname, sep='\t')
        rows = np.array(rows)
    
    # Handles data missing Avg and PI
    if rows.shape[1] == 3:
        # time, 11 Env U, mark
        rows = np.concatenate((rows,np.zeros((rows.shape[0],15))),axis=1)
        rows[:,17] = rows[:,2]
        rows[:,2] = rows[:,1]
        rows[:,1] = 0
        test = rows[:,2]==''
        rows = rows[~test, :]
    elif rows.shape[1] == 5:
        rows = np.concatenate((rows,np.zeros((rows.shape[0],13))),axis=1)
        # time, 1-1 Mean U, 1-1 Env U, 1-1 PI U, mark
        rows[:,17] = rows[:,4]  # mark
        rows[:,16] = rows[:,3]  # PI
        # rows[:,7] = rows[:,0] # RespRate
        # rows[:,6] = rows[:,0] # etCO2
        rows[:,4] = ''
        rows[:,3] = 0
        rows[:,2] = rows[:,2]   # env
        rows[:,1] = rows[:,1]   # mean
    
    marks = np.array(TCDmarks)
    marksStart = marks[0]
    # nan-pads time if it starts before optical
    if marksStart < 0:
        spacer = np.zeros((-marksStart,rows.shape[1]))
        spacer[:,:] = np.nan
        timespace = np.arange(-marksStart)/125
        timespace = timespace - timespace[-1]
        rows = np.concatenate((spacer,rows),axis=0)
        marks = marks - marks[0]
    meanTCD = rows[marks[0]:marks[-1],1].astype(np.float)
    envTCDraw = rows[marks[0]:marks[-1],2].astype(np.float)
    envTCD = rows[marks[0]:marks[-1],2].astype(np.float)
    etCO2 = rows[marks[0]:marks[-1],6].astype(np.float)
    respRate = rows[marks[0]:marks[-1],7].astype(np.float)
    pulIndTCD = rows[marks[0]:marks[-1],16].astype(np.float)
    
    # Converting any value of the evelope 1 or less to a linearly interpolated value
    dataIn = copy.deepcopy(envTCD) # Variable to be fixed (in this case envTCD)
    zeroCount = dataIn[dataIn <= 1].shape[0]
    if zeroCount > 0:
        print('WARNING: ' + str(zeroCount) + ' TCD 0-points linearly interpolated')
        tIn = np.arange(dataIn.shape[0])
        tOut = np.arange(dataIn.shape[0])
        dataOut = copy.deepcopy(dataIn)
        zerosInds = dataOut <= 1
        tIn = tIn[~zerosInds]
        dataOut = dataOut[~zerosInds]
        f = interp1d(tIn,dataOut,kind='linear',fill_value="extrapolate")
        dataOut = f(tOut)
        envTCD = copy.deepcopy(dataOut)
        
        fig, ax = plt.subplots()
        ax.plot(tOut,dataIn,linewidth=0.25)
        ax.scatter(tOut[zerosInds],dataOut[zerosInds],s=np.ones(tOut[zerosInds].shape[0])*0.5,c='red')
        ax.set_xlim(tOut[zerosInds].min()-10,tOut[zerosInds].max()+10)
        ax.set_title(TCDname + '\nPoints Linearly Interpolated: ' + str(zeroCount))
        ax.legend(['Raw data','Linearly Interpolated Fixes'])
        # fig.savefig('/Users/brad/Desktop/' + TCDname + '_missingData.png', dpi=600)
    
    timeTCD = np.zeros((rows.shape[0]))
    for rowInd in range(rows.shape[0]):
        if str(rows[rowInd,0]) == 'nan':
            timeTCD[rowInd] = timespace[rowInd] + float(rows[-marksStart,0][3:5])*60+float(rows[-marksStart,0][6:8]+'.'+rows[-marksStart,0][9:]) - 1/125
        elif str(rows[rowInd,0]).replace('.', '').isnumeric():
             timeTCD[rowInd] = float(rows[rowInd,0])
        else:
            timeTCD[rowInd] = float(rows[rowInd,0][3:5])*60+float(rows[rowInd,0][6:8]+'.'+rows[rowInd,0][9:])
        # print([rowInd,rows[rowInd,0],timeTCD[rowInd]])
    if np.diff(timeTCD).min() < 0:
        timeTCD[(np.diff(timeTCD).argmin()+1):] = timeTCD[(np.diff(timeTCD).argmin()+1):] + timeTCD[np.diff(timeTCD).argmin()]
    timeTCD = timeTCD - timeTCD[marks[0]]
    timeTCD = timeTCD[marks[0]:marks[-1]]
    marks = marks - marks[0]
    respRate[marks[1:-1]] = np.nanmax(respRate)*1.25+1
    respRate[marks[1:-1]+1] = 0
    
    return timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw

def longScanPlot(contMerged,meanMerged,labels,freq,savename,title):
    fig, ax = plt.subplots(nrows=4, ncols=1)
    t = np.arange(contMerged.shape[0])/freq
    ax[0].plot(t,contMerged[:,1],linewidth=0.5)
    ax[0].set_ylabel(labels[0] + '\n(Left)')
    #ax[0].legend(['Left'],loc='lower left')
    ax[1].plot(t,meanMerged[:,1],'r',linewidth=0.5)
    ax[1].set_ylabel(labels[1] + '\n(Left)')
    #ax[1].legend(['Left'],loc='lower left')
    ax[2].plot(t,contMerged[:,0],linewidth=0.5)
    ax[2].set_ylabel(labels[0] + '\n(Right)')
    #ax[2].legend(['Right'],loc='lower left')
    ax[3].plot(t,meanMerged[:,0],'r',linewidth=0.5)
    ax[3].set_ylabel(labels[1] + '\n(Right)')
    ax[3].set_xlabel('Time (s)')
    #ax[3].legend(['Right'],loc='lower left')
    for i in range(0,3):
        ax[i].set_xticks([])
        ax[i].set_xlim([t.min(),t.max()])
    ax[3].set_xlim([t.min(),t.max()])
    fig.suptitle(title)
    fig.savefig(savename + '.png',dpi=300,bbox_inches='tight')

def longScanPlotTCDside(rBFIMerged,rBVIMerged,rBfiAvgMerged,rBviAvgMerged,rBfiAmpMerged,rBviAmpMerged,hRateMerged,labels,freq,savename,title,TCDname,TCDmarks,camInd): 
    # plots all optical and TCD outputs for one side of the head
    # camInd: 0=right, 1=left
    
    fig, ax = plt.subplots(nrows=7, ncols=1, gridspec_kw={'height_ratios': [1,1,1,1,1,1,1]})
    t = np.arange(rBFIMerged.shape[0])/freq
    ax[0].plot(t,rBFIMerged[:,camInd],'b',linewidth=0.5)
    ax[0].set_ylabel('rBFI')
    ax[0].tick_params(axis='y', colors='b')
    
    ax[1].plot(t,rBfiAvgMerged[:,camInd],'b',linewidth=0.5)
    ax[1].set_ylabel('Avg.')
    ax[1].tick_params(axis='y', colors='b')
    ax2 = ax[1].twinx()
    ax2.plot(t,rBfiAmpMerged[:,camInd],'k',linewidth=0.5)
    ax2.set_ylabel('Amp.')
    
    ax[2].plot(t,rBVIMerged[:,camInd],'r',linewidth=0.5)
    ax[2].set_ylabel('rBVI')
    ax[2].tick_params(axis='y', colors='r')
    
    ax[3].plot(t,rBviAvgMerged[:,camInd],'r',linewidth=0.5)
    ax[3].set_ylabel('Avg.')
    ax[3].tick_params(axis='y', colors='r')
    ax2 = ax[3].twinx()
    ax2.plot(t,rBviAmpMerged[:,camInd],'k',linewidth=0.5)
    ax2.set_ylabel('Amp.')
    
    freqTCD = 125
    timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = loadTCDdata(TCDname,TCDmarks)
    meanTCDow, ampTCDow, periodTCDow, medTCDow = getMetrics(envTCD,freqTCD,TCDname)
    
    ax[4].plot(timeTCD,envTCD,'g',linewidth=0.5)
    ax[4].set_ylabel('TCD\nEnv.')
    ax[4].tick_params(axis='y', colors='g')
    
    ax[5].plot(timeTCD,meanTCDow,'g',linewidth=0.5)
    ax[5].set_ylabel('Avg.')
    ax[5].tick_params(axis='y', colors='g')
    ax2 = ax[5].twinx()
    ax2.plot(timeTCD,ampTCDow/meanTCDow,'k',linewidth=0.5)
    ax2.set_ylabel('PI')
    
    ax[6].plot(timeTCD,etCO2,'m',linewidth=0.5)    
    ax[6].set_ylabel('ETCO2\n(mmHg)')
    ax[6].set_xlabel('Time (s)')
    ax[6].tick_params(axis='y', colors='m')
    ax2 = ax[6].twinx()
    holder  = respRate/np.nanmax(respRate)*(np.nanmax(hRateMerged)-np.nanmin(hRateMerged))+np.nanmin(hRateMerged)
    ax2.plot(timeTCD,holder,'--k',linewidth=0.5,alpha=0.2)
    ax2.plot(t,hRateMerged[:,camInd],'k',linewidth=0.5)
    ax2.set_ylabel('HR &\nRR')
    
    # Plotting shading for breath holds
    freqTCD = 125
    freqOpt = 40
    
    if len(TCDmarks) == 8:
        holds = [1,3,5]
    elif len(TCDmarks) == 6:
        holds = [1,3]
    elif len(TCDmarks) == 4:
        holds = [1]
    else:
        holds = [1]
        print('WARNING: ' + str(TCDmarks) + ' MARKS GIVEN, NOT 8, 6, OR 4')
    
    for holdInd in holds:
        TCDmarks = np.array(TCDmarks)
        TCDmarks = TCDmarks - TCDmarks[0]
        ptsTCD = [TCDmarks[holdInd],TCDmarks[holdInd+1]]
        print(ptsTCD)
        for axInd in range(0,4):
            ymin, ymax = ax[axInd].get_ylim()
            nx = np.array([[t[ptsTCD[0]],ymin],[t[ptsTCD[0]],ymax],[t[ptsTCD[1]],ymax],[t[ptsTCD[1]],ymin]])
            ax[axInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='r'))
        for axInd in range(4,7):
            ymin, ymax = ax[axInd].get_ylim()
            nx = np.array([[timeTCD[ptsTCD[0]],ymin],[timeTCD[ptsTCD[0]],ymax],[timeTCD[ptsTCD[1]],ymax],[timeTCD[ptsTCD[1]],ymin]])
            ax[axInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='r'))
        
        # displaying average windows
        # for axInd in range(0,4):
        #     ymin, ymax = ax[axInd].get_ylim()
        #     for ind in range(2):
        #         nx = np.array([[t[ptsOpt[ind]]-5,ymin],[t[ptsOpt[ind]]-5,ymax],[t[ptsOpt[ind]]+5,ymax],[t[ptsOpt[ind]]+5,ymin]])
        #         ax[axInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='b'))
        # for axInd in range(4,7):
        #     ymin, ymax = ax[axInd].get_ylim()
        #     for ind in range(2):
        #         nx = np.array([[timeTCD[ptsTCD[ind]]-5,ymin],[timeTCD[ptsTCD[ind]]-5,ymax],[timeTCD[ptsTCD[ind]]+5,ymax],[timeTCD[ptsTCD[ind]]+5,ymin]])
        #         ax[axInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='b'))
    
    for i in range(0,6):
        ax[i].set_xticks([])
        ax[i].set_xlim([t.min(),t.max()])
        #ax[i].set_xlim([203,210])
    ax[6].set_xlim([t.min(),t.max()])
    #ax[6].set_xlim([203,210])
    if camInd == 0:
        fig.suptitle(title + ' R')
        fig.savefig(savename + '_allRight.png',dpi=300,bbox_inches='tight')
    else:
        fig.suptitle(title + ' L')
        fig.savefig(savename + '_allLeft.png',dpi=300,bbox_inches='tight')

def longScanPlotTCDside2(rBFIMerged,rBVIMerged,rBfiAvgMerged,rBviAvgMerged,rBfiAmpMerged,rBviAmpMerged,hRateMerged,labels,freq,savename,title,TCDname,TCDmarks,camInd): 
#%%
    # plots all optical and TCD outputs for one side of the head
    # camInd: 0=right, 1=left
    
    # rBFI
    # rBVI
    # TCD
    # [rBFI avg, rBVI avg], TCD avg
    # rBFI amp, TCD PI
    
    fig, ax = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [1,1,1,1]},figsize=(4.8,4.8)) # 6.4,4.8
    t = np.arange(rBFIMerged.shape[0])/freq
    timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = loadTCDdata(TCDname,TCDmarks)

    ax[0].plot(t,rBFIMerged[:,camInd],'b',linewidth=0.5)
    ax[0].set_ylabel('rBFI', color='b')
    ax[0].tick_params(axis='y', colors='b')
    
    # ax[1].plot(t,rBVIMerged[:,camInd],'r',linewidth=0.5)
    # ax[1].set_ylabel('rBVI', color='r')
    # ax[1].tick_params(axis='y', colors='r')
    
    ax[1].plot(timeTCD,envTCD,'g',linewidth=0.5)
    ax[1].set_ylabel('TCD', color='g')
    ax[1].tick_params(axis='y', colors='g')
    
    # ax[3].plot(t,rBfiAvgMerged[:,camInd]*1000,'b',linewidth=0.5)
    # ax[3].plot(t,rBviAvgMerged[:,camInd],'r',linewidth=0.5)
    # ax[3].set_ylabel('Optical\nAverage')
    # ax[3].tick_params(axis='y', colors='k')
    # ax[3].legend(['rBFI (x1000)','rBVI'], fontsize=6)
    # ax2 = ax[3].twinx()
    # ax2.plot(timeTCD,meanTCD,'g',linewidth=0.5)
    # ax2.set_ylabel('TCD\nAverage', color='g')
    # ax2.tick_params(axis='y', colors='g')
    
    ax[2].plot(t,rBfiAvgMerged[:,camInd],'b',linewidth=0.5)
    ax[2].set_ylabel('rBFI\nAverage', color='b')
    ax[2].tick_params(axis='y', colors='b')
    ax2 = ax[2].twinx()
    ax2.plot(timeTCD,meanTCD,'g',linewidth=0.5)
    ax2.set_ylabel('TCD\nAverage', color='g')
    ax2.tick_params(axis='y', colors='g')
    
    ax[3].plot(t,rBfiAmpMerged[:,camInd],'b',linewidth=0.5)
    ax[3].set_ylabel('rBVI\nAmp.', color='b')
    ax[3].tick_params(axis='y', colors='b')
    ax2 = ax[3].twinx()
    ax2.plot(timeTCD,pulIndTCD,'g',linewidth=0.5)
    ax2.set_ylabel('TCD\nPulsatility\nIndex', color='g')
    ax2.tick_params(axis='y', colors='g')
  
    ax[3].set_xlabel('Time (s)')
    
    # Plotting shading for breath holds
    freqTCD = 125
    freqOpt = 40
    
    if len(TCDmarks) == 8:
        holds = [1,3,5]
    elif len(TCDmarks) == 6:
        holds = [1,3]
    elif len(TCDmarks) == 4:
        holds = [1]
    else:
        holds = [1]
        print('WARNING: ' + str(TCDmarks) + ' MARKS GIVEN, NOT 8, 6, OR 4')
    
    for holdInd in holds:
        TCDmarks = np.array(TCDmarks) - TCDmarks[0]
        ptsTCD = [TCDmarks[holdInd],TCDmarks[holdInd+1]]
        for axInd in range(0,4):
            ymin, ymax = ax[axInd].get_ylim()
            nx = np.array([[t[ptsTCD[0]],ymin],[t[ptsTCD[0]],ymax],[t[ptsTCD[1]],ymax],[t[ptsTCD[1]],ymin]])
            ax[axInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='r'))
    
    for i in range(0,3):
        ax[i].set_xticks([])
        # ax[i].set_xlim([t.min(),t.max()])
        ax[i].set_xlim([0,90])
        # ax[i].set_xlim([150,234.5])
    # ax[3].set_xlim([t.min(),t.max()])
    ax[3].set_xlim([0,90])
    # ax[3].set_xlim([150,234.5])
    # ax[2].set_ylim([-0.002,0.014])
    
    if camInd == 0:
        fig.suptitle(title + ' R')
        fig.savefig(savename + '_allRight2.png',dpi=300,bbox_inches='tight')
    else:
        fig.suptitle(title + ' L')
        fig.savefig(savename + '_allLeft2_crop1.png',dpi=300,bbox_inches='tight')

#%%

def longScanMergeResamp(dataIn,freqIn,freqOut,lsType):
    # dataIn expecting shape of [# of hist points (~600), # cams (2), # module scans (13-20)]
    
    # Resampling data if desired
    if freqOut != freqIn:
        dataNewFreq = np.zeros((np.round(dataIn.shape[0]*freqOut/freqIn).astype(int),dataIn.shape[1],dataIn.shape[2]))
        tIn = np.linspace(0,dataIn.shape[0]/freqIn,dataIn.shape[0],endpoint=False)
        tOut = np.linspace(0,dataIn.shape[0]/freqIn,dataNewFreq.shape[0],endpoint=False)
        for camInd in range(2):
            for modInd in range(dataIn.shape[2]):
                f = interp1d(tIn,dataIn[:,camInd,modInd],kind='linear',fill_value='extrapolate')
                dataNewFreq[:,camInd,modInd] = f(tOut)
    else:
        dataNewFreq = copy.deepcopy(dataIn)
                
    # plt.plot(tIn,dataIn[:,0,0])
    # plt.plot(tOut,dataNewFreq[:,0,0])
    # plt.xlim(0,2)
    # plt.show()
    
    # Creating nan blocks to fill in where sensor not on (empirically determined), plus fudge factor (ff)
    if lsType == 1:
        modTime = 18.00
        ff = 4
    elif lsType == 2:
        modTime = 15.64
        ff = 4
    elif lsType == 3:
        modTime = 15.714
        ff = 3
    else:
        modTime = 0
        ff = 0
    nanBlock = np.zeros((int(np.round((modTime-dataIn.shape[0]/freqIn)*freqOut))+ff,2))*np.nan
    
    # Merging data and nan blocks
    modStartTimes = np.zeros(dataIn.shape[1:3])
    dataMerged = copy.deepcopy(nanBlock)
    for modInd in range(dataNewFreq.shape[2]-1):
        modStartTimes[:,modInd] = dataMerged.shape[0]
        dataMerged = np.append(dataMerged, dataNewFreq[:,:,modInd], axis=0)
        dataMerged = np.append(dataMerged, nanBlock, axis=0)
    modInd += 1
    modStartTimes[:,modInd] = dataMerged.shape[0]
    dataMerged = np.append(dataMerged, dataNewFreq[:,:,modInd], axis=0)
    
    return dataMerged, modStartTimes
        
def TCDstats(dataTCD,dataOpt,ptsTCD,ptsOpt,freqTCD,freqOpt,avgWin,dataMinMax,slope):
    # dataMinMax is variable to determine time point of min & max values, assumed to be TCD variable!
    
    tOpt = np.arange(dataOpt.shape[0])/freqOpt
    
    # TCD data (resamples to optical data)
    if dataTCD != []:
        tTCD = np.arange(dataTCD.shape[0])/freqTCD
        
        f = interp1d(tTCD, dataTCD, fill_value='extrapolate')
        dataTCD_resam = f(tOpt)
        
        preWin_data = dataTCD_resam[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[0]+avgWin*freqOpt)]
        postWin_data = dataTCD_resam[(ptsOpt[1]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        holdWin_data = dataTCD_resam[ptsOpt[0]:ptsOpt[1]]
        preToPostWin_data = dataTCD_resam[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        
        preWin_data_time = tOpt[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[0]+avgWin*freqOpt)]
        postWin_data_time = tOpt[(ptsOpt[1]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        holdWin_data_time = tOpt[ptsOpt[0]:ptsOpt[1]]
        preToPostWin_data_time = tOpt[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        
        # plt.plot(tTCD,dataTCD)
        # plt.plot(tOpt[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)],preToPostWin_data)
        # plt.plot(tTCD[(ptsTCD[0]-avgWin*freqTCD):(ptsTCD[1]+avgWin*freqTCD)],dataTCD[(ptsTCD[0]-avgWin*freqTCD):(ptsTCD[1]+avgWin*freqTCD)])
        # plt.show()
        
    # Optical data (takes as-is)
    else:
        preWin_data = dataOpt[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[0]+avgWin*freqOpt)]
        postWin_data = dataOpt[(ptsOpt[1]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        holdWin_data = dataOpt[ptsOpt[0]:ptsOpt[1]]
        preToPostWin_data = dataOpt[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
        
    tTCD = np.arange(dataMinMax.shape[0])/freqTCD
    f = interp1d(tTCD, dataMinMax, fill_value='extrapolate')
    dataMinMax_resam = f(tOpt)
        
    preWin_dataMinMax = dataMinMax_resam[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[0]+avgWin*freqOpt)]
    postWin_dataMinMax = dataMinMax_resam[(ptsOpt[1]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
    holdWin_dataMinMax = dataMinMax_resam[ptsOpt[0]:ptsOpt[1]]
    preToPostWin_dataMinMax = dataMinMax_resam[(ptsOpt[0]-avgWin*freqOpt):(ptsOpt[1]+avgWin*freqOpt)]
    
    minLocTCD = np.nanargmin(preWin_dataMinMax)
    maxLocTCD = np.nanargmax(postWin_dataMinMax)
    
    if slope > 0:
        statsOut = [np.nanmean(preWin_data),preWin_data[minLocTCD],np.nanmin(preWin_data),np.nanmean(postWin_data),postWin_data[maxLocTCD],np.nanmax(postWin_data)]
    elif slope < 0:
        statsOut = [np.nanmean(preWin_data),preWin_data[minLocTCD],np.nanmax(preWin_data),np.nanmean(postWin_data),postWin_data[maxLocTCD],np.nanmin(postWin_data)]
        
    return holdWin_data, preToPostWin_data, statsOut

def TCDcrop(dataIn,ptsTCD,freqTCD,avgWin):
    # assumes all data is at freqTCD already
    baseWin = 30 # 30 seconds of baseline before breath hold
    baseline_data = dataIn[(ptsTCD[0]-baseWin*freqTCD):ptsTCD[0]]
    holdWin_data = dataIn[(ptsTCD[0]-avgWin*freqTCD):(ptsTCD[1]+avgWin*freqTCD)]
    
    return baseline_data, holdWin_data

def getTcdPeriod(x,freq,filename):
    
    # plt.plot(x)
    # plt.xlim(1000,1250)
    # plt.grid()
    
    if len(np.argwhere(np.isnan(x))) > 0:
        nanlocs = np.argwhere(np.isnan(x))
        fig, ax = plt.subplots()
        ax.plot(nanlocs, np.ones(nanlocs.shape),'o')
        ax.set_ylabel('1 = location of NaN')
        ax.set_xlabel('Time (TCD pts)')
        ax.set_title(filename + '\nWarning, ' + str(len(np.argwhere(np.isnan(x)))) + ' Nans removed for FFT calc.')
        x = x[~np.isnan(x)]
    
    # N=1024
    N = int(np.power(2,np.ceil(np.log2(x.shape[0]))))  # nearest power of 2 length for FFT
    N1 = int(N/256) # 4
    N2 = int(N/4) # 512
    m = np.nanmean(x, axis = 0)
    f = np.abs(np.fft.fft(x-m, n=N, axis=0))
    t = np.arange(N)
    freq = np.fft.fftfreq(t.shape[-1])*freq
    f[N2:] = 0
    f[:N1] = 0
    ind = np.argmax(f, axis=0)
    period = N/(ind+1)
    
    # plt.plot(f)
    # plt.plot(ind,f[ind],'ro',markersize=10)
    # plt.show()
    
    return period

def tcdPeriod(x,freq,filename):
    period = getTcdPeriod(x,freq,filename)
    minPeriod = int(np.floor(period*0.75))
    xpeaks = find_peaks(-x, distance=minPeriod)[0]
    return xpeaks

def getMetrics(data,freq,filename):
    
    xpeaks = tcdPeriod(data,freq,filename)
    dataAvg = copy.deepcopy(data)*np.nan
    dataMed = copy.deepcopy(data)*np.nan
    dataAmp = copy.deepcopy(data)*np.nan
    dataPeriod = copy.deepcopy(data)*np.nan
    for peakInd in range(len(xpeaks)-2):
        ind1 = xpeaks[peakInd]
        ind2 = xpeaks[peakInd+1]
        dataAvg[ind1:ind2] = data[ind1:(ind2-1)].mean()
        dataMed[ind1:ind2] = np.median(data[ind1:(ind2-1)])
        dataAmp[ind1:ind2] = data[ind1:(ind2-1)].max() - data[ind1:(ind2-1)].min()
        dataPeriod[ind1:ind2] = (ind2 - ind1)
        
    return dataAvg, dataAmp, dataPeriod, dataMed
    
def longScanTCDwaveform(rBFI,rBVI,hRate,labels,freq,savename,title,TCDname,TCDmarks):
    # plots zoomed in waveforms of both rBFI and TCD
    # leave TCD as empty inputs [] to just plot optical
    
    if rBFI.shape[2] == 13:
        modTime = 18
    elif rBFI.shape[2] == 15:
        modTime = 15.64
    else:
        modTime = 0
    freqTCD =  125
    
    if len(TCDname) > 0:
        timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = loadTCDdata(TCDname,TCDmarks)
        tLen = freqTCD*2
        tLen2 = freq*2
        t = np.arange(tLen)/freqTCD
        t2 = np.arange(tLen2)/freqTCD
        tStart = 0
        tInt = round(freqTCD*modTime)
        fig, ax = plt.subplots(nrows=3, ncols=rBFI.shape[2],figsize=(rBFI.shape[2],3))
        for timInd in range(rBFI.shape[2]):
            if t.shape == envTCD[(tStart+timInd*tInt):(tStart+timInd*tInt+tLen)].shape:
                ax[2,timInd].plot(timeTCD[(tStart+timInd*tInt):(tStart+timInd*tInt+tLen)],envTCD[(tStart+timInd*tInt):(tStart+timInd*tInt+tLen)],linewidth=0.5)
            ax[0,timInd].plot(t2,rBFI[0:tLen2,1,timInd],linewidth=0.5)
            ax[1,timInd].plot(t2,rBFI[0:tLen2,0,timInd],linewidth=0.5)
            ax[2,timInd].set_xlabel(str(timInd*tInt/freqTCD) + ' s')
            ax[0,timInd].set_xticks([])
            ax[0,timInd].set_yticks([])
            ax[1,timInd].set_xticks([])
            ax[1,timInd].set_yticks([])
            ax[2,timInd].set_xticks([])
            ax[2,timInd].set_yticks([])
        ax[0,0].set_ylabel('Optical\nLeft')
        ax[1,0].set_ylabel('Optical\nRight')
        ax[2,0].set_ylabel('TCD')
        fig.suptitle(title)
        fig.savefig(savename + 'WaveformsTCDoptical.png',dpi=300,bbox_inches='tight')
    else:
        tLen2 = freq*2
        fig, ax = plt.subplots(nrows=5, ncols=3)
        t = np.arange(rBFI.shape[0])/freq
        for scanInd in range(rBVI.shape[2]):
            ax[scanInd//3,scanInd%3].plot(t[0:tLen2],rBFI[0:tLen2,0,scanInd],linewidth=1)
            ax[scanInd//3,scanInd%3].plot(t[0:tLen2],rBFI[0:tLen2,1,scanInd],linewidth=1)
            ax[scanInd//3,scanInd%3].set_ylim(rBFI.min(),rBFI.max())
            if scanInd%3 > 0:
                ax[scanInd//3,scanInd%3].set_yticks([])
        for scanInd in range(rBVI.shape[2]):
            ax[scanInd//3,scanInd%3].set_yticks([])
        fig.suptitle(title)
        fig.savefig(savename + 'WaveformsOptical.png',dpi=300,bbox_inches='tight')

def longScanPlotTCD(contMerged,meanMerged,labels,freq,savename,title,TCDname,TCDmarks):
    # plots raw optical and raw TCD
    
    fig, ax = plt.subplots(nrows=6, ncols=1, gridspec_kw={'height_ratios': [1,1,1,1,2,1]})
    t = np.arange(contMerged.shape[0])/freq
    ax[0].plot(t,contMerged[:,1],linewidth=0.5)
    ax[0].set_ylabel(labels[0] + '\n(Left)')
    #ax[0].legend(['Left'],loc='lower left')
    ax[1].plot(t,meanMerged[:,1],'r',linewidth=0.5)
    ax[1].set_ylabel(labels[1] + '\n(Left)')
    #ax[1].legend(['Left'],loc='lower left')
    ax[2].plot(t,contMerged[:,0],linewidth=0.5)
    ax[2].set_ylabel(labels[0] + '\n(Right)')
    #ax[2].legend(['Right'],loc='lower left')
    ax[3].plot(t,meanMerged[:,0],'r',linewidth=0.5)
    ax[3].set_ylabel(labels[1] + '\n(Right)')
    # ax[3].set_xlabel('Time (s)')
    #ax[3].legend(['Right'],loc='lower left')
    
    timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = loadTCDdata(TCDname,TCDmarks)
    ax[4].plot(timeTCD,envTCD,'g',linewidth=0.5)
    ax[4].set_ylabel('TCD\nEnvelope')
    ax[5].plot(timeTCD,etCO2,'k',linewidth=1)    
    ax[5].set_ylabel('ETCO2\n(mmHg)')
    ax[5].set_xlabel('Time (s)')
    ax2 = ax[5].twinx()
    ax2.plot(timeTCD,respRate,linewidth=0.5)
    ax2.set_ylabel('Resp. Rate')
    
    for i in range(0,5):
        ax[i].set_xticks([])
        ax[i].set_xlim([t.min(),t.max()])
    ax[5].set_xlim([t.min(),t.max()])
    fig.suptitle(title)
    fig.savefig(savename + '_TCDraw.png',dpi=300,bbox_inches='tight')
    
def longScanPlotTCDmetrics(rBfiAvgMerged,rBfiAmpMerged,labels,freq,savename,title,TCDname,TCDmarks):
    # plots avg & amp metrics for both optical sides and TCD
    
    timeTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = loadTCDdata(TCDname,TCDmarks)
    
    meanTCDoffset = np.nanmean(meanTCD)
    pulIndTCDoffset = np.nanmean(pulIndTCD)
    TCDscaling = np.nanmax(np.abs(pulIndTCD-pulIndTCDoffset))/np.nanmax(np.abs(rBfiAvgMerged))*50
    
    fig, ax = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [1,1,1,1]})
    t = np.arange(rBfiAvgMerged.shape[0])/freq
    ax[0].plot(t,rBfiAvgMerged[:,1],linewidth=0.5)
    ax[0].plot(t,rBfiAmpMerged[:,1]-np.nanmean(rBfiAmpMerged[0:1000,0]),'r',linewidth=0.5)
    ax[0].set_ylabel('Optical\n(Left)')
    ax[0].legend([labels[0],labels[1]],loc='lower right',fontsize=4)
    # ax[1].plot(timeTCD,(meanTCD-meanTCDoffset),'g',linewidth=0.5)
    # ax[1].plot(timeTCD,(pulIndTCD-pulIndTCDoffset)*100,'m',linewidth=0.5)
    # ax[1].set_ylabel('TCD')
    # ax[1].legend(['TCD Average','TCD PI (x100)'],loc='lower right',fontsize=4)

    ax[1].plot(t,rBfiAvgMerged[:,0],linewidth=0.5)
    ax[1].plot(t,rBfiAmpMerged[:,0]-np.nanmean(rBfiAmpMerged[0:1000,1]),'r',linewidth=0.5)
    ax[1].set_ylabel('Optical\n(Right)')
    ax[1].legend([labels[0],labels[1]],loc='lower right',fontsize=4)
    ax[2].plot(timeTCD,(meanTCD-meanTCDoffset),'g',linewidth=0.5)
    ax[2].plot(timeTCD,(pulIndTCD-pulIndTCDoffset)*100,'m',linewidth=0.5)
    ax[2].set_ylabel('TCD')
    ax[2].legend(['TCD Average','TCD PI (x100)'],loc='lower right',fontsize=4)

    ax[3].plot(timeTCD,etCO2,'k',linewidth=1)
    ax[3].plot(timeTCD,respRate,linewidth=0.5)
    ax[3].legend(['ETCO2 (mmHg)','Resp. Rate, start/end'],loc='lower right',fontsize=4)
    ax[3].set_ylabel('CO2')
    ax[3].set_xlabel('Time (s)')
        
    for i in range(0,3):
        ax[i].set_xticks([])
        ax[i].set_xlim([t.min(),t.max()])
    ax[3].set_xlim([t.min(),t.max()])
    fig.suptitle(title)
    fig.savefig(savename + '_TCDmetrics.png',dpi=300,bbox_inches='tight')

def sortCamMod(inputData,bilat):
    # converts unilat/bilat data so that data follows L-R, and Hor-Near-Ver-XXX (where XXX is Near or Ver)
    # also handles long scans (follows L-R)
    
    # input data dimensions expected to be: cameras, modules, scans(optional)
    
    if inputData.shape[1] < 4:
        outputData = []
        print('!!!WARNING, LESS THAN 4 MODULES PRESENT, UNCLEAR HOW IT SHOULD BE SORTED!!!')
    
    # flips left and right for long scans, assumes long scans are never less than 3
    if bilat == 2 or inputData.shape[1] > 4:
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 2:
            outputData = np.flipud(outputData)
        elif len(inputData.shape) == 3:
            outputData = np.flip(outputData,0)
        else:
            outputData = []
    
    elif bilat == 1 and inputData.shape[1] == 4:
        # converts data so that 0 is left, 1 is right (swaps hor, ver)
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 2:
            outputData[:,[0,2,3]] = np.flipud(outputData[:,[0,2,3]])
        elif len(inputData.shape) == 3:
            outputData[:,[0,2,3],:] = np.flip(outputData[:,[0,2,3],:],0)
        else:
            outputData = []
    
    elif bilat == 0 and inputData.shape[1] == 4:
        # converts data so that 0 is left, 1 is right (order is hor,near,ver,near)
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 2:
            outputData = np.flipud(np.concatenate((outputData[0:2,0:2].T,outputData[0:2,2:4].T),axis=1))
        elif len(inputData.shape) == 3:
            for ind in range(inputData.shape[2]):
                outputData[:,:,ind] = np.flipud(np.concatenate((outputData[0:2,0:2,ind].T,outputData[0:2,2:4,ind].T),axis=1))
        else:
            outputData = []
    return outputData

def sortCamModTimetrace(inputData,bilat):
    # converts unilat/bilat data so that data follows L-R, and Hor-Near-Ver-XXX (where XXX is Near or Ver)
    # also handles long scans (follows L-R)
    
    # input data dimensions expected to be: time, cameras, modules
    
    if inputData.shape[2] < 4:
        outputData = []
        print('!!!WARNING, LESS THAN 4 MODULES PRESENT, UNCLEAR HOW IT SHOULD BE SORTED!!!')
    
    # flips left and right for long scans, assumes long scans are never less than 3
    if  bilat == 2 or inputData.shape[2] > 4:
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 3:
            outputData = np.flip(outputData,1)
        else:
            outputData = []
    
    elif bilat == 1 and inputData.shape[2] == 4:
        # converts data so that 0 is left, 1 is right (swaps hor, ver)
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 3:
            outputData[:,:,[0,2,3]] = np.flip(outputData[:,:,[0,2,3]],1)
        else:
            outputData = []
    
    elif bilat == 0 and inputData.shape[2] == 4:
        # converts data so that 0 is left, 1 is right (order is hor,near,ver,near)
        outputData = copy.deepcopy(inputData)
        if len(inputData.shape) == 3:
            for ind in range(inputData.shape[0]):
                outputData[ind,:,:] = np.flipud(np.concatenate((outputData[ind,0:2,0:2].T,outputData[ind,0:2,2:4].T),axis=1))
        else:
            outputData = []
    return outputData

def noiseMetricBH(inputData,periodCount): 
    # Number of times the signal changes directions
    inputData_deriv = np.diff(inputData,n=1)
    noiseMetric1 = np.absolute(np.diff(np.heaviside(inputData_deriv,1))).sum()
    
    # Path length signal takes
    #noiseMetric = np.absolute(inputData_deriv).sum()
    
    # Peak-to-trough single side height (shoulderHeights), summed beyond 2 peaks
    peaks1, _ = find_peaks(inputData,distance=1)
    peaks2, _ = find_peaks(-inputData,distance=1)
    peaks = np.sort(np.concatenate((peaks1,peaks2)))
    
    # amp_mean = -np.sort(np.diff(inputData[peaks]))[0:16].mean()
    
    shoulderHeights = np.flipud(np.sort(np.abs(np.diff(inputData[peaks]))))
    noiseMetric2 = np.sum(shoulderHeights[int(periodCount*4):])
    
    return noiseMetric1, noiseMetric2, shoulderHeights

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def getFileNames(mypath, key, histChUse): # ['scan_ch_0','scan_ch_1']
    
    fnames0 = []
    fnames1 = []
    fnames = os.listdir(mypath)
    N = len(fnames)
    for i in range(N):
        test = fnames[i]
        if bool(re.findall(key, test)):
            if bool(re.findall(histChUse[0], test)): 
                fnames0.append(test)
            elif bool(re.findall(histChUse[1], test)):
                fnames1.append(test)
    # BHadd
    # fnames0.sort()
    # fnames1.sort()   
    fnames0 = natural_sort(fnames0)
    fnames1 = natural_sort(fnames1)
    
    return fnames0, fnames1         

def getImgNames(mypath, key, imgChUse): # ['ch_0','ch_1']
    
    fnames0 = []
    fnames1 = []
    fnames = os.listdir(mypath)
    N = len(fnames)
    for i in range(N):
        test = fnames[i]
        if bool(re.findall(key, test)):
            if bool(re.findall(imgChUse[0], test)): 
                fnames0.append(test)
            elif bool(re.findall(imgChUse[1], test)):
                fnames1.append(test)
    # BHadd
    # fnames0.sort()
    # fnames1.sort()    
    fnames0 = natural_sort(fnames0)
    fnames1 = natural_sort(fnames1)
    
    return fnames0, fnames1


def getHistData2(mypath, fnames0, fnames1, nt):
    
    # if len(fnames0) != 4 or len(fnames1) != 4:
    #     print('WARNING, ' + str(len(fnames0)+len(fnames1)) + ' HISTOGRAM FILES EXIST IN ' + mypath + ' FOLDER, PROCESSING SKIPPED')
    #     return 
    N = len(fnames0)
    data_hist    = np.zeros((nt, 1028, 2, N))
    data_meanStd = np.zeros((nt, 1028, 2, N))
    data_camTemps = np.zeros((nt, 1028*4, 2, N))
    for i in range(N):
        histName = mypath + '/' + fnames0[i]
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.uint32, count=1028*nt)
            data_hist[:, :, 0, i] = data_temp.reshape((nt,1028))
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.float32, count=1028*nt)
            data_meanStd[:, :, 0, i] = data_temp.reshape((nt,1028))
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.uint8, count=1028*4*nt)
            data_camTemps[:, :, 0, i] = data_temp.reshape((nt,1028*4))
        
        histName = mypath + '/' + fnames1[i]
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.uint32, count=1028*nt)
            data_hist[:, :, 1, i] = data_temp.reshape((nt,1028))
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.float32, count=1028*nt)
            data_meanStd[:, :, 1, i] = data_temp.reshape((nt,1028))
        with open(histName, mode='rb') as f:
            data_temp = np.fromfile(f, dtype=np.uint8, count=1028*4*nt)
            data_camTemps[:, :, 1, i] = data_temp.reshape((nt,1028*4))
    print('total hist count 0-1023: ' + str(data_hist[0,0:1023, :, :].sum()))
    
    # 16/17 OB hist mean, standard deviation
    # 1024/1025 Main rows corrected?? mean, standard deviation
    # 1026 temp

    data_meanStd = data_meanStd[:,[16,17,1024,1025], :, :]
    data_hist = data_hist[:,0:1024, :, :]
    data_hist[:,14:18, :, :] = 0
    data_camTemps = np.flip(data_camTemps[:,4104:4108,:,:],axis=1)
    data_camTemps[:,0:2,1,:] = data_camTemps[:,2:4,1,:]
    data_camTemps[:,2:4,:,:] = 0
    data_camTemps[:,[1],:,:] = data_camTemps[:,[1],:,:] * 1.5625 - 45
    
    return data_hist, data_meanStd, data_camTemps


def loadHistograms3(mypath,histName,histChUse):

    fnames0, fnames1 = getFileNames(mypath,histName,histChUse)    
    nt = os.path.getsize(mypath + '/' + fnames0[0])/1028/4
    if nt%1 == 0:
        nt = int(nt)
    else:
        print('WARNING, FILE ' + mypath + '/' + fnames0[0] + ' IS ' + str(nt*1028*4) + ' BYTES')
        nt = int(np.floor(nt))
    histData, meanStdData, camTempsData = getHistData2(mypath, fnames0, fnames1, nt)
    
    return histData, meanStdData, camTempsData, nt

def smoothMeanStdData(meanStdData, t):
    
    s = meanStdData.shape #(t, m&v00, camera, module)
    meanStdDataSmooth = np.zeros(s)
    for i in range(2):
        for j in range(s[2]):
            for k in range(s[3]):
                a, b = linearfit(meanStdData[:,i ,j ,k], t)
                meanStdDataSmooth[:, i, j, k] = a*t + b
    
    return meanStdDataSmooth            

def linearfit(y, x):
    
    s = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x**2)
    sxy = np.sum(x*y)
    a = (sxy - sx*sy/s)/(sxx - sx*sx/s)
    b = (sy - a*sx)/s

    return a, b



def plotMeanStdData(meanStdData, meanStdDataSmooth, t):
    #shape(t,m&v,f&n,rh&lh&rv&lv)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Raw OB Stats')
    legend = ['R Hor', 'L Hor', 'R Ver', 'L Ver']
    ax[1, 0].plot(t, meanStdData[:, 0, 0, :]) #mean, all far cameras
    ax[1, 0].plot(t, meanStdDataSmooth[:, 0, 0, :], 'k')
    ax[1, 0].set_title('Far Mean')
    ax[1, 0].legend(legend)
    ax[0, 0].plot(t, meanStdData[:, 0, 1, :]) #mean, all near cameras
    ax[0, 0].plot(t, meanStdDataSmooth[:, 0, 1, :], 'k')
    ax[0, 0].set_title('Near Mean')
    ax[0, 0].legend(legend)
    ax[1, 1].plot(t, meanStdData[:, 1, 0, :]) #variance, all far cameras
    ax[1, 1].plot(t, meanStdDataSmooth[:, 1, 0, :], 'k')
    ax[1, 1].set_title('Far Standard Deviation')
    ax[1, 1].legend(legend)
    ax[0, 1].plot(t, meanStdData[:, 1, 1, :]) #variance, all near cameras
    ax[0, 1].plot(t, meanStdDataSmooth[:, 1, 1, :], 'k')
    ax[0, 1].set_title('Near Standard Deviation')
    ax[0, 1].legend(legend)



def GetOffsetFromImage(histData, scanname, w, h, bins,histChUse):

    maxfar = 250
    maxnear = 150
    mean_dark = np.zeros((2, 4))
    var_dark = np.zeros((2, 4))
    mypath = scanname + '/'    
    fnames0, fnames1 = getFileNames(mypath, 'bayer.y',histChUse)
    for i in range(4):
        if len(fnames0) == 16:
            namInd = 4*i+2
        elif len(fnames0) == 8:
            namInd = 2*i+1
        else:
            print('Number of images found not 8 or 16')
        img_dark, hist_dark, img_bright, hist_bright = getImgData(mypath + fnames0[namInd], w, h, bins)    
        mean_dark[0, i] = np.mean(img_dark[img_dark<maxfar])
        var_dark[0, i] = np.var(img_dark[img_dark<maxfar])
        img_dark, hist_dark, img_bright, hist_bright = getImgData(mypath + fnames1[namInd], w, h, bins)
        mean_dark[1, i] = np.mean(img_dark[img_dark<maxnear])
        var_dark[1, i] = np.var(img_dark[img_dark<maxnear])
    
    return mean_dark, var_dark

def subtractImagesShow(scanname, w, h, histChUse):
    bins = np.arange(-201, 201)+0.5
    gl = bins[:-1]+0.5
    mypath = scanname + '/'    
    fnames0, fnames1 = getFileNames(mypath, 'bayer.y',histChUse)
    for j in range(2):
        if j==0:
            fnames=fnames0
        elif j==1:
            fnames=fnames1
        
        if len(fnames0) == 16:
            ind = 2
        elif len(fnames0) == 8:
            ind = 1
        else:
            print('Number of images found not 8 or 16')
        
        for i in range(0, len(fnames0), ind):
            
            img_ob, hist_ob, img_main, hist_main = getImgData(mypath + fnames[i], w, h, bins)
            img_ob1, hist_ob1, img_main1, hist_main1 = getImgData(mypath + fnames[i+1], w, h, bins)
            img_obdiff = img_ob1.astype('float64')-img_ob.astype('float64')
            img_maindiff = img_main1.astype('float64')-img_main.astype('float64')
            hist_obdiff = np.histogram(img_obdiff[:], bins)
            hist_obdiff = hist_obdiff[0]
            hist_maindiff = np.histogram(img_maindiff[:], bins)
            hist_maindiff = hist_maindiff[0]
        
            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax[0, 1].imshow(img_obdiff,aspect='auto')
            ax[1, 1].semilogy(gl, hist_obdiff)
            #ax[1, 1].set_xlim((0, 200))
            ax[0, 0].imshow(img_maindiff)
            ax[1, 0].semilogy(gl, hist_maindiff)
            #ax[1, 0].set_xlim((0, 200))
            ax[0, 0].set_title('Main rows')
            ax[0, 1].set_title('OB rows')

def getImageStats2(scanname, w, h, bins, histChUse):

    maxfar = 256
    maxnear = 128
    mean_obRows_dark = np.zeros((2, 4))
    var_obRows_dark = np.zeros((2, 4))
    mean_mainRows_dark = np.zeros((2, 4))
    var_mainRows_dark = np.zeros((2, 4))
    mean_obRows_hotPxlCor = np.zeros((2, 4))
    var_obRows_hotPxlCor = np.zeros((2, 4))
    
    #for foldInd in range(Nfolders):
    mypath = scanname + '/'
    fnames0, fnames1 = getFileNames(mypath, 'bayer.y', histChUse)

    for moduleInd in range(4):
        if len(fnames0) == 16:
            namInd = moduleInd*4
        elif len(fnames0) == 8:
            namInd = moduleInd*2
        else:
            print('Number of images found not 8 or 16')
        
        # Calculating mean & variance for OB rows and main rows
        img_obRows_dark, hist_obRows_dark, img_mainRows_dark, hist_mainRows_dark = getImgData(mypath + fnames0[namInd], w, h, bins)
        mean_obRows_dark[0, moduleInd] = np.mean(img_obRows_dark[img_obRows_dark<maxfar])
        var_obRows_dark[0, moduleInd] = np.var(img_obRows_dark[img_obRows_dark<maxfar])
        mean_mainRows_dark[0, moduleInd] = np.mean(img_mainRows_dark[img_mainRows_dark<maxfar])
        var_mainRows_dark[0, moduleInd] = np.var(img_mainRows_dark[img_mainRows_dark<maxfar])
            
        # Corrections for the OB rows mean & variance due to hot pixels
        mean_obRows_hotPxlCor[0, moduleInd] = np.mean(img_obRows_dark[img_obRows_dark<maxfar]) - np.mean(img_obRows_dark)
        var_obRows_hotPxlCor[0, moduleInd] = np.var(img_obRows_dark[img_obRows_dark<maxfar]) - np.var(img_obRows_dark)
            
        # Calculating mean & variance for OB rows and main rows
        img_obRows_dark, hist_obRows_dark, img_mainRows_dark, hist_mainRows_dark = getImgData(mypath + fnames1[namInd], w, h, bins)
        mean_obRows_dark[1, moduleInd] = np.mean(img_obRows_dark[img_obRows_dark<maxnear])
        var_obRows_dark[1, moduleInd] = np.var(img_obRows_dark[img_obRows_dark<maxnear])
        mean_mainRows_dark[1, moduleInd] = np.mean(img_mainRows_dark[img_mainRows_dark<maxnear])
        var_mainRows_dark[1, moduleInd] = np.var(img_mainRows_dark[img_mainRows_dark<maxnear])
            
        # Corrections for the OB rows mean & variance due to hot pixels
        mean_obRows_hotPxlCor[1, moduleInd] = np.mean(img_obRows_dark[img_obRows_dark<maxnear]) - np.mean(img_obRows_dark)
        var_obRows_hotPxlCor[1, moduleInd] = np.var(img_obRows_dark[img_obRows_dark<maxnear]) - np.var(img_obRows_dark)

    return mean_obRows_dark, var_obRows_dark, mean_mainRows_dark, var_mainRows_dark, mean_obRows_hotPxlCor, var_obRows_hotPxlCor

def getImageStats3(scanname, w, h, bins, obmax, expN, imgChUse):

    #obmax: hot pixel cutoff for [near, far] cameras
    #expN: which image (0-3) for each camera/module combo (0,1)=dark (2,3)=bright
    
    Ncameras = 2
    Nmodules = 4

    fnames0, fnames1 = getImgNames(scanname + '/', 'bayer.y',imgChUse)
    
    if len(fnames0) == 16 and obmax.shape[1] == 4:
        camImgs = [0,4,8,12]
    elif len(fnames0) == 8  and obmax.shape[1] == 4:
        camImgs = [0,2,4,6]
    # elif len(fnames0) == 26:
    #     Nmodules = 13
    #     Nmodules2 = 13
    else:
        # print('Number of images found not 8 or 16')
        Nmodules = obmax.shape[1]
        Nmodules2 = obmax.shape[1]
        
    imgStats = {
            'mean_ob': np.zeros((Ncameras, Nmodules)),
            'var_ob': np.zeros((Ncameras, Nmodules)),   
            'mean_main': np.zeros((Ncameras, Nmodules)),
            'var_main': np.zeros((Ncameras, Nmodules)),   
            'mean_ob_all': np.zeros((Ncameras, Nmodules)),
            'var_ob_all': np.zeros((Ncameras, Nmodules)),
            'mean_main_all': np.zeros((Ncameras, Nmodules)),
            'var_main_all': np.zeros((Ncameras, Nmodules)),
            'hist_ob': np.zeros((1024, Ncameras, Nmodules)),
            'hist_main': np.zeros((1024, Ncameras, Nmodules))
            }
    
    for camInd in range(Ncameras):
        if camInd == 0:
            fnames = fnames0
        elif camInd == 1:
            fnames = fnames1
        for modInd in range(Nmodules):
            # Calculating mean & variance for OB rows and main rows
            if len(fnames0) == 16 and obmax.shape[1] == 4:
                # print('Ncameras2 16img: ' + str(camImgs[modInd]+expN))
                img_ob, hist_ob, img_main, hist_main = getImgData(scanname + '/' + fnames[camImgs[modInd]+expN], w, h, bins)
            elif len(fnames0) == 8  and obmax.shape[1] == 4:
                # print('Ncameras2 8img: ' + str(camImgs[modInd]+expN))
                img_ob, hist_ob, img_main, hist_main = getImgData(scanname + '/' + fnames[camImgs[modInd]+expN], w, h, bins)
            else:
                # print('Ncameras  long: ' + str(modInd*Nmodules+expN))
                img_ob, hist_ob, img_main, hist_main = getImgData(scanname + '/' + fnames[modInd*Ncameras+expN], w, h, bins)
            
            #img_ob[img_ob>255]=255
            #print('change img stats 3')
            
            #print([np.mean(img_ob), np.mean(img_main)])
            
            imgStats['mean_ob'][camInd, modInd] = np.mean(img_ob[img_ob<obmax[camInd, modInd]]) 
            imgStats['var_ob'][camInd, modInd]   = np.var(img_ob[img_ob<obmax[camInd, modInd]])
            imgStats['mean_ob_all'][camInd, modInd] = np.mean(img_ob) 
            imgStats['var_ob_all'][camInd, modInd]   = np.var(img_ob)
            imgStats['mean_main'][camInd, modInd] = np.mean(img_main[img_main<obmax[camInd, modInd]])  #inconsistant change to histogram
            imgStats['var_main'][camInd, modInd]   = np.var(img_main[img_main<obmax[camInd, modInd]])  #inconsistant change to histogram
            imgStats['mean_main_all'][camInd, modInd] = np.mean(img_main)  #inconsistant change to histogram
            imgStats['var_main_all'][camInd, modInd]   = np.var(img_main)  #inconsistant change to histogram
            imgStats['hist_ob'][:, camInd, modInd] = hist_ob
            imgStats['hist_main'][:, camInd, modInd] = hist_main
            
    # fig, ax = plt.subplots(nrows=4, ncols=2)
    # fig.suptitle('Dark Histograms')
    # for camInd in range(Ncameras):
    #     for modInd in range(Nmodules):
    #         print(str([camInd, modInd]))
    #         img_ob, hist_ob, img_main, hist_main = getImgData(scanname + '/' + fnames[modInd*Nmodules2+expN], w, h, bins)
    #         #ax[modInd,camInd].hist(img_main.ravel(), 1024, range=[0,1023], density=True, facecolor='blue', alpha=0.75, log=True)
    #         ax[modInd,camInd].semilogy(hist_main)
    
    # BHadd
    if 1 == 0:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.suptitle('Dark Histograms\n' + scanname)
        for camInd in range(Ncameras):
            for modInd in range(Nmodules):
                img_ob, hist_ob, img_main, hist_main = getImgData(scanname + '/' + fnames[modInd*Nmodules2+expN], w, h, bins)
                if modInd == 0 or modInd == 2:
                    ax[camInd].semilogy(hist_main)
                ax[1].legend(['Horizontal','Vertical'])
            
    return imgStats

        
def getImgData(imgName, w, h, bins):
    
    with open(imgName, mode='rb') as f:
        temp = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

    #plt.figure()
    #im=plt.imshow(temp)
    #plt.colorbar(im)
    
    # format dark and bright images
    img_dark = temp[:20, :]
    img_bright = temp[20:, :]
        
    # get histograms
    hist_dark = np.histogram(img_dark[:], bins)
    hist_dark = hist_dark[0]
    hist_bright = np.histogram(img_bright[:], bins)
    hist_bright = hist_bright[0]
    
    return img_dark, hist_dark, img_bright, hist_bright



def quickPlot(histData, titles, gl, mean_dark, t):
    
    Nfolders = histData.shape[-1]
    fig, ax = plt.subplots(nrows=2, ncols=Nfolders)
    for i in range(Nfolders):
        
        temp = np.sum(histData[:, :, 1, i], axis=0)
        inds = np.where(temp>0)[0]
        a = int(inds[0])
        b = int(inds[-1])
        x = gl[a:b]-mean_dark[1, i]
        ax[0, i].pcolormesh(x, t, np.log(histData[:, a:b, 1, i]), cmap='jet')
        #ax[0, i].imshow(np.log(histData[:, a:b, 1, i]), aspect='auto', cmap='jet')
        
        temp = np.sum(histData[:, :, 0, i], axis=0)
        inds = np.where(temp>0)[0]
        a = int(inds[0])
        b = int(inds[-1])
        x = gl[a:b]-mean_dark[0, i]
        ax[1, i].pcolormesh(x, t, np.log(histData[:, a:b, 0, i]), cmap='jet')
        #ax[1, i].imshow(np.log(histData[:, a:b, 0, i]), aspect='auto', cmap='jet')        
    
        ax[0, i].set_title(titles[i])
        if i==0:
            ax[0, 0].set_ylabel('Near')
            ax[1, 0].set_ylabel('Far')
    
    fig.suptitle('Histograms')
    
def hist2meanVar(histData, gl):
    
    m1 = []
    m2 = []
    var_raw = []
    
    hist = histData #single histogram
    probs = hist/np.sum(hist) #normalized single histogram
    
    m1 = np.sum(gl * probs)
        
    m2 = np.sum((gl**2) * probs)
    
    var_raw = m2 - m1**2
    
    return m1, var_raw

def hist2contrast5(histData, histDataDark, imgStats_dark, gl, gain, nt, meanStdData, meanStdDataDark, histCutoff,flag):
    
    # histData: histograms from scan. shape (t, gl, camera, module)
    # imgStats_dark: statistics from prescan dark image
    # gl: grey levels corresponding to histograms
    # gain: camera gain settings for far and near
    # nt: number of time points in scan (not used)
    # meanStdData: mean and std calculated on TDA4 for during the scan. shape (t, m&v00, camera, module)
    
    modTime = 18 # 15.6 # total time in seconds (acq. takes 15 seconds)
    freq = 40
    
    #flag = 2
    s = histData.shape #(t, gl, camera, module)
    m1 = np.zeros((s[0], s[2], s[3]))  #(t, camInd, module)
    m2 = np.zeros((s[0], s[2], s[3]))
    m3 = np.zeros((s[0], s[2], s[3]))
    dv = np.zeros((s[0], s[2], s[3]))
    
    
    histImg_dark = imgStats_dark['hist_main']
    histImg_dark[histImg_dark<histCutoff]=0
    histImg_dark[1023, :, :]=0
    mean_histImg_dark = np.zeros((2,histData.shape[3]))
    var_histImg_dark = np.zeros((2,histData.shape[3]))
    for camInd in range(2):
        for modInd in range(histData.shape[3]):
            mean_histImg_dark[camInd,modInd], var_histImg_dark[camInd,modInd] = hist2meanVar(histImg_dark[:,camInd,modInd],gl)
    
    # Uses
    if np.array(histDataDark).size != 0:
        histDataDark[histDataDark<histCutoff]=0
        histDataDark[:, 1023, :, :]=0
        mean_histDataDark = np.zeros((histDataDark.shape[0],2,histData.shape[3]))
        var_histDataDark = np.zeros((histDataDark.shape[0],2,histData.shape[3]))
        for camInd in range(2):
            for modInd in range(histData.shape[3]):
                if np.array(histDataDark).size != 0:
                    for tInd in range(histDataDark.shape[0]):
                        mean_histDataDark[tInd,camInd,modInd], var_histDataDark[tInd,camInd,modInd] = hist2meanVar(histDataDark[tInd,:,camInd,modInd],gl)
        mean_histDataDark = mean_histDataDark.mean(axis=0)
        var_histDataDark = var_histDataDark.mean(axis=0)
    
        meanStdData2 = np.zeros(meanStdData.shape)
        t = np.arange(nt)
        for camInd in range(2):
            for modInd in range(s[3]):
                polyFit2 = np.poly1d(np.polyfit(np.array([1,600]), [mean_histImg_dark[camInd,modInd], mean_histDataDark[camInd,modInd]],1))
                meanStdData2[:,0,camInd,modInd] = polyFit2(t)
                polyFit2 = np.poly1d(np.polyfit(np.array([1,600]), np.array([var_histImg_dark[camInd,modInd], var_histDataDark[camInd,modInd]])**0.5,1))
                meanStdData2[:,1,camInd,modInd] = polyFit2(t)
    
    # BHadd
    # flag 5 is for long scans only and uses interpolation of dark images for dark variancne correction
    if flag  == 5:
        meanStdData2 = copy.deepcopy(meanStdData)
        x = np.linspace(0, histData.shape[3]*modTime, num=histData.shape[3], endpoint=False)
        xx = np.linspace(0, histData.shape[3]*modTime, num=int(histData.shape[3]*modTime*freq), endpoint=False)
        for camInd in range(2):
            # y1 = copy.deepcopy(imgStats_dark['mean_main'][camInd, :])
            # f = interpolate.interp1d(x, y, fill_value='extrapolate')
            y1 = mean_histImg_dark[camInd,:]
            polyFit1 = np.poly1d(np.polyfit(x, y1, 4))
            polyFit1 = np.reshape(polyFit1(xx),(histData.shape[3],int(modTime*freq)))
            meanStdData2[:,0,camInd,:] = polyFit1[:,:histData.shape[0]].T
            
            # y2 = copy.deepcopy(imgStats_dark['var_main'][camInd, :])
            # f = interpolate.interp1d(x, y, fill_value='extrapolate')
            y2 = var_histImg_dark[camInd,:]
            polyFit2 = np.poly1d(np.polyfit(x, y2, 4))
            polyFit2 = np.reshape(polyFit2(xx),(histData.shape[3],int(modTime*freq)))**0.5
            meanStdData2[:,1,camInd,:] = polyFit2[:,:histData.shape[0]].T
        
    if flag  == 7:
        meanStdData2 = copy.deepcopy(meanStdData)
        x1 = np.linspace(0, histData.shape[3]*modTime, num=histData.shape[3], endpoint=False)
        x2 = np.linspace(0, histData.shape[3]*modTime, num=histData.shape[3], endpoint=False)+15
        x = np.reshape(np.stack((x1,x2),axis=1),(-1))
        xx = np.linspace(0, histData.shape[3]*modTime, num=int(histData.shape[3]*modTime*freq), endpoint=False)
        for camInd in range(2):
            y1 = mean_histImg_dark[camInd,:]
            y2 = mean_histDataDark[camInd,:]
            y = np.reshape(np.stack((y1,y2),axis=1),(-1))
            # f = interpolate.interp1d(x, y, fill_value='extrapolate')
            polyFit1 = np.poly1d(np.polyfit(x, y, 4))
            polyFit1 = np.reshape(polyFit1(xx),(histData.shape[3],int(modTime*freq)))
            meanStdData2[:,0,camInd,:] = polyFit1[:,:histData.shape[0]].T
            
            y1 = var_histImg_dark[camInd,:]
            y2 = var_histDataDark[camInd,:]
            y = np.reshape(np.stack((y1,y2),axis=1),(-1))
            # f = interpolate.interp1d(x, y, fill_value='extrapolate')
            polyFit2 = np.poly1d(np.polyfit(x, y, 4))
            polyFit2 = np.reshape(polyFit2(xx),(histData.shape[3],int(modTime*freq)))**0.5
            meanStdData2[:,1,camInd,:] = polyFit2[:,:histData.shape[0]].T
    
    for camInd in range(s[2]): #camera index (0 or 1)
        for modInd in range(s[3]): #module index
            for t in range(s[0]): #time    
                hist = histData[t, :, camInd, modInd] #single histogram
                probs = hist/np.sum(hist)             #normalized single histogram
                
                if flag==0:
                    # mean     corrections from dark imgOB rows
                    # variance corrections from dark imgOB rows
                    x = gl - imgStats_dark['mean_ob'][camInd, modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = imgStats_dark['var_ob'][camInd, modInd] + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                if flag==1:
                    # mean     corrections from dark imgMain
                    # variance corrections from dark imgMain
                    x = gl - imgStats_dark['mean_main'][camInd, modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = imgStats_dark['var_main'][camInd, modInd] + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                if flag==2:
                    # mean     correction from histOB + (dark imgMain - dark imgOB)
                    # variance correction from dark imgMain
                    x = gl - meanStdData[t,0,camInd,modInd] - imgStats_dark['mean_main'][camInd, modInd] + imgStats_dark['mean_ob'][camInd, modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = imgStats_dark['var_main'][camInd, modInd] + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                if flag==3:
                    # mean     correction from histOB + (dark imgMain - dark imgOB)
                    # variance correction from histOB + (dark imgMain - dark imgOB)
                    x = gl - meanStdData[t,0,camInd,modInd] - imgStats_dark['mean_main'][camInd, modInd] + imgStats_dark['mean_ob'][camInd, modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = meanStdData[t,1,camInd,modInd]**2 + imgStats_dark['var_main'][camInd, modInd] - imgStats_dark['var_ob'][camInd, modInd] + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd] 
                if flag==4:
                    # mean     correction from scan OB + (imgMain-imgOB)
                    # variance correction from histOB * dark imgMain / dark imgOB (no shot noise correction)
                    x = gl - meanStdData[t,0,camInd,modInd] - imgStats_dark['mean_main'][camInd, modInd] + imgStats_dark['mean_ob'][camInd, modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = meanStdData[t,1,camInd,modInd]**2 * imgStats_dark['var_main'][camInd, modInd] / imgStats_dark['var_ob'][camInd, modInd] #+ 0.12*gain[camInd, modInd]*m1[t, camInd, modInd] 
                
                if flag==5:
                    # flat 5 works for long scans without dark histogram frames
                    # mean     corrections from fitted dark imgMain
                    # variance corrections from fitted dark imgMain
                    x = gl - meanStdData2[t,0,camInd,modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = meanStdData2[t,1,camInd,modInd]**2 + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                    
                if flag==6:
                    # flag 6 works for all data types and uses interpolation of dark image and dark frames for each acquisition
                    # mean     corrections from fitted dark imgMain and dark hist
                    # variance corrections from fitted dark imgMain and dark hist
                    x = gl - meanStdData2[t,0,camInd,modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = meanStdData2[t,1,camInd,modInd]**2 + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                
                if flag==7:
                    # flag 7 works for long scans with dark histogram frames
                    # mean     corrections from fitted dark imgMain and dark hist
                    # variance corrections from fitted dark imgMain and dark hist
                    x = gl - meanStdData2[t,0,camInd,modInd]
                    m1[t, camInd, modInd] = np.sum(x * probs)
                    dv[t, camInd, modInd] = meanStdData2[t,1,camInd,modInd]**2 + 0.12*gain[camInd, modInd]*m1[t, camInd, modInd]
                    
                #m1[t, camInd, modInd] = np.sum(x * probs)
                m2[t, camInd, modInd] = np.sum((x**2) * probs)
                m3[t, camInd, modInd] = np.sum((x**3) * probs)
    
    var_raw = m2 - m1**2
    var = var_raw - dv
    toosmall = np.min(var, axis=0)
    for camInd in range(s[2]):
        for modInd in range(s[3]):            
            if toosmall[camInd, modInd] < 0:
                var[:, camInd, modInd] -= (toosmall[camInd, modInd] - 0.01)
    contrast = np.sqrt(var)/m1           
    skew = (m3 - 3*m1*(m2-m1**2) - m1**3)/((m2-m1**2)**(3/2))    #(m3-3*m1*var_raw-m1**3)/(var_raw**(3/2))
    
    #from Bandyopadhyay 2005 (Durian group)
    v2 = m2/(m1**2) - 1 #2nd reduced moment (uncorrected contrast)
    v3 = m3/(m1**3) - 1 #3rd reduced moment
    v32 = v3-3*v2 # see formula (11)
    

            
                
    """    
                elif flag==1: #mean correction from scan OB & image additive, variance from image ob only
                    mean_correction = (imgStats_dark['mean_main'][camInd, modInd] - imgStats_dark['mean_ob'][camInd, modInd])
                    mean[t, camInd, modInd] = np.sum(gl*hist/np.sum(hist)) - (meanStdData[t,0,camInd,modInd] + mean_correction) 
                    var_light = np.sum(hist/np.sum(hist)*(gl-np.sum(gl*hist/np.sum(hist)))**2)
                    var[t, camInd, modInd] = var_light - imgStats_dark['var_ob'][camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd]
                    
                elif flag==2: #fix it so that 1st correction is from image and scan ob's just adjust it
                    mean_correction = meanStdData[t, 0, camInd, modInd] - meanStdData[0, 0, camInd, modInd]
                    var_correction  = meanStdData[t, 1, camInd, modInd]**2 - meanStdData[0, 1, camInd, modInd]**2
                    mean[t, camInd, modInd] = np.sum((gl-imgStats_dark['mean_main'][camInd, modInd]) * (hist/np.sum(hist))) - mean_correction
                    var[t, camInd, modInd] = np.sum(hist/np.sum(hist)*(gl-np.sum(gl*hist/np.sum(hist)))**2)  - imgStats_dark['var_main'][camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd] - var_correction
                
            toosmall = np.min(var[:, camInd, modInd])
            if toosmall<0:
                var[:, camInd, modInd] -= (toosmall - 0.01)
    """    
    
    stats = {
            'm1': m1,
            'm2': m2,
            'm3': m3,
            'dv': dv,
            'var_raw': var_raw,
            'var': var,
            'contrast': contrast}
    
    return m1, contrast, stats

def hist2contrast4(histData, imgStats_dark, gl, gain, nt, meanVarData):
    
    #shape of meanVarData (t, m&v00, camera, module)
    
    flag = 1
    s = histData.shape #(t, gl, camera, module)       
    mean = np.zeros((s[0], s[2], s[3]))  #(t, camInd, module)
    var =  np.zeros((s[0], s[2], s[3]))
    holder =  np.zeros((s[0], s[2], 3))
    
    for camInd in range(s[2]): #camera index (0 or 1)
        for modInd in range(s[3]): #module index
            for t in range(s[0]): #time    
                hist = histData[t, :, camInd, modInd]
                
                if flag==0: #correction from image ob only
                    mean[t, camInd, modInd] = np.sum((gl-imgStats_dark['mean_ob'][camInd, modInd]) * (hist/np.sum(hist)))        
                    m2 = np.sum(((gl-imgStats_dark['mean_ob'][camInd, modInd])**2) * (hist/np.sum(hist)))
                    var[t, camInd, modInd] = m2 - mean[t, camInd, modInd]**2 - imgStats_dark['var_ob'][camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd]
                    
                elif flag==1: #mean correction from scan OB & image additive, variance from image ob only
                    mean_correction = (imgStats_dark['mean_main'][camInd, modInd] - imgStats_dark['mean_ob'][camInd, modInd])
                    mean[t, camInd, modInd] = np.sum(gl*hist/np.sum(hist)) - (meanVarData[t,0,camInd,modInd] + mean_correction) 
                    var_light = np.sum(hist/np.sum(hist)*(gl-np.sum(gl*hist/np.sum(hist)))**2)
                    var[t, camInd, modInd] = var_light - imgStats_dark['var_ob'][camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd]
                    
                elif flag==2: #fix it so that 1st correction is from image and scan ob's just adjust it
                    mean_correction = meanVarData[t, 0, camInd, modInd] - meanVarData[0, 0, camInd, modInd]
                    var_correction  = meanVarData[t, 1, camInd, modInd]**2 - meanVarData[0, 1, camInd, modInd]**2
                    mean[t, camInd, modInd] = np.sum((gl-imgStats_dark['mean_main'][camInd, modInd]) * (hist/np.sum(hist))) - mean_correction
                    var[t, camInd, modInd] = np.sum(hist/np.sum(hist)*(gl-np.sum(gl*hist/np.sum(hist)))**2)  - imgStats_dark['var_main'][camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd] - var_correction
                
            toosmall = np.min(var[:, camInd, modInd])
            if toosmall<0:
                var[:, camInd, modInd] -= (toosmall - 0.01)
                
    contrast = np.sqrt(var)/mean
    
    
    return mean, contrast, holder


            
def hist2contrast3(histData, mean_obRows_dark, var_obRows_dark, gl, gain, nt, meanVarData, mean_obRows_hotPxlCor, var_obRows_hotPxlCor):
    
    flag = 1
    s = histData.shape #(t, gl, camera, module)       
    mean = np.zeros((s[0], s[2], s[3]))  #(t, camInd, module)
    var =  np.zeros((s[0], s[2], s[3]))
    holder =  np.zeros((s[0], s[2], 3))
    
    for camInd in range(s[2]): #camera index (0 or 1)
        for modInd in range(s[3]): #module index
            for t in range(s[0]): #time    
                hist = histData[t, :, camInd, modInd]
                
                if flag==0: #correction from image only
                    mean[t, camInd, modInd] = np.sum((gl-mean_obRows_dark[camInd, modInd]) * (hist/np.sum(hist)))        
                    m2 = np.sum(((gl-mean_obRows_dark[camInd, modInd])**2) * (hist/np.sum(hist)))
                    var[t, camInd, modInd] = m2 - mean[t, camInd, modInd]**2 - var_obRows_dark[camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd]
                elif flag==1: #from mean OB & image additive, variance from image
                    mean[t, camInd, modInd] = np.sum(gl*hist/np.sum(hist)) - (meanVarData[t,0,camInd,modInd] + mean_obRows_hotPxlCor[camInd, modInd])
                    var_light = np.sum(hist/np.sum(hist)*(gl-np.sum(gl*hist/np.sum(hist)))**2)
                    # corrected variance = histogram variance - (kedar's S.D. squared + offset to correct for hot pixels in 20 OB rows) - shot noise
                    # var_darkCorrection = (meanVarData[t,1,camInd,f]**2 + var_obRows_hotPxlCor[camInd, f])
                    var[t, camInd, modInd] = var_light - var_obRows_dark[camInd, modInd] - 0.12*gain[camInd, modInd]*mean[t, camInd, modInd]# - var_darkCorrection
                elif flag==2: #
                    print('asdf')
                elif flag==3:
                    print('fix it so first one is same as image')
                
            toosmall = np.min(var[:, camInd, modInd])
            if toosmall<0:
                var[:, camInd, modInd] -= (toosmall - 0.01)
                
    contrast = np.sqrt(var)/mean
    
    
    return mean, contrast, holder



def getBVI(mean, meanRef, muaRef, musp, rho, ext):
    
    #mean:    mean of subject camera image (can be vector or scalar)
    #meanRef: mean of calibration block camera image
    #muaRef:  mua of calibration block (mm-1)
    #muspRef: musp of calibration block (mm-1)
    #rho:     source detector separation (mm)
    #ext:     extinction coefficient of HbT at our wavelength assuming some StO2 (mm-1/uM)
    
    #mua:     absorption coefficient of subject
    #bvi:     blood volume index (uM concentration of HbT under certain assumptions)
    
    mua = (1/3/musp) * ( np.sqrt(3*muaRef*musp) - np.log(mean/meanRef)/rho )**2
    bvi = mua/ext
    
    return mua, bvi

def simContrast(bT, a):
    
    # Returns speckle contrast (beta=1) for semi-infinite media with scatterers 
    # undergoing Brownian motion
    # a = 3 * mua * musp * (2*rho)^2
    # bT = musp^2 * k0^2 * alpha * 6 * Db * (2*rho)^2
        
    pre = (2/bT)**2
    term1 = (2*(a+bT) + 6*np.sqrt(a+bT) + 6)*np.exp(np.sqrt(a)-np.sqrt(a+bT))
    term2 = 2*a      + 6*np.sqrt(a)    + 6 - (np.sqrt(a)+1)*bT
    terms = term1 - term2
    
    
    return np.sqrt(np.maximum(pre*terms, 0))

def residual_contrast(bT, a, data):
    
    return simContrast(bT, a) - data

def getBFI(contrast, contrastRef, offset, aDb0, mua, musp, wv, n, rho, T):

    # compute the Brownian diffusion coefficient for light scatterers
    # contrast: measured contrast from patient (can be vector or scalar)
    # contrastRef: measured contrast from static calibration block
    # aDb0: initial guess for scatterer motion (mm^2/s) (alpha * Db)
    # mua: absorption coefficient (mm-1)(can be vector or scalar, can be from getBVI or calibration block)
    # musp: reduced scattering coefficient of calibration block (mm-1)
    # wv: wavelength of light (mm)
    # n: index of refraction
    # rho: source detector separation (mm)
    # T: laser pulse width (s)
    
    # aDb: result for scatterer motion (mm^2/s) (BFI)
    
    nt = len(contrast) if isinstance(contrast, Sized) else 1
    b = np.zeros((nt,))
    k0 = 2*np.pi*n/wv
    a = 3 * mua * musp * (2*rho)**2
    bT = musp**2 * k0**2 * 6 * aDb0 * (2*rho)**2
    data = (contrast-offset)/contrastRef
    if any(data>1):
        print('Warning: Contrast > 1 !!!!')
        data[data>1]=1
    
    if nt>1:
        for i in range(nt):
            if np.isfinite(data[i]):
                out = least_squares(residual_contrast, bT, bounds=(0, np.inf), args=(a[i], data[i]) )
                b[i] = out.x[0]
            else:
                b[i] = np.nan
    else:
        out = least_squares(residual_contrast, bT, bounds=(0, np.inf), args=(a, data) )
        b = out.x[0]
    
    aDb = b/(6*(musp**2)*(k0**2)*4*(rho**2)*T)
    
    return aDb 

def BFIBVI(mean, contrast, mean_timeaverage_Cal, contrast_timeaverage_Cal, params):
    
    mua = np.zeros(mean.shape)
    bvi = np.zeros(mean.shape)
    bfi = np.zeros(mean.shape)

    Nmodules = mean.shape[2]
    Ncameras = mean.shape[1]
    

    print('Using offset2 = ' + str(params['offset2']) + ' to get BFI')
    for modInd in range(Nmodules):
        for camInd in range(Ncameras):
                meanPat = mean[:, camInd, modInd]
                meanRef = mean_timeaverage_Cal[camInd, modInd]
                rho = params['rhos'][camInd]
                mua[:, camInd, modInd], bvi[:, camInd, modInd] = getBVI(meanPat, meanRef, params['muaRef'], params['musp'], rho, params['ext'])
                contrastPat = contrast[:, camInd, modInd]
                contrastRef = contrast_timeaverage_Cal[camInd, modInd]                
                bfi[:, camInd, modInd] = 1e7*getBFI(contrastPat, contrastRef, params['offset2'][camInd], params['aDb0'], mua[:, camInd, modInd], params['musp'], params['wv'], params['indexrefract'], rho, params['T'])
 
                bvi[:, camInd, modInd] = gaussian_filter1d(bvi[:, camInd, modInd], 1)

    return mua, bfi, bvi
        
def plotContMean2axis(x, y, x_mean, y_mean, t, legend, scanname, saveimages, savename, bilat, noiseMetricBH, heartRate, crop=0):
    # Put in raw data format and bilat variable, will auto place in correct plot locations
    
    x = sortCamModTimetrace(x,bilat)
    y = sortCamModTimetrace(y,bilat)
    
    if len(x_mean) != 0:
        x_mean = sortCamMod(x_mean.mean(axis=0),bilat)
        y_mean = sortCamMod(y_mean.mean(axis=0),bilat)
    
    if crop>0:
        x=x[crop:-crop, :, :]
        y=y[crop:-crop, :, :]
        t=t[crop:-crop]
    
    s=x.shape #(t,c,f)
    
    figwidth = 18
    if len(t)<100:
        figwidth = 9
    
    fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
    # fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
    #      'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)   ---   Avg. BHnoiseMetric: \n' + str(np.round(noiseMetricBH.T,1)))
    fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
         'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)')
    fig.tight_layout(w_pad=3.5)
    plt.subplots_adjust(top=0.85)
    
    if bilat == 0:
        for camInd in range(s[1]):
            for modInd in range(s[2]):
                ax[modInd, camInd].plot(t, x[:, camInd, modInd], 'k', linewidth=2)
                if len(x_mean) != 0:
                    ax[modInd, camInd].legend(['Avg. ' + str(x_mean[camInd, modInd].round(3))],loc='upper right')
                ax2 = ax[modInd, camInd].twinx()
                ax2.plot(t, y[:, camInd, modInd], 'r', linewidth=1)
                ax2.tick_params(axis='y', colors='red')
                if len(x_mean) != 0:
                    ax2.legend(['Avg. ' + str(int(y_mean[camInd, modInd]))],loc='lower right')
        ax[0, 0].set_title('Left')
        ax[0, 1].set_title('Right')
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 1].set_xlabel('Time (s)')
        
        ax[0, 0].set_ylabel('Horizontal')
        ax[1, 0].set_ylabel('Surface')
        ax[2, 0].set_ylabel('Vertical')
        ax[3, 0].set_ylabel('Surface')
        
    elif bilat == 1:
        for camInd in range(s[1]):
            for modInd in range(s[2]):
                ax[modInd, camInd].plot(t, x[:, camInd, modInd], 'k', linewidth=2)
                if len(x_mean) != 0:
                    ax[modInd, camInd].legend(['Avg. ' + str(x_mean[camInd, modInd].round(3))],loc='upper right')
                ax2 = ax[modInd, camInd].twinx()
                ax2.plot(t, y[:, camInd, modInd], 'r', linewidth=1)
                ax2.tick_params(axis='y', colors='red')
                if len(x_mean) != 0:
                    ax2.legend(['Avg. ' + str(int(y_mean[camInd, modInd]))],loc='lower right')
        ax[0, 0].set_title('Left')
        ax[0, 1].set_title('Right')
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 1].set_xlabel('Time (s)')
        
        ax[0, 0].set_ylabel('Horizontal')
        ax[1, 0].set_ylabel('Surface')
        ax[2, 0].set_ylabel('Vertical')
        ax[3, 0].set_ylabel('Horizontal')
        
    elif bilat == 2:
        for camInd in range(s[1]):
            for modInd in range(s[2]):
                ax[modInd, camInd].plot(t, x[:, camInd, modInd], 'k', linewidth=2)
                if len(x_mean) != 0:
                    ax[modInd, camInd].legend(['Avg. ' + str(x_mean[camInd, modInd].round(3))],loc='upper right')
                ax2 = ax[modInd, camInd].twinx()
                ax2.plot(t, y[:, camInd, modInd], 'r', linewidth=1)
                ax2.tick_params(axis='y', colors='red')
                if len(x_mean) != 0:
                    ax2.legend(['Avg. ' + str(int(y_mean[camInd, modInd]))],loc='lower right')
        ax[0, 0].set_title('Left')
        ax[0, 1].set_title('Right')
        ax[s[1], 0].set_xlabel('Time (s)')
        ax[s[1], 1].set_xlabel('Time (s)')
    
    if saveimages:
        plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
def plotLeftRight1axis(x, y, x_mean, y_mean, t, legend, scanname, saveimages, savename, bilat, noiseMetricBH, heartRate, crop=0):
    # Put in raw data format and bilat variable, will auto place in correct plot locations
    
    x = sortCamModTimetrace(x,bilat)
    y = sortCamModTimetrace(y,bilat)
    
    if len(x_mean) != 0:
        x_mean = sortCamMod(x_mean.mean(axis=0),bilat)
        y_mean = sortCamMod(y_mean.mean(axis=0),bilat)
    else:
        x_mean = np.zeros(x.shape[1:3])
        y_mean = np.zeros(y.shape[1:3])
    
    if crop>0:
        x=x[crop:-crop, :, :]
        y=y[crop:-crop, :, :]
        t=t[crop:-crop]
    
    s=x.shape #(t,c,f)
    
    figwidth = 18
    if len(t)<100:
        figwidth = 9
    
    fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
    # fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
    #      'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)   ---   Avg. BHnoiseMetric: \n' + str(np.round(noiseMetricBH.T,1)))
    fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
         'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)')
    fig.tight_layout(w_pad=3.5)
    plt.subplots_adjust(top=0.85)
    
    if bilat == 0:
        for modInd in range(s[2]):
            ax[modInd, 0].plot(t, x[:, 0, modInd], 'b', linewidth=1)
            ax[modInd, 0].plot(t, x[:, 1, modInd], 'k', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 0].legend(['L Avg. ' + str(x_mean[0, modInd].round(3)),'R Avg. ' + str(x_mean[1, modInd].round(3))],loc='upper right')
            ax[modInd, 1].plot(t, y[:, 0, modInd], 'r', linewidth=1)
            ax[modInd, 1].plot(t, y[:, 1, modInd], 'm', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 1].legend(['L Avg. ' + str(int(y_mean[0, modInd])),'R Avg. ' + str(int(y_mean[1, modInd]))],loc='upper right')
        ax[0, 0].set_title('Left')
        ax[0, 1].set_title('Right')
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 1].set_xlabel('Time (s)')
        
        ax[0, 0].set_ylabel('Horizontal')
        ax[1, 0].set_ylabel('Surface')
        ax[2, 0].set_ylabel('Vertical')
        ax[3, 0].set_ylabel('Surface')
    
    elif bilat == 1:
        for modInd in range(s[2]):
            ax[modInd, 0].plot(t, x[:, 0, modInd], 'b', linewidth=1)
            ax[modInd, 0].plot(t, x[:, 1, modInd], 'k', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 0].legend(['L Avg. ' + str(x_mean[0, modInd].round(3)),'R Avg. ' + str(x_mean[1, modInd].round(3))],loc='upper right')
            ax[modInd, 1].plot(t, y[:, 0, modInd], 'r', linewidth=1)
            ax[modInd, 1].plot(t, y[:, 1, modInd], 'm', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 1].legend(['L Avg. ' + str(int(y_mean[0, modInd])),'R Avg. ' + str(int(y_mean[1, modInd]))],loc='upper right')
        ax[0, 0].set_title(legend[0])
        ax[0, 1].set_title(legend[1])
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 1].set_xlabel('Time (s)')
        
        ax[0, 0].set_ylabel('Horizontal')
        ax[1, 0].set_ylabel('Surface')
        ax[2, 0].set_ylabel('Vertical')
        ax[3, 0].set_ylabel('Horizontal')
        
    elif bilat == 2:
        for modInd in range(s[2]):
            ax[modInd, 0].plot(t, x[:, 0, modInd], 'b', linewidth=1)
            ax[modInd, 0].plot(t, x[:, 1, modInd], 'k', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 0].legend(['L Avg. ' + str(x_mean[0, modInd].round(3)),'R Avg. ' + str(x_mean[1, modInd].round(3))],loc='upper right')
            ax[modInd, 1].plot(t, y[:, 0, modInd], 'r', linewidth=1)
            ax[modInd, 1].plot(t, y[:, 1, modInd], 'm', linewidth=1)
            if len(x_mean) != 0:
                ax[modInd, 1].legend(['L Avg. ' + str(int(y_mean[0, modInd])),'R Avg. ' + str(int(y_mean[1, modInd]))],loc='upper right')
        ax[0, 0].set_title(legend[0])
        ax[0, 1].set_title(legend[1])
        ax[3, 0].set_xlabel('Time (s)')
        ax[3, 1].set_xlabel('Time (s)')
    
    if saveimages:
        plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
# def plotResults(x, y, t, legend, scanname, saveimages, savename, noiseMetricBH, heartRate, crop=0):
    
#     if crop>0:
#         x=x[crop:-crop, :, :]
#         y=y[crop:-crop, :, :]
#         t=t[crop:-crop]
    
#     s=x.shape #(t,c,f)
#     # acquisition  position          (c,f)     subplot
#     # 0,           right horizontal  (1,0)     [0, 1] near
#     #                                (0,0)     [1, 1] far
#     # 1,           left horizontal   (1,1)     [0, 0]
#     #                                (0,1)     [1, 0]
#     # 2,           right vertical    (1,2)     [2, 1]   
#     #                                (0,2)     [3, 1]
#     # 3,           left vertical     (1,3)     [2, 0]
#     #                                (0,3)     [3, 0]
    
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
    
#     fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
#     #noiseMetricBH = np.round(noiseMetricBH.mean().T,1)
#     #noiseMetricBH = []
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
#          'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)   ---   Avg. BHnoiseMetric: \n' + str(np.round(noiseMetricBH.T,1)))
#     fig.tight_layout(w_pad=3.5)
#     plt.subplots_adjust(top=0.85)
    
#     #rh
#     ax[0, 1].plot(t, x[:, 1, 0], 'k', linewidth=2) 
#     ax2 = ax[0, 1].twinx()
#     ax2.plot(t, y[:, 1, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')

#     ax[1, 1].plot(t, x[:, 0, 0], 'k', linewidth=2)
#     ax2 = ax[1, 1].twinx()
#     ax2.plot(t, y[:, 0, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lh
#     ax[0, 0].plot(t, x[:, 1, 1], 'k', linewidth=2)
#     ax2 = ax[0, 0].twinx()
#     ax2.plot(t, y[:, 1, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[1, 0].plot(t, x[:, 0, 1], 'k', linewidth=2)
#     ax2 = ax[1, 0].twinx()
#     ax2.plot(t, y[:, 0, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #rv
#     ax[2, 1].plot(t, x[:, 1, 2], 'k', linewidth=2)
#     ax2 = ax[2, 1].twinx()
#     ax2.plot(t, y[:, 1, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[3, 1].plot(t, x[:, 0, 2], 'k', linewidth=2)
#     ax2 = ax[3, 1].twinx()
#     ax2.plot(t, y[:, 0, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lv
#     ax[2, 0].plot(t, x[:, 1, 3], 'k', linewidth=2) 
#     ax2 = ax[2, 0].twinx()
#     ax2.plot(t, y[:, 1, 3], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[3, 0].plot(t, x[:, 0, 3], 'k', linewidth=2)
#     ax2 = ax[3, 0].twinx()
#     ax2.plot(t, y[:, 0, 3], 'r', linewidth=1) 
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[0, 0].set_title('Left')
#     ax[0, 1].set_title('Right')
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Surface')
#     ax[1, 0].set_ylabel('Horizontal')
#     ax[2, 0].set_ylabel('Surface')
#     ax[3, 0].set_ylabel('Vertical')
    
#     if saveimages:
#         plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
# def plotResults_bilat(x, y, t, legend, scanname, saveimages, savename, noiseMetricBH, heartRate, crop=0):
    
#     if crop>0:
#         x=x[crop:-crop, :, :]
#         y=y[crop:-crop, :, :]
#         t=t[crop:-crop]
    
#     s=x.shape #(t,c,f)
#     # acquisition  position          (c,f)     subplot
#     # 0,           left horizontal   (1,0)     [0, 0]
#     #              right horitontal  (0,0)     [0, 1]
#     # 1,           left near         (0,1)     [1, 0]
#     #              right near        (1,1)     [1, 1]
#     # 2,           left vertical     (1,2)     [2, 0]   
#     #              right vertical    (0,2)     [2, 1]
#     # 3,           left horizontal   (1,3)     [3, 0]
#     #              right horizontal  (0,3)     [3, 1]
    
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
    
#     fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
#     #noiseMetricBH = np.round(noiseMetricBH.mean().T,1)
#     #noiseMetricBH = []
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1] + '\n HR: ' + str(int(heartRate)) +\
#          'bpm (T=' + str(np.round(60/heartRate,1)) + 'sec)   ---   Avg. BHnoiseMetric: \n' + str(np.round(noiseMetricBH.T,1)))
#     fig.tight_layout(w_pad=3.5)
#     plt.subplots_adjust(top=0.85)
    
#     #rh
#     ax[0, 0].plot(t, x[:, 1, 0], 'k', linewidth=2) 
#     ax2 = ax[0, 0].twinx()
#     ax2.plot(t, y[:, 1, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')

#     ax[0, 1].plot(t, x[:, 0, 0], 'k', linewidth=2)
#     ax2 = ax[0, 1].twinx()
#     ax2.plot(t, y[:, 0, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lh
#     ax[1, 0].plot(t, x[:, 0, 1], 'k', linewidth=2)
#     ax2 = ax[1, 0].twinx()
#     ax2.plot(t, y[:, 0, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[1, 1].plot(t, x[:, 1, 1], 'k', linewidth=2)
#     ax2 = ax[1, 1].twinx()
#     ax2.plot(t, y[:, 1, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #rv
#     ax[2, 0].plot(t, x[:, 1, 2], 'k', linewidth=2)
#     ax2 = ax[2, 0].twinx()
#     ax2.plot(t, y[:, 1, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[2, 1].plot(t, x[:, 0, 2], 'k', linewidth=2)
#     ax2 = ax[2, 1].twinx()
#     ax2.plot(t, y[:, 0, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lv
#     ax[3, 0].plot(t, x[:, 1, 3], 'k', linewidth=2) 
#     ax2 = ax[3, 0].twinx()
#     ax2.plot(t, y[:, 1, 3], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[3, 1].plot(t, x[:, 0, 3], 'k', linewidth=2)
#     ax2 = ax[3, 1].twinx()
#     ax2.plot(t, y[:, 0, 3], 'r', linewidth=1) 
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[0, 0].set_title('Left')
#     ax[0, 1].set_title('Right')
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Horizontal')
#     ax[1, 0].set_ylabel('Surface')
#     ax[2, 0].set_ylabel('Vertical')
#     ax[3, 0].set_ylabel('Horizontal')
    
#     if saveimages:
#         plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
# def plotResults_bilat2(x, y, t, legend, scanname, saveimages, savename, crop=0):
    
#     if crop>0:
#         x=x[crop:-crop, :, :]
#         y=y[crop:-crop, :, :]
#         t=t[crop:-crop]
    
#     s=x.shape #(t,c,f)
#     # acquisition  position          (c,f)     subplot
#     # 0,           left horizontal   (1,0)     [0, 0:1]
#     #              right horitontal  (0,0)     [0, 0:1
#     # 1,           left near         (0,1)     [1, 0:1]
#     #              right near        (1,1)     [1, 0:1]
#     # 2,           left vertical     (1,2)     [2, 0:1]   
#     #              right vertical    (0,2)     [2, 0:1]
#     # 3,           left horizontal   (1,3)     [3, 0:1]
#     #              right horizontal  (0,3)     [3, 0:1]
    
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
#     fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
#     fig.tight_layout(w_pad=3.5)
#     plt.subplots_adjust(top=0.85)
#     #rh
#     ax[0, 0].plot(t, x[:, 1, 0], 'k', linewidth=2) 
#     ax2 = ax[0, 0].twinx()
#     ax2.plot(t, x[:, 0, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')

#     ax[0, 1].plot(t, y[:, 1, 0], 'k', linewidth=2)
#     ax2 = ax[0, 1].twinx()
#     ax2.plot(t, y[:, 0, 0], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lh
#     ax[1, 0].plot(t, x[:, 0, 1], 'k', linewidth=2)
#     ax2 = ax[1, 0].twinx()
#     ax2.plot(t, x[:, 1, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[1, 1].plot(t, y[:, 0, 1], 'k', linewidth=2)
#     ax2 = ax[1, 1].twinx()
#     ax2.plot(t, y[:, 1, 1], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #rv
#     ax[2, 0].plot(t, x[:, 1, 2], 'k', linewidth=2)
#     ax2 = ax[2, 0].twinx()
#     ax2.plot(t, x[:, 0, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[2, 1].plot(t, y[:, 1, 2], 'k', linewidth=2)
#     ax2 = ax[2, 1].twinx()
#     ax2.plot(t, y[:, 0, 2], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
#     #lv
#     ax[3, 0].plot(t, x[:, 1, 3], 'k', linewidth=2) 
#     ax2 = ax[3, 0].twinx()
#     ax2.plot(t, x[:, 0, 3], 'r', linewidth=1)
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[3, 1].plot(t, y[:, 1, 3], 'k', linewidth=2)
#     ax2 = ax[3, 1].twinx()
#     ax2.plot(t, y[:, 0, 3], 'r', linewidth=1) 
#     ax2.tick_params(axis='y', colors='red')
    
#     ax[0, 0].set_title('Contrast (R=red, L=black)')
#     ax[0, 1].set_title('Image Mean (R=red, L=black)')
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Horizontal')
#     ax[1, 0].set_ylabel('Surface')
#     ax[2, 0].set_ylabel('Vertical')
#     ax[3, 0].set_ylabel('Horizontal')
    
#     if saveimages:
#         plt.savefig(savename + '2.png',dpi=300,bbox_inches='tight')

# def plotResults1axis(x, y, t, legend, scanname, saveimages, savename, crop=0):
    
#     if crop>0:
#         x=x[crop:-crop, :, :]
#         y=y[crop:-crop, :, :]
#         t=t[crop:-crop]
    
#     s=x.shape #(t,c,f)
#     # acquisition  position          (c,f)     subplot
#     # 0,           right horizontal  (1,0)     [0, 1]
#     #                                (0,0)     [1, 1]
#     # 1,           left horizontal   (1,1)     [0, 0]
#     #                                (0,1)     [1, 0]
#     # 2,           right vertical    (1,2)     [2, 1]   
#     #                                (0,2)     [3, 1]
#     # 3,           left vertical     (1,3)     [2, 0]
#     #                                (0,3)     [3, 0]
    
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
    
#     fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth,9))
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
#     #x -= np.mean(x, axis=0)
#     #y -= np.mean(y, axis=0)
    
#     #rh
#     ax[0, 1].plot(t, x[:, 1, 0], 'k', linewidth=1) 
#     ax[0, 1].plot(t, y[:, 1, 0], 'r', linewidth=1)
    
#     ax[1, 1].plot(t, x[:, 0, 0], 'k', linewidth=1)
#     ax[1, 1].plot(t, y[:, 0, 0], 'r', linewidth=1)
    
#     #lh
#     ax[0, 0].plot(t, x[:, 1, 1], 'k', linewidth=1)
#     ax[0, 0].plot(t, y[:, 1, 1], 'r', linewidth=1)
    
#     ax[1, 0].plot(t, x[:, 0, 1], 'k', linewidth=1)
#     ax[1, 0].plot(t, y[:, 0, 1], 'r', linewidth=1)

#     #rv
#     ax[2, 1].plot(t, x[:, 1, 2], 'k', linewidth=1)
#     ax[2, 1].plot(t, y[:, 1, 2], 'r', linewidth=1)
    
#     ax[3, 1].plot(t, x[:, 0, 2], 'k', linewidth=1)
#     ax[3, 1].plot(t, y[:, 0, 2], 'r', linewidth=1)

#     #lv
#     ax[2, 0].plot(t, x[:, 1, 3], 'k', linewidth=1) 
#     ax[2, 0].plot(t, y[:, 1, 3], 'r', linewidth=1)
    
#     ax[3, 0].plot(t, x[:, 0, 3], 'k', linewidth=1)
#     ax[3, 0].plot(t, y[:, 0, 3], 'r', linewidth=1) 

    
#     ax[0, 0].set_title('Left')
#     ax[0, 1].set_title('Right')
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Surface')
#     ax[1, 0].set_ylabel('Horizontal')
#     ax[2, 0].set_ylabel('Surface')
#     ax[3, 0].set_ylabel('Vertical')
    
#     if saveimages:
#         plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')




# def plotResultsRightLeft(x, y, t, legend, scanname, saveimages, savename):
    
#     s=x.shape #(t,c,f)
    
        
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
    
#     fig, ax = plt.subplots(nrows=s[2], ncols=s[1], figsize=(figwidth, 9))
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
#     #hn
#     ax[0, 0].plot(t, x[:, 1, 0], 'k') 
#     ax[0, 0].plot(t, x[:, 1, 1], 'b')
#     ax[0, 1].plot(t, y[:, 1, 0], 'r')
#     ax[0, 1].plot(t, y[:, 1, 1], 'm')
#     #hf
#     ax[1, 0].plot(t, x[:, 0, 0], 'k') 
#     ax[1, 0].plot(t, x[:, 0, 1], 'b')
#     ax[1, 1].plot(t, y[:, 0, 0], 'r')
#     ax[1, 1].plot(t, y[:, 0, 1], 'm')
#     #vn
#     ax[2, 0].plot(t, x[:, 1, 2], 'k') 
#     ax[2, 0].plot(t, x[:, 1, 3], 'b')
#     ax[2, 1].plot(t, y[:, 1, 2], 'r')
#     ax[2, 1].plot(t, y[:, 1, 3], 'm')
#     #vf
#     ax[3, 0].plot(t, x[:, 0, 2], 'k') 
#     ax[3, 0].plot(t, x[:, 0, 3], 'b')
#     ax[3, 1].plot(t, y[:, 0, 2], 'r')
#     ax[3, 1].plot(t, y[:, 0, 3], 'm')

#     ax[0, 0].set_title(legend[0])
#     ax[0, 1].set_title(legend[1])
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Near Hor')
#     ax[1, 0].set_ylabel('Far Hor')
#     ax[2, 0].set_ylabel('Near Ver')
#     ax[3, 0].set_ylabel('Far Ver')

#     for i in range(s[2]):
#         for j in range(s[1]):
#             ax[i, j].legend(('Right', 'Left'))

#     if saveimages:
#         plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
# def plotResultsRightLeft_bilat(x, y, t, legend, scanname, saveimages, savename):
    
#     s=x.shape #(t,c,f)
    
#     # acquisition  position          (c,f)     subplot
#     # 0,           left horizontal   (1,0)     [0, 0]
#     #              right horitontal  (0,0)     [0, 1]
#     # 1,           left near         (0,1)     [1, 0]
#     #              right near        (1,1)     [1, 1]
#     # 2,           left vertical     (1,2)     [2, 0]   
#     #              right vertical    (0,2)     [2, 1]
#     # 3,           left horizontal   (1,3)     [3, 0]
#     #              right horizontal  (0,3)     [3, 1]
        
#     figwidth = 18
#     if len(t)<100:
#         figwidth = 9
    
#     fig, ax = plt.subplots(nrows=s[2], ncols=s[1], figsize=(figwidth, 9))
#     fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
#     #hn
#     ax[0, 0].plot(t, x[:, 1, 0], 'k') 
#     ax[0, 0].plot(t, x[:, 0, 0], 'b')
#     ax[0, 1].plot(t, y[:, 1, 0], 'r')
#     ax[0, 1].plot(t, y[:, 0, 0], 'm')
#     #hf
#     ax[1, 0].plot(t, x[:, 0, 1], 'k') 
#     ax[1, 0].plot(t, x[:, 1, 1], 'b')
#     ax[1, 1].plot(t, y[:, 0, 1], 'r')
#     ax[1, 1].plot(t, y[:, 1, 1], 'm')
#     #vn
#     ax[2, 0].plot(t, x[:, 1, 2], 'k') 
#     ax[2, 0].plot(t, x[:, 0, 2], 'b')
#     ax[2, 1].plot(t, y[:, 1, 2], 'r')
#     ax[2, 1].plot(t, y[:, 0, 2], 'm')
#     #vf
#     ax[3, 0].plot(t, x[:, 1, 3], 'k') 
#     ax[3, 0].plot(t, x[:, 0, 3], 'b')
#     ax[3, 1].plot(t, y[:, 1, 3], 'r')
#     ax[3, 1].plot(t, y[:, 0, 3], 'm')

#     ax[0, 0].set_title(legend[0])
#     ax[0, 1].set_title(legend[1])
#     ax[3, 0].set_xlabel('Time (s)')
#     ax[3, 1].set_xlabel('Time (s)')
    
#     ax[0, 0].set_ylabel('Far Hor')
#     ax[1, 0].set_ylabel('Near Hor')
#     ax[2, 0].set_ylabel('Far Ver')
#     ax[3, 0].set_ylabel('Far Hor')

#     for i in range(s[2]):
#         for j in range(s[1]):
#             ax[i, j].legend(('Left', 'Right'))

#     if saveimages:
#         plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')

def plotResultsNearFar(x, y, t, legend, scanname, saveimages, savename, crop=0):
    
    if crop>0:
        x=x[crop:-crop, :, :]
        y=y[crop:-crop, :, :]
        t=t[crop:-crop]
    
    s=x.shape #(t,c,f)
    linthick = 1
        
    figwidth = 18
    if len(t)<100:
        figwidth = 9
    
    fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth, 9))
    fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
    #rh
    ax[0, 1].plot(t, x[:, 1, 0], 'k',linewidth=linthick) 
    ax[0, 1].plot(t, x[:, 0, 0], 'b',linewidth=linthick)
    ax[1, 1].plot(t, y[:, 1, 0], 'r',linewidth=linthick)
    ax[1, 1].plot(t, y[:, 0, 0], 'm',linewidth=linthick)
    #lh
    ax[0, 0].plot(t, x[:, 1, 1], 'k',linewidth=linthick) 
    ax[0, 0].plot(t, x[:, 0, 1], 'b',linewidth=linthick)
    ax[1, 0].plot(t, y[:, 1, 1], 'r',linewidth=linthick)
    ax[1, 0].plot(t, y[:, 0, 1], 'm',linewidth=linthick)
    #rv
    ax[2, 1].plot(t, x[:, 1, 2], 'k',linewidth=linthick) 
    ax[2, 1].plot(t, x[:, 0, 2], 'b',linewidth=linthick)
    ax[3, 1].plot(t, y[:, 1, 2], 'r',linewidth=linthick)
    ax[3, 1].plot(t, y[:, 0, 2], 'm',linewidth=linthick)
    #lv
    ax[2, 0].plot(t, x[:, 1, 3], 'k',linewidth=linthick) 
    ax[2, 0].plot(t, x[:, 0, 3], 'b',linewidth=linthick)
    ax[3, 0].plot(t, y[:, 1, 3], 'r',linewidth=linthick)
    ax[3, 0].plot(t, y[:, 0, 3], 'm',linewidth=linthick)
    
    ax[0, 0].set_title('Left')
    ax[0, 1].set_title('Right')
    ax[3, 0].set_xlabel('Time (s)')
    ax[3, 1].set_xlabel('Time (s)')
    
    ax[0, 0].set_ylabel(legend[0] + ' H')
    ax[1, 0].set_ylabel(legend[1] + ' H')
    ax[2, 0].set_ylabel(legend[0] + ' V')
    ax[3, 0].set_ylabel(legend[1] + ' V')
    
    for i in range(s[2]):
        for j in range(s[1]):
            ax[i, j].legend(('Near', 'Far'))

    if saveimages:
        plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
        

def plotResultsNearFarBilat(x, y, t, legend, scanname, saveimages, savename, bilat, crop=0):
    
    # if crop>0:
    #     x=x[crop:-crop, :, :]
    #     y=y[crop:-crop, :, :]
    #     t=t[crop:-crop]
    
    x = sortCamModTimetrace(x,bilat)
    y = sortCamModTimetrace(y,bilat)
    
    # valData = np.tile(np.array([[1,2,3,4],[5,6,7,8]]),(600,1,1))
    # x = valData/10
    # y = valData*10
    freq = 40
        
    # x = copy.deepcopy(1-contrast2)
    # y = copy.deepcopy(1-mean2)
    # legend = ['cont','mean']
    # near = 0
    # vert = 2
    # plt.plot(t[150:450],contrast2[150:450,0,near])
    # plt.plot(t[150:450]+(delay/freq),contrast2[150:450,0,vert])
    # t_win = t[150:450]+(delay/freq/2)
    # plt.xlim(t_win[0],t_win[-1])
    # plt.show()
    
    s=x.shape #(t,c,f)
    linthick = 1
        
    figwidth = 18
    if len(t)<100:
        figwidth = 9
    
    fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth, 9))
    fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
    
    delay1,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,0], y=x[:,0,1], detrend=True)
    ax[0, 0].plot(t+delay1/freq, x[:, 0, 1], 'k',linewidth=linthick)
    ax[0, 0].plot(t, x[:, 0, 0], 'b',linewidth=linthick) 
    ax[0, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    delay2,b,c,d = biosppy.signals.tools.synchronize(x=y[:,1,0], y=y[:,1,1], detrend=True)
    ax[0, 1].plot(t+delay2/freq, x[:, 1, 1], 'k',linewidth=linthick)
    ax[0, 1].plot(t, x[:, 1, 0], 'b',linewidth=linthick)
    ax[0, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,0], y=x[:,0,1], detrend=True)
    ax[1, 0].plot(t+delay1/freq, y[:, 0, 1], 'r',linewidth=linthick)
    ax[1, 0].plot(t, y[:, 0, 0], 'm',linewidth=linthick) 
    ax[1, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,0], y=x[:,1,1], detrend=True)
    ax[1, 1].plot(t+delay2/freq, y[:, 1, 1], 'r',linewidth=linthick)
    ax[1, 1].plot(t, y[:, 1, 0], 'm',linewidth=linthick)
    ax[1, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    delay1,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,2], y=x[:,0,1], detrend=True)
    ax[2, 0].plot(t+delay1/freq, x[:, 0, 1], 'k',linewidth=linthick)
    ax[2, 0].plot(t, x[:, 0, 2], 'b',linewidth=linthick) 
    ax[2, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    delay2,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,2], y=x[:,1,1], detrend=True)
    ax[2, 1].plot(t+delay2/freq, x[:, 1, 1], 'k',linewidth=linthick)
    ax[2, 1].plot(t, x[:, 1, 2], 'b',linewidth=linthick)
    ax[2, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,2], y=x[:,0,1], detrend=True)
    ax[3, 0].plot(t+delay1/freq, y[:, 0, 1], 'r',linewidth=linthick)
    ax[3, 0].plot(t, y[:, 0, 2], 'm',linewidth=linthick) 
    ax[3, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,2], y=x[:,1,1], detrend=True)
    ax[3, 1].plot(t+delay2/freq, y[:, 1, 1], 'r',linewidth=linthick)
    ax[3, 1].plot(t, y[:, 1, 2], 'm',linewidth=linthick)
    ax[3, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    # #rh
    # ax[0, 0].plot(t, x[:, 0, 0], 'k',linewidth=linthick) 
    # ax[0, 0].plot(t, x[:, 0, 1], 'b',linewidth=linthick)
    # ax[0, 1].plot(t, x[:, 1, 0], 'k',linewidth=linthick)
    # ax[0, 1].plot(t, x[:, 1, 1], 'b',linewidth=linthick)
    # #lh
    # ax[1, 0].plot(t, y[:, 0, 0], 'r',linewidth=linthick) 
    # ax[1, 0].plot(t, y[:, 0, 1], 'm',linewidth=linthick)
    # ax[1, 1].plot(t, y[:, 1, 0], 'r',linewidth=linthick)
    # ax[1, 1].plot(t, y[:, 1, 1], 'm',linewidth=linthick)
    # #rv
    # ax[2, 0].plot(t, x[:, 0, 2], 'k',linewidth=linthick) 
    # ax[2, 0].plot(t, x[:, 0, 1], 'b',linewidth=linthick)
    # ax[2, 1].plot(t, x[:, 1, 2], 'k',linewidth=linthick)
    # ax[2, 1].plot(t, x[:, 1, 1], 'b',linewidth=linthick)
    # #lv
    # ax[3, 0].plot(t, y[:, 0, 2], 'r',linewidth=linthick) 
    # ax[3, 0].plot(t, y[:, 0, 1], 'm',linewidth=linthick)
    # ax[3, 1].plot(t, y[:, 1, 2], 'r',linewidth=linthick)
    # ax[3, 1].plot(t, y[:, 1, 1], 'm',linewidth=linthick)
    
    ax[0, 0].set_title('Left')
    ax[0, 1].set_title('Right')
    ax[3, 0].set_xlabel('Time (s)')
    ax[3, 1].set_xlabel('Time (s)')
    
    ax[0, 0].set_ylabel(legend[0] + ' H')
    ax[1, 0].set_ylabel(legend[1] + ' H')
    ax[2, 0].set_ylabel(legend[0] + ' V')
    ax[3, 0].set_ylabel(legend[1] + ' V')
    
    for i in range(s[2]):
        for j in range(s[1]):
            ax[i, j].legend(('Near', 'Far'))

    if saveimages:
        plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')
        
def plotResultsHorVerBilat(x, y, t, legend, scanname, saveimages, savename, bilat, crop=0):
    
    # if crop>0:
    #     x=x[crop:-crop, :, :]
    #     y=y[crop:-crop, :, :]
    #     t=t[crop:-crop]
    
    x = sortCamModTimetrace(x,bilat)
    y = sortCamModTimetrace(y,bilat)
    
    # valData = np.tile(np.array([[1,2,3,4],[5,6,7,8]]),(600,1,1))
    # x = valData/10
    # y = valData*10
    freq = 40
        
    # x = copy.deepcopy(1-contrast2)
    # y = copy.deepcopy(1-mean2)
    # legend = ['cont','mean']
    # near = 0
    # vert = 2
    # plt.plot(t[150:450],contrast2[150:450,0,near])
    # plt.plot(t[150:450]+(delay/freq),contrast2[150:450,0,vert])
    # t_win = t[150:450]+(delay/freq/2)
    # plt.xlim(t_win[0],t_win[-1])
    # plt.show()
    
    s=x.shape #(t,c,f)
    linthick = 1
        
    figwidth = 18
    if len(t)<100:
        figwidth = 9
    
    fig, ax = plt.subplots(nrows=4, ncols=s[1], figsize=(figwidth, 9))
    fig.suptitle(scanname + '   ' + legend[0] + '   ' + legend[1])
    
    
    delay1,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,0], y=x[:,0,2], detrend=True)
    ax[0, 0].plot(t+delay1/freq, x[:, 0, 2], 'k',linewidth=linthick)
    ax[0, 0].plot(t, x[:, 0, 0], 'b',linewidth=linthick) 
    ax[0, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    delay2,b,c,d = biosppy.signals.tools.synchronize(x=y[:,1,0], y=y[:,1,2], detrend=True)
    ax[0, 1].plot(t+delay2/freq, x[:, 1, 2], 'k',linewidth=linthick)
    ax[0, 1].plot(t, x[:, 1, 0], 'b',linewidth=linthick)
    ax[0, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,0], y=x[:,0,1], detrend=True)
    ax[1, 0].plot(t+delay1/freq, y[:, 0, 2], 'r',linewidth=linthick)
    ax[1, 0].plot(t, y[:, 0, 0], 'm',linewidth=linthick) 
    ax[1, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,0], y=x[:,1,1], detrend=True)
    ax[1, 1].plot(t+delay2/freq, y[:, 1, 2], 'r',linewidth=linthick)
    ax[1, 1].plot(t, y[:, 1, 0], 'm',linewidth=linthick)
    ax[1, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    # delay1,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,2], y=x[:,0,1], detrend=True)
    # ax[2, 0].plot(t+delay1/freq, x[:, 0, 1], 'k',linewidth=linthick)
    # ax[2, 0].plot(t, x[:, 0, 2], 'b',linewidth=linthick) 
    # ax[2, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    # delay2,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,2], y=x[:,1,1], detrend=True)
    # ax[2, 1].plot(t+delay2/freq, x[:, 1, 1], 'k',linewidth=linthick)
    # ax[2, 1].plot(t, x[:, 1, 2], 'b',linewidth=linthick)
    # ax[2, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    # #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,0,2], y=x[:,0,1], detrend=True)
    # ax[3, 0].plot(t+delay1/freq, y[:, 0, 1], 'r',linewidth=linthick)
    # ax[3, 0].plot(t, y[:, 0, 2], 'm',linewidth=linthick) 
    # ax[3, 0].set_xlim((t[-1]+abs(delay1/freq))/2-3.5,(t[-1]+abs(delay1/freq))/2+3.5)
    # #delay,b,c,d = biosppy.signals.tools.synchronize(x=x[:,1,2], y=x[:,1,1], detrend=True)
    # ax[3, 1].plot(t+delay2/freq, y[:, 1, 1], 'r',linewidth=linthick)
    # ax[3, 1].plot(t, y[:, 1, 2], 'm',linewidth=linthick)
    # ax[3, 1].set_xlim((t[-1]+abs(delay2/freq))/2-3.5,(t[-1]+abs(delay2/freq))/2+3.5)
    
    ax[0, 0].set_title('Left')
    ax[0, 1].set_title('Right')
    ax[3, 0].set_xlabel('Time (s)')
    ax[3, 1].set_xlabel('Time (s)')
    
    ax[0, 0].set_ylabel(legend[0] + ' H')
    ax[1, 0].set_ylabel(legend[1] + ' H')
    ax[2, 0].set_ylabel(legend[0] + ' V')
    ax[3, 0].set_ylabel(legend[1] + ' V')
    
    for i in range(s[2]):
        for j in range(s[1]):
            ax[i, j].legend(('Vert', 'Hor'))

    if saveimages:
        plt.savefig(savename + '.png',dpi=300,bbox_inches='tight')

def vertNormWaveform(x, trofs_loc, peaks_loc):

    #get bottom peaks
    # trofs_loc = find_peaks(-x[minPeakLoc:], distance=minPeriod)[0] + minPeakLoc
    # if trofs_loc.shape[0]<2:
    #     return x
    trofs_val = x[trofs_loc]
    f = interp1d(trofs_loc, trofs_val, bounds_error=False, fill_value=(trofs_val[0], trofs_val[-1]))
    valleyRidgeline = f(np.arange(len(x)))
    valleyRidgeline[:trofs_loc[0]]=trofs_val[0]
    valleyRidgeline[trofs_loc[-1]:]=trofs_val[-1]
    dataZeroBottom = x - valleyRidgeline
    
    # p = find_peaks(-x, distance=d)[0]
    # if len(p)<2:
    #     return x
    # z = x[p]
    # f = interp1d(p, z, bounds_error=False, fill_value='extrapolate')
    # #f = interp1d(p, z, bounds_error=False, fill_value=0)
    # ff = f(np.arange(len(x)))
    # ff[:p[0]]=z[0]
    # ff[p[-1]:]=z[-1]
    # x -= ff
    
    #get top peaks
    # peaks_loc = find_peaks(dataZeroBottom, distance=minPeriod)[0]
    # Finds peaks between all troughs
    #   important to re-find because peaks_loc from dataIn aren't always the maxes after normalization process
    peaks_loc = []
    for trofInd in range(len(trofs_loc)-1):
        peaks_loc.append(np.nanargmax(x[trofs_loc[trofInd]:trofs_loc[trofInd+1]])+trofs_loc[trofInd])
    # if len(peaks_loc)<2:
    #     return dataZeroBottom
    peaks_val = dataZeroBottom[peaks_loc]
    f = interp1d(peaks_loc, peaks_val, bounds_error=False, fill_value=(peaks_val[0], peaks_val[-1]))
    peakRidgeline = f(np.arange(len(dataZeroBottom)))
    peakRidgeline[:peaks_loc[0]]=peaks_val[0]
    peakRidgeline[peaks_loc[-1]:]=peaks_val[-1]
    if np.count_nonzero(peakRidgeline) != peakRidgeline.shape[0]:
        np.where(peakRidgeline == 0)[0]
        print('Warning')
    dataVertNorm = dataZeroBottom/peakRidgeline
    
    # p = find_peaks(x, distance=d)    
    # p = p[0]
    # z = x[p]
    # g = interp1d(p, z, bounds_error=False, fill_value='extrapolate')
    # #g = interp1d(p, z, bounds_error=False, fill_value=0)
    # gg = g(np.arange(len(x)))
    # gg[:p[0]]=z[0]
    # gg[p[-1]:]=z[-1]
    # m = np.mean(gg)
    # x *= m/gg
    # x += np.mean(ff)
    
    return dataVertNorm


def flattenbottom(x, d):
    
    # Finds troughs of each pulse, fits a line connecting all troughs, then subtracts it from waveform
    
    p = find_peaks(-x, distance=d)
    trofs_loc = p[0]
    if len(trofs_loc)<2:
        return x
    trofs_val = x[trofs_loc]
    f = interp1d(trofs_loc, trofs_val, bounds_error=False, fill_value=(trofs_val[0], trofs_val[-1]))
    valleyRidgeline = f(np.arange(len(x)))
    x = x - valleyRidgeline
    
    return x
    
def containwaveform2(x, period):
    # x: taken as raw contrast (not inverted)
    
    d = np.floor(0.75*period)
    p1 = find_peaks(-x, distance=d)
    p1 = p1[0]
    z1 = x[p1]
    p2 = find_peaks(x, distance=d)
    p2 = p2[0]
    z2 = x[p2]
    if len(p1)<2 or len(p2)<2:
        meanAmplitude = 0
    else:
        meanAmplitude = z2.mean() - z1.mean()
    #print(meanAmplitude)

    d = np.floor(0.75*period)
    x = flattenbottom(x, d)
    x = flattenbottom(x, d)

    x[x<0]=0
    #get top peaks
    p = find_peaks(x, distance=d)
    p = p[0]
    if len(p)<2:
        return x, meanAmplitude
    z = x[p]
    
    g = interp1d(p, z, bounds_error=False, fill_value=(z[0], z[-1]))
    gg = g(np.arange(len(x)))
    gg[:p[0]]=z[0]
    gg[p[-1]:]=z[-1]
    x /= gg
    x[x>1]=1
    
    #x *= np.mean(z)
    #x += m
    # print('HR from Peaks: ' + str(40/np.array([np.diff(p1).mean(),np.diff(p2).mean()]).mean()*60))
    
    return x, meanAmplitude

def platinumPeriod(x, y, period):
    
    period = getperiod(x).mean()
    minPeriod = int(np.floor(period*0.75))
    
    xpeaks = find_peaks(-x, distance=minPeriod)[0]
    ypeaks = find_peaks(-y, distance=minPeriod)[0]
    
    xWavelets = []
    yWavelets = []
    for pkInd in range(len(xpeaks)-1):
        xWavelets.append(x[xpeaks[pkInd]:(xpeaks[pkInd+1]+1)])
    for pkInd in range(len(ypeaks)-1):
        yWavelets.append(y[ypeaks[pkInd]:(ypeaks[pkInd+1]+1)])
        
    # for i in range(len(xWavelets)):
    #     plt.plot(xWavelets[i])
    
    return xpeaks, ypeaks, xWavelets, yWavelets

def platinumPulse(x, y, period):
    # Updated to Soren's V6 platinumPulse version on 3/15/2023
    # takes in vertically normalized contrast (not inverted), returns inverted version
    
    period = getperiod(x)
    
    ### get dividing pts
    xp = x[2:]-x[:-2]  #use derivative
    minPeriod = int(np.floor(period-1))
    peaks = find_peaks(xp**2, distance=minPeriod)[0] 
    lengths = peaks[1:]-peaks[:-1]

    ### get segments: only pick ones that are close < +/-0.5 of the median length
    medianlength = mode(lengths)[0]#np.median(lengths)
    maxlength = medianlength+0.5
    minlength = medianlength-0.5
    L = int(maxlength+4)
    usesegment = ( (minlength<=np.array(lengths)).astype(int) + (maxlength>=np.array(lengths)).astype(int) ) == 2
    nsegments =int(np.sum(usesegment))
    if nsegments<2:
        return np.zeros((30,)), np.zeros((30,))
        
    #print('nsegments = ' + str(nsegments))
    #print('medianlength = ' + str(medianlength))
    #print('lengths =')
    #print(lengths)
    xsegments = np.zeros((L, nsegments))
    ysegments = np.zeros((L, nsegments))
    ind=0
    order = np.argsort(lengths)
    for j in range(len(lengths)):
        jj = order[j]
        
        if lengths[jj]==medianlength:
            t1 = peaks[jj]-2
            t2 = peaks[jj+1]+2
            if t1>=0 and t2<=(len(x)-1):
                xsegments[:, ind] = x[t1:t2]
                ysegments[:, ind] = y[t1:t2]
                ind += 1
        """
        if lengths[jj]==maxlength:
            t1 = peaks[jj]-2
            t2 = peaks[jj+1]+2
            if t1>=0 and t2<=(len(x)-1):
                xsegments[:, ind] = x[t1:t2]
                ysegments[:, ind] = y[t1:t2]
                ind += 1        
        elif lengths[jj]==medianlength:
            t1 = peaks[jj]-2
            t2 = peaks[jj+1]+2
            if t1>=0 and t2<=(len(x)-1):
                xsegments[:, ind] = x[t1:t2]
                ysegments[:, ind] = y[t1:t2]
                ind += 1        
        elif lengths[jj]==minlength:
            t1 = peaks[jj]-3
            t2 = peaks[jj+1]+2
            if t1>=0 and t2<=(len(x)-1):
                xsegments[:, ind] = x[t1:t2]
                ysegments[:, ind] = y[t1:t2]
                ind += 1
        """
    #plt.figure()
    #plt.plot(xsegments,'k')
    #plt.plot(ysegments,'r')
    
    ### subpixel alignment
    movingInd = -1
    moving = xsegments[2:-2, movingInd] #align everything to the last one
    ncc = np.zeros((5,))
    alignedxsegments = np.zeros((len(moving), nsegments))
    alignedysegments = np.zeros((len(moving), nsegments))
    #alignedsegments[:, movingInd] = moving
    xx=np.arange(L)

    for j in range(nsegments):
        fixed = xsegments[:, j]
        for k in range(5): #from -2 to 2
            out = pearsonr(fixed[k:k+len(moving)], moving)
            ncc[k]=out[0]
        indmax = np.argmax(ncc)
        if indmax>0 and indmax<4:
            a = 0.5*(ncc[indmax-1]-2*ncc[indmax]+ncc[indmax+1])
            b = 0.5*(ncc[indmax+1]-ncc[indmax-1])
            if np.abs(a)>np.abs(b):  #not sure if this is the correct condition
                dk = -b/a/2
            else:
                print('Not able to align waveforms !!!!!!!!!!!!!!!!!!')
                dk = 0
        else:
            dk = 0
        kmax = indmax + dk

        f = interp1d(xx, fixed, kind='linear')
        xnew = np.arange(kmax, kmax+len(moving))
        if len(xnew)>alignedxsegments.shape[0]:
            xnew = xnew[:-1]
        alignedxsegments[:, j] = f(xnew)
        
        yfixed = ysegments[:, j]
        f = interp1d(xx, yfixed, kind='linear')
        alignedysegments[:, j] = f(xnew)

    xmean = np.mean(alignedxsegments, axis=1)
    ymean = np.mean(alignedysegments, axis=1)
    
    #plt.figure()
    #plt.plot(alignedysegments, color=[1, 0.7, 0.7])
    #plt.plot(ymean, 'r')
    #plt.plot(alignedxsegments, color=[0.7, 0.7, 0.7])
    #plt.plot(xmean, 'k')
    
    while xmean[-1]>xmean[-2]:
        xmean = np.roll(xmean, 1)
        ymean = np.roll(ymean, 1)
    while xmean[0]>xmean[1]:
        xmean = np.roll(xmean, -1)
        ymean = np.roll(ymean, -1)
    
    return xmean, ymean

def ComputeVCI(goldenPulseIn,samplingIn,hanningFilter=False):
    # Yoinked from Kedar's ReadGen2Data.py
    '''
        Computes the velocity curve index for the input waveform

        Parameters
        ----------
        goldenPulseIn : 1D numpy array
            Input goldenPulse
        hanningFilter : bool
            If true, apply a hanning filter to the input waveform

        Returns
        -------
        canopyVCI : float
            Velocity curve index for the canopy
        pulseLengthNormalizedCanopyVCI : float
            Velocity curve index for the canopy normalized by the pulse length
    '''
    #Strech pulse between 0-1 (vertically)
    goldenPulse = copy.deepcopy(goldenPulseIn)
    goldenPulse -= np.amin(goldenPulse)
    goldenPulse /= np.amax(goldenPulse)
    goldenPulse = 1-goldenPulse
    gPIndex = np.arange(len(goldenPulse))*samplingIn
    
    #We use 4x oversampling to get a smoother gradient
    gPIndexUpSamp = np.arange(len(goldenPulse)*4)*samplingIn/4
    gPUpSample = np.interp(gPIndexUpSamp, gPIndex, goldenPulse)
    if hanningFilter:
        #90 ms window is the length we are shooting for
        window = round(0.09/(samplingIn/4))
        gPUpSample = signal.convolve(gPUpSample,signal.hann(window)/np.sum(signal.hann(window)),'same')
    grad  = np.gradient(gPUpSample,samplingIn/4)
    grad2 = np.gradient(grad,samplingIn/4)
    k = np.abs(grad2)/((1+grad**2)**1.5)
    indsInCanopy = gPUpSample > 0.25
    indsInCanopy = np.flatnonzero(indsInCanopy)
    canopyVCI = np.sum(k[indsInCanopy])
    
    return canopyVCI, canopyVCI/len(gPIndexUpSamp)

def padnans(x, N):
    #BHadd updated to not break when x[i][j] is > N
    
    ls = np.zeros((2, 4))
    for i in range(2):
        for j in range(4):
            ls[i, j] = len(x[i][j])
    M = int(np.maximum(N, np.amax(ls)))+1
    M = N + 1
    #print(N)    
    out = np.nan*np.zeros((M, 2, 4))
    for i in range(2):
        for j in range(4):
            out[:len(x[i][j][0:N]), i, j] = x[i][j][0:N]
            # out[int(ls[i, j])+1, i, j] = x[i][j][0]
            out[len(x[i][j][0:N]), i, j] = x[i][j][0]
            
    return out, M

def getperiod(x):

    N=1024
    N1 = int(N/256)
    N2 = int(N/2)
    m = np.mean(x, axis = 0)
    f = np.abs(np.fft.fft(x-m, n=N, axis=0))
    t = np.arange(1024)
    freq = np.fft.fftfreq(t.shape[-1])*40
    xaxis2 = np.array([0,1,2,3,4,5,10]) # tick labels to display
    if len(f.shape)==3:
        f[N2:, :, :] = 0
        f[:N1, :, :] = 0
        # BHadd
        # fig, ax = plt.subplots(nrows=4, ncols=2)
        # for camInd in range(2):
        #     for modInd in range(4):
        #         ax[modInd,camInd].plot(freq[:250],f[:250,camInd,modInd])
        #         ax[modInd,camInd].set_xticks(xaxis2)
        # ax[3,0].set_xlabel('Freq. (Hz)')
        # ax[3,1].set_xlabel('Freq. (Hz)')
    elif len(f.shape)==1:
        f[N2:] = 0
        f[:N1:] = 0
    ind = np.argmax(f, axis=0)
    period = N/(ind+1)
    # if len(f.shape)==3:
        # fig.suptitle(str(ind.T) + '\n' + str(np.median(ind[:, 1])))
        # period = np.median(period[:, 1]) # BHadd, switched to only #BHadd
    #print('Period = ' + np.array2string(period) + ' exposures')
    
    return period

def getderivatives(x):
    
    d1=np.zeros(x.shape)
    d2=np.zeros(x.shape)
    
    d1[1:-1, :, :] = x[2:, :, :] - x[:-2, :, :]
    d2[1:-1, :, :] = x[2:, :, :] - 2*x[1:-1, :, :] + x[:-2, :, :]
    
    d1[:50, :, :] = 0#d1[1, :, :]
    d2[:50, :, :] = 0#d2[1, :, :]
    
    d1[-50:, :, :] = 0#d1[:, :, -2]
    d2[-50:, :, :] = 0#d2[:, :, -2]
       
    return d1, d2
    
def fithistogram(histData, mean, gl, mean_dark, var_dark):
    
    flag = 2
    #setting   e-/adu   noise (e-)
    #1         8.7      1.67
    #16        0.58     6.3
    cam = 0
    folder = 0
    if cam==0:
        gain = 0.58
        #readnoise = 6.3
    elif cam == 1:        
        gain      = 8.7
        #readnoise = 1.67
    
    test = histData[10, :, cam, folder]  #(N, 1024, camera, Nfolders)
    u = mean[100, cam, folder]  #mean in GL
    offset = mean_dark[cam, folder]
    readnoise = np.sqrt(var_dark[cam, folder])
    
    # shift histogram by offset
    gl_temp = gl-offset
    gl = np.round(gl_temp)
    f = interp1d(gl_temp, test, bounds_error=False, fill_value='extrapolate')
    #test = f(gl)
    
    #get range of DN in histogram
    inds = np.where(test>0)[0]
    a = int(inds[0])
    b = int(inds[-1])
    D = gl[a:b]
    
    # crop histogram to match list of DN
    test = test[a:b]
    test /= np.sum(test)
    
    #calculate range of possible e- numbers
    a = np.floor(D[0]*gain - 2*readnoise)
    b = np.ceil(D[-1]*gain + 2*readnoise)
    K = np.arange(a, b)  
    
    e2d = ADUmatrix2(K, gain, readnoise, D)
    plt.figure()
    plt.pcolormesh(K, D, e2d)
    plt.gca().invert_yaxis()
    plt.ylabel('DN')
    plt.xlabel('e-')
    print(K)
      
    #meanK = u*gain  
    bins  = [1]
    probs = [1]
    M0 = 40
    meanK0 = u*gain
    print('meanK0 = ')
    print(meanK0)
    

    if flag == 1:
        guess = (M0)
        out = least_squares(residual_histogram, guess, bounds=(2, np.inf), args=(meanK0, test, e2d, K, gain, readnoise, bins, probs) )
        M = out.x
        meanK = meanK0
    elif flag == 2:
        guess = (M0, meanK0)
        out = least_squares(residual_histogram2, guess, bounds=(2, np.inf), args=(test, e2d, K, gain, readnoise, bins, probs) )
        M = out.x[0]
        meanK = out.x[1]
    pdsim = getP_D(M, meanK, K, bins, probs, e2d)
    """
    if N>1:
        for i in range(N):
            out = least_squares(residual_histogram, M, bounds=(1, np.inf), args=(u[i], histData[i]) )
            b[i] = out.x[0]
    else:
        out = least_squares(residual_histogram, M, bounds=(1, np.inf), args=(u[i], histData[i]) )
        b = out.x[0]
    """
    """ 
    plt.figure()
    plt.plot(D, out.fun)
    """
    plt.figure()
    plt.plot(D, pdsim)
    plt.plot(D, test)
    plt.legend('Simulation', 'Data')
    
    return M, meanK



def residual_histogram(M, meanK, data, e2d, K, gain, readnoise, bins, probs):
    
    return getP_D(M, meanK, K, bins, probs, e2d) - data

def residual_histogram2(guess, data, e2d, K, gain, readnoise, bins, probs):
    
    M = guess[0]
    meanK = guess[1]

    return getP_D(M, meanK, K, bins, probs, e2d) - data


def getP_D(M, meanK, K, bins, probs, e2d):
    
    P_K = np.zeros((len(K),))
    for j in range(len(bins)): 
        P_temp = negativeBinomial(K, bins[j]*meanK, M)
        P_K += probs[j]*np.array(P_temp)
    P_D = e2d @ P_K # Go from e- to DN, and list of digital values

    return P_D

def negativeBinomial(K, meanK, M):
    
    part1 = comb(K+M-1, K)
    part2 = (1+(M/meanK))**(-K)
    part3 = (1+(meanK/M))**(-M)

    return part1*part2*part3    

def ADUmatrix2(K, gain, s, D):
    
    #K: vector of number of electrons 
    #gain: e-/DN
    #s: read noise in e-
    #D: vector of digital numbers
    
    KK, dd = np.meshgrid(K, D)
    if s==0:
        diff = KK/gain - dd
        e2d = np.greater_equal(diff, -0.5) * np.less(diff, 0.5)
    else:
        high = ((dd+0.5)*gain - KK)/(np.sqrt(2)*s)
        low  = ((dd-0.5)*gain - KK)/(np.sqrt(2)*s)
        e2d  = 0.5*erf(high) - 0.5*erf(low)

    return e2d





        