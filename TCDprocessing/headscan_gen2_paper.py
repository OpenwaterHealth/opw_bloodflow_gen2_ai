#%% Set which sections to run

processCont     = True
processPulses   = True
processAgPulses = True # Must be set to True to do peak analysis plots
processPlots    = False

scanRange = [1,2,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] # full paper

#%% Start up
import copy,csv,re,pickle,scipy
import batchfilePaper
import matplotlib
import headscan_gen2_fcns_paper as fcns
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from matplotlib.lines import Line2D

# %matplotlib qt
plt.close('all')

moduleData = [ # (cam/mod/timePoint)
    'moduleData_timeRaw',
    'moduleData_meanRaw',
    'moduleData_rBVIRaw',
    'moduleData_contRaw',
    'moduleData_rBFIRaw',
    
    'moduleData_time',
    'moduleData_timeAbsolute', # calculated after merging
    
    'moduleData_mean',
    'moduleData_rBVI',
    'moduleData_rBVI_vertNorm',
    'moduleData_rBVI_avg',
    'moduleData_rBVI_amp',
    'moduleData_rBVI_period',
    
    'moduleData_cont',
    'moduleData_rBFI',
    'moduleData_rBFI_vertNorm',
    'moduleData_rBFI_avg',
    'moduleData_rBFI_amp',
    'moduleData_rBFI_period',
    'moduleData_rBFI_pulsesDataStats', # currently called rBfiDataStats (x17 currently)
    ]

pulsesData = [ # (cam/mod/pulse)
    'pulsesData_mean_starts',
    'pulsesData_mean_toUse',
    'pulsesData_mean_time',
    'pulsesData_mean',
    'pulsesData_rBVI',
    'pulsesData_rBVI_vertNorm',
    'pulsesData_rBVI_vertHorNorm',
    'pulsesData_rBVI_avg',
    'pulsesData_rBVI_amp',
    'pulsesData_rBVI_period',
    'pulsesData_rBVI_stats',
    
    'pulsesData_cont_starts',
    'pulsesData_cont_toUse',
    'pulsesData_cont_time',
    'pulsesData_cont',
    'pulsesData_rBFI',
    'pulsesData_rBFI_vertNorm',
    'pulsesData_rBFI_vertHorNorm',
    'pulsesData_rBFI_avg',
    'pulsesData_rBFI_amp',
    'pulsesData_rBFI_period',
    'pulsesData_rBFI_stats', # currently a x17 array
    ]
    
agPulseData = [ # (cam/mod)
    'agPulse_rBVI_timeVertNorm',
    'agPulse_rBVI_vertNorm',
    'agPulse_rBVI_timeVertHorNorm',
    'agPulse_rBVI_vertHorNorm', # pulsesVertHorNormAvg_parts
    
    'agPulse_rBFI_timeVertNorm',
    'agPulse_rBFI_vertNorm',
    'agPulse_rBFI_timeVertHorNorm',
    'agPulse_rBFI_vertHorNorm', # pulsesVertHorNormAvg_parts
    'agPulse_rBFI_vertHorNorm_prePostPeaks',
    ]
    
mergedLongScanData = [ # (cam)
    'mergedData_time',
    
    'mergedData_mean',
    'mergedData_mean_avg',
    'mergedData_rBVI',
    'mergedData_rBVI_vertNorm',
    'mergedData_rBVI_avg',
    'mergedData_rBVI_amp',
    'mergedData_rBVI_stats',
    
    'mergedData_cont',
    'mergedData_cont_avg',
    'mergedData_rBFI',
    'mergedData_rBFI_vertNorm',
    'mergedData_rBFI_avg',
    'mergedData_rBFI_amp',
    'mergedData_rBFI_stats', # currently called rBfiDataStats (x17 currently)
    'moduleData_rBFI_hr',
    ]

batch = True
waveform = True
saveImages = True
plotImages = False

smoothData = True
bilat = 0 # 0=unilat, 1=bilat, 2=long
varCor = 0 # 0=no dark hist, 1=has dark hist, 2=long scans w/o dark hist
processNewSubject = False

histChUse = ['scan_ch_0','scan_ch_1'] # ['scan_ch_0','scan_ch_1'] ['scan_ch_0','scan_ch_2']
imgChUse = ['ch_0','ch_1'] #['ch_0','ch_1'] ['ch_0','ch_2']

# experimental constants
freqOpt = 40   
freqTCD = 125
freqOut = 125
bins = np.arange(0, 1025)-0.5 # edges of bins in histogram
gl = bins[1:] - 0.5 # grey level values in bins
titles = ['Right horizontal', 'Left Horizontal', 'Right Vertical', 'Left Vertical'] #labels for scan order
if bilat == 0:
    gain = np.array([[16, 16, 16, 16],[1, 1, 1, 1]])
    obmax = np.array([[256, 256, 256, 256],[128, 128, 128, 128]])
if bilat == 1:
    gain = np.array([[16, 1, 16, 16],[16, 1, 16, 16]])
    obmax = np.array([[256, 128, 256, 256],[256, 128, 256, 256]])

w, h = 2720, 1450 # camera dimensions
histCutoff = 100 # set histogram bin with < this # of pixels to zero

shortNames,scanNames,bilatTypes,varCorTypes,lsTypes,tcdNames,tcdMarks,tcdMarksOffset,peakLocs,trofLocs = batchfilePaper.LongScan_breathHoldPaper()

batchName = 'LongScan_breathHoldPaper'
scanPath = '/Users/brad/Desktop/gen2 data/TCD'
savePath = '/Users/brad/Desktop/gen2 results/' + batchName

numMod = 4
# if batchName[0:8] == 'LongScan' or bilatTypes[0] == 2:
#     fnames0, fnames1 = fcns.getImgNames(scanPath + '/' + scanNames[scanInd] + '/', 'bayer.y',['ch_0','ch_1'])
#     numMod = int(len(fnames0)/2)
numCam = 2
numHis = 600

# Code to load TCD file for first time to get tcdMarks
if processNewSubject:
    # up to CVRhBH05
    file = open(scanPath + '/' + tcdNames[0])
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    rows = np.array(rows)
    notes = rows[:,18] 
    markCsv = []
    for rowInd in range(len(notes)):
        if notes[rowInd] != '-':
            print([rowInd,notes[rowInd]])
            markCsv.append([rowInd,notes[rowInd]])
          
    # after CVRhBH05
    file = open(scanPath + '/' + tcdNames[26])
    csvreader = csv.reader(file, delimiter = '\t')
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    rows = np.array(rows)
    notes = rows[:,4]
    markCsv = []
    for rowInd in range(len(notes)):
        if notes[rowInd] != '':
            markCsv.append([rowInd,notes[rowInd]])
    marksPrint = []
    marksNamesPrint = []
    for ind in range(len(markCsv)):
        marksPrint.append(markCsv[ind][0])
        marksNamesPrint.append(markCsv[ind][1])
    print(marksPrint)
    print(marksNamesPrint)

numFeats = 16
rows, cols = (len(scanNames), numFeats)
p1p2ratio = np.zeros((len(scanNames),15))
peakTiming = np.zeros((len(scanNames),15,5))
p2p1diffAmp = np.ones((len(scanNames),15))

#%% processCont
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
if processCont:
    scans = [[] for rowInd in range(len(scanNames))]
    for scanInd in scanRange: #range(0,len(scanNames)):
        fnames0, fnames1 = fcns.getImgNames(scanPath + '/' + scanNames[scanInd] + '/', 'bayer.y',['ch_0','ch_1'])
        numMod = int(len(fnames0)/2)
        
        if lsTypes[scanInd] == 3:
            histChUse = ['scan_ch_0','scan_ch_2']
            imgChUse = ['ch_0','ch_2']
        else:
            histChUse = ['scan_ch_0','scan_ch_1']
            imgChUse = ['ch_0','ch_1']
        
        if batch==True:
            scanName = scanPath + '/' + scanNames[scanInd]
            
            #BHadd
            if len(bilatTypes) == 1:
                bilatTypes = np.ones((len(scanNames)))*np.array(bilatTypes)
            elif len(bilatTypes) > 1:
                bilatTypes = np.array(bilatTypes)
            bilat = bilatTypes[scanInd]
            gain = np.array([[16, 16, 16, 16],[1, 1, 1, 1]])
            obmax = np.array([[256, 256, 256, 256],[128, 128, 128, 128]])
            if bilat == 1:
                gain = np.array([[16, 1, 16, 16],[16, 1, 16, 16]])
                obmax = np.array([[256, 128, 256, 256],[256, 128, 256, 256]])
            
            if len(varCorTypes) == 1:
                varCorTypes = np.ones((len(scanNames)))*np.array(varCorTypes)
            elif len(varCorTypes) > 1:
                varCorTypes = np.array(varCorTypes)
            # flag 2 works for all data typees and uses dark images for variance correction
            # flag 6 works for all data types and uses interpolation of dark image and dark frames for each acquisition
            # flag 5 works for long scans without dark histogram frames
            varCor = varCorTypes[scanInd]
        if varCor == 0:
            flag = 2
        elif varCor == 1:
            flag = 6
        elif varCor == 2:
            flag = 5
        
        print('Scan Number: ' + str(scanInd) + ' ' + shortNames[scanInd])
        print(scanName)
        
        savename = savePath + '/' + re.split('/', scanName)[-2] + '_' + re.split('/', scanName)[-1]
    
        # patient data
        if batchName[0:8] == 'LongScan' or bilat == 2:
            histData, meanStdData, camTempsData, numHis = fcns.loadHistograms3(scanName,'histo_output_long',histChUse)
        else:
            histData, meanStdData, camTempsData, numHis = fcns.loadHistograms3(scanName,'histo_output_full',histChUse)
        
        # BHadd 
        fnames0, fnames1 = fcns.getImgNames(scanPath + '/' + scanNames[scanInd] + '/', 'histo_output_dark',['ch_0','ch_1'])
        histDataDark = []
        meanStdDataDark = []
        if len(fnames0) > 0:
            histDataDark, meanStdDataDark, camTempsDataDark  = fcns.loadHistograms3(scanName,'histo_output_dark',histChUse)[0:3]
            histDataDark[histDataDark<histCutoff]=0
            histDataDark[:, 1023, :, :]=0
        t = np.arange(numHis)/freqOpt
        meanStdDataSmooth = fcns.smoothMeanStdData(meanStdData, t) 
        fnames0, fnames1 = fcns.getImgNames(scanName + '/', 'bayer.y',['ch_0','ch_1'])
        numMod = int(len(fnames0)/2)
        if len(fnames0) == 16:
            expN = [0,2]
        elif len(fnames0) == 8:
            expN = [0,1]
        elif len(fnames0) == numMod*2:
            expN = [0,1]
            gain = np.ones((2,numMod))*16
            obmax = np.ones((2,numMod))*256
        else:
            print('Number of images found not 8 or 16')
        
        histData[histData<histCutoff]=0
        histData[:, 1023, :, :]=0
        imgStats_dark = fcns.getImageStats3(scanName, w, h, bins, obmax, expN[0], imgChUse)
        imgStats_bright = fcns.getImageStats3(scanName, w, h, bins, obmax, expN[1], imgChUse)
        mean, contrast, stats = fcns.hist2contrast5(histData, histDataDark, imgStats_dark, gl, gain, numHis, meanStdDataSmooth, meanStdDataDark, histCutoff,flag)
        numHis = mean.shape[0]
        # Overview of processing steps
        # loadHistograms3 (no processing, meanStdData are from the hist calcs output)
        # getImageStats3 (only removes values above 128/256, needs to filter 1023 and <100 count bins)
        # hist2contrast5
        
        #M = fcns.fithistogram(histData, mean, gl, imgStats_dark['mean_ob'], imgStats_dark['mean_ob'])
        
        if batchName[0:8] == 'LongScan':
            contrast_mean = np.tile(contrast[:,:,0].mean(axis=0),(numHis,numMod,1))
            contrast_mean = np.transpose(contrast_mean, (0, 2, 1))
            mean_mean = np.tile(mean[:,:,0].mean(axis=0),(numHis,numMod,1))
            mean_mean = np.transpose(mean_mean, (0, 2, 1))
        else:
            contrast_mean = np.tile(contrast.mean(axis=0),(numHis,1,1))
            mean_mean = np.tile(mean.mean(axis=0),(numHis,1,1))
        rBFI = contrast_mean-contrast
        rBVI = mean_mean-mean
        
        np.set_printoptions(precision=2, suppress=True)
        print('PreScan Images')    
        print('Mean light level with laser off. i.e. stray light')    
        print(imgStats_dark['mean_main'] - imgStats_dark['mean_ob'])
        print('Mean light level with laser on')
        print(imgStats_bright['mean_main_all'] - imgStats_bright['mean_ob'])
        
        # code to correct for drift / spurious pulses in process
        # may make it easer to see waveform morphology & timing
        # then get representative pulse for each camera    
        mean2 = copy.deepcopy(mean)
        contrast2 = copy.deepcopy(contrast)
        periodAvg = fcns.getperiod(contrast2)
        mean1 = [[0] * numMod, [0] * numMod]
        contrast1 = [[0] * numMod, [0] * numMod]
        meanAmp_timeaverage = np.zeros(obmax.shape)
        contrastAmp_timeaverage = np.zeros(obmax.shape)
        
        #BHadd
        bviPeaks = [[0] * numMod, [0] * numMod]
        bfiPeaks = [[0] * numMod, [0] * numMod]
        bviWavelets = [[0] * numMod, [0] * numMod]
        bfiWavelets = [[0] * numMod, [0] * numMod]
        rBfiAvg = copy.deepcopy(rBFI)*np.nan
        rBviAvg = copy.deepcopy(rBVI)*np.nan
        rBfiAmp = copy.deepcopy(rBFI)*np.nan
        rBviAmp = copy.deepcopy(rBVI)*np.nan
        rBfiPeriod = copy.deepcopy(mean)*np.nan
        rBviPeriod = copy.deepcopy(mean)*np.nan
        rPeriod = copy.deepcopy(mean)*np.nan # heart rate in BPM = freq/batch_period*60
        contAvg = copy.deepcopy(contrast)*np.nan
        meanAvg = copy.deepcopy(mean)*np.nan
        
        rBfiStats = np.zeros((numHis, numCam, 15,20))*np.nan
        
        if waveform:
            for modInd in range(numMod):
                for camInd in range(numCam):
                    temp = gaussian_filter1d(mean2[:, camInd, modInd], 1)
                    mean2[:, camInd, modInd], meanAmp_timeaverage[camInd, modInd] = fcns.containwaveform2(temp, periodAvg[camInd, modInd])
                    contrast2[:, camInd, modInd], contrastAmp_timeaverage[camInd, modInd] = fcns.containwaveform2(contrast2[:, camInd, modInd], periodAvg[camInd, modInd])
                    
                    contrast1[camInd][modInd], mean1[camInd][modInd] = fcns.platinumPulse(1-contrast2[:,camInd,modInd],1-mean2[:,camInd,modInd], periodAvg[:,modInd].mean(axis=0))   
                    bfiPeaks[camInd][modInd], bviPeaks[camInd][modInd], bfiWavelets[camInd][modInd], bviWavelets[camInd][modInd] = fcns.platinumPeriod(1-contrast2[:,camInd,modInd],1-mean2[:,camInd,modInd], periodAvg[:,modInd].mean(axis=0)) 
                
            contrastwaveform2plot, ntwaveform = fcns.padnans(contrast1, 60)
            meanwaveform2plot, ntwaveform = fcns.padnans(mean1, 60) 
            twaveform = np.arange(ntwaveform)/freqOpt
            
            rPeriod[:,0,:] = rBfiPeriod[:,:,:].mean(axis=1)
            rPeriod[:,1,:] = rBfiPeriod[:,:,:].mean(axis=1)
    
        #dcontrast, ddcontrast = fcns.getderivatives(contrast2)
        
        # BHadd (smoothing phantom data for better analysis)
        if smoothData:
            mean_smoothed = np.zeros((numHis,numCam,numMod))
            contrast_smoothed = np.zeros((numHis,numCam,numMod))
            for camInd in range(numCam):
                for modInd in range(numMod):
                    # Low pass filter (mirroring data method)
                    # soshp = signal.butter(2,0.1,'hp',fs=freq,output='sos')
                    # filtered = signal.sosfilt(soshp, np.concatenate((np.flip(mean[:,camInd,modInd]),mean[:,camInd,modInd])))
                    # mean_smoothed[:,camInd,modInd] = filtered[numHis:] + mean[:,camInd,modInd].mean()
                    # filtered = signal.sosfilt(soshp, np.concatenate((np.flip(contrast[:,camInd,modInd]),contrast[:,camInd,modInd])))
                    # contrast_smoothed[:,camInd,modInd] = filtered[numHis:] + contrast[:,camInd,modInd].mean()
                    
                    # Low pass filter (DC removal method)
                    soshp = signal.butter(2,1/10,'hp',fs=freqOpt,output='sos')
                    mean_smoothed[:,camInd,modInd] = signal.sosfilt(soshp, mean[:,camInd,modInd]-mean[:,camInd,modInd].mean())+mean[:,camInd,modInd].mean()
                    contrast_smoothed[:,camInd,modInd] = signal.sosfilt(soshp, contrast[:,camInd,modInd]-contrast[:,camInd,modInd].mean())+contrast[:,camInd,modInd].mean()
                    
                    # Polynomial fit
                    # polyFit = np.poly1d(np.polyfit(t, mean[:,camInd,modInd], 2))
                    # mean_smoothed[:,camInd,modInd] = mean[:,camInd,modInd] - polyFit(t) + mean[:,camInd,modInd].mean()
                    # polyFit = np.poly1d(np.polyfit(t, contrast[:,camInd,modInd], 2))
                    # contrast_smoothed[:,camInd,modInd] = contrast[:,camInd,modInd] - polyFit(t) + contrast[:,camInd,modInd].mean()
                    
                    # ax[modInd,camInd].plot(mean[:,camInd,modInd],linewidth=0.5)
                    # ax[modInd,camInd].plot(mean_smoothed[:,camInd,modInd],linewidth=0.5)
        heartRt = freqOpt/periodAvg.mean()*60
        
        
        normLength = int(np.round(numHis/freqOpt*freqTCD))
        scanArrayEmpty = np.zeros((normLength,numCam,numMod))*np.nan
        scanListEmpty = [[[] for colInd in range(numMod)] for rowInd in range(numCam)]
        camsEmpty = [[] for rowInd in range(numCam)]
        
        scans[scanInd] = {}
        for varInd in range(len(moduleData)):
            scans[scanInd][moduleData[varInd]] = copy.deepcopy(scanArrayEmpty)
        for varInd in range(len(pulsesData)):
            scans[scanInd][pulsesData[varInd]] = copy.deepcopy(scanListEmpty)
        # for varInd in range(len(agPulseData)):
        #     scans[scanInd][agPulseData[varInd]] = copy.deepcopy(scanListEmpty)
        for varInd in range(1,len(mergedLongScanData)): # skips time
            scans[scanInd][mergedLongScanData[varInd]] = copy.deepcopy(camsEmpty)
            
        meanRaw = copy.deepcopy(mean)
        rBviRaw = copy.deepcopy(rBVI)
        contrastRaw = copy.deepcopy(contrast)
        rBfiRaw = copy.deepcopy(rBFI)
        scans[scanInd]['moduleData_timeRaw'] = np.arange(numHis)/freqOpt
        scans[scanInd]['moduleData_meanRaw'] = meanRaw
        scans[scanInd]['moduleData_rBVIRaw'] = rBviRaw
        scans[scanInd]['moduleData_contRaw'] = contrastRaw
        scans[scanInd]['moduleData_rBFIRaw'] = rBfiRaw
        
        # Resampling pulse to TCD freq
        mean = np.zeros((normLength,numCam,numMod))
        rBVI = np.zeros((normLength,numCam,numMod))
        contrast = np.zeros((normLength,numCam,numMod))
        rBFI = np.zeros((normLength,numCam,numMod))
        tOut = []
        for camInd in range(numCam):
            for modInd in range(numMod):                
                mean[:,camInd,modInd],_ = fcns.resampleData(meanRaw[:,camInd,modInd],freqOpt,freqTCD)
                rBVI[:,camInd,modInd],_ = fcns.resampleData(rBviRaw[:,camInd,modInd],freqOpt,freqTCD)
                contrast[:,camInd,modInd],tOut = fcns.resampleData(contrastRaw[:,camInd,modInd],freqOpt,freqTCD)
                rBFI[:,camInd,modInd],_ = fcns.resampleData(rBfiRaw[:,camInd,modInd],freqOpt,freqTCD)
        scans[scanInd]['moduleData_time'] = tOut
        scans[scanInd]['moduleData_mean'] = mean
        scans[scanInd]['moduleData_rBVI'] = rBVI
        scans[scanInd]['moduleData_cont'] = contrast
        scans[scanInd]['moduleData_rBFI'] = rBFI
    
    userInput = input('WARNING: Overwriting allData_contMean file! Enter any character to cancel.')
    if userInput == '':
        print('Writing allData_contMean file')
        dataToPickle = {'scans': scans}
        f = open('allData_contMean.pkl', 'wb')
        pickle.dump(dataToPickle, f)
        f.close()
else:
    f = open('allData_contMean.pkl','rb')
    stupidDict = pickle.load(f)
    f.close()
    scans = stupidDict['scans']
    
#%% processPulses
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

dropsSubj = [10, 16, 19,] #  subjects that require manual TCD segmentation
dropsRng = [[[10940,12712]], [[10443,10733],[11860,12232],[12870,13100],[14112,14840]], [[10585,11305],[23197,23760]],]

if processPulses:
    for scanInd in scanRange: #range(0,len(scanNames)):
        print('Scan Number: ' + str(scanInd) + ' ' + shortNames[scanInd])
        
        # TEMPORARY, NEED TO REMOVE LOCAL VARIABLES AT SOME POINT
        mean = copy.deepcopy(scans[scanInd]['moduleData_mean'])
        rBVI = copy.deepcopy(scans[scanInd]['moduleData_rBVI'])
        contrast = copy.deepcopy(scans[scanInd]['moduleData_cont'])
        rBFI = copy.deepcopy(scans[scanInd]['moduleData_rBFI'])
        rBfiAvg = copy.deepcopy(rBFI)*np.nan
        rBviAvg = copy.deepcopy(rBVI)*np.nan
        rBfiAmp = copy.deepcopy(rBFI)*np.nan
        rBviAmp = copy.deepcopy(rBVI)*np.nan
        contAvg = copy.deepcopy(rBVI)*np.nan
        meanAvg = copy.deepcopy(rBVI)*np.nan
        numSam = copy.deepcopy(scans[scanInd]['moduleData_mean'].shape[0])
        rBfiStats = np.zeros((numSam, numCam, 15,20))*np.nan
        scanName = scanPath + '/' + scanNames[scanInd]
        
        subj_pulsesVertHorNorm = [[0 for colInd in range(rBFI.shape[2])] for rowInd in range(numCam)]
        subj_pulsesToUse = [[0 for colInd in range(rBFI.shape[2])] for rowInd in range(numCam)]
        
        for camInd in range(numCam):
            for modInd in range(scans[scanInd]['moduleData_mean'].shape[2]):
                plotOutputs = True
                plotTitle = shortNames[scanInd] + '_Cam' + str(camInd) + '_Mod' + str(modInd)
                
                # modInd = 14
                # print(camInd,modInd)
                dataVertNorm,pulsesStarts,pulsesToUse,pulses,pulsesVertNorm,pulsesVertHorNorm,pulsesStats,rBfiDataStats = fcns.processWaveforms(rBFI[:,camInd,modInd],freqTCD,0,freqOut,rBFI[:,:,modInd].mean(axis=1),plotTitle,plotOutputs)
                rBfiAvg[:,camInd,modInd] = rBfiDataStats[:,1]
                rBfiAmp[:,camInd,modInd] = rBfiDataStats[:,2]
                hRate = freqTCD/rBfiDataStats[:,0]*60
                rBfiStats[:,camInd,modInd,:] = rBfiDataStats
                subj_pulsesVertHorNorm[camInd][modInd] = pulsesVertHorNorm
                subj_pulsesToUse[camInd][modInd] = pulsesToUse
                _,_,_,_,_,_,_,contDataStats = fcns.processWaveforms(contrast[:,camInd,modInd],freqTCD,1,freqOut,rBFI[:,:,modInd].mean(axis=1),plotTitle,False)
                contAvg[:,camInd,modInd] = contDataStats[:,1]
                                
                scans[scanInd]['pulsesData_cont_starts'][camInd][modInd] = pulsesStarts
                scans[scanInd]['pulsesData_cont_toUse'][camInd][modInd] = pulsesToUse
                scans[scanInd]['pulsesData_cont_time'][camInd][modInd] = []
                scans[scanInd]['pulsesData_cont'][camInd][modInd] = []
                scans[scanInd]['pulsesData_rBFI'][camInd][modInd] = pulses
                scans[scanInd]['pulsesData_rBFI_vertNorm'][camInd][modInd] = pulsesVertNorm
                scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][modInd] = pulsesVertHorNorm
                scans[scanInd]['pulsesData_rBFI_avg'][camInd][modInd] = pulsesStats[:,1]
                scans[scanInd]['pulsesData_rBFI_amp'][camInd][modInd] = pulsesStats[:,2]
                scans[scanInd]['pulsesData_rBFI_period'][camInd][modInd] = pulsesStats[:,0]
                scans[scanInd]['pulsesData_rBFI_stats'][camInd][modInd] = pulsesStats
                
                scans[scanInd]['moduleData_rBFI_vertNorm'][:,camInd,modInd] = dataVertNorm
                scans[scanInd]['moduleData_rBFI_avg'][:,camInd,modInd] = rBfiDataStats[:,1]
                scans[scanInd]['moduleData_rBFI_amp'][:,camInd,modInd] = rBfiDataStats[:,2]
                scans[scanInd]['moduleData_rBFI_period'][:,camInd,modInd] = rBfiDataStats[:,0]
                #scans[scanInd]['moduleData_rBFI_pulsesDataStats'][:,camInd,modInd] = rBfiDataStats
                
                dataVertNorm,pulsesStarts,pulsesToUse,pulses,pulsesVertNorm,pulsesVertHorNorm,pulsesStats,rBviDataStats = fcns.processWaveforms(rBVI[:,camInd,modInd],freqTCD,0,freqOut,rBVI[:,:,modInd].mean(axis=1),plotTitle,False)
                rBviAvg[:,camInd,modInd] = rBviDataStats[:,1]
                rBviAmp[:,camInd,modInd] = rBviDataStats[:,2]
                _,_,_,_,_,_,_,meanDataStats = fcns.processWaveforms(mean[:,camInd,modInd],freqTCD,1,freqOut,rBVI[:,:,modInd].mean(axis=1),plotTitle,False)
                meanAvg[:,camInd,modInd] = meanDataStats[:,1]

                
                                
                scans[scanInd]['pulsesData_mean_starts'][camInd][modInd] = pulsesStarts
                scans[scanInd]['pulsesData_mean_toUse'][camInd][modInd] = pulsesToUse
                scans[scanInd]['pulsesData_mean_time'][camInd][modInd] = []
                scans[scanInd]['pulsesData_mean'][camInd][modInd] = []
                scans[scanInd]['pulsesData_rBVI'][camInd][modInd] = pulses
                scans[scanInd]['pulsesData_rBVI_vertNorm'][camInd][modInd] = pulsesVertNorm
                scans[scanInd]['pulsesData_rBVI_vertHorNorm'][camInd][modInd] = pulsesVertHorNorm
                scans[scanInd]['pulsesData_rBVI_avg'][camInd][modInd] = pulsesStats[:,1]
                scans[scanInd]['pulsesData_rBVI_amp'][camInd][modInd] = pulsesStats[:,2]
                scans[scanInd]['pulsesData_rBVI_stats'][camInd][modInd] = pulsesStats
                
                scans[scanInd]['moduleData_rBVI_vertNorm'][:,camInd,modInd] = dataVertNorm
                scans[scanInd]['moduleData_rBVI_avg'][:,camInd,modInd] = rBviDataStats[:,1]
                scans[scanInd]['moduleData_rBVI_amp'][:,camInd,modInd] = rBviDataStats[:,2]
                scans[scanInd]['moduleData_rBVI_period'][:,camInd,modInd] = rBviDataStats[:,0]
                # scans[scanInd]['moduleData_rBVI_pulsesDataStats'][:,camInd,modInd] = rBviDataStats
        periodAvg = np.nanmean(scans[scanInd]['moduleData_rBFI_period'],axis=0)
        
        # Module Scanning Time Calculation
        fnames0, fnames1 = fcns.getFileNames(scanName, 'histo_output_long',histChUse)
        histTimes = np.zeros((len(fnames0)))
        for modInd in range(len(fnames0)):
            histTimes[modInd] = float(fnames0[modInd][-8:-6])*60 + float(fnames0[modInd][-6:-4])
        histTimes = histTimes - histTimes[0]
        print('Average module scan time (s): ' + str(np.diff(histTimes).mean()))
        
        # Noise Metric
        noiseMetricBH1 = np.zeros((2,scans[scanInd]['moduleData_mean'].shape[2]))
        noiseMetricBH2 = np.zeros((2,scans[scanInd]['moduleData_mean'].shape[2]))
        # fig, ax = plt.subplots(nrows=scans[scanInd]['moduleData_mean'].shape[2], ncols=2)
        for camInd in range(numCam):
            for modInd in range(scans[scanInd]['moduleData_mean'].shape[2]):
                noiseMetricBH1[camInd,modInd], noiseMetricBH2[camInd,modInd], shoulderHeights = fcns.noiseMetricBH(contrast[:,camInd,modInd],(numSam/periodAvg[camInd,modInd])) # normalized by heart rate
                noiseMetricBH1[camInd,modInd] = noiseMetricBH1[camInd,modInd]/2/(numSam/periodAvg[camInd,modInd])
                #noiseMetricBH[camInd,modInd] = noiseMetricBH[camInd,modInd]/(numHisnumHis/period[camInd,modInd])-4 # offsetting by min of 4
                noiseMetricBH2[camInd,modInd] = noiseMetricBH2[camInd,modInd]/(numSam/periodAvg[camInd,modInd])*100
                
                if shoulderHeights.shape[0] > 200:
                    shoulderHeights = shoulderHeights[:200]
        #         ax[modInd,camInd].plot(np.pad(shoulderHeights, (0, 200-shoulderHeights.shape[0]), 'constant'))
        # fig.suptitle(noiseMetricBH1)
                
        # Converting to correct locations for bilat plots
        noiseMetricBH1 = fcns.sortCamMod(noiseMetricBH1,bilat)
        t = np.arange(numSam)/freqTCD
            
        # Generating merged data formats for optical
        contMerged,modStartTimes = fcns.longScanMergeResamp(contrast,freqTCD,freqTCD,lsTypes[scanInd])
        meanMerged,_    = fcns.longScanMergeResamp(mean,freqTCD,freqTCD,lsTypes[scanInd])
        contAvgMerged,_ = fcns.longScanMergeResamp(contAvg,freqTCD,freqTCD,lsTypes[scanInd])
        meanAvgMerged,_ = fcns.longScanMergeResamp(meanAvg,freqTCD,freqTCD,lsTypes[scanInd])
        rBfiVertNormMerged,_ = fcns.longScanMergeResamp(scans[scanInd]['moduleData_rBFI_vertNorm'],freqTCD,freqTCD,lsTypes[scanInd])
        rBviVertNormMerged,_ = fcns.longScanMergeResamp(scans[scanInd]['moduleData_rBVI_vertNorm'],freqTCD,freqTCD,lsTypes[scanInd])
        
        tOpt = np.arange(contMerged.shape[0])/freqTCD
        rBfiMerged,_    = fcns.longScanMergeResamp(rBFI,freqTCD,freqTCD,lsTypes[scanInd])
        rBfiAvgMerged,_ = fcns.longScanMergeResamp(rBfiAvg,freqTCD,freqTCD,lsTypes[scanInd])
        rBfiAmpMerged,_ = fcns.longScanMergeResamp(rBfiAmp,freqTCD,freqTCD,lsTypes[scanInd])
        rBviMerged,_    = fcns.longScanMergeResamp(rBVI,freqTCD,freqTCD,lsTypes[scanInd])
        rBviAvgMerged,_ = fcns.longScanMergeResamp(rBviAvg,freqTCD,freqTCD,lsTypes[scanInd])
        rBviAmpMerged,_ = fcns.longScanMergeResamp(rBviAmp,freqTCD,freqTCD,lsTypes[scanInd])
        hRate = freqTCD/scans[scanInd]['moduleData_rBFI_period']*60
        hRateMerged,_   = fcns.longScanMergeResamp(hRate,freqTCD,freqTCD,lsTypes[scanInd])
        
        rBfiStatsMerged,_ = fcns.longScanMergeResamp(rBfiStats[:,:,:,0],freqTCD,freqTCD,lsTypes[scanInd])
        rBfiStatsAllMerged = np.zeros((rBfiStatsMerged.shape[0],rBfiStatsMerged.shape[1],17))
        rBfiStatsAllMerged[:,:,0] = rBfiStatsMerged
        for statInd in range(1,17):
            rBfiStatsAllMerged[:,:,statInd],_ = fcns.longScanMergeResamp(rBfiStats[:,:,:,statInd],freqTCD,freqTCD,lsTypes[scanInd])
        
        for camInd in range(numCam):
            scans[scanInd]['mergedData_time'] = np.arange(contMerged.shape[0])/freqTCD
            
            scans[scanInd]['mergedData_mean'][camInd] = meanMerged[:,camInd]
            scans[scanInd]['mergedData_mean_avg'][camInd] = contAvgMerged[:,camInd]
            scans[scanInd]['mergedData_rBVI'][camInd] = rBviMerged[:,camInd]
            scans[scanInd]['mergedData_rBVI_vertNorm'][camInd] = rBviVertNormMerged[:,camInd]
            scans[scanInd]['mergedData_rBVI_avg'][camInd] = rBviAvgMerged[:,camInd]
            scans[scanInd]['mergedData_rBVI_amp'][camInd] = rBviAmpMerged[:,camInd]
            # scans[scanInd]['mergedData_rBVI_stats'][camInd] = []
        
            scans[scanInd]['mergedData_cont'][camInd] = contMerged[:,camInd]
            scans[scanInd]['mergedData_cont_avg'][camInd] = meanAvgMerged[:,camInd]
            scans[scanInd]['mergedData_rBFI'][camInd] = rBfiMerged[:,camInd]
            scans[scanInd]['mergedData_rBFI_vertNorm'][camInd] = rBfiVertNormMerged[:,camInd]
            scans[scanInd]['mergedData_rBFI_avg'][camInd] = rBfiAvgMerged[:,camInd]
            scans[scanInd]['mergedData_rBFI_amp'][camInd] = rBfiAmpMerged[:,camInd]
            scans[scanInd]['mergedData_rBFI_stats'][camInd] = rBfiStatsAllMerged[:,camInd,:]
            scans[scanInd]['moduleData_rBFI_hr'][camInd] = hRateMerged
                    
        scans[scanInd]['moduleData_time'] = np.zeros((numSam,numCam,scans[scanInd]['moduleData_mean'].shape[2]))
        for camInd in range(numCam):
            for modInd in range(scans[scanInd]['moduleData_mean'].shape[2]):
                scans[scanInd]['moduleData_time'][:,camInd,modInd] = np.arange(numSam)/freqTCD
                scans[scanInd]['moduleData_timeAbsolute'][:,camInd,modInd] = np.arange(numSam)/freqTCD + modStartTimes[camInd,modInd]/freqTCD
        
        
        
        # Generating merged data formats for TCD
        tcdMarks[scanInd][0] = tcdMarks[scanInd][0] + tcdMarksOffset[scanInd][0]
        tTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = fcns.loadTCDdata(scanPath,tcdNames[scanInd],tcdMarks[scanInd])
        meanTCDow, _, _, _ = fcns.getMetrics(envTCD,freqTCD,tcdNames[scanInd])
        
        plotTitle = shortNames[scanInd]
        dataTcdVertNorm,pulsesTcdStarts,pulsesTcdToUse,pulsesTcd,pulsesTcdVertNorm,pulsesTcdVertHorNorm,pulsesTcdStats,dataTcdStats = fcns.processWaveforms(envTCD,freqTCD,0,freqOut,envTCD,plotTitle,plotOutputs)
        meanTCDow = dataTcdStats[:,1]
        ampTCDow = dataTcdStats[:,2]
        periodTCDow = dataTcdStats[:,0]
        hRateTCD = freqTCD/periodTCDow*60
        
        if scanInd in dropsSubj:
            for dropInd in range(len(dropsRng[dropsSubj.index(scanInd)])):
                rng = np.array(dropsRng[dropsSubj.index(scanInd)][dropInd])
                meanTCDow[rng[0]:rng[1]] = np.nan
                ampTCDow[rng[0]:rng[1]] = np.nan
                hRateTCD[rng[0]:rng[1]] = np.nan
                print(shortNames[scanInd],' TCD dropped range: ',rng)
        
        if processNewSubject:
            timeDelay = fcns.calcTimeDelay(hRateTCD,hRateMerged[:,0],5000,0)
        
        scans[scanInd]['data_TCD_raw'] = []
        scans[scanInd]['data_TCD'] = envTCD
        scans[scanInd]['data_TCD_vertNorm'] = dataTcdVertNorm
        scans[scanInd]['data_TCD_avg'] = meanTCDow
        scans[scanInd]['data_TCD_amp'] = ampTCDow
        scans[scanInd]['data_TCD_period'] = periodTCDow
        scans[scanInd]['data_TCD_stats'] = dataTcdStats
        
        scans[scanInd]['pulsesData_TCD_starts'] = pulsesTcdStarts
        scans[scanInd]['pulsesData_TCD_toUse'] = pulsesTcdToUse
        scans[scanInd]['pulsesData_TCD_time'] = []
        scans[scanInd]['pulsesData_TCD'] = pulsesTcd
        scans[scanInd]['pulsesData_TCD_vertNorm'] = pulsesTcdVertNorm
        scans[scanInd]['pulsesData_TCD_vertHorNorm'] = pulsesTcdVertHorNorm
        scans[scanInd]['pulsesData_TCD_avg'] = pulsesTcdStats[:,1]
        scans[scanInd]['pulsesData_TCD_amp'] = pulsesTcdStats[:,2]
        scans[scanInd]['pulsesData_TCD_period'] = pulsesTcdStats[:,2]
        scans[scanInd]['pulsesData_TCD_stats'] = pulsesTcdStats
        
    userInput = input('WARNING: Overwriting allData_contMeanPulses file! Enter any character to cancel.')
    if userInput == '':
        print('Writing processCont file')
        dataToPickle = {'scans': scans}
        f = open('allData_contMeanPulses.pkl', 'wb')
        pickle.dump(dataToPickle, f)
        f.close()
else:
    f = open('allData_contMeanPulses.pkl','rb')
    stupidDict = pickle.load(f)
    f.close()
    scans = stupidDict['scans']
    
    

#%% processAgPulses
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

if processAgPulses:
    # scanListEmpty = [[[] for colInd in range(numMod)] for rowInd in range(numCam)]
    # for varInd in range(len(agPulseData)):
    #     scans[scanInd][agPulseData[varInd]] = copy.deepcopy(scanListEmpty)
    
    for scanInd in scanRange: #range(0,len(scanNames)):
        print('Scan Number: ' + str(scanInd) + ' ' + shortNames[scanInd])
        plotTitle = shortNames[scanInd]
        
        modStartTimes = copy.deepcopy(scans[scanInd]['moduleData_timeAbsolute'][0,:,:])
        
        baseWin = [-5,5] #[-30,0]
        postWin = [0,10] # sec pre post end of hold
        peakCutoffs = np.array([
                [0.000,0.000,0.000,0.000,0.000],
                [1.000,1.000,1.000,0.000,0.000],])
        camInd = 1
        holdInds = [1,2]
        
        if scanInd == 9:  #sub 5
            postWin = [-10,0]
        if scanInd == 10: #sub 6
            postWin = [-10,-5]
        if scanInd == 11: #sub 7
            postWin = [5,15]
        if scanInd == 14: #sub 10
            postWin = [-20,-15]
        if scanInd == 16: #sub 13
            holdInds = [3,4]
        tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
        ptsTCD = np.array([tcdMarksZerod[holdInds[0]],tcdMarksZerod[holdInds[1]]])
        
        if 0:
            # rBFI Ag Pulses
            pulsesVertHorNormAvg = np.zeros(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][0][0].shape[0])
            pulsesVertHorNorm_parts = []
            rngs = [ptsTCD[0] + baseWin[0]*freqTCD, ptsTCD[0] + baseWin[1]*freqTCD]
            for modInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd])):
                for pulseInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd][modInd])):
                    startTime = scans[scanInd]['pulsesData_cont_starts'][camInd][modInd][pulseInd] + modStartTimes[camInd,modInd]*freqTCD
                    usePulse = scans[scanInd]['pulsesData_cont_toUse'][camInd][modInd][pulseInd]
                    if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                        pulsesVertHorNormAvg = pulsesVertHorNormAvg + scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][modInd][pulseInd]
                        pulsesVertHorNorm_parts.append(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][modInd][pulseInd])
            pulsesVertHorNormAvg -= pulsesVertHorNormAvg.min()
            pulsesVertHorNormAvg /= pulsesVertHorNormAvg.max()
            print('pulseCount rBFI pre:' + str(len(pulsesVertHorNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertHorNormAvg,plotTitle + '_rBFI_baseline',peakLocsOpt[scanInd][0:2],pulsesVertHorNorm_parts,[],peakCutoffs)
            # peaksLocs,peaksVals,peaksLocsMan,peaksValsMan
            # p1p2ratio[scanInd,0] = peaksValsMan[1]/peaksValsMan[0]
            
            pulsesVertHorNormAvg = np.zeros(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][0][0].shape[0])
            pulsesVertHorNorm_parts = []
            rngs = [ptsTCD[1] + postWin[0]*freqTCD, ptsTCD[1] + postWin[1]*freqTCD]
            for modInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd])):
                for pulseInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd][modInd])):
                    startTime = scans[scanInd]['pulsesData_cont_starts'][camInd][modInd][pulseInd] + modStartTimes[camInd,modInd]*freqTCD
                    usePulse = scans[scanInd]['pulsesData_cont_toUse'][camInd][modInd][pulseInd]
                    if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                        pulsesVertHorNormAvg = pulsesVertHorNormAvg + scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][modInd][pulseInd]
                        pulsesVertHorNorm_parts.append(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][modInd][pulseInd])
            pulsesVertHorNormAvg -= pulsesVertHorNormAvg.min()
            pulsesVertHorNormAvg /= pulsesVertHorNormAvg.max()
            print('pulseCount rBFI post:' + str(len(pulsesVertHorNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertHorNormAvg,plotTitle + '_rBFI_postHold',peakLocsOpt[scanInd][2:4],pulsesVertHorNorm_parts,[],peakCutoffs,peakCutoffs)
            # p1p2ratio[scanInd,1] = peaksValsMan[1]/peaksValsMan[0]
            
            # TCD Ag Pulses
            pulsesVertHorNormAvg = np.zeros(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][0][0].shape[0])
            pulsesVertHorNorm_parts = []
            rngs = [ptsTCD[0] + baseWin[0]*freqTCD, ptsTCD[0] + baseWin[1]*freqTCD]
            for pulseInd in range(len(scans[scanInd]['pulsesData_TCD_starts'])):
                startTime = scans[scanInd]['pulsesData_TCD_starts'][pulseInd]
                usePulse = scans[scanInd]['pulsesData_TCD_toUse'][pulseInd]
                if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                    pulsesVertHorNormAvg = pulsesVertHorNormAvg + scans[scanInd]['pulsesData_TCD_vertHorNorm'][pulseInd]
                    pulsesVertHorNorm_parts.append(scans[scanInd]['pulsesData_TCD_vertHorNorm'][pulseInd])
            pulsesVertHorNormAvg -= pulsesVertHorNormAvg.min()
            pulsesVertHorNormAvg /= pulsesVertHorNormAvg.max()
            print('pulseCount TCD pre:' + str(len(pulsesVertHorNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertHorNormAvg,plotTitle + '_TCD_baseline',peakLocsTcd[scanInd][0:2],pulsesVertHorNorm_parts,[],peakCutoffs)
            # p1p2ratio[scanInd,2] = peaksValsMan[1]/peaksValsMan[0]
            
            pulsesVertHorNormAvg = np.zeros(scans[scanInd]['pulsesData_rBFI_vertHorNorm'][camInd][0][0].shape[0])
            pulsesVertHorNorm_parts = []
            rngs = [ptsTCD[1] + postWin[0]*freqTCD, ptsTCD[1] + postWin[1]*freqTCD]
            for pulseInd in range(len(scans[scanInd]['pulsesData_TCD_starts'])):
                startTime = scans[scanInd]['pulsesData_TCD_starts'][pulseInd]
                usePulse = scans[scanInd]['pulsesData_TCD_toUse'][pulseInd]
                if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                    pulsesVertHorNormAvg = pulsesVertHorNormAvg + scans[scanInd]['pulsesData_TCD_vertHorNorm'][pulseInd]
                    pulsesVertHorNorm_parts.append(scans[scanInd]['pulsesData_TCD_vertHorNorm'][pulseInd])
            pulsesVertHorNormAvg -= pulsesVertHorNormAvg.min()
            pulsesVertHorNormAvg /= pulsesVertHorNormAvg.max()
            print('pulseCount TCD post:' + str(len(pulsesVertHorNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertHorNormAvg,plotTitle + '_TCD_postHold',peakLocsTcd[scanInd][2:4],pulsesVertHorNorm_parts,[],peakCutoffs)
            # p1p2ratio[scanInd,3] = peaksValsMan[1]/peaksValsMan[0]
            
        
        
        
        
        maxPulLen = 1.6 # in seconds, zeros out data beyond this time point
        pulAlign = 30 # in time points, where left FWHM is aligned to
        peakCutoffs = np.array([
                [0.020,0.100,0.248,0.000,0.000],
                [0.100,0.248,0.398,0.000,0.000],]) + pulAlign/freqTCD
        
        # Vert only normalized average waveform
        pulsesVertNormShiftedBoth = []
        manualPointsBoth = []
        pulsesVertNorm_parts_newBoth = []
        pulsesVertNorm_parts_namesBoth = []
        
        for ind in range(2):
            if ind == 0:
                rngs = [ptsTCD[0] + baseWin[0]*freqTCD, ptsTCD[0] + baseWin[1]*freqTCD]
                peakInd = 0
                trofInd = 0
                saveStr = '_rBFI_vertNorm_baseline_NewMeth'
            else:
                rngs = [ptsTCD[1] + postWin[0]*freqTCD, ptsTCD[1] + postWin[1]*freqTCD]
                peakInd = 3
                trofInd = 2
                saveStr = '_rBFI_vertNorm_postHold_NewMeth'
            
            pulsesVertNormAvg = np.zeros((1,freqTCD*2))*np.nan
            pulCount = 0
            pulsesVertNorm_parts = []
            pulsesVertNormShifted = copy.deepcopy(pulsesVertNormAvg)
            pulsesVertNorm_parts_new = []
            pulsesVertNorm_parts_names = []
            
            for modInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd])):
                for pulseInd in range(len(scans[scanInd]['pulsesData_cont_starts'][camInd][modInd])):
                    startTime = scans[scanInd]['pulsesData_cont_starts'][camInd][modInd][pulseInd] + modStartTimes[camInd,modInd]*freqTCD
                    usePulse = scans[scanInd]['pulsesData_cont_toUse'][camInd][modInd][pulseInd]
                    if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                        pulsesVertNorm_parts.append(scans[scanInd]['pulsesData_rBFI_vertNorm'][camInd][modInd][pulseInd])   
                        pulsesVertNormShifted[pulCount,:] = fcns.alignLeftFwhm(scans[scanInd]['pulsesData_rBFI_vertNorm'][camInd][modInd][pulseInd],pulAlign,freqTCD)
                        pulsesVertNorm_parts_new.append(pulsesVertNormShifted[pulCount,:])
                        pulsesVertNorm_parts_names.append(['c'+str(camInd)+'_m'+str(modInd)+'_p'+str(pulseInd)])
                        pulsesVertNormAvg = np.append(pulsesVertNormAvg,np.zeros((1,pulsesVertNormAvg.shape[1]))*np.nan,axis=0)
                        pulsesVertNormShifted = np.append(pulsesVertNormShifted,np.zeros((1,pulsesVertNormAvg.shape[1]))*np.nan,axis=0)
                        pulCount += 1
            pulsesVertNorm_parts_newBoth.append(pulsesVertNorm_parts_new)
            pulsesVertNorm_parts_namesBoth.append(pulsesVertNorm_parts_names)
            
            pulsesVertNormAvgCount = np.sum(~np.isnan(pulsesVertNormAvg)*1,axis=0)
            pulsesVertNormAvg = np.nanmean(pulsesVertNormAvg,axis=0)
            pulsesVertNormAvg = pulsesVertNormAvg - np.nanmin(pulsesVertNormAvg)
            pulsesVertNormAvg = pulsesVertNormAvg / np.nanmax(pulsesVertNormAvg)
        
            # plt.plot(pulsesVertNormAvg)
            # ax2 = plt.twinx()
            # ax2.plot(pulsesVertNormAvgCount[:np.sum(~np.isnan(pulsesVertNormAvg)*1)],'g')
            print('pulseCount rBFI ' + str(ind) + ':' + str(len(pulsesVertNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertNormAvg,plotTitle + '_rBFI_vertNorm_postHold',peakLocsOpt[scanInd][peakInd:peakInd+2],pulsesVertNorm_parts,freqTCD)
            
            pulsesVertNormShifted[:,int(maxPulLen*freqTCD):] = np.nan
            pulsesVertNormShifted = np.nanmean(pulsesVertNormShifted,axis=0)
            pulsesVertNormShifted = pulsesVertNormShifted - np.nanmin(pulsesVertNormShifted)
            pulsesVertNormShifted = pulsesVertNormShifted / np.nanmax(pulsesVertNormShifted)
            pulsesVertNormShiftedBoth.append(pulsesVertNormShifted)
            # plt.plot(pulsesVertNormShifted)
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertNormShifted,plotTitle + saveStr,peakLocsOpt[scanInd][peakInd:peakInd+2],pulsesVertNorm_parts_new,freqTCD)
            manualPoints = peakLocs[scanInd][0][peakInd:peakInd+3]
            manualPoints.append(trofLocs[scanInd][0][trofInd])
            manualPoints.append(trofLocs[scanInd][0][trofInd+1])
            manualPointsBoth.append(manualPoints)
            _,_,peaksLocsMan,peaksValsMan,trofsLocs,startLoc = fcns.plotAgPulse(pulsesVertNormShifted,plotTitle + saveStr,manualPoints,pulsesVertNorm_parts_new,freqTCD,peakCutoffs)
            p1p2ratio[scanInd,0+ind] = peaksValsMan[1]/peaksValsMan[0]
            # peakTiming[scanInd,0+ind,0:3] = peaksLocsMan-pulAlign/freqTCD
            # peakTiming[scanInd,0+ind,3:5] = trofsLocs-pulAlign/freqTCD
            peakTiming[scanInd,0+ind,0:3] = peaksLocsMan-startLoc
            peakTiming[scanInd,0+ind,3:5] = trofsLocs-startLoc
            
            p2p1diffAmp[scanInd,0+ind] = np.nanmean(scans[scanInd]['mergedData_rBFI_amp'][camInd][rngs[0]:rngs[1]])
        
        saveStr = '_rBFI_vertNorm_Both_NewMeth'
        _,_,_,_,_,_ = fcns.plotAgPulseDouble(pulsesVertNormShiftedBoth,plotTitle + saveStr,manualPointsBoth,pulsesVertNorm_parts_newBoth,pulsesVertNorm_parts_namesBoth,freqTCD,peakCutoffs)
        
         
        # Vert only normalized average waveform
        pulsesVertNormShiftedBoth = []
        manualPointsBoth = []
        pulsesVertNorm_parts_newBoth = []
        pulsesVertNorm_parts_namesBoth = []
        
        for ind in range(2):
            if ind == 0:
                rngs = [ptsTCD[0] + baseWin[0]*freqTCD, ptsTCD[0] + baseWin[1]*freqTCD]
                peakInd = 0
                trofInd = 0
                saveStr = '_TCD_vertNorm_baseline_NewMeth'
            else:
                rngs = [ptsTCD[1] + postWin[0]*freqTCD, ptsTCD[1] + postWin[1]*freqTCD]
                peakInd = 3
                trofInd = 2
                saveStr = '_TCD_vertNorm_postHold_NewMeth'
            
            pulsesVertNormAvg = np.zeros((len(scans[scanInd]['pulsesData_TCD_starts']),freqTCD*2))*np.nan
            pulsesVertNorm_parts = []
            pulsesVertNormShifted = copy.deepcopy(pulsesVertNormAvg)
            pulsesVertNorm_parts_new = []
            pulsesVertNorm_parts_names = []
            
            for pulseInd in range(len(scans[scanInd]['pulsesData_TCD_starts'])):
                startTime = scans[scanInd]['pulsesData_TCD_starts'][pulseInd]
                usePulse = scans[scanInd]['pulsesData_TCD_toUse'][pulseInd]
                if startTime > rngs[0] and startTime < rngs[1] and usePulse:
                    # pulLen = len(scans[scanInd]['pulsesData_TCD_vertNorm'][pulseInd])
                    # pulsesVertNormAvg[pulseInd,:pulLen] = scans[scanInd]['pulsesData_TCD_vertNorm'][pulseInd]
                    pulsesVertNorm_parts.append(scans[scanInd]['pulsesData_TCD_vertNorm'][pulseInd])
                    pulsesVertNormShifted[pulseInd,:] = fcns.alignLeftFwhm(scans[scanInd]['pulsesData_TCD_vertNorm'][pulseInd],pulAlign,freqTCD)
                    # results_full = signal.peak_widths(pulsesVertNormAvg[pulseInd,:], [np.nanargmax(pulsesVertNormAvg[pulseInd,:])], rel_height=0.5)
                    # # widths = results_full[0]
                    # # width_heights = results_full[1]
                    # # left_ips = results_full[2]
                    # # right_ips = results_full[3]
                    # pulsesVertNormShifted[pulseInd,:] = np.roll(pulsesVertNormAvg[pulseInd,:],int(pulAlign-np.round(results_full[2][0])),axis=0)
                    pulsesVertNorm_parts_new.append(pulsesVertNormShifted[pulseInd,:])
                    pulsesVertNorm_parts_names.append(['p'+str(pulseInd)])
            pulsesVertNorm_parts_newBoth.append(pulsesVertNorm_parts_new)
            pulsesVertNorm_parts_namesBoth.append(pulsesVertNorm_parts_names)
            
            pulsesVertNormAvgCount = np.sum(~np.isnan(pulsesVertNormAvg)*1,axis=0)
            pulsesVertNormAvg = np.nanmean(pulsesVertNormAvg,axis=0)
            pulsesVertNormAvg = pulsesVertNormAvg - np.nanmin(pulsesVertNormAvg)
            pulsesVertNormAvg = pulsesVertNormAvg / np.nanmax(pulsesVertNormAvg)
            
            # plt.plot(pulsesVertNormAvg)
            # ax2 = plt.twinx()
            # ax2.plot(pulsesVertNormAvgCount[:np.sum(~np.isnan(pulsesVertNormAvg)*1)],'g')
            print('pulseCount TCD post:' + str(len(pulsesVertNorm_parts)))
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertNormAvg,plotTitle + saveStr,peakLocsTcd[scanInd][peakInd:peakInd+2],pulsesVertNorm_parts,freqTCD)
            
            pulsesVertNormShifted[:,int(maxPulLen*freqTCD):] = np.nan
            pulsesVertNormShifted = np.nanmean(pulsesVertNormShifted,axis=0)
            pulsesVertNormShifted = pulsesVertNormShifted - np.nanmin(pulsesVertNormShifted)
            pulsesVertNormShifted = pulsesVertNormShifted / np.nanmax(pulsesVertNormShifted)
            pulsesVertNormShiftedBoth.append(pulsesVertNormShifted)
            # plt.plot(pulsesVertNormShifted)
            # _,_,_,peaksValsMan = fcns.plotAgPulse(pulsesVertNormShifted,plotTitle + saveStr,peakLocsTcd[scanInd][peakInd:peakInd+2],pulsesVertNorm_parts_new,freqTCD)            
            manualPoints = peakLocs[scanInd][1][peakInd:peakInd+3]
            manualPoints.append(trofLocs[scanInd][1][trofInd])
            manualPoints.append(trofLocs[scanInd][1][trofInd+1])
            manualPointsBoth.append(manualPoints) 
            _,_,peaksLocsMan,peaksValsMan,trofsLocs,startLoc = fcns.plotAgPulse(pulsesVertNormShifted,plotTitle + saveStr,manualPoints,pulsesVertNorm_parts_new,freqTCD,peakCutoffs)
            p1p2ratio[scanInd,2+ind] = peaksValsMan[1]/peaksValsMan[0]
            # peakTiming[scanInd,2+ind,0:3] = peaksLocsMan-pulAlign/freqTCD
            # peakTiming[scanInd,2+ind,3:5] = trofsLocs-pulAlign/freqTCD
            peakTiming[scanInd,2+ind,0:3] = peaksLocsMan-startLoc
            peakTiming[scanInd,2+ind,3:5] = trofsLocs-startLoc
            
            p2p1diffAmp[scanInd,2+ind] = np.nanmean(scans[scanInd]['data_TCD_amp'][rngs[0]:rngs[1]])
            
            # Getting Cutoffs for Peak Timing
            # peakTimingTemp = copy.deepcopy(peakTiming)
            # peakTimingTemp[peakTimingTemp == 0] = np.nan
            # mins = np.nanmin(peakTimingTemp[:,:,:],axis=(0,1))*1000
            # maxs = np.nanmax(peakTimingTemp[:,:,:],axis=(0,1))*1000
            # peakCutoffs = np.array([
            #     [0.024,0.120,0.280,0.040,0.232],
            #     [0.080,0.216,0.376,0.176,0.352],])
        
        saveStr = '_TCD_vertNorm_Both_NewMeth'
        _,_,_,_,_,_ = fcns.plotAgPulseDouble(pulsesVertNormShiftedBoth,plotTitle + saveStr,manualPointsBoth,pulsesVertNorm_parts_newBoth,pulsesVertNorm_parts_namesBoth,freqTCD,peakCutoffs)
        
        
    
        #%%
        if 0: # testing filtering methods
            dataIn = pulsesVertNormShifted[~np.isnan(pulsesVertNormShifted)]
            # dataIn = scans[scanInd]['moduleData_rBFI'][0:1250,0,5]
            # dataIn = scans[scanInd]['data_TCD'][0:1250]
            soshp = signal.butter(2,15,'lp',fs=freqTCD,output='sos')
            dataOut = signal.sosfilt(soshp, dataIn-np.nanmean(dataIn))+np.nanmean(dataIn)
            plt.plot(dataIn)
            plt.plot(dataOut)
            plt.xlim(0,450)
            plt.show()
            #%%
            from skimage.restoration import denoise_wavelet
            dataOut = denoise_wavelet(dataIn,method='BayesShrink', mode='soft', wavelet_levels=6, wavelet='sym3', rescale_sigma='True')
            plt.plot(dataIn)
            plt.plot(dataOut)
            plt.show()
            #%%
            import biosppy
            dataIn = scans[scanInd]['data_TCD'][0:1250] - np.nanmean(scans[scanInd]['data_TCD'][0:1250])
            _, dataOut, onsetsCh, _, hr = biosppy.signals.ppg.ppg(signal=dataIn, sampling_rate=freqTCD, show=False)
            plt.plot(dataIn)
            plt.plot(dataOut)
            plt.xlim(0,450)
            plt.show()
            #%%
            
    p2p1diffAmp[:,1] = p2p1diffAmp[:,1]/p2p1diffAmp[:,0]
    p2p1diffAmp[:,3] = p2p1diffAmp[:,3]/p2p1diffAmp[:,2]
    p2p1diffAmp[:,[0,2]] = 1

    userInput = input('WARNING: Overwriting allData_contMeanPulsesAgPulses file! Enter any character to cancel.')
    if userInput == '':
        print('Writing processCont file')
        dataToPickle = {'scans': scans}
        f = open('allData_contMeanPulsesAgPulses.pkl', 'wb')
        pickle.dump(dataToPickle, f)
        f.close()
else:
    f = open('allData_contMeanPulsesAgPulses.pkl','rb')
    stupidDict = pickle.load(f)
    f.close()
    scans = stupidDict['scans']
    


#%% processPlots
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
        
if processPlots:
    for scanInd in scanRange: #range(0,len(scanNames)):
        if 1:
            # fcns.longScanPlot(contMerged,meanMerged,['contrast','mean'],freqTCD,savename,scanNames[scanInd])
            # fcns.longScanPlot(camTempsData[:,1,:,:],meanStdData[:,0,:,:],['contrast','mean'],freq,savename,scanNames[scanInd])
            labels = ['rBFI','rBVI','rBFI Avg.','rBVI Avg.','rBFI Amp.','rBVI Amp.']
            #fcns.longScanPlotTCDside(rBfiMerged,rBviMerged,rBfiAvgMerged,rBviAvgMerged,rBfiAmpMerged,rBviAmpMerged,hRateMerged,labels,freqTCD,savename,scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd],0)
            #fcns.longScanPlotTCDside(rBfiMerged,rBviMerged,rBfiAvgMerged,rBviAvgMerged,rBfiAmpMerged,rBviAmpMerged,hRateMerged,labels,freqTCD,savename,scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd],1)
            
            # fcns.longScanPlotTCDside(rBfiMerged,rBviMerged,rBfiAvgMerged,rBviAvgMerged,rBfiStatsAllMerged[:,:,7],rBfiStatsAllMerged[:,:,7],hRateMerged,labels,freqTCD,savename,scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd],1)
            
            #fcns.longScanTCDwaveform(rBFI,rBVI,hRate,['rBFI','rBVI'],freq,savename,scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd])
            
            # fcns.longScanPlotTCDside2(rBfiMerged,rBviMerged,rBfiAvgMerged,rBviAvgMerged,rBfiAmpMerged,rBviAmpMerged,hRateMerged,labels,freqTCD,savename,scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd],1)
            
            # fcns.longScanPlotTCD(rBfiMerged,rBviMerged,['rBFI','rBVI'],freq,savename + '_rBFIrBVI',scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd])
            # fcns.longScanPlottcdMetrics(rBfiAvgMerged,rBfiAmpMerged,['rBFI Average','rBFI Amplitude'],freq,savename + '_rBfiAvgAmp',scanNames[scanInd],tcdNames[scanInd],tcdMarks[scanInd])
    
            # fcns.longScanPlot(contMerged,meanMerged,['contrast','mean'],freq,savename + '_CM',scanNames[scanInd])
        elif batchName[0:8] == 'LongScan':
            fcns.longScanPlot(contMerged,meanMerged,['contrast','mean'],freqTCD,savename + '_CM',scanNames[scanInd])
        else: 
            if bilat == 0:
                fcns.plotResultsNearFar(1-contrast2, 1-mean2, t, ['rBFI','rBVI'], scanName, saveImages, savename + '_nearFar_rBFIrBVI_crop',150)
                # fcns.plotResultsNearFar(contrastwaveform2plot, meanwaveform2plot, twaveform, ['BFIish', 'BVIish'], scanName, saveImages, savename + '_waveformNF')
            
            # valData = np.tile(np.array([[1,2,3,4],[5,6,7,8]]),(numHis,1,1))
            # bilat = 1
            # # valData/10, valData*10
            # fcns.plotResultsNearFarBilat(valData/10, valData*10, t, ['cont','mean'], scanName, 0, savename + '_nearFar_rBFIrBVI_crop',bilat,150)
            
            #fcns.plotContMean2axis(contrast[:,:,:4], mean[:,:,:4], contrast[:,:,:4], mean[:,:,:4], t, ['cont', 'mean'], scanName, saveImages, savename + '_CM', bilat, noiseMetricBH1,heartRt)
            #fcns.plotLeftRight1axis(contrast[:,:,:4], mean[:,:,:4], contrast[:,:,:4], mean[:,:,:4], t, ['cont', 'mean'], scanName, saveImages, savename + '_LR_CM', bilat, noiseMetricBH1,heartRt)
            
            #fcns.plotContMean2axis(rBFI, rBVI, contrast, mean, t, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_rBFIrBVI', bilat, noiseMetricBH1,heartRt)
            #fcns.plotContMean2axis(rBFI, rBVI, contrast, mean, t, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_rBFIrBVI_crop', bilat, noiseMetricBH1,heartRt,260)
            # fcns.plotLeftRight1axis(rBFI, rBVI, contrast, mean, t, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_LR_rBFIrBVI', bilat, noiseMetricBH1,heartRt)
            # fcns.plotLeftRight1axis(contrastwaveform2plot, meanwaveform2plot, contrast, mean, twaveform, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_LR_WavrBFIrBVI', bilat, noiseMetricBH1,heartRt)
    
        
        if 0: #plotImages or saveImages: # Soren's plots turned off
            #fcns.plotMeanStdData(meanStdData, meanStdDataSmooth, t)
            fcns.quickPlot(histData, titles, gl, imgStats_dark['mean_ob'], t)
            fcns.plotResults(contrast, mean, t, ['Contrast', 'Intensity'], scanName, saveImages, savename + '_CM',noiseMetricBH1,heartRt)
            #fcns.plotResults(np.sqrt(stats['var_raw']), stats['m1'], t, ['Std', 'Mean'], scanName, saveImages, savename + '_SM')
            #fcns.plotResults(np.sqrt(stats['var']), stats['m1'], t, ['Std', 'Mean'], scanName, saveImages, savename + '_SM')
            #fcns.plotResults1axis(np.sqrt(stats['var_raw']), np.sqrt(stats['dv']), t, ['Main var', 'OB var'], scanName, saveImages, savename + '_VV')
            fcns.plotResultsRightLeft(contrast, mean, t, ['Contrast', 'Mean'], scanName, saveImages, savename + '_CM_RL')
        
            #normalized waveforms
            fcns.plotResults(1-contrast2, 1-mean2, t, ['Contrast2', 'Intensity2'], scanName, saveImages, savename + '_CM2')
            fcns.plotResultsNearFar(1-contrast2, 1-mean2, t, ['Contrast2', 'Intensity2'], scanName, saveImages, savename + '_CM2_NF')
            
            #golden waveforms
            fcns.plotResults(contrastwaveform2plot, meanwaveform2plot, twaveform, ['BFIish', 'BVIish'], scanName, saveImages, savename + '_waveformCM')
            fcns.plotResultsNearFar(contrastwaveform2plot, meanwaveform2plot, twaveform, ['BFIish', 'BVIish'], scanName, saveImages, savename + '_waveformNF')
            fcns.plotResultsRightLeft(contrastwaveform2plot, meanwaveform2plot, twaveform, ['BFIish', 'BVIish'], scanName, saveImages, savename + '_waveformRL')
    
            #derivatives
            #fcns.plotResults(0.5-contrast2, -dcontrast, t, ['Contrast2', 'Slope'], scanName, saveImages, savename + '_D')
            #fcns.plotResults(0.5-contrast2, -ddcontrast, t, ['Contrast2', 'Curvature'], scanName, saveImages, savename + '_DD')
            
        if scanInd == 0:
            camTempsDataStart = camTempsData[0:10,1,:,:].mean(axis=0)
            camTempsDataStart[:,3] = camTempsDataStart[:,0]
            camTempsDataStart = fcns.sortCamMod(camTempsDataStart,bilat)

#%% Example waveform plots

# Example full time trace
for scanInd in [26]: #range(0,len(scanNames)):

    holdInds = [1,2]
    if scanInd == 16: #sub 13
        holdInds = [3,4]
    camInd = 1
    tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
    ptsTCD = np.array([tcdMarksZerod[holdInds[0]],tcdMarksZerod[holdInds[1]]])
    ptInds = [ptsTCD[0] - 30*freqTCD, ptsTCD[1] + 20*freqTCD]
    
    fig,ax = plt.subplots(ncols=1,nrows=3,figsize=(3.5,3.06))
    # fig.tight_layout(h_pad=0.5)
    
    time = scans[scanInd]['mergedData_time'] - ptsTCD[0]/freqTCD + 30
    
    movOff = 300 # time in seconds to move left axis data off the plot so it doesn't screw with final renderings
    
    C_min = 0.09 # Contrast value of demodulated seed
    C_baseline = np.nanmean(scans[scanInd]['mergedData_cont'][camInd][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]]) # 30 sec prior to hold
    C_data = scans[scanInd]['mergedData_cont'][camInd] # hold data plus 5 sec before and after
    C_toPlot = (1+(C_baseline-C_data)/(C_baseline-C_min))*100
    
    I_baseline = np.nanmean(scans[scanInd]['mergedData_mean'][camInd][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]])
    I_data = scans[scanInd]['mergedData_mean'][camInd]
    I_toPlot = (1+((I_baseline-I_data)/I_baseline))*100
    
    T_baseline = np.nanmean(scans[scanInd]['data_TCD'][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]])
    T_data = scans[scanInd]['data_TCD']
    T_toPlot = (T_data/T_baseline)*100
    
    ax[0].plot(time[ptInds[0]:ptInds[1]],C_data[ptInds[0]:ptInds[1]],'b',linewidth=0.5)
    ax[0].invert_yaxis()
    ax0 = plt.twinx(ax[0])
    ax0.plot(time[ptInds[0]:ptInds[1]]+movOff,C_toPlot[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax[0].set_ylabel('Contrast')
    ax0.set_ylabel('rBF (%)')
    ax[0].set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax[0].set_xticklabels([])
    
    ax[1].plot(time[ptInds[0]:ptInds[1]],I_data[ptInds[0]:ptInds[1]],'r',linewidth=0.5)
    ax[1].invert_yaxis()
    ax1 = plt.twinx(ax[1])
    ax1.plot(time[ptInds[0]:ptInds[1]]+movOff,I_toPlot[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax[1].set_ylabel('Intensity')
    ax1.set_ylabel('rBV (%)')
    ax[1].set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax[1].set_xticklabels([])
    
    ax[2].plot(time[ptInds[0]:ptInds[1]],T_data[ptInds[0]:ptInds[1]],'g',linewidth=0.5)
    ax2 = plt.twinx(ax[2])
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,T_toPlot[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax[2].set_ylabel('CBFv (cm/s)')
    ax2.set_ylabel('rCBFv (%)')
    ax[2].set_xlim(time[ptInds[0]],time[ptInds[1]])
    
    for subInd in range(3): 
        ymin, ymax = ax[subInd].get_ylim()
        nx = np.array([[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymin],[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymin]])
        nx[:,0] = nx[:,0] - ptsTCD[0]/freqTCD + 30
        ax[subInd].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    ax[2].set_xlabel('Time (s)')
    
    ax[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax[2].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax0.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    
    if  scanInd == 26:
        ax[0].set_yticks([0.10,0.15])
        ax[1].set_yticks([80,100])
        ax[2].set_yticks([40,160])
        ax0.set_yticks([100,180])
        ax1.set_yticks([100,120])
        ax2.set_yticks([100,200])
     
    fig.savefig(savePath + '/' + 'ExampleWaveform_' + shortNames[scanInd] + '_V2.png',dpi=300,bbox_inches='tight')


# Example full time trace with zoomed in below
for scanInd in [26]: #range(0,len(scanNames)):

    holdInds = [1,2]
    if scanInd == 16: #sub 13
        holdInds = [3,4]
    camInd = 1 
    tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
    ptsTCD = np.array([tcdMarksZerod[holdInds[0]],tcdMarksZerod[holdInds[1]]])
    ptInds = [ptsTCD[0] - 30*freqTCD, ptsTCD[1] + 20*freqTCD]
    
    plt.figure(figsize = (6, 9))
    grid = plt.GridSpec(6, 2, wspace =0.4, hspace = 0.5)
    
    time = scans[scanInd]['mergedData_time'] - ptsTCD[0]/freqTCD + 30
    
    movOff = 0 # time in seconds to move left axis data off the plot so it doesn't screw with final renderings
    
    C_min = 0.09 # Contrast value of demodulated seed
    C_baseline = np.nanmean(scans[scanInd]['mergedData_cont'][camInd][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]]) # 30 sec prior to hold
    C_data = scans[scanInd]['mergedData_cont'][camInd] # hold data plus 5 sec before and after
    C_toPlot = (1+(C_baseline-C_data)/(C_baseline-C_min))*100
    
    I_baseline = np.nanmean(scans[scanInd]['mergedData_mean'][camInd][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]])
    I_data = scans[scanInd]['mergedData_mean'][camInd]
    I_toPlot = (1+((I_baseline-I_data)/I_baseline))*100
    
    T_baseline = np.nanmean(scans[scanInd]['data_TCD'][(ptsTCD[0] - 30*freqTCD):ptsTCD[0]])
    T_data = scans[scanInd]['data_TCD']
    T_toPlot = (T_data/T_baseline)*100
    
    ax00 = plt.subplot(grid[0,:])
    ax00.plot(time[ptInds[0]:ptInds[1]],C_data[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax00.invert_yaxis()
    ax00b = plt.twinx(ax00)
    ax00b.plot(time[ptInds[0]:ptInds[1]]+movOff,C_toPlot[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax00b.set_ylabel('rBF (%)')
    ax00.set_ylabel('Contrast')
    ax00.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax00.set_xticklabels([])
    
    ax10 = plt.subplot(grid[1,:])
    ax10.plot(time[ptInds[0]:ptInds[1]],I_data[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax10.invert_yaxis()
    ax10b = plt.twinx(ax10)
    ax10b.plot(time[ptInds[0]:ptInds[1]]+movOff,I_toPlot[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax10b.set_ylabel('rBV (%)')
    ax10.set_ylabel('Intensity')
    ax10.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax10.set_xticklabels([])
    
    ax20 = plt.subplot(grid[2,:])
    ax20.plot(time[ptInds[0]:ptInds[1]],T_data[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax20b = plt.twinx(ax20)
    ax20b.plot(time[ptInds[0]:ptInds[1]]+movOff,T_toPlot[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax20b.set_ylabel('rCBFv (%)')
    ax20.set_xlabel('Time (s)')
    ax20.set_ylabel('CBFv (cm/s)')
    ax20.set_xlim(time[ptInds[0]],time[ptInds[1]])
    
    ymin, ymax = ax00.get_ylim()
    nx = np.array([[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymin],[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymin]])
    nx[:,0] = nx[:,0] - ptsTCD[0]/freqTCD + 30
    ax00.add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    
    ymin, ymax = ax10.get_ylim()
    nx = np.array([[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymin],[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymin]])
    nx[:,0] = nx[:,0] - ptsTCD[0]/freqTCD + 30
    ax10.add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    
    ymin, ymax = ax20.get_ylim()
    nx = np.array([[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymin],[scans[scanInd]['mergedData_time'][ptsTCD[0]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymax],[scans[scanInd]['mergedData_time'][ptsTCD[1]],ymin]])
    nx[:,0] = nx[:,0] - ptsTCD[0]/freqTCD + 30
    ax20.add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    
    ptInds = [ptsTCD[0] - 30*freqTCD + 60, ptsTCD[0] - 28*freqTCD + 55]
    
    ax30 = plt.subplot(grid[3,0])
    ax30.plot(time[ptInds[0]:ptInds[1]],C_data[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax30.invert_yaxis()
    ax2b = plt.twinx(ax30)
    ax2b.plot(time[ptInds[0]:ptInds[1]]+movOff,C_toPlot[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax30.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax30.set_ylabel('Contrast')
    ax30.set_xticklabels([])
    
    ax40 = plt.subplot(grid[4,0])
    ax40.plot(time[ptInds[0]:ptInds[1]],I_data[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax40.invert_yaxis()
    ax2 = plt.twinx(ax40)
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,I_toPlot[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax40.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax40.set_ylabel('Intensity')
    ax40.set_xticklabels([])
    
    ax50 = plt.subplot(grid[5,0])
    ax50.plot(time[ptInds[0]:ptInds[1]],T_data[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax2 = plt.twinx(ax50)
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,T_toPlot[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax50.set_xlabel('Time (s)')
    ax50.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax50.set_ylabel('CBFv (cm/s)')

    
    
    ptInds = [ptsTCD[1] + 0*freqTCD - 25, ptsTCD[1] + 2*freqTCD - 30]
    
    ax31 = plt.subplot(grid[3,1])
    ax31.plot(time[ptInds[0]:ptInds[1]],C_data[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax31.invert_yaxis()
    ax2 = plt.twinx(ax31)
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,C_toPlot[ptInds[0]:ptInds[1]],'b',linewidth=1)
    ax31.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax2.set_ylabel('rBF (%)')
    ax31.set_xticklabels([])
    
    ax41 = plt.subplot(grid[4,1])
    ax41.plot(time[ptInds[0]:ptInds[1]],I_data[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax41.invert_yaxis()
    ax2 = plt.twinx(ax41)
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,I_toPlot[ptInds[0]:ptInds[1]],'r',linewidth=1)
    ax41.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax2.set_ylabel('rBV (%)')
    ax41.set_xticklabels([])
    
    ax51 = plt.subplot(grid[5,1])
    ax51.plot(time[ptInds[0]:ptInds[1]],T_data[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax2 = plt.twinx(ax51)
    ax2.plot(time[ptInds[0]:ptInds[1]]+movOff,T_toPlot[ptInds[0]:ptInds[1]],'g',linewidth=1)
    ax51.set_xlim(time[ptInds[0]],time[ptInds[1]])
    ax51.set_xlabel('Time (s)')
    ax2.set_ylabel('rCBFv (%)')
    
    # Set all right scaling to be the same
    # newLimY = [0,1]
    # newLimY[0] = np.min([ax00b.get_ylim()[0],ax10b.get_ylim()[0],ax20b.get_ylim()[0]])
    # newLimY[1] = np.max([ax00b.get_ylim()[1],ax10b.get_ylim()[1],ax20b.get_ylim()[1]])
    # ax00b.set_ylim(newLimY)
    # ax10b.set_ylim(newLimY)
    # ax20b.set_ylim(newLimY)
    
    ax00.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax10.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax20.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax00b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax10b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    ax20b.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2,min_n_ticks=2))
    
    # for rowInd in range(3):
    #     for colInd in range(1,3):
    #         ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))        
     
    plt.savefig(savePath + '/' + 'ExampleWaveform_' + shortNames[scanInd] + '.png',dpi=300,bbox_inches='tight')
    
#%%s Zoomed in Plot
for scanInd in [20]: #range(0,len(scanNames)): # 13 18 20
    
    holdInds = [1,2]
    if scanInd == 16: #sub 13
        holdInds = [3,4]
    camInd = 1 
    tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
    ptsTCD = np.array([tcdMarksZerod[holdInds[0]],tcdMarksZerod[holdInds[1]]])
    
    fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(6,3.75))
    fig.tight_layout(h_pad=2.5,w_pad=1.5)
    
    
    
    C_data = scans[scanInd]['mergedData_rBFI_vertNorm'][camInd] # hold data plus 5 sec before and after
    I_data = scans[scanInd]['mergedData_rBVI_vertNorm'][camInd]
    T_data = scans[scanInd]['data_TCD_vertNorm']
    
    if scanInd == 13:
        ptInds = np.array([[ptsTCD[0] + 0*freqTCD + 103, ptsTCD[0] + 1*freqTCD + 84],
                           [ptsTCD[1] + 0*freqTCD + 39, ptsTCD[1] + 1*freqTCD + 23]])
    elif scanInd == 20:
        ptInds = np.array([[ptsTCD[0] + 0*freqTCD + 98, ptsTCD[0] + 2*freqTCD - 37],
                           [ptsTCD[1] + 3*freqTCD + 34, ptsTCD[1] + 4*freqTCD + 12]])
        # 5 seconds for reviewer follow up
        ptInds = np.array([[ptsTCD[0] + 0*freqTCD + 98, ptsTCD[0] + 0*freqTCD + 98 + 125*5],
                           [ptsTCD[1] + 3*freqTCD + 34, ptsTCD[1] + 3*freqTCD + 34 + 125*5]])
    else:
        ptInds = np.array([[ptsTCD[0] + 0*freqTCD + 0, ptsTCD[0] + 2*freqTCD + 0],
                           [ptsTCD[1] + 6*freqTCD + 0, ptsTCD[1] + 9*freqTCD + 0]])
    
    
    for colInd in range(2):
        
        time = np.arange(np.diff(ptInds[colInd,:]))/freqTCD
        ax[0,colInd].plot(time,1-C_data[ptInds[colInd,0]:ptInds[colInd,1]],'b',linewidth=1)
        ax[0,colInd].invert_yaxis()
        ax[0,colInd].set_ylabel('Contrast (a.u.)')
        ax[0,colInd].set_xlabel('Time (s)')
        ax[0,colInd].spines[['right', 'top']].set_visible(False)
        ax[0,colInd].set_yticks([0,0.5,1])
        ax[0,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
        ax[0,colInd].set_xlim(time[0],time[-1])
        # ax[0,colInd].set_ylim(1,0)
        
        ax[1,colInd].plot(time,T_data[ptInds[colInd,0]:ptInds[colInd,1]],'darkorange',linewidth=1)
        ax[1,colInd].set_ylabel('CBFv (a.u.)')
        ax[1,colInd].set_xlabel('Time (s)')
        ax[1,colInd].spines[['right', 'top']].set_visible(False)
        ax[1,colInd].set_yticks([0,0.5,1])
        ax[1,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
        ax[1,colInd].set_xlim(time[0],time[-1])
        # ax[1,colInd].set_ylim(0,1)
         
        fig.savefig(savePath + '/' + 'ExampleWaveformZoomed_' + shortNames[scanInd] + '.png',dpi=300,bbox_inches='tight')

#%% Peak/Trough Analysis Plots

# P2/P1 Plots
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
fig.tight_layout(w_pad=3.5,h_pad=5.25)
plt.subplots_adjust(top=0.85)
# colors = plt.get_cmap('gist_rainbow')(np.linspace(0, 1, len(scansToUse))) # jet gist_rainbow tab20
colors = np.concatenate((plt.get_cmap('tab10')(np.linspace(0, 1, 10)),plt.get_cmap('Set1')(np.linspace(0, 1, 9)),plt.get_cmap('Dark2')(np.linspace(0, 1,8))),axis=0)
colors = ['black','grey','lightcoral','maroon','red','sienna','darkorange','yellowgreen','cyan','gold', #1-10
          'tan','olive','yellow','forestgreen','lime','turquoise','cadetblue','deepskyblue','cornflowerblue','blue', #11-20
          'darkviolet','magenta','pink'] # 21-22

feats = ['rCBFv Env.','rCBFv','rCBFv PI','rCBFv HR',
         'rBF Env.','rBF','rBF Amp.',
         'rBV Env.','rBV','rBV Amp.',
         'rBF HR',
         'Contrast','Mean','Contrast Avg.','Mean Avg.',
         'rCBFv stat','rBF stat','rCBFv stat2','rBF stat2',
         'rCBFv Amp.','rBF PI','rBV PI',
         ]
shortNames = np.array(shortNames)

useAugInd = 0
axLabel = ' P2/P1 Height'

pkRatioData = copy.deepcopy(p1p2ratio)
if useAugInd:
    axLabel = ' Augmentation Index'
    p2p1ratio = copy.deepcopy(p1p2ratio)
    p2p1_AI = copy.deepcopy(p1p2ratio)
    for rowInd in range(p1p2ratio.shape[0]):
        for colInd in range(p1p2ratio.shape[1]):
            if p2p1_AI[rowInd,colInd] == 0:
                p2 = 0
                p1 = 0
            elif p2p1_AI[rowInd,colInd] > 1:
                p2 = 1
                p1 = 1/p2p1_AI[rowInd,colInd]
            else:
                p1 = 1
                p2 = p2p1_AI[rowInd,colInd]
            p2p1_AI[rowInd,colInd] = (p2-p1)/p2p1diffAmp[rowInd,colInd]
    pkRatioData = copy.deepcopy(p2p1_AI)

# diffOpt_all = (1/pkRatioData[:,1])/(1/pkRatioData[:,0])
# diffTcd_all = (1/pkRatioData[:,3])/(1/pkRatioData[:,2])
diffOpt_all = pkRatioData[:,1]/pkRatioData[:,0]
diffTcd_all = pkRatioData[:,3]/pkRatioData[:,2]
for colInd,scanInd in enumerate(scanRange):
    ax[0,1].plot(diffTcd_all[scanInd]*100,diffOpt_all[scanInd]*100,'o',color=colors[colInd])
ax[0,1].set_xlabel('\u0394 ' + feats[0][:5] + axLabel + ' (%)')
ax[0,1].set_ylabel('\u0394 ' + feats[5] + axLabel + ' (%)')
ax[0,1].set_title('%\u0394 in P2/P1 Before & After Breath Hold\n' + 'Correlation (Pearson): %.2f' % scipy.stats.pearsonr(diffTcd_all[scanRange], diffOpt_all[scanRange]).statistic)
# ax[0,1].legend(shortNames[scansToUse])    
# ax[0,1].set_xlim(90,190)
# ax[0,1].set_ylim(90,190)

minMaxTcdNorm_all = []
minMaxOptNorm_all = []
for colInd,scanInd in enumerate(scanRange):
    ax[0,0].plot(100,100,'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
    ax[0,0].plot([100,diffTcd_all[scanInd]*100],[100,diffOpt_all[scanInd]*100],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
    minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,[100,diffTcd_all[scanInd]*100],axis=0)
    minMaxOptNorm_all = np.append(minMaxOptNorm_all,[100,diffOpt_all[scanInd]*100],axis=0)
ax[0,0].set_xlabel(feats[0][:5] + axLabel + ' (%)')
ax[0,0].set_ylabel(feats[5] + axLabel + ' (%)')
ax[0,0].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
# ax[0,0].set_xlim(90,190)
# ax[0,0].set_ylim(90,190)

minMaxTcdNorm_all = []
minMaxOptNorm_all = []
for colInd,scanInd in enumerate(scanRange):
    # ax[0,2].plot(1/pkRatioData[scanInd,2],1/pkRatioData[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
    # ax[0,2].plot(1/pkRatioData[scanInd,2:4],1/pkRatioData[scanInd,0:2],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
    # minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,1/pkRatioData[scanInd,2:4],axis=0)
    # minMaxOptNorm_all = np.append(minMaxOptNorm_all,1/pkRatioData[scanInd,0:2],axis=0)
    ax[0,2].plot(pkRatioData[scanInd,2],pkRatioData[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
    ax[0,2].plot(pkRatioData[scanInd,2:4],pkRatioData[scanInd,0:2],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
    minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,pkRatioData[scanInd,2:4],axis=0)
    minMaxOptNorm_all = np.append(minMaxOptNorm_all,pkRatioData[scanInd,0:2],axis=0)
ax[0,2].set_xlabel(feats[0][:5] + axLabel)
ax[0,2].set_ylabel(feats[5] + axLabel)
ax[0,2].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
# ax[0,2].set_xlim(90,190)
# ax[0,2].set_ylim(90,190)


minMaxTcdNorm_all = []
minMaxOptNorm_all = []
for colInd,scanInd in enumerate(scanRange):
    ax[1,0].plot(pkRatioData[scanInd,2],pkRatioData[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
    minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,pkRatioData[scanInd,2])
    minMaxOptNorm_all = np.append(minMaxOptNorm_all,pkRatioData[scanInd,0])
ax[1,0].set_xlabel(feats[0][:5] + axLabel)
ax[1,0].set_ylabel(feats[5] + axLabel)
ax[1,0].set_title('Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
 

minMaxBothNorm_mean_all = []
minMaxBothNorm_diff_all = []
for colInd,scanInd in enumerate(scanRange):
    
    # minMaxBothNorm_mean = (1/pkRatioData[scanInd,0:2]+1/pkRatioData[scanInd,2:4])/2
    # minMaxBothNorm_diff = 1/pkRatioData[scanInd,0:2]-1/pkRatioData[scanInd,2:4]
    minMaxBothNorm_mean = (pkRatioData[scanInd,0:2]+pkRatioData[scanInd,2:4])/2
    minMaxBothNorm_diff = pkRatioData[scanInd,0:2]-pkRatioData[scanInd,2:4]
    
    ax[1,2].plot(minMaxBothNorm_mean[0],minMaxBothNorm_diff[0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
    ax[1,2].plot(minMaxBothNorm_mean,minMaxBothNorm_diff,'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
    minMaxBothNorm_mean_all = np.append(minMaxBothNorm_mean_all,minMaxBothNorm_mean,axis=0)
    minMaxBothNorm_diff_all = np.append(minMaxBothNorm_diff_all,minMaxBothNorm_diff,axis=0)

xlims11 = ax[1,2].get_xlim()
ax[1,2].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
ax[1,2].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
ax[1,2].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
ax[1,2].set_xlim(xlims11)

ax[1,2].set_xlabel('Mean of ' + feats[0][:5] + ' and ' + feats[5] + axLabel)
ax[1,2].set_ylabel('Diff. of ' + feats[0][:5] + ' and ' + feats[5] + axLabel)
ax[1,2].set_title('Mean +/- 1.96 S.D. shown in black\n(values = %+.2f, %+.2f, %+.2f)' % 
  (np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all),np.nanmean(minMaxBothNorm_diff_all),np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all)))


for colInd,scanInd in enumerate(scanRange):
    ax[1,1].plot(diffTcd_all[scanInd]*0,diffOpt_all[scanInd]*0,'o',color=colors[colInd])
ax[1,1].legend(shortNames[scanRange])

for rowInd in range(2):
    for colInd in range(3):
        # ax[rowInd,colInd].set_aspect('equal', adjustable='box')
        ax[rowInd,colInd].set_box_aspect(1)
        ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
        ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
        
lims = [np.min([ax[0,2].get_xlim()[0],ax[0,2].get_ylim()[0]])*0.95,np.max([ax[0,2].get_xlim()[1],ax[0,2].get_ylim()[1]])*1.05]
ax[0,2].set_xlim(lims)
ax[0,2].set_ylim(lims)

legend_elements = [
                    Line2D([0],[0],marker='o',color='w',label='Pre Hold',markerfacecolor='k',markersize=7),
                    Line2D([0],[0],marker='s',color='w',label='Post Hold',markerfacecolor='k',markersize=7),
                    ]
ax[0,0].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
ax[0,2].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
ax[1,0].legend(handles=[legend_elements[0]], loc='lower right', fontsize=10,frameon=False,handletextpad=0)

if useAugInd:
    fig.savefig(savePath + '/' + 'WaveformP1P2_aligned_AI.png',dpi=300,bbox_inches='tight')
else:
    fig.savefig(savePath + '/' + 'WaveformP1P2_aligned.png',dpi=300,bbox_inches='tight')



# Peak & Trough Individual Plots
scansToUse = np.array(np.flip(scanRange))
for peakNum in  range(5):
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
    fig.tight_layout(w_pad=3.5,h_pad=5.25)
    plt.subplots_adjust(top=0.85)
    
    diffOpt_all = peakTiming[:,1,peakNum]/peakTiming[:,0,peakNum]
    diffTcd_all = peakTiming[:,3,peakNum]/peakTiming[:,2,peakNum]
    for colInd,scanInd in enumerate(scansToUse):
        ax[0,1].plot(diffTcd_all[scanInd]*100,diffOpt_all[scanInd]*100,'o',color=colors[colInd])
    ax[0,1].set_xlabel('\u0394 ' + feats[0][:5] + ' P' + str(peakNum+1) + ' Time (%)')
    ax[0,1].set_ylabel('\u0394 ' + feats[5] + ' P' + str(peakNum+1) + ' Time (%)')
    ax[0,1].set_title('\u0394 in P' + str(peakNum+1) + ' Time Before & After Breath Hold\n' + 'Correlation (Pearson): %.2f' % scipy.stats.pearsonr(diffTcd_all[scansToUse], diffOpt_all[scansToUse]).statistic)
    
    minMaxTcdNorm_all = []
    minMaxOptNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[0,0].plot(100,100,'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        ax[0,0].plot([100,diffTcd_all[scanInd]*100],[100,diffOpt_all[scanInd]*100],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,[100,diffTcd_all[scanInd]*100],axis=0)
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,[100,diffOpt_all[scanInd]*100],axis=0)
    ax[0,0].set_xlabel(feats[0][:5] + 'P' + str(peakNum+1) + ' Timing ' + ' (%)')
    ax[0,0].set_ylabel(feats[5] + 'P' + str(peakNum+1) + ' Timing ' + ' (%)')
    ax[0,0].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[0,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        ax[0,2].plot(minMaxTcdNorm[scanInd,:],minMaxOptNorm[scanInd,:],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,:],axis=0)
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,:],axis=0)
    ax[0,2].set_xlabel(feats[0][:5] + ' P' + str(peakNum+1) + ' Time (s)')
    ax[0,2].set_ylabel(feats[5] + ' P' + str(peakNum+1) + ' Time (s)')
    ax[0,2].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[1,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,0])
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,0])
    ax[1,2].set_xlabel(feats[0][:5] + ' P' + str(peakNum+1) + ' Time (s)')
    ax[1,2].set_ylabel(feats[5] + ' P' + str(peakNum+1) + ' Time (s)')
    # ax[1,2].set_title('Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(minMaxTcdNorm_all,minMaxOptNorm_all)
    ax[1,2].plot(x_line, y_line, color = 'k',linestyle='--',linewidth=1,zorder=0.95)
    ax[1,2].fill_between(x_line, y_line + ci, y_line - ci, color = [0.925,0.925,0.925],zorder=0.9)
    ax[1,2].set_title('Slope: y = %.2f x + %.2f\n Correlation (Pearson): %.2f' % (slope,intercept,r_value))
    
    for colInd,scanInd in enumerate(scansToUse):
        ax[1,1].plot(diffTcd_all[scanInd]*0,diffOpt_all[scanInd]*0,'o',color=colors[colInd])
    ax[1,1].legend(shortNames[scansToUse])
    
    for rowInd in range(2):
        lims = [np.min([ax[rowInd,2].get_xlim()[0],ax[rowInd,2].get_ylim()[0]])*1,np.max([ax[rowInd,2].get_xlim()[1],ax[rowInd,2].get_ylim()[1]])*1]
        ax[rowInd,2].set_xlim(lims)
        ax[rowInd,2].set_ylim(lims)
     
    for rowInd in range(2):
        for colInd in range(3):
            # ax[rowInd,colInd].set_aspect('equal', adjustable='box')
            ax[rowInd,colInd].set_box_aspect(1)
            ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
            ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
            
    legend_elements = [
                    Line2D([0],[0],marker='o',color='w',label='Pre Hold',markerfacecolor='k',markersize=7),
                    Line2D([0],[0],marker='s',color='w',label='Post Hold',markerfacecolor='k',markersize=7),
                    ]
    ax[0,0].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
    ax[0,2].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
    ax[1,2].legend(handles=[legend_elements[0]], loc='lower right', fontsize=10,frameon=False,handletextpad=0)
    
    fig.savefig(savePath + '/' + 'WaveformP' + str(peakNum+1) + '_aligned.png',dpi=300,bbox_inches='tight')



# All Peak Timing Aggregates
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
fig.tight_layout(w_pad=3.5,h_pad=5.25)
plt.subplots_adjust(top=0.85)

for peakNum in  range(3):
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[0,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        ax[0,2].plot(minMaxTcdNorm[scanInd,:],minMaxOptNorm[scanInd,:],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,:],axis=0)
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,:],axis=0)
    ax[0,2].set_xlabel('Peak Time (s) ' + feats[0][:5]) 
    ax[0,2].set_ylabel('Peak Time (s) ' + feats[5])
    # ax[0,2].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[1,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,0])
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,0])
    ax[1,2].set_xlabel('Peak Time (s) ' + feats[0][:5])
    ax[1,2].set_ylabel('Peak Time (s) ' + feats[5])
    # ax[1,2].set_title('Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(minMaxTcdNorm_all,minMaxOptNorm_all)
    ax[1,2].plot(x_line, y_line, color = 'k',linestyle='--',linewidth=1,zorder=2.5)
    ax[1,2].fill_between(x_line, y_line + ci, y_line - ci, color = 'k',edgecolor='k',alpha=0.2 ,zorder=2.4)  
    
for rowInd in range(2):
    #lims = [np.min([ax[rowInd,2].get_xlim()[0],ax[rowInd,2].get_ylim()[0]])*0.95,np.max([ax[rowInd,2].get_xlim()[1],ax[rowInd,2].get_ylim()[1]])*1.05]
    lims = [0.0,0.50]
    ax[rowInd,2].set_xlim(lims)
    ax[rowInd,2].set_ylim(lims)
 
for rowInd in range(2):
    for colInd in range(3):
        # ax[rowInd,colInd].set_aspect('equal', adjustable='box')
        ax[rowInd,colInd].set_box_aspect(1)
        ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
        ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
legend_elements = [
                    Line2D([0],[0],marker='o',color='w',label='Pre Hold',markerfacecolor='k',markersize=7),
                    Line2D([0],[0],marker='s',color='w',label='Post Hold',markerfacecolor='k',markersize=7),
                    ]
ax[0,2].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
# ax[1,2].legend(handles=[legend_elements[0]], loc='lower right', fontsize=10,frameon=False,handletextpad=0)

fig.savefig(savePath + '/' + 'WaveformAllPeaks_aligned.png',dpi=300,bbox_inches='tight')



# All Peak & All Trough Timing Aggregates
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
fig.tight_layout(w_pad=3.5,h_pad=5.25)
plt.subplots_adjust(top=0.85)

for peakNum in  range(4,5):
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[0,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        ax[0,2].plot(minMaxTcdNorm[scanInd,:],minMaxOptNorm[scanInd,:],'s',ls='-', ms=5, markevery=[-1], color=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,:],axis=0)
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,:],axis=0)
    ax[0,2].set_xlabel('Dicrotic Notch Time (s) ' + feats[0][:5]) 
    ax[0,2].set_ylabel('Dicrotic Notch Time (s) ' + feats[5])
    # ax[0,2].set_title('\u25CF avg. of pre hold\n\u25A0 avg. of post hold\n' + 'Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    minMaxOptNorm = peakTiming[:,0:2,peakNum]
    minMaxTcdNorm = peakTiming[:,2:4,peakNum]
    minMaxOptNorm_all = []
    minMaxTcdNorm_all = []
    for colInd,scanInd in enumerate(scansToUse):
        ax[1,2].plot(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],'o',ls='-', ms=5, markevery=[0], color=colors[colInd])
        ax[1,1].scatter(minMaxTcdNorm[scanInd,0],minMaxOptNorm[scanInd,0],s=20, facecolors='none', edgecolors=colors[colInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[scanInd,0])
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[scanInd,0])
    ax[1,2].set_xlabel('Dicrotic Notch Time (s) ' + feats[0][:5])
    ax[1,2].set_ylabel('Dicrotic Notch Time (s) ' + feats[5])
    # ax[1,2].set_title('Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(minMaxTcdNorm_all,minMaxOptNorm_all)
    ax[1,2].plot(x_line, y_line, color = 'k',linestyle='--',linewidth=1,zorder=0.95)
    ax[1,2].fill_between(x_line, y_line + ci, y_line - ci, color = [0.925,0.925,0.925],zorder=0.9)
    
for rowInd in range(2):
    #lims = [np.min([ax[rowInd,2].get_xlim()[0],ax[rowInd,2].get_ylim()[0]])*1,np.max([ax[rowInd,2].get_xlim()[1],ax[rowInd,2].get_ylim()[1]])*1]
    lims = [0.25,0.40]
    ax[rowInd,2].set_xlim(lims)
    ax[rowInd,2].set_ylim(lims)
 
for rowInd in range(2):
    for colInd in range(3):
        ax[rowInd,colInd].set_box_aspect(1)
        ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
        ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3,min_n_ticks=3))
        
legend_elements = [
                    Line2D([0],[0],marker='o',color='w',label='Pre Hold',markerfacecolor='k',markersize=7),
                    Line2D([0],[0],marker='s',color='w',label='Post Hold',markerfacecolor='k',markersize=7),
                    ]
ax[0,2].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
# ax[1,2].legend(handles=[legend_elements[0]], loc='lower right', fontsize=10,frameon=False,handletextpad=0)

fig.savefig(savePath + '/' + 'WaveformAllTrofs_aligned.png',dpi=300,bbox_inches='tight')



#%% Plotting validation
# valData = np.tile(np.array([[1,2,3,4],[5,6,7,8]]),(numHis,1,1))
# bilat = 0
# # valData/10, valData*10
# fcns.plotContMean2axis(valData/10, valData*10, valData/10+1, valData*10+1, t, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_rBFIrBVI', bilat, noiseMetricBH1,heartRt)
# fcns.plotLeftRight1axis(valData/10, valData*10, valData/10+1, valData*10+1, t, ['rBFI', 'rBVI'], scanName, saveImages, savename + '_LR_rBFIrBVI', bilat, noiseMetricBH1,heartRt)
