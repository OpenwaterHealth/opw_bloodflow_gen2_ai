
import copy, pickle, scipy
import batchfilePaper
import headscan_gen2_fcns_paper as fcns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from tkinter.filedialog import askdirectory
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy import signal
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load data file output from headscan_gen2_paper.py
f = open('allData_contMeanPulses.pkl','rb')
stupidDict = pickle.load(f)
f.close()
scans = stupidDict['scans']

scanPath = '/Users/brad/Desktop/gen2 data/TCD'
batchName = 'LongScanUPenn_all'
savePath = '/Users/brad/Desktop/gen2 results/' + batchName
shortNames,scanNames,bilatTypes,varCorTypes,lsTypes,tcdNames,tcdMarks,tcdMarksOffset,peakLocs,trofLocs = batchfilePaper.LongScan_breathHoldPaper()
scanPath = '/Users/brad/Desktop/gen2 data/TCD'

shortNames = np.array(shortNames)

scanRange = [1,2,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26] # full paper
hold2use =  [1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dataNames = []
for nameInd in range(len(scanRange)):
    dataNames.append(shortNames[scanRange[nameInd]]+'.'+str(hold2use[nameInd]))

dropsSubj = [10, 16, 19,] #  subjects that require manual segmentation
dropsRng = [[[10940,12712]], [[10443,10733],[11860,12232],[12870,13100],[14112,14840]], [[10585,11305],[23197,23760]],]

# 0 envTCD
# 1 meanTCD
# 2 pulIndTCD
# 3 hRateTCD

# 4 rBfiMerged
# 5 rBfiAvgMerged
# 6 rBfiAmpMerged
# 7 rBviMerged
# 8 rBviAvgMerged
# 9 rBviAmpMerged
# 10 hRateMerged

# 11 contMerged
# 12 meanMerged
# 13 contAvgMerged
# 14 meanAvgMerged

feats = ['TCD Env.','TCD Avg.','TCD PI','TCD HR',
         'rBFI','rBFI Avg.','rBFI Amp.',
         'rBVI','rBVI Avg.','rBVI Amp.',
         'rBFI HR',
         'Contrast','Mean','Contrast Avg.','Mean Avg.',
         'TCD stat','rBFI stat','TCD stat2','rBFI stat2',
         'TCD Amp.','rBFI PI','rBVI PI',
         'rBFI (right)','BVI (right)','Contrast Avg. (right)','Mean Avg. (right)',
         ]
feats = ['rCBFv Env.','rCBFv','rCBFv PI','rCBFv HR',
         'rBF Env.','rBF','rBF Amp.',
         'rBV Env.','rBV','rBV Amp.',
         'rBF HR',
         'Contrast','Mean','Contrast Avg.','Mean Avg.',
         'rCBFv stat','rBF stat','rCBFv stat2','rBF stat2',
         'rCBFv Amp.','rBF PI','rBV PI',
         'rBF (right)','rBV (right)','Contrast Avg. (right)','Mean Avg. (right)',
         ]

numFeats = len(feats)
rows, cols = (len(dataNames), numFeats)
baselineData = [[0 for i in range(cols)] for j in range(rows)] # length of 8 attributes
holdWinData = [[0 for i in range(cols)] for j in range(rows)] # length of 8 attributes
holdWinLength = [[] for j in range(rows)]

#%%

for listInd,scanInd in enumerate(scanRange):
    # print(listInd,scanInd)
    print('Scan Number: ' + str(scanInd) + ' ' + shortNames[scanInd])
    
    holdInd = hold2use[listInd]
    
    freqTCD = 125
    freqOpt = 40
    avgWin = 5
    
    tcdMarks[scanInd][0] = tcdMarks[scanInd][0] + tcdMarksOffset[scanInd][0]
    tTCD, meanTCD, envTCD, etCO2, respRate, pulIndTCD, envTCDraw = fcns.loadTCDdata(scanPath,tcdNames[scanInd],tcdMarks[scanInd])
    
    meanTCDow, ampTCDow, periodTCDow, medTCDow = fcns.getMetrics(envTCD,freqTCD,tcdNames[scanInd])
    hRateTCD = freqTCD/periodTCDow*60
    
    plotTitle = shortNames[scanInd]
    dataVertNorm,pulsesStarts,pulsesToUse,pulses,pulsesVertNorm,pulsesVertHorNorm,pulsesStats,dataTcdStats = fcns.processWaveforms(envTCD,freqTCD,0,freqTCD,envTCD,plotTitle,False)
    meanTCDow = dataTcdStats[:,1]
    ampTCDow = dataTcdStats[:,2]
    periodTCDow = dataTcdStats[:,0]
    hRateTCD = freqTCD/periodTCDow*60
    
    # Setting poor TCD signal area to NaNs
    if scanInd in dropsSubj:
        for dropInd in range(len(dropsRng[dropsSubj.index(scanInd)])):
            rng = np.array(dropsRng[dropsSubj.index(scanInd)][dropInd])
            print(scanInd,rng)
            
            meanTCDow[rng[0]:rng[1]] = np.nan
            ampTCDow[rng[0]:rng[1]] = np.nan
            medTCDow[rng[0]:rng[1]] = np.nan
            hRateTCD[rng[0]:rng[1]] = np.nan
    
    camInd = 1 # 1=left 0=right side of head
    stat1 = 3
    stat2 = 7
    
    rBfiMerged = scans[scanInd]['mergedData_rBFI'][camInd]
    rBfiAvgMerged = scans[scanInd]['mergedData_rBFI_avg'][camInd]
    rBfiAmpMerged = scans[scanInd]['mergedData_rBFI_amp'][camInd]
    rBviMerged = scans[scanInd]['mergedData_rBVI'][camInd]
    rBviAvgMerged = scans[scanInd]['mergedData_rBVI_avg'][camInd]
    rBviAmpMerged = scans[scanInd]['mergedData_rBVI_amp'][camInd]
    hRateMerged = scans[scanInd]['moduleData_rBFI_hr'][camInd]
    contMerged = scans[scanInd]['mergedData_cont'][camInd]
    meanMerged = scans[scanInd]['mergedData_mean'][camInd]
    contAvgMerged = scans[scanInd]['mergedData_mean_avg'][camInd]
    meanAvgMerged = scans[scanInd]['mergedData_cont_avg'][camInd]
    
    rBfiStatMerged = scans[scanInd]['mergedData_rBFI_stats'][camInd][:,stat1]
    rBfiStatMerged2 = scans[scanInd]['mergedData_rBFI_stats'][camInd][:,stat2]
    
    tcdStats = dataTcdStats[:,stat1]
    tcdStats2 = dataTcdStats[:,stat2]
    
    # 0 pulsePeriod,pulseAvg,pulseAmp,
    # 3 pulseVertHorNorm_Skew,pulseVertHorNorm_Kurt,pulseVertNormAuc_Skew,pulseVertNormAuc_Kurt,
    # 7 pulseVertNorm_Centroid[0],pulseVertNorm_Centroid[1],
    # 9 canopy,onset,onsetProp,secMoment,
    # 13 velCurInd,velCurIndNorm,velCurIndHann,velCurIndHannNorm,
    
    # hRateTCD = 1/(dataPeriod/freqTCD)*60
    C_min = 0.09
    C_max = 0.30
    
    print('Hold:' + str(holdInd) + ' listInd:' + str(listInd) + ' ' + str(dataNames[listInd]))
            
    tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
    ptsTCD = np.array([tcdMarksZerod[holdInd],tcdMarksZerod[holdInd+1]])
    # ptsOpt = np.array([tcdMarksZerod[holdInd]/freqTCD*freqOpt,tcdMarksZerod[holdInd+1]/freqTCD*freqOpt]).astype(int)
    
    holdWinLength[listInd] = np.diff(ptsTCD)[0]/freqTCD
    
    baselineData[listInd][0], holdWinData[listInd][0] = fcns.TCDcrop(envTCD,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][1], holdWinData[listInd][1] = fcns.TCDcrop(meanTCDow,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][2], holdWinData[listInd][2] = fcns.TCDcrop(ampTCDow/meanTCDow,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][3], holdWinData[listInd][3] = fcns.TCDcrop(hRateTCD,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][16], holdWinData[listInd][16] = fcns.TCDcrop(tcdStats,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][18], holdWinData[listInd][18] = fcns.TCDcrop(tcdStats2,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][19], holdWinData[listInd][19] = fcns.TCDcrop(ampTCDow,ptsTCD,freqTCD,avgWin) #-1
    
    baselineData[listInd][4], holdWinData[listInd][4] = fcns.TCDcrop(rBfiMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][5], holdWinData[listInd][5] = fcns.TCDcrop(rBfiAvgMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][6], holdWinData[listInd][6] = fcns.TCDcrop(rBfiAmpMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][7], holdWinData[listInd][7] = fcns.TCDcrop(rBviMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][8], holdWinData[listInd][8] = fcns.TCDcrop(rBviAvgMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][9], holdWinData[listInd][9] = fcns.TCDcrop(rBviAmpMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][10], holdWinData[listInd][10] = fcns.TCDcrop(hRateMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][11], holdWinData[listInd][11] = fcns.TCDcrop(contMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][12], holdWinData[listInd][12] = fcns.TCDcrop(meanMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][13], holdWinData[listInd][13] = fcns.TCDcrop(contAvgMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][14], holdWinData[listInd][14] = fcns.TCDcrop(meanAvgMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][15], holdWinData[listInd][15] = fcns.TCDcrop(rBfiStatMerged,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][17], holdWinData[listInd][17] = fcns.TCDcrop(rBfiStatMerged2,ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][20], holdWinData[listInd][20] = fcns.TCDcrop(rBfiAmpMerged/((C_max-contAvgMerged)/(C_max-C_min)),ptsTCD,freqTCD,avgWin) #-1
    baselineData[listInd][21], holdWinData[listInd][21] = fcns.TCDcrop(rBviAmpMerged/(1/meanAvgMerged),ptsTCD,freqTCD,avgWin) #-1
    
    # Updates to allow comparison to right side data
    rBfiAvgMerged = scans[scanInd]['mergedData_rBFI_avg'][0]
    rBviAvgMerged = scans[scanInd]['mergedData_rBVI_avg'][0]
    contAvgMerged = scans[scanInd]['mergedData_mean_avg'][0]
    meanAvgMerged = scans[scanInd]['mergedData_cont_avg'][0]
    baselineData[listInd][22], holdWinData[listInd][22] = fcns.TCDcrop(rBfiAvgMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][23], holdWinData[listInd][23] = fcns.TCDcrop(rBviAvgMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][24], holdWinData[listInd][24] = fcns.TCDcrop(contAvgMerged,ptsTCD,freqTCD,avgWin) #+1
    baselineData[listInd][25], holdWinData[listInd][25] = fcns.TCDcrop(meanAvgMerged,ptsTCD,freqTCD,avgWin) #+1

#%% Finding short drop outs in TCD data for removal
    
dropsHold = [15, 27,27,27,27, 33,34,] #  holds that require manual segmentation
dropsSubj1 = [10, 16,16,16,16, 19,19,] #  subjects that require manual segmentation
dropsRng1 = [[11054,12712], [10547,10733],[11860,12190],[12782,13100],[14122,14840], [10585,11305], [23270,23755],] # based on evelope
dropsRng1 = [[10940,12712], [10443,10733],[11860,12232],[12870,13100],[14112,14840], [10585,11305], [23197,23760],] # based on PI/pulses
# set these regions to NaNs

holdInd = 1
# plt.plot(scans[dropsSubj1[scanInd]]['data_TCD'])
# plt.show()

for scanInd in range(0): #range(len(dropsRng1)):
    # ptsTCD = np.array([tcdMarksZerod[holdInd],tcdMarksZerod[holdInd+1]])
    # plt.plot(scans[dropsSubj1[scanInd]]['data_TCD'][(ptsTCD[0]-avgWin*freqTCD):(ptsTCD[1]+avgWin*freqTCD),1])
    plt.plot(scans[dropsSubj1[scanInd]]['data_TCD'])
    plt.plot([dropsRng1[scanInd][0],dropsRng1[scanInd][0]],[0,100],'k')
    plt.plot([dropsRng1[scanInd][1],dropsRng1[scanInd][1]],[0,100],'k')
    
    tcdMarksZerod = np.array(tcdMarks[dropsSubj1[scanInd]]) - tcdMarks[dropsSubj1[scanInd]][0]
    plt.plot(scans[dropsSubj1[scanInd]]['data_TCD_avg'])
    ax2 = plt.twinx()
    ax2.plot(scans[dropsSubj1[scanInd]]['data_TCD_amp']/scans[dropsSubj1[scanInd]]['data_TCD_avg'])
    
    plt.xlim(dropsRng1[scanInd][0]-250,dropsRng1[scanInd][1]+250)
    #plt.xlim(tcdMarksZerod[holdInd]-625-125,tcdMarksZerod[holdInd+1]+625+125)
    # plt.xlim(14700,14940)
    plt.title(tcdNames[dropsSubj1[scanInd]])
    plt.show()
    
#%% Plotting and saving all TCD Waveforms
for scanInd in range(0): #range(3,26):
# ind = 26

    tcdMarks[scanInd][0] = tcdMarks[scanInd][0] + tcdMarksOffset[scanInd][0]
    tcdMarksZerod = np.array(tcdMarks[scanInd]) - tcdMarks[scanInd][0]
    tTCD = np.arange(scans[scanInd]['data_TCD'].shape[0])/freqTCD
    holdsData = np.zeros(tTCD.shape)
    
    if len(tcdMarks[scanInd]) == 8:
        holds = [1,3,5]
    elif len(tcdMarks[scanInd]) == 6:
        holds = [1,3]
    elif len(tcdMarks[scanInd]) == 4:
        holds = [1]
    for ind, holdInd in enumerate(holds):
        # print(ind,holdInd)
        holdsData[tcdMarksZerod[holdInd]:tcdMarksZerod[holdInd+1]] = 1
    
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,13))
    for tInd in range(5):
        cTime = [tInd*24*125,(tInd+1)*24*125]
        cTimeStr = '_diff' + str(cTime[0]).zfill(3) + '-' + str(cTime[1]).zfill(3)
        ax[tInd].plot(tTCD[cTime[0]:cTime[1]],scans[scanInd]['data_TCD'][cTime[0]:cTime[1],1])
        if scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0]!=[]:
            holdsMax = scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0].max()
            holdsMin = scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0].min()
            holdsNorm = holdsData[cTime[0]:cTime[1]]*holdsMax
            holdsNorm[holdsNorm==0] = holdsMin
        ax[tInd].plot(tTCD[cTime[0]:cTime[1]],holdsNorm)
        ax[tInd].set_xlim(tTCD[cTime[0]],tTCD[cTime[1]])
        fig.suptitle(tcdNames[scanInd][4:])
    fig.savefig(savePath + '/' + tcdNames[scanInd][4:] + '_1.png',dpi=300)
    
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(24,13))
    for tInd in range(5,10):
        cTime = [tInd*24*125,(tInd+1)*24*125]
        cTimeStr = '_diff' + str(cTime[0]).zfill(3) + '-' + str(cTime[1]).zfill(3)
        ax[tInd-5].plot(tTCD[cTime[0]:cTime[1]],scans[scanInd]['data_TCD'][cTime[0]:cTime[1],1])
        if scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0]!=[]:
            holdsMax = scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0].max()
            holdsMin = scans[scanInd]['data_TCD'][cTime[0]:cTime[1],0].min()
            holdsNorm = holdsData[cTime[0]:cTime[1]]*holdsMax
            holdsNorm[holdsNorm==0] = holdsMin
        ax[tInd-5].plot(tTCD[cTime[0]:cTime[1]],holdsNorm)
        ax[tInd-5].set_xlim(cTime[0]/125,cTime[1]/125)
        fig.suptitle(tcdNames[scanInd][4:])
    fig.savefig(savePath + '/' + tcdNames[scanInd][4:] + '_2.png',dpi=300)

#%% Plots for comparing Multigon vs OW method for calculating mean
from scipy.signal import convolve

tOpt = np.arange(meanTCDow.shape[0])/freqTCD
tTCD = np.arange(envTCD.shape[0])/freqTCD

for tInd in range(0): #range(3):
    plt.plot(tTCD,envTCD, linewidth=0.5)
    plt.plot(tOpt+0,meanTCDow)
    wind = np.concatenate((np.zeros((125*5)),np.ones((125*5))),axis=0)/(125*5)
    dataMedSmooth = convolve(medTCDow,wind,mode='same',method='direct')
    plt.plot(tTCD,np.round(dataMedSmooth))
    plt.plot(tTCD,meanTCD)
    # plt.title([np.nanmean(hRateTCD),np.nanmean(hRateMerged)])
    plt.legend(['TCD env','TCD Mean','TCD env calc. median, asym filt', 'TCD env calc. mean']) # ,'TCD env calc. median'
    plt.xlim(tInd*80,tInd*80+80)
    plt.savefig(savePath + '/' + 'img_' + str(tInd) + '.png',dpi=300)
    plt.show()

#%% Determining Optical Time Shift Per Subject
if 0:
    scanInd=14
    ind=1
    timeDelay = fcns.calcTimeDelay(freqTCD/scans[scanInd]['data_TCD_period']*60,scans[scanInd]['moduleData_rBFI_hr'][0],250,1)
    
    data1 = holdWinData[scanInd*2+ind][3]
    data2 = holdWinData[scanInd*2+ind][10]
    
    tdata1 = np.arange(data1.shape[0])/freqTCD
    plt.plot(tdata1,data1)
    plt.plot(tdata1,data2)
    # plt.xlim(3,18)
    plt.grid()
  
#%% Plots of individual holds with color-code timing
if 0:
    scanInd=11
    ind=1
    data1 = holdWinData[scanInd*2+ind][0]
    data2 = holdWinData[scanInd*2+ind][10]
    
    tdata1 = np.arange(data1.shape[0])/freqTCD
    plt.plot(tdata1,data1)
    plt.xlim(30,40)
    plt.grid()
    plt.show()

    scanInd=11
    ind=1
    cmp = [13,1]
    for scanInd in range(2):
        
        data1 = holdWinData[scanInd*2+ind][cmp[0]]
        data2 = holdWinData[scanInd*2+ind][cmp[1]]
        
        tOpt = np.arange(data1.shape[0])/freqOpt
        
        plt.scatter(data2, data1, c=tOpt, s=10, cmap='jet')
        plt.colorbar()
        plt.show()    

#%%
def quickScatter(baselineData,holdWinData,dataNames,cmp,feats,saveName,expChg,avgWin):
    
    # Get alternative variable for when calculating rBF & rBV
    cmp2 = copy.deepcopy(np.array(cmp))
    cmp2[cmp2 == 5] = 13
    cmp2[cmp2 == 8] = 14
    cmp2[cmp2 == 22] = 24
    cmp2[cmp2 == 23] = 25
    
    freqTCD = 125
    C_min = 0.09
    
    chgNaming = ['Maximum ','Maximum ']
    if expChg[0] == -1:
        chgNaming[0] = 'Minimum '
    if expChg[1] == -1:
        chgNaming[1] = 'Minimum '
    
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11,8))
    fig.tight_layout(w_pad=3.5,h_pad=5.25)
    plt.subplots_adjust(top=0.85)
    colors = np.concatenate((plt.get_cmap('tab10')(np.linspace(0, 1, 10)),plt.get_cmap('Set1')(np.linspace(0, 1, 9)),plt.get_cmap('Dark2')(np.linspace(0, 1,8))),axis=0)
    colors = ['black','grey','lightcoral','maroon','red','sienna','darkorange','yellowgreen','cyan','gold', #1-10
              'tan','olive','yellow','forestgreen','lime','turquoise','cadetblue','deepskyblue','cornflowerblue','blue', #11-20
              'darkviolet','magenta','pink'] # 21-22
    # colors = plt.get_cmap('tab20')(np.linspace(0, 1, holdWinData)) # jet gist_rainbow tab20
    
    
    # Plots all values 5sec before hold to 5 sec after hold
    dataTcd_all = []
    dataOpt_all = []
    rVals = []
    for scanInd in range(len(holdWinData)):
        dataTcd = np.array(holdWinData[scanInd][cmp[0]])
        dataOpt = np.array(holdWinData[scanInd][cmp[1]])
        
        ax[0,0].scatter(dataTcd,dataOpt,s=1,color=colors[scanInd])
        dataTcd = dataTcd[~np.isnan(dataOpt)]
        dataOpt = dataOpt[~np.isnan(dataOpt)]
        dataOpt = dataOpt[~np.isnan(dataTcd)]
        dataTcd = dataTcd[~np.isnan(dataTcd)]
        regr = linear_model.LinearRegression()
        regr.fit(dataTcd.reshape(-1, 1),dataOpt.reshape(-1, 1))
        dataOpt_pred = regr.predict(dataTcd.reshape(-1, 1))
        ax[0,0].plot(dataTcd,dataOpt_pred,color=colors[scanInd])
        # ax[0,0].annotate("   R:%.2f" % (r2_score(dataOpt.reshape(-1, 1),dataOpt_pred))**0.5, (dataTcd.mean(), dataOpt_pred.mean()),color=colors[scanInd])
        dataTcd_all = np.append(dataTcd_all,dataTcd,axis=0)
        dataOpt_all = np.append(dataOpt_all,dataOpt,axis=0)
        rVals = np.append(rVals,(r2_score(dataOpt.reshape(-1, 1),dataOpt_pred))**0.5)
    ax[0,0].set_xlabel(feats[cmp[0]])
    ax[0,0].set_ylabel(feats[cmp[1]])
    dataTcd_all = dataTcd_all[~np.isnan(dataOpt_all)]
    dataOpt_all = dataOpt_all[~np.isnan(dataOpt_all)]
    #ax[0,0].set_title('All data points within hold +/-5sec\n' + 'Correlation (Pearson): %.2f' % scipy.stats.pearsonr(dataTcd_all, dataOpt_all).statistic)
    ax[0,0].set_title('Correlation (All): %.2f,\nCorrelation (Individuals): %.2f +/- %.2f' 
                      % (scipy.stats.pearsonr(dataTcd_all, dataOpt_all).statistic,rVals.mean(),rVals.std()))
    
    # Plots 2 points, baseline is avg of 30 sec before hold, second is max or min of +/- 5 sec at end of hold
    minMaxTcd_all = []
    minMaxOpt_all = []
    for scanInd in range(len(holdWinData)):
        if expChg[0] > 0:
            minMaxTcd = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
            # *** TypeError: 'int' object is not subscriptable ------>>>>>
            # means data doesn't exist in holdWinData, likely mismatch between dataRng and scanRange
        else:
            minMaxTcd = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        if expChg[1] > 0:
            minMaxOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            minMaxOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        minMaxTcd = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxTcd])
        minMaxOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt])
        
        #ax[0,1].quiver(data3[0],data4[0],np.diff(data3),np.diff(data4),scale_units='xy', angles='xy', scale=1, color=colors[scanInd])
        ax[0,1].plot(minMaxTcd[0],minMaxOpt[0],'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
        ax[0,1].plot(minMaxTcd,minMaxOpt,'s',ls='-', ms=5, markevery=[-1], color=colors[scanInd])
        minMaxTcd_all = np.append(minMaxTcd_all,minMaxTcd,axis=0)
        minMaxOpt_all = np.append(minMaxOpt_all,minMaxOpt,axis=0)
    ax[0,1].set_xlabel(feats[cmp[0]])
    ax[0,1].set_ylabel(feats[cmp[1]])
    minMaxTcd_all = minMaxTcd_all[~np.isnan(minMaxOpt_all)]
    minMaxOpt_all = minMaxOpt_all[~np.isnan(minMaxOpt_all)]
    minMaxOpt_all = minMaxOpt_all[~np.isnan(minMaxTcd_all)]
    minMaxTcd_all = minMaxTcd_all[~np.isnan(minMaxTcd_all)]
    ax[0,1].set_title('\u25CF avg. of start +/-5sec of hold\n\u25A0 min/max of end +/-5sec of hold\n' + 'Correlation (Pearson): %.2f' % scipy.stats.pearsonr(dataTcd_all, dataOpt_all).statistic)
    
    topRow_xlims = np.min([np.nanmin(dataTcd_all),np.nanmin(minMaxTcd_all)]),np.max([np.nanmax(dataTcd_all),np.nanmax(minMaxTcd_all)])
    topRow_ylims = np.min([np.nanmin(dataOpt_all),np.nanmin(minMaxOpt_all)]),np.max([np.nanmax(dataOpt_all),np.nanmax(minMaxOpt_all)])
    ax[0,0].set_xlim(topRow_xlims)
    ax[0,0].set_ylim(topRow_ylims)
    ax[0,1].set_xlim(topRow_xlims)
    ax[0,1].set_ylim(topRow_ylims)
    
    # Plots difference of previous plot's two points
    diffTcd_all = []
    diffOpt_all = []
    for scanInd in range(len(holdWinData)):
        if expChg[0] > 0:
            minMaxTcd = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        else:
            minMaxTcd = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        if expChg[1] > 0:
            minMaxOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            minMaxOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        
        diffTcd = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxTcd])
        diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt])
        ax[0,2].plot(np.diff(diffTcd),np.diff(diffOpt),'o', color=colors[scanInd])
        diffTcd_all = np.append(diffTcd_all,diffTcd,axis=0)
        diffOpt_all = np.append(diffOpt_all,diffOpt,axis=0)
    ax[0,2].set_xlabel('\u0394 ' + feats[cmp[0]])
    ax[0,2].set_ylabel('\u0394 ' + feats[cmp[1]])
    diffTcd_all = diffTcd_all[~np.isnan(diffOpt_all)]
    diffOpt_all = diffOpt_all[~np.isnan(diffOpt_all)]
    diffOpt_all = diffOpt_all[~np.isnan(diffTcd_all)]
    diffTcd_all = diffTcd_all[~np.isnan(diffTcd_all)]
    ax[0,2].set_title('Corresponding \u0394 Values\n' + 'Correlation (Pearson): %.2f' % scipy.stats.pearsonr(diffTcd_all, diffOpt_all).statistic)
    # ax[0,2].legend(dataNames[dataRng])
    
    
    
    # Plots all values 5sec before hold to 5 sec after hold, normalized to 30 second baseline before first hold
    dataTcdNorm_all = []
    dataOptNorm_all = []
    rVals = []
    for scanInd in range(len(holdWinData)):
        # normFactTcd = np.nanmean(baselineData[scanInd][cmp[0]])
        # data1 = (np.array(holdWinData[scanInd][cmp[0]])/normFactTcd)*100
        
        
        if cmp[0] in [5,22]: # rBFI average
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp2[0]])
            # data2 = (1+((normFactOpt-holdWinData[scanInd][cmp2[0]])/(normFactOpt-conMin)))*100
            
            # 1 + ((C_baseline-C_min) - (C_data-C_min))/(C_baseline-C_min) = 1 + (C_baseline - C_data)/(C_baseline-C_min)
            T_min = 0.09 # Contrast value of demodulated seed
            T_max = [] # Phantom data, unused
            T_baseline = np.nanmean(baselineData[scanInd][cmp2[0]]) # 30 sec prior to hold
            T_data = holdWinData[scanInd][cmp2[0]] # hold data plus 5 sec before and after
            dataTcdNorm = ( 1+( (T_baseline-T_data) / (T_baseline-T_min) ) )*100
        elif cmp[0] in [8,23]: # rBVI average
            # 1 + (I_baseline - I_data) / I_baseline
            T_baseline = np.nanmean(baselineData[scanInd][cmp2[0]])
            T_data = holdWinData[scanInd][cmp2[0]]
            dataTcdNorm = ( 1+( (T_baseline-T_data) / T_baseline ) )*100
        else:
            # 1 + (T_data - T_baseline) / T_baseline = T_data / T_baseline
            T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
            T_data = np.array(holdWinData[scanInd][cmp[0]])
            dataTcdNorm = ( T_data / T_baseline )*100
        
        if cmp[1] in [5,22]: # rBFI average
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp2[1]])
            # data2 = (1+((normFactOpt-holdWinData[scanInd][cmp2[1]])/(normFactOpt-conMin)))*100
            
            # 1 + ((C_baseline-C_min) - (C_data-C_min))/(C_baseline-C_min) = 1 + (C_baseline - C_data)/(C_baseline-C_min)
            C_min = 0.09 # Contrast value of demodulated seed
            C_max = [] # Phantom data, unused
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]]) # 30 sec prior to hold
            C_data = holdWinData[scanInd][cmp2[1]] # hold data plus 5 sec before and after
            dataOptNorm = ( 1+( (C_baseline-C_data) / (C_baseline-C_min) ) )*100
        elif cmp[1] in [8,23]: # rBVI average
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp2[1]])
            # data2 = (1+((normFactOpt-holdWinData[scanInd][cmp2[1]])/normFactOpt))*100
            
            # 1 + (I_baseline - I_data) / I_baseline
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_data = holdWinData[scanInd][cmp2[1]]
            dataOptNorm = ( 1+( (I_baseline-I_data) / I_baseline ) )*100
        else:
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp[1]])
            # data2 = ((holdWinData[scanInd][cmp[1]])/normFactOpt)*100
            
            # 1 + (X_data - X_baseline) / X_baseline = X_data / X_baseline
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_data = holdWinData[scanInd][cmp[1]]
            dataOptNorm = ( X_data / X_baseline )*100
        
        ax[1,0].scatter(dataTcdNorm,dataOptNorm,s=1,color=colors[scanInd])
        # Plotting linear regression for each subject
        dataTcdNorm = dataTcdNorm[~np.isnan(dataOptNorm)]
        dataOptNorm = dataOptNorm[~np.isnan(dataOptNorm)]
        dataOptNorm = dataOptNorm[~np.isnan(dataTcdNorm)]
        dataTcdNorm = dataTcdNorm[~np.isnan(dataTcdNorm)]
        regr = linear_model.LinearRegression()
        regr.fit(dataTcdNorm.reshape(-1, 1),dataOptNorm.reshape(-1, 1))
        dataOptNorm_pred = regr.predict(dataTcdNorm.reshape(-1, 1))
        ax[1,0].plot(dataTcdNorm,dataOptNorm_pred,color=colors[scanInd])
        # print(dataNames[scanInd],int(np.nanmax(dataTcdNorm)-np.nanmin(dataTcdNorm)),int(np.nanmax(dataOptNorm_pred)-np.nanmin(dataOptNorm_pred)))
        # print(dataNames[scanInd],slope)

        # ax[1,0].annotate("   R:%.2f" % (r2_score(dataOptNorm.reshape(-1, 1),dataOptNorm_pred))**0.5, (dataTcdNorm.mean(), dataOptNorm_pred.mean()),color=colors[scanInd])
        # slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(dataTcdNorm,dataOptNorm)
        
        
        dataTcdNorm_all = np.append(dataTcdNorm_all,dataTcdNorm,axis=0)
        dataOptNorm_all = np.append(dataOptNorm_all,dataOptNorm,axis=0)
        rVals = np.append(rVals,(r2_score(dataOptNorm.reshape(-1, 1),dataOptNorm_pred))**0.5)
    if cmp[0] in [1,5,8,22,23]:
        ax[1,0].set_xlabel(feats[cmp[0]] + ' Beat-to-Beat Mean (%)')
    else:
        ax[1,0].set_xlabel(feats[cmp[0]] + ' (%)')
    if cmp[1] in [5,8,22,23]:
        ax[1,0].set_ylabel(feats[cmp[1]] + ' Beat-to-Beat Mean (%)')
    else:
        ax[1,0].set_ylabel(feats[cmp[1]] + ' (%)')
    
    dataTcdNorm_all = dataTcdNorm_all[~np.isnan(dataOptNorm_all)]
    dataOptNorm_all = dataOptNorm_all[~np.isnan(dataOptNorm_all)]
    #ax[1,0].set_title('Correlation (Pearson): %.2f,\ntest' % scipy.stats.pearsonr(dataTcdNorm_all, dataOptNorm_all).statistic)
    ax[1,0].set_title('Correlation (All): %.2f,\nCorrelation (Individuals): %.2f +/- %.2f' 
                      % (scipy.stats.pearsonr(dataTcdNorm_all, dataOptNorm_all).statistic,rVals.mean(),rVals.std()))
    
    # Plots 2 points, baseline is avg of 30 sec before hold, second is max or min of +/- 5 sec at end of hold, normalized to 30 second baseline before first hold
    minMaxTcdNorm_all = []
    minMaxOptNorm_all = []
    for scanInd in range(len(holdWinData)):
        if expChg[0] > 0:
            minMaxTcd = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        else:
            minMaxTcd = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        if expChg[1] > 0:
            minMaxOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            minMaxOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        
        # X-AXIS normalizations
        if cmp[0] in [5,22]:
            T_baseline = np.nanmean(baselineData[scanInd][cmp2[0]])
            T_minMaxTcd = [np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxTcd]-np.nanmean(baselineData[scanInd][cmp[0]])
            minMaxTcdNorm = (1+T_minMaxTcd/(T_baseline-C_min))*100
        elif cmp[0] in [8,23]:
            T_baseline = np.nanmean(baselineData[scanInd][cmp2[0]])
            T_minMaxTcd = [np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxTcd]-np.nanmean(baselineData[scanInd][cmp[0]])
            minMaxTcdNorm = (1+T_minMaxTcd/T_baseline)*100
        else:
            T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
            T_minMaxTcd = [np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxTcd]
            minMaxTcdNorm = T_minMaxTcd/T_baseline*100
        
        # Y-AXIS normalization
        if cmp[1] in [5,22]:
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            C_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]-np.nanmean(baselineData[scanInd][cmp[1]])
            minMaxOptNorm = (1+C_minMaxOpt/(C_baseline-C_min))*100
        elif cmp[1] in [8,23]:
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]-np.nanmean(baselineData[scanInd][cmp[1]])
            minMaxOptNorm = (1+I_minMaxOpt/I_baseline)*100
        else:
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]
            minMaxOptNorm = X_minMaxOpt/X_baseline*100
        
        # ax[1,1].quiver(minMaxTcdNorm[0],minMaxOptNorm[0],np.diff(minMaxTcdNorm),np.diff(minMaxOptNorm),scale_units='xy', angles='xy', scale=1, color=colors[scanInd])
        ax[1,1].plot(minMaxTcdNorm[0],minMaxOptNorm[0],'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
        ax[1,1].plot(minMaxTcdNorm,minMaxOptNorm,'s',ls='-', ms=5, markevery=[-1], color=colors[scanInd])
        minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm,axis=0)
        minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm,axis=0)
        # If only interested in the baseline data point
        if 0:
            ax[1,1].plot(minMaxTcdNorm[0],minMaxOptNorm[0],'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
            minMaxTcdNorm_all = np.append(minMaxTcdNorm_all,minMaxTcdNorm[0])
            minMaxOptNorm_all = np.append(minMaxOptNorm_all,minMaxOptNorm[0])
        
    ax[1,1].set_xlabel(feats[cmp[0]] + ' (%)')
    ax[1,1].set_ylabel(feats[cmp[1]] + ' (%)')
    # minMaxTcdNorm = minMaxTcdNorm.reshape((-1,1))
    # minMaxOptNorm = minMaxOptNorm.reshape((-1,1))
    minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxOptNorm_all)]
    minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxOptNorm_all)]
    minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxTcdNorm_all)]
    minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxTcdNorm_all)]
    ax[1,1].set_title('Correlation (All): %.2f' % scipy.stats.pearsonr(minMaxTcdNorm_all, minMaxOptNorm_all).statistic)
    
    
    botRow_xlims = np.min([np.nanmin(dataTcdNorm_all),np.nanmin(minMaxTcdNorm_all)]),np.max([np.nanmax(dataTcdNorm_all),np.nanmax(minMaxTcdNorm_all)])
    botRow_ylims = np.min([np.nanmin(dataOptNorm_all),np.nanmin(minMaxOptNorm_all)]),np.max([np.nanmax(dataOptNorm_all),np.nanmax(minMaxOptNorm_all)])
    ax[1,0].set_xlim(botRow_xlims)
    ax[1,0].set_ylim(botRow_ylims)
    ax[1,1].set_xlim(botRow_xlims)
    ax[1,1].set_ylim(botRow_ylims)
    
    # Plots difference of previous plot's two points
    diffTcdNorm_all = []
    diffOptNorm_all = []
    for scanInd in range(len(holdWinData)):
        if expChg[0] > 0:
            diffTcdNorm = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        else:
            diffTcdNorm = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        if expChg[1] > 0:
            diffOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            diffOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        
        T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
        diffTcdNorm = copy.deepcopy([np.nanmin(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),diffTcdNorm])
        diffTcdNorm = np.diff(diffTcdNorm/T_baseline*100)
        
        if cmp[1] in [5,22]:
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            C_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])-np.nanmean(baselineData[scanInd][cmp[1]])
            diffOptNorm = np.diff((1+C_diffOpt/(C_baseline-C_min))*100)
        elif cmp[1] in [8,23]:
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])-np.nanmean(baselineData[scanInd][cmp[1]])
            diffOptNorm = np.diff((1+I_diffOpt/I_baseline)*100)
        else:
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])
            diffOptNorm = np.diff(X_diffOpt/X_baseline*100)
        
        ax[1,2].plot(diffTcdNorm,diffOptNorm,'o', color=colors[scanInd])
        diffTcdNorm_all = np.append(diffTcdNorm_all,diffTcdNorm,axis=0)
        diffOptNorm_all = np.append(diffOptNorm_all,diffOptNorm,axis=0)
    ax[1,2].set_xlabel('\u0394 ' + feats[cmp[0]] + ' (%)')
    ax[1,2].set_ylabel('\u0394 ' + feats[cmp[1]] + ' (%)')
    diffTcdNorm_all = diffTcdNorm_all[~np.isnan(diffOptNorm_all)]
    diffOptNorm_all = diffOptNorm_all[~np.isnan(diffOptNorm_all)]
    diffOptNorm_all = diffOptNorm_all[~np.isnan(diffTcdNorm_all)]
    diffTcdNorm_all = diffTcdNorm_all[~np.isnan(diffTcdNorm_all)]
    ax[1,2].set_title('Correlation (Pearson): %.2f' % scipy.stats.pearsonr(diffTcdNorm_all, diffOptNorm_all).statistic)
    ax[1,2].legend(dataNames)
    
    # holder = ax[0,0].get_xticks()
    # ax[0,0].set_xticks(holder)
    # ax[0,0].set_yticks(holder)
    
    for rowInd in range(2):
        for colInd in range(3):
            # ax[rowInd,colInd].set_aspect('equal', adjustable='box')
            ax[rowInd,colInd].set_box_aspect(1)
            ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
            ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
            
    # topRow_lims = [np.min([topRow_xlims[0],topRow_ylims[0]]),np.max([topRow_xlims[1],topRow_ylims[1]])]
    # ax[0,0].set_xlim(topRow_lims)
    # ax[0,0].set_ylim(topRow_lims)
    # ax[0,1].set_xlim(topRow_lims)
    # ax[0,1].set_ylim(topRow_lims)
    botRow_lims = [np.min([botRow_xlims[0],botRow_ylims[0]])*0.9,np.max([botRow_xlims[1],botRow_ylims[1]])*1.05]
    ax[1,0].set_xlim(botRow_lims)
    ax[1,0].set_ylim(botRow_lims)
    if cmp[1] == 20 or cmp[1] == 21:
        # Caveat for PI plots
        lims = [np.min([np.nanmin(minMaxOptNorm_all),np.nanmin(minMaxTcdNorm_all)])*0.9,np.max([np.nanmax(minMaxOptNorm_all),np.nanmax(minMaxTcdNorm_all)])*1.05]
        ax[1,1].set_xlim(lims)
        ax[1,1].set_ylim(lims)
    else:
        ax[1,1].set_xlim(botRow_lims)
        ax[1,1].set_ylim(botRow_lims)
        
    legend_elements = [
                    Line2D([0],[0],marker='o',color='w',label='Pre Hold',markerfacecolor='k',markersize=7),
                    Line2D([0],[0],marker='s',color='w',label='Post Hold',markerfacecolor='k',markersize=7),
                    ]
    ax[0,1].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
    ax[1,1].legend(handles=legend_elements, loc='lower right', fontsize=10,frameon=False,handletextpad=0)
    
    fig.savefig(saveName + '_combined.png',dpi=300,bbox_inches='tight')
    
    
    
    # Plot Time to Min/Max Value
    timeToMaxTCD_all = []
    timeToMaxOpt_all = []
    fig, ax = plt.subplots(ncols=2, figsize=(9,5))
    for scanInd in range(len(holdWinData)):
        dataTCD = np.array(holdWinData[scanInd][cmp[0]])
        dataOpt = np.array(holdWinData[scanInd][cmp[1]])
        timeToMaxTCD = (np.nanargmax(dataTCD[(-freqTCD*10):])+(dataTCD.shape[0]-freqTCD*10))/freqTCD
        timeToMaxOpt = (np.nanargmax(dataOpt[(-freqTCD*10):])+(dataOpt.shape[0]-freqTCD*10))/freqTCD
        ax[0].scatter(timeToMaxTCD,timeToMaxOpt,s=20,color=colors[scanInd])
        timeToMaxTCD_all.append(timeToMaxTCD)
        timeToMaxOpt_all.append(timeToMaxOpt)
    ax[0].set_xlabel('Time to ' + chgNaming[0] + feats[cmp[0]] + ' (seconds)')
    ax[0].set_ylabel('Time to ' + chgNaming[1] + feats[cmp[1]] + ' (seconds)')
    xmin, xmax = ax[0].get_xlim()
    ymin, ymax = ax[0].get_ylim()
    allmin = np.min([xmin,ymin])
    allmax = np.max([xmax,ymax])
    slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(timeToMaxTCD_all,timeToMaxOpt_all)
    ax[0].plot(x_line, y_line, color = 'k',linestyle='--',linewidth=1,zorder=0.95)
    ax[0].fill_between(x_line, y_line + ci, y_line - ci, color = [0.925,0.925,0.925],zorder=0.9)
    ax[0].set_title('Slope: y = %.2f x + %.2f\n Correlation (Pearson): %.2f' % (slope,intercept,r_value))
    
    
    
    minMaxBothNorm_mean_all = []
    minMaxBothNorm_diff_all = []
    for scanInd in range(diffTcdNorm_all.shape[0]):        
        minMaxBothNorm_mean = (timeToMaxOpt_all[scanInd]+timeToMaxTCD_all[scanInd])/2
        minMaxBothNorm_diff = timeToMaxOpt_all[scanInd]-timeToMaxTCD_all[scanInd]
        ax[1].plot(minMaxBothNorm_mean,minMaxBothNorm_diff,'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
        minMaxBothNorm_mean_all = np.append(minMaxBothNorm_mean_all,minMaxBothNorm_mean)
        minMaxBothNorm_diff_all = np.append(minMaxBothNorm_diff_all,minMaxBothNorm_diff)
    xlims11 = ax[1].get_xlim()
    ax[1].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all),(2,1)),'k',linestyle='--',linewidth=1.0,zorder=0.95)
    lowerCI = np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all)
    upperCI = np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all)
    xmin, xmax = ax[1].get_xlim()
    nx = np.array([[xmin,lowerCI],[xmin,upperCI],[xmax,upperCI],[xmax,lowerCI]])
    ax[1].add_patch(plt.Polygon(nx,facecolor=[0.925,0.925,0.925],zorder=0.9))
    ax[1].set_xlim(xlims11)
    
    ax[1].set_xlabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Mean')
    ax[1].set_ylabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Difference')
    ax[1].set_title('Mean +/- 1.96 S.D. shown in black\n(values = %+.1f, %+.1f, %+.1f)' % 
          (lowerCI,np.nanmean(minMaxBothNorm_diff_all),upperCI))
    
    for subInd in range(2):
        ax[subInd].set_box_aspect(1)
        ax[subInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
        ax[subInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
        if subInd == 0:
            plot_lims = [allmin,allmax]
            ax[subInd].set_xlim(plot_lims)
            ax[subInd].set_ylim(plot_lims)
    
    fig.savefig(saveName + '_TimeToMax.png',dpi=300,bbox_inches='tight')



quickScatter(baselineData,holdWinData,dataNames,[1,5],feats,savePath+'/'+batchName+'TCDavg-rBFIavg_max',[+1,+1],avgWin)
quickScatter(baselineData,holdWinData,dataNames,[1,8],feats,savePath+'/'+batchName+'TCDavg-rBVIavg_max',[+1,+1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[2,20],feats,savePath+'/'+batchName+'TCDpi-rBFIpi_max',[-1,-1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[2,21],feats,savePath+'/'+batchName+'TCDpi-rBVIpi_max',[-1,-1],avgWin)
quickScatter(baselineData,holdWinData,dataNames,[22,5],feats,savePath+'/'+batchName+'rBFIavgR-rBFIavgL_max',[+1,+1],avgWin)
quickScatter(baselineData,holdWinData,dataNames,[23,8],feats,savePath+'/'+batchName+'rBVIavgR-rBVIavgL_max',[+1,+1],avgWin)


# quickScatter(baselineData,holdWinData,dataNames,[2,6],feats,savePath+'/'+batchName+'TCDpi-rBFIamp_max',[-1,-1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[2,9],feats,savePath+'/'+batchName+'TCDpi-rBVIamp_max',[-1,-1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[3,10],feats,savePath+'/'+batchName+'OptHR-TCDHR_max',[-1,-1],avgWin)

# quickScatter(baselineData,holdWinData,dataNames,[8,5],feats,savePath+'/'+batchName+'rBVIavg-rBFIavg_max',[+1,+1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[16,15],feats,savePath+'/'+batchName+'statsTest_3_max',[+1,+1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[18,17],feats,savePath+'/'+batchName+'statsTest_7_max',[+1,+1],avgWin)

# quickScatter(baselineData,holdWinData,dataNames,[2,5],feats,savePath+'/'+batchName+'TCDpi-rBFIavg_max',[-1,+1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[2,8],feats,savePath+'/'+batchName+'TCDpi-rBVIavg_max',[-1,+1],avgWin)

# quickScatter(baselineData,holdWinData,dataNames,[19,6],feats,savePath+'/'+batchName+'TCDamp-rBFIamp_max',[-1,-1],avgWin)
# quickScatter(baselineData,holdWinData,dataNames,[19,9],feats,savePath+'/'+batchName+'TCDamp-rBVIamp_max',[-1,-1],avgWin)

#%%
def quickScatterBHI(baselineData,holdWinData,dataNames,cmp,feats,saveName,expChg,avgWin,holdWinLength):
    
    # Get alternative variable for when calculating rBF & rBV
    cmp2 = copy.deepcopy(np.array(cmp))
    cmp2[cmp2 == 5] = 13
    cmp2[cmp2 == 8] = 14
    cmp2[cmp2 == 22] = 24
    cmp2[cmp2 == 23] = 25
    
    freqTCD = 125
    C_min = 0.09
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11,8))
    fig.tight_layout(w_pad=3.5,h_pad=5.25)
    plt.subplots_adjust(top=0.85)
    colors = np.concatenate((plt.get_cmap('tab10')(np.linspace(0, 1, 10)),plt.get_cmap('Set1')(np.linspace(0, 1, 9)),plt.get_cmap('Dark2')(np.linspace(0, 1,8))),axis=0)
    # colors = plt.get_cmap('tab20')(np.linspace(0, 1, holdWinData)) # jet gist_rainbow tab20
    colors = ['black','grey','lightcoral','maroon','red','sienna','darkorange','yellowgreen','cyan','gold', #1-10
              'tan','olive','yellow','forestgreen','lime','turquoise','cadetblue','deepskyblue','cornflowerblue','blue', #11-20
              'darkviolet','magenta','pink'] # 21-22
    
    # Plots difference of previous plot's two points
    diffTcdNorm_all = []
    diffOptNorm_all = []
    for scanInd in range(len(holdWinData)):
        if expChg[0] > 0:
            diffTcdNorm = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        else:
            diffTcdNorm = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
        if expChg[1] > 0:
            diffOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            diffOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        
        T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
        diffTcdNorm = copy.deepcopy([np.nanmin(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),diffTcdNorm])
        diffTcdNorm = np.diff(diffTcdNorm/T_baseline*100)
        
        if cmp[1] == 5:
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            C_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])-np.nanmean(baselineData[scanInd][cmp[1]])
            diffOptNorm = np.diff((1+C_diffOpt/(C_baseline-C_min))*100)
        elif cmp[1] == 8:
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])-np.nanmean(baselineData[scanInd][cmp[1]])
            diffOptNorm = np.diff((1+I_diffOpt/I_baseline)*100)
        else:
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_diffOpt = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),diffOpt])
            diffOptNorm = np.diff(X_diffOpt/X_baseline*100)
        
        ax[1,2].plot(diffTcdNorm/holdWinLength[scanInd],diffOptNorm/holdWinLength[scanInd],'o', color=colors[scanInd])
        diffTcdNorm_all = np.append(diffTcdNorm_all,diffTcdNorm/holdWinLength[scanInd],axis=0)
        diffOptNorm_all = np.append(diffOptNorm_all,diffOptNorm/holdWinLength[scanInd],axis=0)
        
    print('TCD Mean SD: %.2f %.2f' % (diffTcdNorm_all.mean(),diffTcdNorm_all.std()))
    print('Opt Mean SD: %.2f %.2f' % (diffOptNorm_all.mean(),diffOptNorm_all.std()))
    
    ax[1,2].set_xlabel(feats[cmp[0]] + ' BHI')
    ax[1,2].set_ylabel(feats[cmp[1]] + ' BHI')
    diffTcdNorm_all = diffTcdNorm_all[~np.isnan(diffOptNorm_all)]
    diffOptNorm_all = diffOptNorm_all[~np.isnan(diffOptNorm_all)]
    diffOptNorm_all = diffOptNorm_all[~np.isnan(diffTcdNorm_all)]
    diffTcdNorm_all = diffTcdNorm_all[~np.isnan(diffTcdNorm_all)]
    
    slope,intercept,x_line,y_line,r_value,ci = fcns.linRegAndCI(diffTcdNorm_all,diffOptNorm_all)
    ax[1,2].plot(x_line, y_line, color = 'k',linestyle='--',linewidth=1,zorder=0.95)
    ax[1,2].fill_between(x_line, y_line + ci, y_line - ci, color = [0.925,0.925,0.925],zorder=0.9)
    ax[1,2].set_title('Slope: y = %.2f x + %.2f\n Correlation (Pearson): %.2f' % (slope,intercept,r_value))
    # ax[1,2].legend(dataNames[dataRng])
    
    ax[1,2].set_box_aspect(1)
    ax[1,2].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
    ax[1,2].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
    
    xmin, xmax = ax[1,2].get_xlim()
    ymin, ymax = ax[1,2].get_ylim()
    allmin = np.min([xmin-0.1,ymin-0.1])
    allmax = np.max([xmax+0.1,ymax+0.1])
    plot_lims = [allmin,allmax]
    ax[1,2].set_xlim(plot_lims)
    ax[1,2].set_ylim(plot_lims)
    
    
    
    minMaxBothNorm_mean_all = []
    minMaxBothNorm_diff_all = []
    for scanInd in range(diffTcdNorm_all.shape[0]):        
        minMaxBothNorm_mean = (diffOptNorm_all[scanInd]+diffTcdNorm_all[scanInd])/2
        minMaxBothNorm_diff = diffOptNorm_all[scanInd]-diffTcdNorm_all[scanInd]
        ax[1,1].plot(minMaxBothNorm_mean,minMaxBothNorm_diff,'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
        minMaxBothNorm_mean_all = np.append(minMaxBothNorm_mean_all,minMaxBothNorm_mean)
        minMaxBothNorm_diff_all = np.append(minMaxBothNorm_diff_all,minMaxBothNorm_diff)
    
    xlims11 = ax[1,1].get_xlim()
    ax[1,1].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=1.5)
    lowerCI = np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all)
    upperCI = np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all)
    if 0:
        ax[1,1].plot(xlims11,np.tile(upperCI,(2,1)),'k',linewidth=1.5)
        ax[1,1].plot(xlims11,np.tile(lowerCI,(2,1)),'k',linewidth=1.5)
    if 1:
        xmin, xmax = ax[1,1].get_xlim()
        nx = np.array([[xmin,lowerCI],[xmin,upperCI],[xmax,upperCI],[xmax,lowerCI]])
        ax[1,1].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    ax[1,1].set_xlim(xlims11)
    
    
        
    ax[1,1].set_xlabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Mean')
    ax[1,1].set_ylabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Difference')
    # minMaxTcdNorm = minMaxTcdNorm.reshape((-1,1))
    # minMaxOptNorm = minMaxOptNorm.reshape((-1,1))
    # minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxOptNorm_all)]
    # minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxOptNorm_all)]
    # minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxTcdNorm_all)]
    # minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxTcdNorm_all)]
    ax[1,1].set_title('Mean +/- 1.96 S.D. shown in black\n(values = %+.1f, %+.1f, %+.1f)' % 
          (lowerCI,np.nanmean(minMaxBothNorm_diff_all),upperCI))
    
    fig.savefig(saveName + '_BHIcombined.png',dpi=300,bbox_inches='tight')

quickScatterBHI(baselineData,holdWinData,dataNames,[1,5],feats,savePath+'/'+batchName+'TCDavg-rBFIavg_max',[+1,+1],avgWin,holdWinLength)
quickScatterBHI(baselineData,holdWinData,dataNames,[1,8],feats,savePath+'/'+batchName+'TCDavg-rBVIavg_max',[+1,+1],avgWin,holdWinLength)

#%%
def blandAltman(baselineData,holdWinData,dataNames,cmp,feats,saveName,expChg,avgWin):
    
    # Get alternative variable for when calculating rBF & rBV
    cmp2 = copy.deepcopy(np.array(cmp))
    cmp2[cmp2 == 5] = 13
    cmp2[cmp2 == 8] = 14
    cmp2[cmp2 == 22] = 24
    cmp2[cmp2 == 23] = 25
    
    freqTCD = 125
    C_min = 0.09
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11,8))
    fig.tight_layout(w_pad=3.5,h_pad=5.25)
    plt.subplots_adjust(top=0.85)
    colors = np.concatenate((plt.get_cmap('tab10')(np.linspace(0, 1, 10)),plt.get_cmap('Set1')(np.linspace(0, 1, 9)),plt.get_cmap('Dark2')(np.linspace(0, 1,8))),axis=0)
    colors = ['black','grey','lightcoral','maroon','red','sienna','darkorange','yellowgreen','cyan','gold', #1-10
              'tan','olive','yellow','forestgreen','lime','turquoise','cadetblue','deepskyblue','cornflowerblue','blue', #11-20
              'darkviolet','magenta','pink'] # 21-22
    
    # Plots all values 5sec before hold to 5 sec after hold, normalized to 30 second baseline before first hold
    dataBothNorm_mean_all = []
    dataBothNorm_diff_all = []
    for scanInd in range(len(holdWinData)):
        # normFactTcd = np.nanmean(baselineData[scanInd][cmp[0]])
        # data1 = (np.array(holdWinData[scanInd][cmp[0]])/normFactTcd)*100
        
        if cmp[0] == 8: # rBVI average            
            # 1 + (I_baseline - I_data) / I_baseline
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[0]])
            I_data = holdWinData[scanInd][cmp2[0]]
            dataTcdNorm = ( 1+( (I_baseline-I_data) / I_baseline ) )*100
        else:
            # 1 + (T_data - T_baseline) / T_baseline = T_data / T_baseline
            T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
            T_data = np.array(holdWinData[scanInd][cmp[0]])
            dataTcdNorm = ( T_data / T_baseline )*100
        
        if cmp[1] == 5: # rBFI average
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp2[1]])
            # data2 = (1+((normFactOpt-holdWinData[scanInd][cmp2[1]])/(normFactOpt-conMin)))*100
            
            # 1 + ((C_baseline-C_min) - (C_data-C_min))/(C_baseline-C_min) = 1 + (C_baseline - C_data)/(C_baseline-C_min)
            C_min = 0.09 # Contrast value of demodulated seed
            C_max = [] # Phantom data, unused
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]]) # 30 sec prior to hold
            C_data = holdWinData[scanInd][cmp2[1]] # hold data plus 5 sec before and after
            dataOptNorm = ( 1+( (C_baseline-C_data) / (C_baseline-C_min) ) )*100
        elif cmp[1] == 8: # rBVI average
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp2[1]])
            # data2 = (1+((normFactOpt-holdWinData[scanInd][cmp2[1]])/normFactOpt))*100
            
            # 1 + (I_baseline - I_data) / I_baseline
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_data = holdWinData[scanInd][cmp2[1]]
            dataOptNorm = ( 1+( (I_baseline-I_data) / I_baseline ) )*100
        else:
            # normFactOpt = np.nanmean(baselineData[scanInd][cmp[1]])
            # data2 = ((holdWinData[scanInd][cmp[1]])/normFactOpt)*100
            
            # 1 + (X_data - X_baseline) / X_baseline = X_data / X_baseline
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_data = holdWinData[scanInd][cmp[1]]
            dataOptNorm = ( X_data / X_baseline )*100
            
        dataTcdNorm = dataTcdNorm[~np.isnan(dataOptNorm)]
        dataOptNorm = dataOptNorm[~np.isnan(dataOptNorm)]
        dataOptNorm = dataOptNorm[~np.isnan(dataTcdNorm)]
        dataTcdNorm = dataTcdNorm[~np.isnan(dataTcdNorm)]
        
        dataBothNorm_mean = (dataOptNorm+dataTcdNorm)/2
        dataBothNorm_diff = dataOptNorm-dataTcdNorm
        ax[1,0].scatter(dataBothNorm_mean,dataBothNorm_diff,s=1,color=colors[scanInd])
        dataBothNorm_mean_all = np.append(dataBothNorm_mean_all,dataBothNorm_mean,axis=0)
        dataBothNorm_diff_all = np.append(dataBothNorm_diff_all,dataBothNorm_diff,axis=0)
        
        # ax[1,0].annotate("   R:%.2f" % (r2_score(dataOptNorm.reshape(-1, 1),dataOptNorm_pred))**0.5, (dataTcdNorm.mean(), dataOptNorm_pred.mean()),color=colors[scanInd])
    
    xlims11 = ax[1,0].get_xlim()
    ax[1,0].plot(xlims11,np.tile(np.nanmean(dataBothNorm_diff_all),(2,1)),'k',linewidth=1.5)
    lowerCI = np.nanmean(dataBothNorm_diff_all)-1.96*np.nanstd(dataBothNorm_diff_all)
    upperCI = np.nanmean(dataBothNorm_diff_all)+1.96*np.nanstd(dataBothNorm_diff_all)
    if 0:
        ax[1,0].plot(xlims11,np.tile(upperCI,(2,1)),'k',linewidth=1.5)
        ax[1,0].plot(xlims11,np.tile(lowerCI,(2,1)),'k',linewidth=1.5)
    if 1:
        xmin, xmax = ax[1,0].get_xlim()
        nx = np.array([[xmin,lowerCI],[xmin,upperCI],[xmax,upperCI],[xmax,lowerCI]])
        ax[1,0].add_patch(plt.Polygon(nx,alpha=0.1,facecolor='k'))
    ax[1,0].set_xlim(xlims11)
    
    # ax[1,0].set_xlabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Mean (%)')
    # ax[1,0].set_ylabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Difference (%)')
    ax[1,0].set_xlabel('Average of the Two Methods (%)')
    ax[1,0].set_ylabel('Difference Between the Two Methods (%)')
    # dataTcdNorm_all = dataTcdNorm_all[~np.isnan(dataOptNorm_all)]
    # dataOptNorm_all = dataOptNorm_all[~np.isnan(dataOptNorm_all)]
    #ax[1,0].set_title('Correlation (Pearson): %.2f,\ntest' % scipy.stats.pearsonr(dataTcdNorm_all, dataOptNorm_all).statistic)
    ax[1,0].set_title('Mean +/- 1.96 S.D. shown in black\n(values = %+.1f, %+.1f, %+.1f)' % 
          (lowerCI,np.nanmean(dataBothNorm_diff_all),upperCI))
    
    # Plots 2 points, baseline is avg of 30 sec before hold, second is max or min of +/- 5 sec at end of hold, normalized to 30 second baseline before first hold
    minMaxBothNorm_mean_all = []
    minMaxBothNorm_diff_all = []
    for scanInd in range(len(holdWinData)):
        if expChg > 0:
            T_minMaxTcd = np.nanmax(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
            minMaxOpt = np.nanmax(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        else:
            T_minMaxTcd = np.nanmin(holdWinData[scanInd][cmp[0]][(-2*avgWin*freqTCD):])
            minMaxOpt = np.nanmin(holdWinData[scanInd][cmp[1]][(-2*avgWin*freqTCD):])
        
        if cmp[0] == 8:
            I_baseline = np.nanmean(baselineData[scanInd][14])
            minMaxTcdNorm = [np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),minMaxOpt]-np.nanmean(baselineData[scanInd][cmp[0]])
            minMaxTcdNorm = (1+minMaxTcdNorm/I_baseline)*100
        else:
            T_baseline = np.nanmean(baselineData[scanInd][cmp[0]])
            T_minMaxTcd = copy.deepcopy([np.nanmean(holdWinData[scanInd][cmp[0]][:(2*avgWin*freqTCD)]),T_minMaxTcd])
            minMaxTcdNorm = T_minMaxTcd/T_baseline*100
        
        if cmp[1] == 5:
            C_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            C_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]-np.nanmean(baselineData[scanInd][cmp[1]])
            minMaxOptNorm = (1+C_minMaxOpt/(C_baseline-C_min))*100
        elif cmp[1] == 8:
            I_baseline = np.nanmean(baselineData[scanInd][cmp2[1]])
            I_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]-np.nanmean(baselineData[scanInd][cmp[1]])
            minMaxOptNorm = (1+I_minMaxOpt/I_baseline)*100
        else:
            X_baseline = np.nanmean(baselineData[scanInd][cmp[1]])
            X_minMaxOpt = [np.nanmean(holdWinData[scanInd][cmp[1]][:(2*avgWin*freqTCD)]),minMaxOpt]
            minMaxOptNorm = X_minMaxOpt/X_baseline*100
        
        # ax[1,1].quiver(minMaxTcdNorm[0],minMaxOptNorm[0],np.diff(minMaxTcdNorm),np.diff(minMaxOptNorm),scale_units='xy', angles='xy', scale=1, color=colors[scanInd])
        minMaxBothNorm_mean = (minMaxOptNorm+minMaxTcdNorm)/2
        minMaxBothNorm_diff = minMaxOptNorm-minMaxTcdNorm
        ax[1,1].plot(minMaxBothNorm_mean[0],minMaxBothNorm_diff[0],'o',ls='-', ms=5, markevery=[0], color=colors[scanInd])
        ax[1,1].plot(minMaxBothNorm_mean,minMaxBothNorm_diff,'s',ls='-', ms=5, markevery=[-1], color=colors[scanInd])
        minMaxBothNorm_mean_all = np.append(minMaxBothNorm_mean_all,minMaxBothNorm_mean,axis=0)
        minMaxBothNorm_diff_all = np.append(minMaxBothNorm_diff_all,minMaxBothNorm_diff,axis=0)
    
    xlims11 = ax[1,1].get_xlim()
    ax[1,1].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
    ax[1,1].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
    ax[1,1].plot(xlims11,np.tile(np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all),(2,1)),'k',linewidth=2)
    ax[1,1].set_xlim(xlims11)
    
    ax[1,1].set_xlabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Mean (%)')
    ax[1,1].set_ylabel(feats[cmp[1]] + ' & ' + feats[cmp[0]] + ' Difference (%)')
    # minMaxTcdNorm = minMaxTcdNorm.reshape((-1,1))
    # minMaxOptNorm = minMaxOptNorm.reshape((-1,1))
    # minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxOptNorm_all)]
    # minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxOptNorm_all)]
    # minMaxOptNorm_all = minMaxOptNorm_all[~np.isnan(minMaxTcdNorm_all)]
    # minMaxTcdNorm_all = minMaxTcdNorm_all[~np.isnan(minMaxTcdNorm_all)]
    ax[1,1].set_title('Mean +/- 1.96 S.D. shown in black\n(values = %+.1f, %+.1f, %+.1f)' % 
      (np.nanmean(minMaxBothNorm_diff_all)-1.96*np.nanstd(minMaxBothNorm_diff_all),np.nanmean(minMaxBothNorm_diff_all),np.nanmean(minMaxBothNorm_diff_all)+1.96*np.nanstd(minMaxBothNorm_diff_all)))
    
    ylimsNow = np.array(ax[1,0].get_ylim()+ax[1,1].get_ylim()) 
    ax[1,0].set_ylim(ylimsNow.min(),ylimsNow.max())
    ax[1,1].set_ylim(ylimsNow.min(),ylimsNow.max())
    
    for rowInd in range(2):
        for colInd in range(3):
            # ax[rowInd,colInd].set_aspect('equal', adjustable='box')
            ax[rowInd,colInd].set_box_aspect(1)
            ax[rowInd,colInd].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
            ax[rowInd,colInd].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,min_n_ticks=4))
    
    fig.savefig(saveName + '_combined_BlandAltman.png',dpi=300,bbox_inches='tight')

blandAltman(baselineData,holdWinData,dataNames,[1,5],feats,savePath+'/'+batchName+'TCDavg-rBFIavg_max',+1,avgWin)
blandAltman(baselineData,holdWinData,dataNames,[1,8],feats,savePath+'/'+batchName+'TCDavg-rBVIavg_max',+1,avgWin)
# blandAltman(baselineData,holdWinData,dataNames,[2,6],feats,savePath+'/'+batchName+'TCDpi-rBFIamp_max',-1,avgWin)
# blandAltman(baselineData,holdWinData,dataNames,[2,9],feats,savePath+'/'+batchName+'TCDpi-rBVIamp_max',-1,avgWin)
# blandAltman(baselineData,holdWinData,dataNames,[3,10],feats,savePath+'/'+batchName+'OptHR-TCDHR_max',-1,avgWin)

# blandAltman(baselineData,holdWinData,dataNames,[8,5],feats,savePath+'/'+batchName+'rBVIavg-rBFIavg_max',+1,avgWin)

blandAltman(baselineData,holdWinData,dataNames,[2,20],feats,savePath+'/'+batchName+'TCDpi-rBFIpi_max',-1,avgWin)
