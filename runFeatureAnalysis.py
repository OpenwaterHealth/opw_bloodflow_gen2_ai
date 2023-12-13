#%%
import math
from ReadGen2Data import ReadGen2Data, PrettyFloat4, ConvenienceFunctions, PulseFeatures
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re, pickle

#%%
plotContrast = False
plotImageMean = False
plotContrastFreq = False
plotGoldenPulses = False
plotInvertedContrastAndMean = False
dt = 1/40

scanIndexJson = 'ScanIndex.json'

dataRoot = '/Users/kedar/Desktop/'
#%% Read in only recently added patients to validate the quality of the scans
if 0:
    patient1 = [[('name','076')],[('name','077')],[('name','078')],[('name','079')],[('name','080')],[('name','081')],
                [('name','082')],[('name','083')],[('name','084')],[('name','085')],[('name','087')],[('name','088')],[('name','090')]]
    for patientFilt in patient1:
        print(patientFilt)
        patientScans = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, patientFilt)
        allScans = []
        for patient in patientScans.iterrows():
            for scan in patient[1].scanNames:
                scanPath = os.path.join(dataRoot, patient[1].site, patient[1].path, scan)
                print(scanPath)
                scan = ReadGen2Data(scanPath, deviceID=patient[1].device, scanTypeIn=int(patient[1].scanType)-1)
                scan.ReadDataAndComputeContrast()
                scan.PlotInvertedAndCenteredContrastAndMean( titleStr=patient[1].site + ' ' + patientFilt[0][1] )
                
#%%
#Read in ScanIndex.json into pandas dataframe
SiteYLVOsFilter = [('scansValid','True'), ('lowNoise','True') , ('site','SiteY'), ('subjectType','LVO'), ('subjectState','PreThrombectomy')]
SiteYIHCsFilter = [('scansValid','True'), ('lowNoise','True'), ('site','SiteY'), ('subjectType','IHC'), ('subjectState','FirstScan')]
SiteXLVOsFilter = [('scansValid','True'), ('lowNoise','True'), ('site','SiteX'), ('subjectType','LVO'), ('subjectState','PreThrombectomy')]
SiteXIHCsFilter = [('scansValid','True'), ('lowNoise','True'), ('site','SiteX'), ('subjectType','IHC'), ('subjectState','FirstScan')]

SiteYLVOs  = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, SiteYLVOsFilter)
SiteYIHCs  = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, SiteYIHCsFilter)
SiteXLVOs = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, SiteXLVOsFilter)
SiteXIHCs = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, SiteXIHCsFilter)

allPatients = pd.concat([SiteYLVOs, SiteYIHCs, SiteXLVOs, SiteXIHCs])
#Reset index for all patients
allPatients = allPatients.reset_index(drop=True)

#%% Check if there is a allScans.pkl file that is newer than the ScanIndex.json, if there is then load it
# Otherwise all scans into a list
allScans = []
if os.path.exists('allScans.pkl'):
    if os.path.getmtime('allScans.pkl') > os.path.getmtime(scanIndexJson):
        print('Loading allScans.pkl')
        with open('allScans.pkl', 'rb') as f:
            allScans = pickle.load(f)
if len(allScans) == 0:
    print('allScans.pkl is older than ScanIndex.json, regenerating allScans.pkl')
    print('Processing data in folder')
    for index, row in allPatients.iterrows():
        scanPath = os.path.join(dataRoot, row['site'], row['path'], row['scanUsed'])
        print(scanPath)
        scan = ReadGen2Data(scanPath, deviceID=row['device'], scanTypeIn=row['scanType']-1)
        scan.ReadDataAndComputeContrast()
        scan.DetectPulseAndFindFit()
        scan.ComputeGoldenPulseFeatures()
        scan.ComputePulseSegmentFeatures()
        allScans.append(scan)
    #Save scans to a pickle file
    with open('allScans.pkl', 'wb') as f:
        pickle.dump(allScans, f)

#%% Generate plots if needed
if plotContrast or plotImageMean or plotContrastFreq or plotGoldenPulses or plotInvertedContrastAndMean:
    for i,scan in enumerate(allScans):
        if plotInvertedContrastAndMean:
            scan.PlotInvertedAndCenteredContrastAndMean( titleStr=allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i] )
        if plotContrast or plotImageMean or plotContrastFreq or plotGoldenPulses:
            scan.PlotContrastMeanAndFrequency( titleStr=allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i],
                                               plotContrast=plotContrast, plotMean=plotImageMean, plotGoldenPulses=plotGoldenPulses,
                                               plotFreq=plotContrastFreq )

# %%
tempPulseFeatures = PulseFeatures()
#The features are implemented for golden pulses and pulse segments
#We keep the list as is to access the feature by name
opticalFeatureList    = tempPulseFeatures.featureList
opticalFeatureNamesGP = tempPulseFeatures.featureNames
#Repeat entries for opticaFeatureNames but add a suffix for pulse segments
opticalFeatureNamesPulses = [name + ' PulseSegments' for name in opticalFeatureNamesGP]
opticalFeatureNamesRange  = [name + ' SegmentsRange' for name in opticalFeatureNamesGP]
#Skip modulation depth for the pulse feature for now which is the first entry
opticalFeatureNamesAll = opticalFeatureNamesGP + opticalFeatureNamesPulses[1:] + opticalFeatureNamesRange[1:]

allFeatures = np.full((len(allScans), len(opticalFeatureNamesAll)), np.nan)

validChannels = ([0,1],[0,3,4,7],[],[0,1,3,8,9,11])
for i,scan in enumerate(allScans):
    scan = allScans[i]
    channels = validChannels[scan.scanType]

    goldenPulseFeaturesForChannels = np.full((len(channels), len(opticalFeatureList)), np.nan)
    #Push golden pulse features features into a single array
    for j,ch in enumerate(channels):
        for k in range(len(opticalFeatureList)):
            goldenPulseFeaturesForChannels[j,k] = getattr(scan.channelData[ch].goldenPulseFeatures, opticalFeatureList[k])
    #Take the min for the last five features and max for the rest
    if np.isnan(goldenPulseFeaturesForChannels).all():
        print('No valid golden pulses for ' + allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i])
        continue
    allFeatures[i,:len(opticalFeatureList)-5] = np.nanmax(goldenPulseFeaturesForChannels[:,:-5], axis=0) 
    allFeatures[i,len(opticalFeatureList)-5:len(opticalFeatureList)] = np.nanmin(goldenPulseFeaturesForChannels[:,-5:], axis=0)

    pulseSegmentFeaturesForChannels = np.full((len(channels), len(opticalFeatureList)-1), np.nan)
    segmentRangeFeaturesForChannels = np.full((len(channels), len(opticalFeatureList)-1), np.nan)
    #Push raw pulse features features into a single array
    for j,ch in enumerate(channels):
        if scan.channelData[ch].pulseSegmentsFeatures is None or np.isnan(scan.channelData[ch].pulseSegmentsFeatures.veloCurveIndex).all():
            continue
        for k in range(len(opticalFeatureList)-1):
            # k+1 because the first entry is the modulation depth and we skip it for now
            featureArray = np.array(getattr(scan.channelData[ch].pulseSegmentsFeatures, opticalFeatureList[k+1]))
            featureArray = featureArray[~np.isnan(featureArray)]
            if len(featureArray) > 0:
                pulseSegmentFeaturesForChannels[j,k] = np.nanmedian(featureArray)
                #Get the 80-20 percentile range
                segmentRangeFeaturesForChannels[j,k] = np.diff(np.percentile(featureArray,[20,80]))[0]

    #Take the min for the last five features and max for the rest
    if np.isnan(pulseSegmentFeaturesForChannels).all() or np.isnan(segmentRangeFeaturesForChannels).all():
        print('No valid pulse segments for ' + allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i])
        continue
    allFeatures[i,len(opticalFeatureList):2*len(opticalFeatureList)-6]     = np.nanmax(pulseSegmentFeaturesForChannels[:,:-5], axis=0)
    allFeatures[i,2*len(opticalFeatureList)-6:2*len(opticalFeatureList)-1] = np.nanmin(pulseSegmentFeaturesForChannels[:,-5:], axis=0)
    allFeatures[i,2*len(opticalFeatureList)-1:-5] = np.nanmax(segmentRangeFeaturesForChannels[:,:-5], axis=0)
    allFeatures[i,-5:]                            = np.nanmin(segmentRangeFeaturesForChannels[:,-5:], axis=0)

#Create data frame with all features
allOpticalFeaturesDf = pd.DataFrame(allFeatures, columns=opticalFeatureNamesAll)

allOpticalFeaturesDf['site'] = allPatients['site']
allOpticalFeaturesDf['name'] = allPatients['name']
allOpticalFeaturesDf['subjectType'] = allPatients['subjectType']
allOpticalFeaturesDf['device'] = allPatients['device']

# %%
print(allOpticalFeaturesDf)
# %%
redcapData = pd.read_csv('redcap_data_values.csv')
# %%
#Start by cleaning the names
subjectIDsRedCap = redcapData['subject_id'].tolist()
namesFeatures    = allOpticalFeaturesDf['name'].tolist()
subjectIDsRedCap = [ ConvenienceFunctions.CleanName(nm) for nm in subjectIDsRedCap ]
namesFeatures    = [ ConvenienceFunctions.CleanName(nm) for nm in namesFeatures ]

#Replace the column with cleaned names and add the subject ID
allOpticalFeaturesDf['name'] = namesFeatures
redcapData['subject_id'] = subjectIDsRedCap

# %%
#Create a new dataframe with the features and redcap data where the subject IDs match names
dfFeaturesRedCapMerge = pd.merge(allOpticalFeaturesDf.copy(), redcapData, left_on='name', right_on='subject_id', how='inner')
#Get the list of subject IDs that are in both the redcap data and the features data
subjectIDsFeaturesRedCap = dfFeaturesRedCapMerge['name'].tolist()
#Get the list of names that are in the feature data but not in the combined features+redcap data
namesRedCapNotFeatures = [nm for nm in namesFeatures if nm not in subjectIDsFeaturesRedCap]
#Get the list of subject IDs that are in the redcap data but not in the combined features+redcap data
namesFeaturesNotRedCap = [nm for nm in subjectIDsRedCap if nm not in subjectIDsFeaturesRedCap]

print('Number of patients in redcap data: ', len(redcapData))
print('Number of patients in feature data: ', len(allOpticalFeaturesDf))
print('Number of patients in merged data: ', len(dfFeaturesRedCapMerge))
print('Patients in feature data but not in redcap features merged data: ', namesRedCapNotFeatures)
print('Patients in redcap data but not in redcap features merged data: ', namesFeaturesNotRedCap)
#Print columns in combined data with index
print('Columns in combined data with index: ', *zip(dfFeaturesRedCapMerge.columns, range(len(dfFeaturesRedCapMerge.columns))))
# %%
redcapCols = ['age','race_score','lams_score','nihss_baseline']
redcapColsAmbulance = ['age','pre_op_dbp','pre_op_sbp']
classificationFeaturesColumns = [i for i in range(0,40)] + [dfFeaturesRedCapMerge.columns.get_loc(colName) for colName in redcapCols]

featureNames = list(dfFeaturesRedCapMerge.columns)
featureNames = [featureNames[i] for i in classificationFeaturesColumns]
print('Feature names used in classification: ', featureNames)

# %%
def RunClassification(dfMergedInput, featureColumns, RFDepth=2):
    featureNames = list(dfMergedInput.columns)
    featureNames = [featureNames[i] for i in featureColumns]
    print('Feature names used in classification: ', featureNames)

    x = dfMergedInput.iloc[:,featureColumns].to_numpy()
    y = dfMergedInput['subjectType'].copy()
    #Find any nan values and drop rows with nan values
    nanRows = np.any(np.isnan(x), axis=1)
    print('Dropping these many rows with nan values: ', np.sum(nanRows))
    x = x[~nanRows,:]
    y = y[~nanRows]
    #Change y from LVO and IHC to 0 and 1
    y[y == 'LVO'] = 1
    y[y == 'IHC'] = 0
    y = y.to_numpy().astype(float)

    meanAuc, aucCurveTup = ConvenienceFunctions.NormalizeFeaturesAndRunKFoldCrossValidationWithRF(x, y, RFDepth=RFDepth, featureNames=featureNames)
    print('Mean AUC: ', meanAuc)

    return aucCurveTup


# %% Site wise classification of optical only data
dfSiteYOptical = allOpticalFeaturesDf[allOpticalFeaturesDf['site'] == 'SiteY'].copy()
dfSiteYOptical = dfSiteYOptical.reset_index(drop=True)
print('SiteY subjects optical features ', dfSiteYOptical.shape)
print(dfSiteYOptical['subjectType'].value_counts())
SiteYAllOpticalAUC = RunClassification(dfSiteYOptical, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

dfSiteYOptical = dfSiteYOptical[dfSiteYOptical['device'] == 2.0].copy()
dfSiteYOptical = dfSiteYOptical.reset_index(drop=True)
print('SiteY subjects optical features with device 002', dfSiteYOptical.shape)
print(dfSiteYOptical['subjectType'].value_counts())
SiteYAllOpticalAUC = RunClassification(dfSiteYOptical, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

dfSiteXOptical = allOpticalFeaturesDf[allOpticalFeaturesDf['site'] == 'SiteX'].copy()
dfSiteXOptical = dfSiteXOptical.reset_index(drop=True)
print('SiteX subjects optical features ', dfSiteXOptical.shape)
print(dfSiteXOptical['subjectType'].value_counts())
SiteXAllOpticalAUC = RunClassification(dfSiteXOptical, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

dfSiteXOptical = allOpticalFeaturesDf[allOpticalFeaturesDf['site'] == 'SiteX'].copy()
dfSiteXOptical = dfSiteXOptical.reset_index(drop=True)
#Drop the columns with SegmentsRange in the name
print('SiteX subjects optical features ', dfSiteXOptical.shape)
print(dfSiteXOptical['subjectType'].value_counts())
SiteXAllOpticalAUC = RunClassification(dfSiteXOptical, classificationFeaturesColumns[:-(len(redcapCols)+len(opticalFeatureList)-1)], RFDepth=-1)

print('All subjects optical features ', allOpticalFeaturesDf.shape)
print(allOpticalFeaturesDf['subjectType'].value_counts())
allOpticalAUC = RunClassification(allOpticalFeaturesDf, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

# %% Site wise classification of optical only data with matching REDCap data
dfSiteYOptical = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['site'] == 'SiteY'].copy()
dfSiteYOptical = dfSiteYOptical.reset_index(drop=True)
print('SiteY subjects(w/matching Redcap Entries) optical features ', dfSiteYOptical.shape)
print(dfSiteYOptical['subjectType'].value_counts())
SiteYRdCpMatchOpticalAUC = RunClassification(dfSiteYOptical, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

dfSiteXOptical = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['site'] == 'SiteX'].copy()
dfSiteXOptical = dfSiteXOptical.reset_index(drop=True)
print('SiteX subjects(w/matching Redcap Entries) optical features ', dfSiteXOptical.shape)
print(dfSiteXOptical['subjectType'].value_counts())
SiteXRdCpMatchOpticalAUC = RunClassification(dfSiteXOptical, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)

print('All subjects(w/matching Redcap Entries) optical features ', dfFeaturesRedCapMerge.shape)
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
allRdCpMatchOpticalAUC = RunClassification(dfFeaturesRedCapMerge, classificationFeaturesColumns[:-len(redcapCols)], RFDepth=-1)


# %% Run Analysis with RedCap features and optical features
print('All subjects with Optical+RedCap features')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
allRdCpOpticalAUC = RunClassification(dfFeaturesRedCapMerge, classificationFeaturesColumns, RFDepth=-1)

# %% Run Analysis with RedCap NIHSS feature and optical features
print('All subjects with RedCap NIHSS Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
allNIHSSAUC = RunClassification(dfFeaturesRedCapMerge, [dfFeaturesRedCapMerge.columns.get_loc('nihss_baseline')], RFDepth=-1)

print('All subjects with Optical+NIHSS Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = classificationFeaturesColumns[:-len(redcapCols)]+[dfFeaturesRedCapMerge.columns.get_loc('nihss_baseline')]
print('curClassificationCols: ', curClassificationCols)
allNIHSSOpticalAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

# %% Run Analysis with RedCap Race Score feature and optical features
print('All subjects with RedCap RACE Score Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
allRACEAUC = RunClassification(dfFeaturesRedCapMerge, [dfFeaturesRedCapMerge.columns.get_loc('race_score')], RFDepth=-1)

print('All subjects with RedCap RACE Score + Age + Blood Pressure')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = [ dfFeaturesRedCapMerge.columns.get_loc(ambFeat) for ambFeat in redcapColsAmbulance ]
curClassificationCols += [dfFeaturesRedCapMerge.columns.get_loc('race_score')]
allRACEAgeBpAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

print('All subjects with Optical+RACE Score Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = classificationFeaturesColumns[:-len(redcapCols)]+[dfFeaturesRedCapMerge.columns.get_loc('race_score')]
allRACEOpticalAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

print('All subjects with Optical+ RACE Score + Age + Blood Pressure')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = [ dfFeaturesRedCapMerge.columns.get_loc(ambFeat) for ambFeat in redcapColsAmbulance ]
curClassificationCols += classificationFeaturesColumns[:-len(redcapCols)]+[dfFeaturesRedCapMerge.columns.get_loc('lams_score')]
allRACEOpticalAgeBpAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

# %% Run Analysis with RedCap Lams Score feature and optical features
print('All subjects with RedCap LAMS Score Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
allLAMSAUC = RunClassification(dfFeaturesRedCapMerge, [dfFeaturesRedCapMerge.columns.get_loc('lams_score')], RFDepth=-1)

print('All subjects with RedCap LAMS Score + Age + Blood Pressure')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = [ dfFeaturesRedCapMerge.columns.get_loc(ambFeat) for ambFeat in redcapColsAmbulance ]
curClassificationCols += [dfFeaturesRedCapMerge.columns.get_loc('lams_score')]
allLAMSAgeBpAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

print('All subjects with Optical+LAMS Score Only')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = classificationFeaturesColumns[:-len(redcapCols)]+[dfFeaturesRedCapMerge.columns.get_loc('lams_score')]
allLAMSOpticalAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

print('All subjects with Optical+LAMS Score + Age + Blood Pressure')
print(dfFeaturesRedCapMerge['subjectType'].value_counts())
curClassificationCols = [ dfFeaturesRedCapMerge.columns.get_loc(ambFeat) for ambFeat in redcapColsAmbulance ]
curClassificationCols += classificationFeaturesColumns[:-len(redcapCols)]+[dfFeaturesRedCapMerge.columns.get_loc('lams_score')]
allLAMSOpticalAgeBpAUC = RunClassification(dfFeaturesRedCapMerge, curClassificationCols, RFDepth=-1)

# %%
###ROC RACE only, RACE + Optical, Optical only
#Compute ROC curve and ROC area for each class
x = dfFeaturesRedCapMerge.iloc[:,dfFeaturesRedCapMerge.columns.get_loc('race_score')].to_numpy()
y = dfFeaturesRedCapMerge['subjectType'].copy()
y[y == 'LVO'] = 1
y[y == 'IHC'] = 0
y = y.to_numpy().astype(int)
ind = np.argsort(x.flatten())
TN = np.sum(y)
TP = np.sum(1-y)
y1 = np.cumsum(1-y[ind])/TP  #sensitivity
x1 = np.cumsum(y[ind])/TN  #1-specificity
auc = np.trapz(y1,x1)

#plt.plot(x1,y1,label='RACE Score') #Five fold cross validation is more fair for now - switch to this later
plt.rcParams.update({'font.size': 16})
plt.plot(allRACEAUC[0],allRACEAUC[1],label='RACE Score')
plt.plot(allOpticalAUC[0], allOpticalAUC[1], label='Optical')
plt.plot([0,1],[0,1],'m-.' ,label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
#Set a higher font size for the legend
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.show()

###ROC RACE only, RACE + Optical + Age + BP, Optical only
#plt.plot(x1,y1,label='RACE Score') #Five fold cross validation is more fair for now - switch to this later
#If file DL_ROC.npy exists, read it in and plot it
if os.path.isfile('DLResults/DL_ROC.npy'):
    DL_ROC = np.load('DLResults/DL_ROC.npy')
    plt.plot(DL_ROC[0],DL_ROC[1],label='Deep Learning - Optical+RACE Score+Age+BP')
plt.plot(allRACEAUC[0],allRACEAUC[1],label='RACE Score')
plt.plot(allRACEAgeBpAUC[0],allRACEAgeBpAUC[1],label='RF RACE Score+Age+BP')
plt.plot(allRACEOpticalAgeBpAUC[0], allRACEOpticalAgeBpAUC[1], label='RF Optical+RACE Score+Age+BP' )
plt.plot([0,1],[0,1],'m-.' ,label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(10, 10)
plt.show()

plt.style.use(['default'])

# %%
allOpticalFeaturesDf.to_csv('all_optical_features.csv')

# %%
#Print DL_ROC[0],DL_ROC[1] by pairing elements
print(list(zip(DL_ROC[0],DL_ROC[1])))
print(list(zip(allRACEAgeBpAUC[0],allRACEAgeBpAUC[1])))
print(list(zip(allRACEOpticalAgeBpAUC[0],allRACEOpticalAgeBpAUC[1])))
# %%
plt.plot(DL_ROC[0],DL_ROC[1],label='Deep Learning')
plt.show()
# %%

trfOptROC = np.load('DLResults/DL_ROC_Trf_optical.npy')

#Plot roc curves
plt.figure(figsize=(5,5))
AUCtrfOpt = np.trapz(trfOptROC[1],trfOptROC[0])
plt.plot(trfOptROC[0], trfOptROC[1])#, label='Openwater Stroke Detection AUC = %0.2f' % AUCtrfOpt)
AUCRACE = np.trapz(allRACEAUC[1],allRACEAUC[0])
plt.plot(allRACEAUC[0],allRACEAUC[1])#,label='RACE Score AUC = %0.2f' % AUCRACE)
AUCLAMS = np.trapz(allLAMSAUC[1],allLAMSAUC[0])
plt.plot(allLAMSAUC[0],allLAMSAUC[1])#,label='LAMS Score AUC = %0.2f' % AUCLAMS)
plt.plot([0,1],[0,1],'m-.')# ,label='Random Guess')
#plt.legend(loc='lower right')
plt.show()
# %%
#print all patients from site SiteX
allPatientsBr = allPatients[allPatients['site'] == 'SiteX']
print(allPatientsBr)
allIHCBr = allPatientsBr[allPatientsBr['subjectType'] == 'IHC']
print(allIHCBr)
# %%
#plot patient index 67, 90, 91 and show the pulse segments with same x axis
ax1 = plt.subplot(3,1,1)
plt.plot(allScans[67].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[67].channelData[0].goldenPulse)
ax2 = plt.subplot(3,1,2)
plt.plot(allScans[90].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[90].channelData[0].goldenPulse)
ax3 = plt.subplot(3,1,3)
plt.plot(allScans[91].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[91].channelData[0].goldenPulse)
#Set the x axis to be the same for all plots
ax1.sharex(ax3)
ax3.sharex(ax2)
#Change x axis to be time in seconds instead of samples
ax3.set_xticks(np.arange(0, 30, 5))
ax3.set_xticklabels(np.arange(0, 30, 5)/40)
ax3.set_xlabel('Time (s)')
# Bring subplots close to each other.
plt.subplots_adjust(hspace=0.08)
#Flip the y axis for all plots
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.set_ylim(ax2.get_ylim()[::-1])
ax3.set_ylim(ax3.get_ylim()[::-1])
fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.show()

# %%
#plot patient index 67, 90, 91 and show the pulse segments with same x axis
ax1 = plt.subplot(3,1,1)
plt.plot(allScans[67].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[67].channelData[0].goldenPulse)
ax2 = plt.subplot(3,1,2)
plt.plot(allScans[90].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[90].channelData[0].goldenPulse)
ax3 = plt.subplot(3,1,3)
plt.plot(allScans[91].channelData[0].pulseSegments,color='0.8')
plt.plot(allScans[91].channelData[0].goldenPulse)
#Set the x axis to be the same for all plots
ax1.sharex(ax3)
ax3.sharex(ax2)
#Change x axis to be time in seconds instead of samples
ax3.set_xticks(np.arange(0, 30, 5))
ax3.set_xticklabels(np.arange(0, 30, 5)/40)
ax3.set_xlabel('Time (s)')
# Bring subplots close to each other.
plt.subplots_adjust(hspace=0.08)
#Flip the y axis for all plots
ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.set_ylim(ax2.get_ylim()[::-1])
ax3.set_ylim(ax3.get_ylim()[::-1])
fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.show()
