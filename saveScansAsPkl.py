#%%
from ReadGen2Data import ReadGen2Data, PrettyFloat2, ConvenienceFunctions
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re, timeit, pickle
from sklearn.metrics import roc_curve, auc

#%%
# Read in the data
LVOsFilter = [('scansValid','True'), ('lowNoise','True'), ('subjectType','LVO'), ('subjectState', 'PreThrombectomy')]
IHCsFilter = [('scansValid','True'), ('lowNoise','True'), ('subjectType','IHC'), ('subjectState', 'FirstScan')]
scanIndexJson = 'ScanIndex.json'

dataRoot = '/Users/kedar/Desktop/'

LVOs  = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, LVOsFilter)
IHCs  = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson, IHCsFilter)

allPatients = pd.concat([LVOs, IHCs])
print(len(LVOs), len(IHCs), len(allPatients))
print(allPatients)
#%%
#Add the redcap data and match
# %% Add the redcap data to the class data
redcapData = pd.read_csv('redcap_data_values.csv')
#Drop the rows with subject_id contianing pattern CVRhBH or EVD which are not in the LVO study
redcapData = redcapData[~redcapData['subject_id'].str.contains('CVRhBH')]
redcapData = redcapData[~redcapData['subject_id'].str.contains('EVD')]
redcapData = redcapData[['subject_id', 'source', 'age', 'race_score', 'pre_op_sbp', 'pre_op_dbp', 'lvo']]
#redcapData = redcapData[~np.any(redcapData == -1, axis=1)] #Remove any rows with -1 for age, race score, pre_op_sbp, pre_op_dbp
# %%
#Start by cleaning the names
subjectIDsRedCap = redcapData['subject_id'].tolist()
namesFeatures    = allPatients['name'].tolist()
subjectIDsRedCap = [ ConvenienceFunctions.CleanName(nm) for nm in subjectIDsRedCap ]
namesFeatures    = [ ConvenienceFunctions.CleanName(nm) for nm in namesFeatures ]

#Replace the column with cleaned names and add the subject ID
allPatients['name'] = namesFeatures
redcapData['subject_id'] = subjectIDsRedCap

#Create a new dataframe with the features and redcap data where the subject IDs match names
dfFeaturesRedCapMerge = pd.merge(allPatients.copy(), redcapData, left_on='name', right_on='subject_id', how='inner')
#Get the list of subject IDs that are in both the redcap data and the features data
subjectIDsFeaturesRedCap = dfFeaturesRedCapMerge['name'].tolist()
#Get the list of names that are in the feature data but not in the combined features+redcap data
namesRedCapNotFeatures = [nm for nm in namesFeatures if nm not in subjectIDsFeaturesRedCap]
#Get the list of subject IDs that are in the redcap data but not in the combined features+redcap data
namesFeaturesNotRedCap = [nm for nm in subjectIDsRedCap if nm not in subjectIDsFeaturesRedCap]

print('Number of patients in redcap data: ', len(redcapData))
print('Number of patients in feature data: ', len(allPatients))
print('Number of patients in merged data: ', len(dfFeaturesRedCapMerge))
print('Patients in feature data but not in redcap features merged data: ', namesRedCapNotFeatures)
print('Patients in redcap data but not in redcap features merged data: ', namesFeaturesNotRedCap)
#Print columns in combined data with index
print('Columns in combined data with index: ', *zip(dfFeaturesRedCapMerge.columns, range(len(dfFeaturesRedCapMerge.columns))))

#Reset index for all patients
allPatients = dfFeaturesRedCapMerge.reset_index(drop=True)
print('Total number of scans to be read: ' + str(len(allPatients)))

dfFeaturesRedCapMerge['subjectTypeCopy'] = dfFeaturesRedCapMerge['subjectType']
dfFeaturesRedCapMerge['subjectTypeCopy'] = dfFeaturesRedCapMerge['subjectTypeCopy'].map({'IHC': 0, 'LVO': 1})
if dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['subjectTypeCopy'] != dfFeaturesRedCapMerge['lvo']] is not None:
    print('Mismatched subject:', dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['subjectTypeCopy'] != dfFeaturesRedCapMerge['lvo']])
    raise ValueError('subjectType in the JSON file and the lvo column in the Redcap data do not match')
dfFeaturesRedCapMerge = dfFeaturesRedCapMerge.drop(columns=['subjectTypeCopy', 'lvo'])

#%% Read all scans into a list
allScans = []
allTime = 0
if os.path.exists('allScansDL.pkl'):
    if os.path.getmtime('allScansDL.pkl') > os.path.getmtime(scanIndexJson):
        print('Loading allScansDL.pkl')
        with open('allScansDL.pkl', 'rb') as f:
            allScans = pickle.load(f)
if len(allScans) == 0:
    print('allScansDL.pkl is older than ScanIndex.json, regenerating allScansDL.pkl')
    print('Processing data in folder')
    for index, row in allPatients.iterrows():
        scanPath = os.path.join(dataRoot, row['site'], row['path'], row['scanUsed'])
        print(scanPath)
        scan = ReadGen2Data(scanPath, deviceID=row['device'], scanTypeIn=row['scanType']-1)
        # Read the data and compute the contrast and time it
        start = timeit.default_timer()
        scan.ReadDataAndComputeContrast()
        stop = timeit.default_timer()
        allTime += stop - start
        allScans.append(scan)
    #Save scans to a pickle file
    with open('allScansDL.pkl', 'wb') as f:
        pickle.dump(allScans, f)
print('Time taken to read each scan on an average: ' + str(*map(PrettyFloat2, ((allTime/len(allScans)),))) + 's')

#%% Plot all scans
#for i,scan in enumerate(allScans):
#    scan.PlotInvertedAndCenteredContrastAndMean( titleStr=allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i] )
#    scan.PlotContrastMeanAndFrequency( titleStr=allPatients['site'].iloc[i]+ ' ' + allPatients['name'].iloc[i], plotContrast=False, plotMean=False, plotGoldenPulses=True, plotFreq=False )

# %% Save all scans as a pickle
scanData = np.zeros( (len(allScans), 16, 600), dtype=np.float32)
scanDataGrayson = np.zeros( (len(allScans), 6, 600), dtype=np.float32)
classData = np.zeros( (len(allScans), 7), dtype=np.object)
subjectID = allPatients['name']
# 0: subjectType, 1: scanType, 2: site, 3: age,4: race score, 5: systolic blood pressure, 6: diastolic blood pressure
scan : ReadGen2Data
cameraOrder1 = ['RH','LH','RV','LV','RN','LN','RN','LN',]
cameraOrder2 = ['RH','LN','RV','RH','LH','RN','LV','LH',]
cameraOrder2Grayson = ['RH','LN','RV','LH','RN','LV',]
cameraNamesGrayson = ['Right Forehead','Left Near','Right Temple','Left Forehead','Right Near','Left Temple',]
cameraOrder3 = ['RH','RH','LN','RH','LN','RV','RV','LN','LH','LH','RN','LH','RN','LV','LV','RN',]
#Map from camera order 2 to camera order 1
cameraMap1 = [cameraOrder1.index(cam) for cam in cameraOrder2]
cameraMap1Grayson = [cameraOrder1.index(cam) for cam in cameraOrder2Grayson]
cameraMap2 = [cameraOrder3.index(cam) for cam in cameraOrder2]
cameraMap2Grayson = [cameraOrder3.index(cam) for cam in cameraOrder2Grayson]
print(cameraMap1)
print(cameraMap2)
#%%
for i, scan in enumerate(allScans):
    if scan.scanType == 0:
        for j,_ in enumerate(cameraMap1):
            ch = scan.channelData[cameraMap1[j]]
            scanData[i, j*2, :] = ch.contrast - np.mean(ch.contrast)
            scanData[i, j*2+1, :] = ch.correctedMean - np.mean(ch.correctedMean)
        for j,_ in enumerate(cameraMap1Grayson):
            ch = scan.channelData[cameraMap1Grayson[j]]
            scanDataGrayson[i, j, :] = ch.contrast
    elif scan.scanType == 3:
        for j,_ in enumerate(cameraMap2):
            ch = scan.channelData[cameraMap2[j]]
            scanData[i, j*2, :] = ch.contrast - np.mean(ch.contrast)
            scanData[i, j*2+1, :] = ch.correctedMean - np.mean(ch.correctedMean)
        for j,_ in enumerate(cameraMap2Grayson):
            ch = scan.channelData[cameraMap2Grayson[j]]
            scanDataGrayson[i, j, :] = ch.contrast
    else:
        for j,ch in enumerate(scan.channelData):
            scanData[i, j*2, :] = ch.contrast - np.mean(ch.contrast)
            scanData[i, j*2+1, :] = ch.correctedMean - np.mean(ch.correctedMean)
        for j,ch in enumerate(scan.channelData):
            if j>2:
                chNu = j-1
            else:
                chNu = j
            if j == 3 or j == 7:
                continue
            scanDataGrayson[i, chNu, :] = ch.contrast
    
for i in range(len(allScans)):
    classData[i, 0] = allPatients['subjectType'][i]
    classData[i, 1] = allPatients['scanType'][i]
    classData[i, 2] = allPatients['source'][i]
    classData[i, 3] = allPatients['age'][i]
    classData[i, 4] = allPatients['race_score'][i]
    classData[i, 5] = allPatients['pre_op_sbp'][i]
    classData[i, 6] = allPatients['pre_op_dbp'][i]

#Round the age to the nearest integer and normalize it for relu
classData[:,3] = np.floor(classData[:,3])
classData[:,3] = classData[:,3]/100
#Max race score is 9 - normalize it
classData[:, 4] = classData[:, 4]/9
#Normalize the blood pressure
classData[:,5] = classData[:,5]/200
classData[:,6] = classData[:,6]/200

# %% Plot the data for first patient from Grayson
'''plt.figure()
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.plot(scanDataGrayson[0,i,:])'''

#Save the data for Grayson as csv stack the first two dimensions
print(scanDataGrayson.shape)
scanDataGrayson = np.reshape(scanDataGrayson, (scanDataGrayson.shape[0]*scanDataGrayson.shape[1],scanDataGrayson.shape[2]))
print(scanDataGrayson.shape)
#Make pandas dataframe with t0 to t599 as columns
scanDataGrayson = pd.DataFrame(scanDataGrayson, dtype=np.float32)
#Set the column names as t0 to t599
scanDataGrayson.columns = ['t'+str(i) for i in range(600)]
#Repeat subject ID for each channel 6 times
subjectIDRep = np.repeat(subjectID, 6)
subjectIDRep = subjectIDRep.to_numpy().astype('<U14')
cameraNamesGraysonRep = np.tile(cameraNamesGrayson, len(subjectID))
scanDataGrayson.insert(0, 'subjectID', subjectIDRep)
scanDataGrayson.insert(1, 'cameraName', cameraNamesGraysonRep)

#Save the pandas dataframe as csv
scanDataGrayson.to_csv('scanDataGrayson.csv', index=False)

#Repeat plot to check
'''plt.figure()
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.plot(scanDataGrayson.iloc[i,2:])'''

# %%
#Randomize the data
randomize = np.arange(len(allScans))
np.random.shuffle(randomize)
scanData = scanData[randomize]
classData = classData[randomize]
subjectID = subjectID[randomize]

# %%
# Save the data
with open('scanData.npy', 'wb') as f:
    np.save(f, scanData)
with open('classData.npy', 'wb') as f:
    np.save(f, classData)
with open('subjectID.npy', 'wb') as f:
    np.save(f, subjectID)
# %%
print('Number of scans used ', allPatients.shape[0])
print(allPatients['subjectType'].value_counts())

# %%
print(classData[:,0])
#Get counts of each class
print('Number of LVOs: ', np.sum(classData[:,0] == 'LVO'))
print('Number of IHCs: ', np.sum(classData[:,0] == 'IHC'))
# %%
print(subjectID)
# %%
#Import csv file redcap_data_labels.csv and filter subject_id column by subjectID array ignoring capitalization
redcapDataLabels = pd.read_csv('redcap_data_labels.csv')
redcapDataLabels = redcapDataLabels[~redcapDataLabels['subject_id'].str.contains('CVRhBH')]
redcapDataLabels = redcapDataLabels[~redcapDataLabels['subject_id'].str.contains('EVD')]
redcapDataLabelsSubjects = redcapDataLabels['subject_id'].tolist()
redcapDataLabelsSubjects = [ ConvenienceFunctions.CleanName(nm) for nm in redcapDataLabelsSubjects ]
redcapDataLabels['subject_id'] = redcapDataLabelsSubjects
#Drop the rows with subject_id not in subjectID
redcapDataLabelsFilt = redcapDataLabels[redcapDataLabels['subject_id'].str.lower().isin(subjectID.str.lower())]
print(redcapDataLabelsFilt.shape)
#Read the DL predictions - fold, IHC_predProb, LVO_predProb
DL_predProb = pd.read_csv('DL_predProb.csv')
#Change column 0 to subject_id
DL_predProb.columns = ['subject_id', DL_predProb.columns[1], DL_predProb.columns[2], DL_predProb.columns[3]]

#Join the two dataframes on subject_id
redcapDataLabelsFilt = pd.merge(redcapDataLabelsFilt, DL_predProb, on='subject_id', how='inner')
print(redcapDataLabelsFilt.shape)

#Drop record_id
redcapDataLabelsFilt = redcapDataLabelsFilt.drop(columns=['record_id'])
#Drop the first unnamed column
redcapDataLabelsFilt = redcapDataLabelsFilt.drop(columns=['Unnamed: 0'])
#Save data to csv
redcapDataLabelsFilt.to_csv('redcap_data_labels_grayson.csv', index=False)

#%%
#Get LVO classification from redcapDataLabelsFilt by argmax of predProb0 and predProb1
redcapDataLabelsFilt['LVO_pred'] = np.argmax(redcapDataLabelsFilt[['predProb0','predProb1']].to_numpy(), axis=1)
#Compare LVO_pred with LVO
print(redcapDataLabelsFilt['LVO_pred'].value_counts())
print(redcapDataLabelsFilt['lvo'].value_counts())
#Print the sensitivity and specificity
TP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 1))
TN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 0))
FP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 0))
FN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 1))
print('Sensitivity: ', TP/(TP+FN))
print('Specificity: ', TN/(TN+FP))
#%%
#Get the rows in redcapDataLabelsFilt where sex column is female
redcapDataLabelsFiltCopy = redcapDataLabelsFilt.copy()
redcapDataLabelsFilt = redcapDataLabelsFilt[redcapDataLabelsFilt['sex'] == 'female'] 
#Get LVO classification from redcapDataLabelsFilt by argmax of predProb0 and predProb1
redcapDataLabelsFilt['LVO_pred'] = np.argmax(redcapDataLabelsFilt[['predProb0','predProb1']].to_numpy(), axis=1)
#Compare LVO_pred with LVO
print(redcapDataLabelsFilt['LVO_pred'].value_counts())
print(redcapDataLabelsFilt['lvo'].value_counts())
#Print the sensitivity and specificity
TP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 1))
TN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 0))
FP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 0))
FN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 1))
print('female')
print('Sensitivity: ', TP/(TP+FN))
print('Specificity: ', TN/(TN+FP))

#%% Calculate area of ROC curve
fpr, tpr, _ = roc_curve(redcapDataLabelsFiltCopy['lvo'].to_numpy(), redcapDataLabelsFiltCopy[['predProb1']].to_numpy(),)
roc_auc = '%.2f' % auc(fpr, tpr)
print('AUC: ', roc_auc)
#%%
#Get the rows in redcapDataLabelsFilt where sex column is female
redcapDataLabelsFilt = redcapDataLabelsFiltCopy[redcapDataLabelsFiltCopy['fitzpatrick_scale']>3] 
#Get LVO classification from redcapDataLabelsFilt by argmax of predProb0 and predProb1
redcapDataLabelsFilt['LVO_pred'] = np.argmax(redcapDataLabelsFilt[['predProb0','predProb1']].to_numpy(), axis=1)
#Compare LVO_pred with LVO
print(redcapDataLabelsFilt['LVO_pred'].value_counts())
print(redcapDataLabelsFilt['lvo'].value_counts())
#Print the sensitivity and specificity
TP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 1))
TN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 0))
FP = np.sum((redcapDataLabelsFilt['LVO_pred'] == 1) & (redcapDataLabelsFilt['lvo'] == 0))
FN = np.sum((redcapDataLabelsFilt['LVO_pred'] == 0) & (redcapDataLabelsFilt['lvo'] == 1))
print('fitzpatrick_scale > 3')
print('Sensitivity: ', TP/(TP+FN))
print('Specificity: ', TN/(TN+FP))


#%%
print(redcapDataLabels['subject_id'].str.lower().isin(subjectID.str.lower()))
print(redcapDataLabelsFilt.shape)
for sub in subjectID.str.lower():
    print(sub)
for sub in redcapDataLabels['subject_id'].str.lower():
    print(sub)
# %%
