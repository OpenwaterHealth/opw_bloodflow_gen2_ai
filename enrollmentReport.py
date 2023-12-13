#%%
import math
from ReadGen2Data import ReadGen2Data, PrettyFloat4, ConvenienceFunctions, PulseFeatures
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import os, re

redcapData = pd.read_csv('redcap_data_values.csv')
scanIndexJson = 'ScanIndex.json'
opticalData = ConvenienceFunctions.ReadJsonAsPdAndFilter(scanIndexJson)

#Drop the rows with subject_id contianing pattern CVRhBH or EVD which are not in the LVO study
redcapData = redcapData[~redcapData['subject_id'].str.contains('CVRhBH')]
redcapData = redcapData[~redcapData['subject_id'].str.contains('EVD')]
#Print the columns of the redcap data
print(redcapData.columns)
print(redcapData['lvo'].value_counts())
#%%
#Print the study_arm column to see if there are any subjects in the IHC study
print(redcapData['study_arm'])

#%%
#Coulumn source has the sites for the LVO study - 0 for SiteY, 1 for SiteX
#Coulumn study_arm has the study arm for the LVO study:
# 0 for no thrombectomy, 1 for LVO with thrombectomy, 2 for Ischemic stroke, 3 for Hemorrhagic stroke, 4 for Stroke mimic
# enrollment_date has the enrollment date

#Get the weekly patient count for SiteY
SiteYData = redcapData[redcapData['source'] == 0]
SiteYData = SiteYData[SiteYData['study_arm'] != 0]
SiteYData = SiteYData[SiteYData['study_arm'] < 5]
SiteYData = SiteYData[SiteYData['enrollment_date'] != '-1']
ser = pd.to_datetime(SiteYData['enrollment_date'])
SiteYData['week'] = ser.dt.isocalendar().week
#Add 52 if the data was in 2023
SiteYData.loc[SiteYData['enrollment_date'].str.contains('2023'), 'week'] += 52
SiteYData['week'] -= min(SiteYData['week'])
SiteYData = SiteYData.sort_values(by=['week'])

#Do the same for SiteX
SiteXData = redcapData[redcapData['source'] == 1]
SiteXData = SiteXData[SiteXData['study_arm'] != 0]
SiteXData = SiteXData[SiteXData['study_arm'] < 5]
SiteXData = SiteXData[SiteXData['enrollment_date'] != '-1']
ser = pd.to_datetime(SiteXData['enrollment_date'])
SiteXData['week'] = ser.dt.isocalendar().week
SiteXData['week'] -= min(SiteXData['week'])
SiteXData = SiteXData.sort_values(by=['week'])

# %%
#Plot the cumulative counts of lvo and non lvo patient count for both sites
SiteYLvo = SiteYData[SiteYData['study_arm'] == 1]
SiteYNonLvo = SiteYData[SiteYData['study_arm'] != 1]
SiteYIschemic = SiteYData[SiteYData['study_arm'] == 2]
SiteYHemorrhagic = SiteYData[SiteYData['study_arm'] == 3]
SiteYMimic = SiteYData[SiteYData['study_arm'] == 4]
SiteYFirstNonLVOWeek = SiteYNonLvo['week'].min()
SiteYCurrentWeek = SiteYData['week'].max()

SiteYHemorrhagicRate = SiteYHemorrhagic.shape[0]/(SiteYCurrentWeek-SiteYFirstNonLVOWeek)
SiteYIschemicRate = SiteYIschemic.shape[0]/(SiteYCurrentWeek-SiteYFirstNonLVOWeek)
SiteYMimicRate = SiteYMimic.shape[0]/(SiteYCurrentWeek-SiteYFirstNonLVOWeek)

SiteYHemorrhagicTrailing4WeekRate = SiteYHemorrhagic[SiteYHemorrhagic['week'] >= SiteYCurrentWeek-4].shape[0]/4
SiteYIschemicTrailing4WeekRate = SiteYIschemic[SiteYIschemic['week'] >= SiteYCurrentWeek-4].shape[0]/4
SiteYMimicTrailing4WeekRate = SiteYMimic[SiteYMimic['week'] >= SiteYCurrentWeek-4].shape[0]/4

SiteYHemorrhagicWeeksLeft = math.ceil((25-SiteYHemorrhagic.shape[0])/SiteYHemorrhagicRate)
SiteYIschemicWeeksLeft = math.ceil((25-SiteYIschemic.shape[0])/SiteYIschemicRate)
SiteYMimicWeeksLeft = math.ceil((25-SiteYMimic.shape[0])/SiteYMimicRate)

SiteYHemorrhagicWeeksLeftTrailing4Week = math.ceil((25-SiteYHemorrhagic.shape[0])/SiteYHemorrhagicTrailing4WeekRate)
SiteYIschemicWeeksLeftTrailing4Week = math.ceil((25-SiteYIschemic.shape[0])/SiteYIschemicTrailing4WeekRate)
SiteYMimicWeeksLeftTrailing4Week = math.ceil((25-SiteYMimic.shape[0])/SiteYMimicTrailing4WeekRate)

plt.figure()
plt.plot(SiteYLvo['week'], np.cumsum(np.ones(len(SiteYLvo))), 'r', label='LVO')
plt.plot(np.arange(SiteYLvo['week'].max()+1), np.arange(SiteYLvo['week'].max()+1), 'r--', label='1 LVO/wk')
#plt.plot(SiteYNonLvo['week'], np.cumsum(np.ones(len(SiteYNonLvo))), 'b', label='IHC')
plt.plot(SiteYIschemic['week'], np.cumsum(np.ones(len(SiteYIschemic))), 'g', label='Ischemic')
plt.plot(SiteYHemorrhagic['week'], np.cumsum(np.ones(len(SiteYHemorrhagic))), 'y', label='Hemorrhagic')
plt.plot(SiteYMimic['week'], np.cumsum(np.ones(len(SiteYMimic))), 'm', label='Mimic')
plt.xlabel('Week')
plt.ylabel('Cumulative Count')
plt.title('SiteY')
plt.legend()

#Print the weeks left for each type of patient at SiteY
print('Projected weeks to complete study for patient type at SiteY:')
print('Hemorrhagic: ' + str(SiteYHemorrhagicWeeksLeft) + ' weeks(' + str(25-SiteYHemorrhagic.shape[0]) + '/25 remaining)')
print('Ischemic: ' + str(SiteYIschemicWeeksLeft) + ' weeks(' + str(25-SiteYIschemic.shape[0]) + '/25 remaining)')
print('Mimic: ' + str(SiteYMimicWeeksLeft) + ' weeks(' + str(25-SiteYMimic.shape[0]) + '/25 remaining)')

print('Projected weeks to complete study for patient type at SiteY using the trailing 4 weeks of enrollment data:')
print('Hemorrhagic: ' + str(SiteYHemorrhagicWeeksLeftTrailing4Week) + ' weeks')
print('Ischemic: ' + str(SiteYIschemicWeeksLeftTrailing4Week) + ' weeks')
print('Mimic: ' + str(SiteYMimicWeeksLeftTrailing4Week) + ' weeks')

SiteXLvo = SiteXData[SiteXData['study_arm'] == 1]
SiteXNonLvo = SiteXData[SiteXData['study_arm'] != 1]
SiteXIschemic = SiteXData[SiteXData['study_arm'] == 2]
SiteXHemorrhagic = SiteXData[SiteXData['study_arm'] == 3]
SiteXMimic = SiteXData[SiteXData['study_arm'] == 4]
SiteXFirstNonLVOWeek = SiteXNonLvo['week'].min()
SiteXCurrentWeek = SiteXData['week'].max()

SiteXLvoRate = SiteXLvo.shape[0]/SiteXCurrentWeek
SiteXHemorrhagicRate = SiteXHemorrhagic.shape[0]/SiteXCurrentWeek
SiteXIschemicRate = SiteXIschemic.shape[0]/SiteXCurrentWeek
SiteXMimicRate = SiteXMimic.shape[0]/SiteXCurrentWeek

SiteXHemorrhagicTrailing4WeekRate = SiteXHemorrhagic[SiteXHemorrhagic['week'] > SiteXCurrentWeek-4].shape[0]/4
SiteXIschemicTrailing4WeekRate = SiteXIschemic[SiteXIschemic['week'] > SiteXCurrentWeek-4].shape[0]/4

SiteXHemorrhagicWeeksLeft = math.ceil((25-SiteXHemorrhagic.shape[0])/SiteXHemorrhagicRate)
SiteXIschemicWeeksLeft = math.ceil((25-SiteXIschemic.shape[0])/SiteXIschemicRate)

SiteXHemorrhagicWeeksLeftTrailing4Week = math.ceil((25-SiteXHemorrhagic.shape[0])/SiteXHemorrhagicTrailing4WeekRate)
SiteXIschemicWeeksLeftTrailing4Week = math.ceil((25-SiteXIschemic.shape[0])/SiteXIschemicTrailing4WeekRate)

plt.figure()
plt.plot(SiteXLvo['week'], np.cumsum(np.ones(len(SiteXLvo))), 'r', label='LVO')
#plt.plot(np.arange(SiteXLvo['week'].max()+1), np.arange(SiteXLvo['week'].max()+1)*1.5, 'r--', label='1.5 LVOs/wk')
plt.plot(np.arange(SiteXLvo['week'].max()+1), np.arange(SiteXLvo['week'].max()+1)*2, 'r.', label='2 LVOs/wk')
#plt.plot(SiteXNonLvo['week'], np.cumsum(np.ones(len(SiteXNonLvo))), 'b', label='IHC')
plt.plot(SiteXIschemic['week'], np.cumsum(np.ones(len(SiteXIschemic))), 'g', label='Ischemic')
plt.plot(SiteXHemorrhagic['week'], np.cumsum(np.ones(len(SiteXHemorrhagic))), 'y', label='Hemorrhagic')
plt.plot(SiteXMimic['week'], np.cumsum(np.ones(len(SiteXMimic))), 'm', label='Mimic')
plt.xlabel('Week')
plt.ylabel('Cumulative Count')
plt.title('SiteX')
plt.legend()
plt.show()

#Print the weeks left for each type of patient at SiteX
print('Projected weeks to complete study for patient type at SiteX:')
print('Hemorrhagic: ' + str(SiteXHemorrhagicWeeksLeft) + ' weeks(' + str(25-SiteXHemorrhagic.shape[0]) + '/25 remaining)')
print('Ischemic: ' + str(SiteXIschemicWeeksLeft) + ' weeks(' + str(25-SiteXIschemic.shape[0]) + '/25 remaining)')

#Print the weeks left for each type of patient at SiteX
print('Projected weeks to complete study for patient type at SiteX using the trailing 4 weeks of enrollment data:')
print('Hemorrhagic: ' + str(SiteXHemorrhagicWeeksLeftTrailing4Week) + ' weeks')
print('Ischemic: ' + str(SiteXIschemicWeeksLeftTrailing4Week) + ' weeks')

# %%
pd.set_option('display.max_rows', None)
subjectIDsRedCap = redcapData['subject_id'].tolist()
namesFeatures    = opticalData['name'].tolist()
subjectIDsRedCap = [ ConvenienceFunctions.CleanName(nm) for nm in subjectIDsRedCap ]
namesFeatures    = [ ConvenienceFunctions.CleanName(nm) for nm in namesFeatures ]

#Replace the column with cleaned names and add the subject ID
opticalData['name'] = namesFeatures
redcapData['subject_id'] = subjectIDsRedCap

#Create a new dataframe with the features and redcap data where the subject IDs match names
dfFeaturesRedCapMerge = pd.merge(opticalData.copy(), redcapData, left_on='name', right_on='subject_id', how='inner')

# %%
#Make a new column for failures - scansValid is False or lowNoise is False
dfFeaturesRedCapMerge["scansValid"] = dfFeaturesRedCapMerge["scansValid"]=='True'
dfFeaturesRedCapMerge["lowNoise"] = dfFeaturesRedCapMerge["lowNoise"]=='True'

#%%
dfFeaturesRedCapMerge['failure'] = ~dfFeaturesRedCapMerge['scansValid'] | ~dfFeaturesRedCapMerge['lowNoise']
print(dfFeaturesRedCapMerge[['name', 'failure', 'scansValid', 'lowNoise']])
#Delete row if enrollment date is -1
dfFeaturesRedCapMerge = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['enrollment_date'] != '-1']
#%%
ser = pd.to_datetime(dfFeaturesRedCapMerge['enrollment_date'])
dfFeaturesRedCapMerge['week'] = ser.dt.isocalendar().week
#If week is less than 26 then add 52
dfFeaturesRedCapMerge.loc[dfFeaturesRedCapMerge['enrollment_date'].str.contains('2023'), 'week'] += 52
dfFeaturesRedCapMerge['week'] -= min(dfFeaturesRedCapMerge['week'])
#Get max number of weeks
maxWeek = dfFeaturesRedCapMerge['week'].max()
failuresPerWeek = np.zeros(maxWeek+1)
totalScansPerWeek = np.zeros(maxWeek+1)
percentageFailurePerWeek = np.zeros(maxWeek+1)
#Get the number of failures per week
for i in range(maxWeek+1):
    failuresPerWeek[i] = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['week'] == i]['failure'].sum()
    totalScansPerWeek[i] = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['week'] == i]['failure'].count()
percentageFailurePerWeek = failuresPerWeek/totalScansPerWeek
#Plot percentage failure per week
plt.figure()
plt.plot(np.arange(maxWeek+1), percentageFailurePerWeek)
#Plot histogram of failures per week
plt.figure()
plt.plot(np.arange(maxWeek+1), failuresPerWeek)
# %%
print(failuresPerWeek)
# %%
print(totalScansPerWeek)
#Remove the rows with failure = True
dfFeaturesRedCapMerge = dfFeaturesRedCapMerge[dfFeaturesRedCapMerge['failure'] == False]
# %%
#Print counts of each type of patient
dfFeaturesRedCapMerge['study_arm'] = dfFeaturesRedCapMerge['study_arm'].astype(int)
print(dfFeaturesRedCapMerge['study_arm'].value_counts())
# %%
