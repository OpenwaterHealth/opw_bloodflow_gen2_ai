#%%
import pandas as pd

#%%
allRedcapData = pd.read_csv('redcap_data_labels_with_classes.csv')

#Print all column names
print(allRedcapData.columns)

# %%
#Get argmax of predProb0 predProb1 and save to new column predClass with values 0 or 1
allRedcapData['predClass'] = allRedcapData[['predProb0', 'predProb1']].apply(lambda x: x.argmax(), axis=1)
# %%
#Get the indices where allRedcapData where race_score column is greater than 4
highRaceDataIndices = allRedcapData.index[allRedcapData['race_score'] > 4].tolist()
print(highRaceDataIndices)
#Create a column for race score lvo prediction called raceLvoPred in allRedcapData and set to 0
allRedcapData['raceLvoPred'] = 0
#Set highRaceDataIndices raceLvoPred column to 1
allRedcapData.loc[highRaceDataIndices, 'raceLvoPred'] = 1
#Write a confusion matrix for raceLvoPred and lvo columns
print(pd.crosstab(allRedcapData['lvo'], allRedcapData['raceLvoPred'], rownames=['Actual'], colnames=['Predicted']))
#%%
#Get counts for lvo column in highRaceData
highRaceData = allRedcapData[highRaceData]
print(highRaceData['lvo'].value_counts())
print(highRaceData['predClass'].value_counts())
print(pd.crosstab(highRaceData['lvo'], highRaceData['predClass'], rownames=['Actual'], colnames=['Predicted']))

# %%
#Filter dataframe by race_score column less than 5 and create new dataframe
lowRaceData = allRedcapData['race_score'] < 5
lowRaceData = allRedcapData[lowRaceData]
#Get counts for lvo column in highRaceData
print(lowRaceData['lvo'].value_counts())
print(lowRaceData['predClass'].value_counts())
print(pd.crosstab(lowRaceData['lvo'], lowRaceData['predClass'], rownames=['Actual'], colnames=['Predicted']))
# %%
#Do the same for lams_score column
#Filter dataframe by lams_score column more than 3 and create new dataframe
highLamsData = allRedcapData['lams_score'] > 3
highLamsDataIndices = highLamsData.index
#Create a column for lams score lvo prediction called raceLvoPred in allRedcapData and set to 0
allRedcapData['lamsLvoPred'] = 0
#Set highLamsDataIndices lamsLvoPred column to 1
allRedcapData.loc[highLamsDataIndices, 'raceLvoPred'] = 1
#Write a confusion matrix for lamsLvoPred and lvo columns
print(pd.crosstab(allRedcapData['lvo'], allRedcapData['lamsLvoPred'], rownames=['Actual'], colnames=['Predicted']))




highLamsData = allRedcapData[highLamsData]
#Get counts for lvo column in highLamsData
print(highLamsData['lvo'].value_counts())
print(highLamsData['predClass'].value_counts())
print(pd.crosstab(highLamsData['lvo'], highLamsData['predClass'], rownames=['Actual'], colnames=['Predicted']))

# %%
#Filter dataframe by lams_score column less than 4 and create new dataframe
lowLamsData = allRedcapData['lams_score'] < 4
lowLamsData = allRedcapData[lowLamsData]
#Get counts for lvo column in highLamsData
print(lowLamsData['lvo'].value_counts())
print(lowLamsData['predClass'].value_counts())
print(pd.crosstab(lowLamsData['lvo'], lowLamsData['predClass'], rownames=['Actual'], colnames=['Predicted']))
# %%
