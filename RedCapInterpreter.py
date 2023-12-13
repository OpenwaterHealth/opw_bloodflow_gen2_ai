#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Mar 17 14:45:20 2023

@author: ethanhead
'''
#%%
import numpy as np
import pandas as pd
import math, os

# Dataset file names
SiteY_datafile= 'REDCapData/COMETSV3-OWAllPatientsAllFiel_DATA_2023-06-14_1928.csv'
SiteX_datafile = 'REDCapData/OpenWaterHeadsetForL_DATA_2023-06-14_1928.csv'
printDebug = False

SiteY_data = pd.read_csv(SiteY_datafile)
SiteX_data = pd.read_csv(SiteX_datafile)

if(printDebug):
    print('SiteY Data\n')
    num_cols = len(SiteY_data.columns)
    for i in range(num_cols):
        header = SiteY_data.columns[i]
        length = SiteY_data[header].size
        print(header + ': ' + str(length))

    print('SiteX Data\n')
    num_cols = len(SiteX_data.columns)
    for i in range(num_cols):
        header = SiteX_data.columns[i]
        length = SiteX_data[header].size
        print(header + ': ' + str(length))

# Drop rows where study arm is null
SiteY_data = SiteY_data[SiteY_data['study_arm'].notna()]

# fix values out of range in SiteY data
SiteY_data["study_arm_mimic"] = SiteY_data["study_arm_mimic"].apply(lambda x: 5 if (x>=5) else x).fillna(0)
SiteY_data["hx_diabetes"] = SiteY_data["hx_diabetes"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_htn"] = SiteY_data["hx_htn"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_afib"] = SiteY_data["hx_afib"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_hld"] = SiteY_data["hx_hld"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_chf"] = SiteY_data["hx_chf"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_carotidstenosis"] = SiteY_data["hx_carotidstenosis"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_icartstenosis"] = SiteY_data["hx_icartstenosis"].apply(lambda x: x if x<=1 else -1)
SiteY_data["hx_cad"] = SiteY_data["hx_cad"].apply(lambda x: x if x<=1 else -1)
SiteY_data["operator"] = 1 # always the same operator at SiteY? 
SiteY_data["operator_multiple"] = 0

# drop empty rows
SiteY_data = SiteY_data.dropna(subset=['subject_id'])
#drop all unmatchable columns from SiteY
clean_SiteY_data = SiteY_data.drop(columns=['lvo_ic_1', 'lvo_ic_2','subj_elig','enrollment_complete','demographics_complete','hx_other',
                          'facial_palsy','arm_motor_impairment','leg_motor_impairment','head_and_gaze_deviation','hemiparesis',
                          'agnosia','aphasia','facial_droop','arm_drift','grip_strength','conjugate_gaze_deviation',   
                            'incorrectly_answers_age_or','arm_falls_in_10_seconds','c_stat_score','facial_palsy_fast_ed','arm_weakness',
                            'speech_changes','eye_deviation','denial_neglect','fast_ed_score','prehospital_scores_complete',
                            'thombolytic_agent','baseline_imaging___1','baseline_imaging___2','baseline_imaging___3','baseline_imaging___9',
                            'baseline_infarct','baseline_infarct_side','ctp_available','perfusion_defect','baseline_stroke_details_complete','headframe_size',
                            'monitor_base_time','monitor_recan_time','monitoring_session_complete',    
                            'reperfusion_time','reperfusion_tici','thrombo_compliations___0','thrombo_compliations___1','thrombo_compliations___2',
                            'thrombo_compliations___3','thrombo_compliations___4','thrombo_compliations___5','thrombo_passes','procedure_details_complete',
                            'ct_post_thrombo','post_thrombo_hemorrhage','followup_complete','discharge_details_complete','ae_reporttype','ae_serious',
                            'ae_outcome','ae_status','ae_severity','ae_related','ae_expected','adverse_events_complete',
                            "gic_1","gic_2","gec_1","gec_2","gec_3","gec_4","gec_5","ais_wo_lvo_ic_1","ais_wo_lvo_ic_2",
                            "hs_ic_1","hs_ic_2","sm_ic_1","sm_ic_2","medical_history_complete",'thrombo_device_technique___1','thrombo_device_technique___2',
                            'thrombo_device_technique___3','dc_dispo', "anesthesia_type","bp_data_method","lkw_date","lkw_time", "monitor_date"])
clean_SiteY_data["source"] = 0
clean_SiteY_data = clean_SiteY_data.drop_duplicates()

clean_SiteY_data["lvo_site___1"] = clean_SiteY_data["lvo_site___1"].fillna(0).astype(int)# = SiteX_data["occlusion_location___0"].astype(int)
clean_SiteY_data["lvo_site___2"] = clean_SiteY_data["lvo_site___2"].fillna(0).astype(int)# = SiteX_data["occlusion_location___1"].astype(int)
clean_SiteY_data["lvo_site___3"] = clean_SiteY_data["lvo_site___3"].fillna(0).astype(int)# = SiteX_data["occlusion_location___2"].astype(int)
clean_SiteY_data["lvo_site___4"] = clean_SiteY_data["lvo_site___4"].fillna(0).astype(int)# = SiteX_data["tandem"].astype(int)


#rename blood pressure columns to be less silly
clean_SiteY_data["pre_op_sbp"] = clean_SiteY_data["bp_at_time_of_preop"]
clean_SiteY_data["post_op_sbp"] = clean_SiteY_data["bp_at_time_of_postop"]
clean_SiteY_data["pre_op_map"] = clean_SiteY_data["map"]

# put bp for nonsurgery into pre-op bp
clean_SiteY_data["pre_op_sbp"] = clean_SiteY_data["pre_op_sbp"].fillna(clean_SiteY_data["blood_pressure_at_time_of"])
clean_SiteY_data["pre_op_dbp"] = clean_SiteY_data["pre_op_dbp"].fillna(clean_SiteY_data["dbp"])

clean_SiteY_data = clean_SiteY_data.drop(["bp_at_time_of_preop","bp_at_time_of_postop", "blood_pressure_at_time_of","dbp","map"], axis=1)

# Switch to SiteX data

# Copy over schema from SiteX data to be removed after it is converted to SiteY style
SiteX_schema = SiteX_data.columns
# Remove columns that we want to keep from SiteX_schema
SiteX_schema = SiteX_schema.drop("record_id")
SiteX_schema = SiteX_schema.drop("fitzpatrick_scale")
SiteX_schema = SiteX_schema.drop("sex")
SiteX_schema = SiteX_schema.drop("subject_id") 
SiteX_schema = SiteX_schema.drop("skull_thickness") 
SiteX_schema = SiteX_schema.drop("scalp_thickness")
SiteX_schema = SiteX_schema.drop("ethnicity")

# Convert SiteX data to SiteY style
SiteX_data["age"] = SiteX_data["age_at_study"]

def columns_to_enum_string(df, cols_to_check, enum_list, output_col):
    # Create a copy of the original DataFrame to avoid modifying it
    new_df = df.copy()
    
    # Iterate over the specified columns
    for i, col in enumerate(cols_to_check):
        # Check if the value in the first column is greater than zero
        mask = new_df[cols_to_check[i]] > 0
        
        # Add the corresponding value from the enumeration list to the output column
        new_df.loc[mask, output_col] = enum_list[i]
        
    return new_df

study_arm_lookup = [1,2,3,4]
SiteX_data = columns_to_enum_string(SiteX_data,['inclusion_3','other_ischemia','hemorrhagic_stroke','stroke_mimic'], study_arm_lookup, "study_arm")

# SiteX_data["study_arm"]  = np.where(SiteX_data["inclusion_3"],1,0) # patient will undergo mechanical thrombectomy
SiteX_data["study_arm_mimic"] = np.where(SiteX_data["inclusion_3"],0,-1)

SiteX_data["race___5"] = 0+(SiteX_data["race"] == 0)  #race = white
SiteX_data["race___3"] = 0+(SiteX_data["race"] == 1) #race = black/ african american
SiteX_data["race___2"] = 0+(SiteX_data["race"] == 2) #race = asian
SiteX_data["race___1"] = 0+(SiteX_data["race"] == 3)#race = native american
SiteX_data["race___4"] = 0+(SiteX_data["race"] == 4)#race = native hawaiian / pacific islander
SiteX_data["race___6"] = 0+(SiteX_data["race"] > 4) # race == other
SiteX_data["race___7"] = 0 #no entry on SiteX data for unknown race
SiteX_data["ethnicity"] = 1- SiteX_data["ethnicity"]

SiteX_data["race_score"] = np.where(SiteX_data["race_arrival_available"],SiteX_data["race_arrival"],-1)
SiteX_data["lams_score"] = np.where(SiteX_data["lams_arrival_available"],SiteX_data["lams_arrival"],-1)
SiteX_data["hx_htn"] = SiteX_data["htn"].apply(lambda x: x if x<=1 else -1)
SiteX_data["hx_hld"] = SiteX_data["dl"].apply(lambda x: x if x<=1 else -1)
SiteX_data["hx_diabetes"] = SiteX_data["dm"].apply(lambda x: x if x<=1 else -1)
SiteX_data["hx_chf"] = SiteX_data["chf"]
SiteX_data["hx_carotidstenosis"] = SiteX_data["carotid_stenosis"]
SiteX_data["hx_icartstenosis"] = SiteX_data["arterial_stenosis"]
SiteX_data["hx_afib"] = SiteX_data["afib"]
SiteX_data["nihss_baseline"] = SiteX_data["nihss_pre"]
SiteX_data["lvo_side"] = np.where(pd.isna(SiteX_data["side"]),-1,SiteX_data["side"]+1) # SiteY style is left is 1 right is 2, 
SiteX_data["lvo_site___1"] = SiteX_data["occlusion_location___0"].fillna(0).astype(int)
SiteX_data["lvo_site___2"] = SiteX_data["occlusion_location___1"].fillna(0).astype(int)
SiteX_data["lvo_site___3"] = SiteX_data["occlusion_location___2"].fillna(0).astype(int)
SiteX_data["lvo_site___4"] = SiteX_data["tandem"].fillna(0).astype(int)
SiteX_data["aspects_baseline"] = SiteX_data["aspects_pre"]
SiteX_data["hypodensity_cortical_front"] = SiteX_data["frontal_hypodensity"]
SiteX_data["post_thrombo_echo_ef"] = SiteX_data["ejection_fraction"]
SiteX_data["dc_nihss"] = np.where(SiteX_data["discharge_nihss_done"],SiteX_data["discharge_nihss"],-1)
SiteX_data["source"] = 1
SiteX_data["iv_thombolytic_given"] = SiteX_data["tpa"]
SiteX_data["monitor_timefromonset"] = SiteX_data["lkw_to_baseline_tracing"]
SiteX_data["enrollment_date"] = SiteX_data["study_date"]

SiteX_data["pre_op_sbp"] = SiteX_data["bp_systolic_baseline"]
SiteX_data["pre_op_dbp"] = SiteX_data["bp_diastolic_baseline"]
SiteX_data["post_op_sbp"] = SiteX_data["bp_systolic_second_tracing"]
SiteX_data["post_op_dbp"] = SiteX_data["bp_diastolic_second_tracing"]
SiteX_data["pre_op_map"] = -1
SiteX_data["post_op_map"] = -1

def columns_to_enum_string(df, cols_to_check, enum_list, output_col):
    # Create a copy of the original DataFrame to avoid modifying it
    new_df = df.copy()
    
    # Iterate over the specified columns
    for i, col in enumerate(cols_to_check):
        # Check if the value in the first column is greater than zero
        mask = new_df[cols_to_check[i]] > 0
        
        # Add the corresponding value from the enumeration list to the output column
        new_df.loc[mask, output_col] = enum_list[i]
        
    return new_df

operator_names = [2,4]
SiteX_data = columns_to_enum_string(SiteX_data,['investigator___16','investigator___18'], operator_names, "operator")
SiteX_data["operator_multiple"] = SiteX_data["investigator___17"].apply(lambda x: 1 if x>0 else 0)

# Put data together
clean_SiteX_data = SiteX_data.drop(SiteX_schema.array, axis = 1)
output_data = pd.concat([clean_SiteY_data, clean_SiteX_data], ignore_index=True)

# Clean up unified data
output_data["fitzpatrick_scale"] = output_data["fitzpatrick_scale"].fillna(-1).astype(int)
output_data["race_score"] = output_data["race_score"].fillna(-1).astype(int)
output_data["lams_score"] = output_data["lams_score"].fillna(-1).astype(int)
output_data["dc_nihss"] = output_data["dc_nihss"].fillna(-1).astype(int)
output_data["nihss_baseline"] = output_data["nihss_baseline"].fillna(-1).astype(int)
output_data["lvo_side"] = output_data["lvo_side"].fillna(-1)
output_data["post_thrombo_echo_ef"] = output_data["post_thrombo_echo_ef"].fillna(-1).astype(int)
output_data["hypodensity_cortical_front"] = output_data["hypodensity_cortical_front"].fillna(-1)
output_data["study_arm_mimic"] = output_data["study_arm_mimic"].fillna(-1)
output_data = output_data.fillna(-1)
#Save unified data

output_data["lvo"] = output_data["study_arm"].apply(lambda x: 1 if x == 1 else 0)

output_data['subject_id'] = output_data['subject_id'].astype(str)

output_data = output_data.sort_values('subject_id')

os.remove('redcap_data_values.csv')
output_data.to_csv("redcap_data_values.csv")


# # Create a new column in the DataFrame based on the values in the four columns
lvo_site_names = ['M1 Occlusion', 'M2 Occlusion', 'ICA Occlusion', 'Tandem Occlusion']
output_data = columns_to_enum_string(output_data,['lvo_site___1','lvo_site___2','lvo_site___3','lvo_site___4'], lvo_site_names, "lvo_site")

output_data["hx_diabetes"] = output_data["hx_diabetes"]# .astype(bool)
output_data["hx_htn"] = output_data["hx_htn"]# .astype(bool)
output_data["hx_hld"] = output_data["hx_hld"]# .astype(bool)
output_data["hx_chf"] = output_data["hx_chf"]# .astype(bool)
output_data["hx_carotidstenosis"] = output_data["hx_carotidstenosis"].apply(lambda x: x if x<=1 else -1)# .astype(bool)
output_data["hx_icartstenosis"] = output_data["hx_icartstenosis"].apply(lambda x: x if x<=1 else -1)# .astype(bool)
output_data["hx_cad"] = output_data["hx_cad"].apply(lambda x: x if x<=1 else -1)# .astype(bool)
output_data["hx_afib"] = output_data["hx_afib"]# .astype(bool)
output_data["hypodensity_cortical_front"] = output_data["hypodensity_cortical_front"]# .astype(bool)

sex_lookup = ["female","male"]
output_data['sex'] = output_data['sex'].apply(lambda x: sex_lookup[int(x)])

study_arm_lookup = ["No thrombectomy","LVO","Ischemic stroke","Hemorrhagic stroke", "Stroke mimic", "EVD","Optical Biomarker"]
output_data['study_arm'] = output_data['study_arm'].apply(lambda x: study_arm_lookup[int(x)])

study_arm_mimic_lookup = ["NA","Seizures","Complex Migraines","Conversion","Recrudescence","Other"]
output_data['study_arm_mimic'] = output_data['study_arm_mimic'].apply(lambda x: study_arm_mimic_lookup[int(x)])

side_of_occlusion_lookup = ["Unknown","NA","Right","Left"]
output_data['lvo_side'] = output_data['lvo_side'].apply(lambda x: side_of_occlusion_lookup[int(x)+1])

test_location_lookup = ["SiteY","SiteX"]
output_data['source'] = output_data['source'].apply(lambda x: test_location_lookup[int(x)])

# convert these to bool types?
# output_data["hx_carotidstenosis"] = output_data["hx_carotidstenosis"]
# output_data["hx_icartstenosis"] = output_data["hx_icartstenosis"]
# output_data["hx_htn"] = output_data["hx_htn"]
# output_data["hx_chf"] = output_data["hx_chf"]
# output_data["hx_afib"] = output_data["hx_afib"]

os.remove('redcap_data_labels.csv')
output_data.to_csv("redcap_data_labels.csv")

# %%
