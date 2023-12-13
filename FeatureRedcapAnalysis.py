#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import PyPDF2
from ReadGen2Data import ConvenienceFunctions

features = pd.read_csv('all_optical_features.csv')
redcap_data = pd.read_csv('redcap_data_values.csv')
redcap_data = redcap_data.rename(columns={"subject_id":"name"})
redcap_data['name'] = redcap_data['name'].apply(lambda x: ConvenienceFunctions.CleanName(x) )
features['name'] = features['name'].apply(lambda x: ConvenienceFunctions.CleanName(x) )

merged_inner = pd.merge(redcap_data, features, on='name',how='inner')
merged_inner.to_csv("redcap_and_features_values.csv")

merged_left = pd.merge(redcap_data, features, on='name',how='left')
unlinked_labels = []
for i, featureRow in merged_left.iterrows():
    if(math.isnan(featureRow["device"])):
        unlinked_labels.append(featureRow["name"])
print(unlinked_labels)

merged_right = pd.merge(redcap_data, features, on='name',how='right')
unlinked_labels = []
for i, featureRow in merged_right.iterrows():
    if(math.isnan(featureRow["age"])):
        unlinked_labels.append(featureRow["name"])
print(unlinked_labels)


print("Number of linked rows/all feature rows: " +  str(len(merged_inner)) + "/" + str(len(features)))
print("Number of linked rows/all redcap rows: " +  str(len(merged_inner)) + "/" + str(len(redcap_data)))
#%%

color_names = ["No thrombectomy (SiteX)","Large Vessel Occlusion (LVO)",
    "Ischemic stroke without LVO",
    "Hemorrhagic stroke",
    "Stroke mimic",]
merged_inner["study_arm_name"] = merged_inner["study_arm"].apply(lambda x: color_names[int(x)])

output_path = 'output/'
def plot_columns(df, x_col_list, y_col_list):
    for x_col in x_col_list:
        for y_col in y_col_list:
            plt.scatter(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.savefig(f"{x_col}_{y_col}.jpeg")
            plt.clf()  # Clear the plot for the next iteration

def get_covariance(df, x_col_list, y_col_list):
    x_df = df[x_col_list]
    y_df = df[y_col_list]
    cov_list = []
    for x_col in x_col_list:
        for y_col in y_col_list:
            cov = x_df[x_col].cov(y_df[y_col])
            cov_list.append(cov)
    return cov_list

def get_corr(df, x_col, y_col):
    # Remove any rows where either x or y is -1
    df = df[(df[x_col] != -1) & (df[y_col] != -1)]
    
    # Calculate the correlation and number of values
    corr = df[x_col].corr(df[y_col])
    n = len(df)
    
    return corr, n

def combine_pdfs(paths, output_path):
    """Combine a list of PDFs into a single PDF with multiple pages."""
    merger = PyPDF2.PdfMerger()
    for path in paths:
        merger.append(path)
    with open(output_path, 'wb') as f:
        merger.write(f)

def plot_dataframe(df, x_col_list, y_col_list, threshold=0,set = 0):
    # Create a dictionary to hold the covariance values for each pair of columns
    corr_dict = []
    filepaths = []
    # Loop over all combinations of x and y columns
    for x_col in x_col_list:
        fig, axs = plt.subplots(nrows=int(1+(len(y_col_list))/5), ncols=5, figsize=(20, 15))
        fig.suptitle(f"Plots of Clinical Feature {x_col} against Optical Features")
        for i, y_col in enumerate(y_col_list):

            corr, num = get_corr(df, x_col, y_col)
            corr_dict.append([x_col, y_col, corr, num])
            ax = axs[int(i/5),i%5]
            ax.set_title("(corr: %.2E, N = %i)" % (corr,num))
            ax.set_ylabel("y_col")

            # Plot the data and calculate the covariance
            df.plot.scatter(x=x_col, y=y_col, ax=ax)

            # Add a blue border to the plot if the covariance is above or below the threshold
            if corr > threshold or corr < -threshold:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')

        # Save the figure with the name of the x_col as the title
        pdf_path = os.path.join(output_path,f"correlations_{x_col}_set{set}.jpg")
        
        fig.savefig(pdf_path, format="pdf")
        filepaths.append(pdf_path)
    combine_pdfs(filepaths, f'output/correlations_{ set}.pdf')

    return corr_dict;

def plot_corr(df, low_threshold, high_threshold):
    # Drop columns with more than 10 instances of -1
    df = df.loc[:, (df == -1).sum() <= 10]

    # Compute correlation matrix
    corr = df.corr()

    # Filter correlations below threshold
    corr = corr[low_threshold < corr  ]
    corr = corr[corr < high_threshold ]
    corr.to_csv("correlations.csv")

    # Create table plot
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=np.round(corr.values, 2), colLabels=corr.columns, loc='center')
    # table.auto_set_font_size(False)
    # table.set_fontsize(14)
    table.scale(1, 1.5)

    # Save table as PDF
    plt.savefig('corr_table.pdf', bbox_inches='tight')

    # Create heatmap plot
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set x and y ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if corr.values[i, j] != 0:
                text = ax.text(j, i, np.round(corr.values[i, j], 2),
                               ha="center", va="center", color="w")

    # Save heatmap as PDF
    plt.savefig('corr_heatmap.pdf', bbox_inches='tight')

    return corr


clinical_features = ["iv_thombolytic_given","age","sex","fitzpatrick_scale","race_score","lams_score","hx_htn",
                     "hx_diabetes","hx_hld","hx_chf","hx_carotidstenosis","hx_icartstenosis",
                     "nihss_baseline","source","study_arm","lvo",
                     "pre_op_sbp","pre_op_dbp","pre_op_map","monitor_timefromonset"]
# clinical_features = ["study_arm"]

optical_features = [[
    "Modulation depth",
    "Area under curve",
    "Area under curve P1",
    "Skewness",
    "Kurtosis",
    "Pulse canopy",
    "Pulse onset",
    "Pulse onset proportion",
    "Second Moment",
    "Amplitude",
    "Velocity curve index",
    "Velocity curve index Hanning",
    "Velocity curve index normalized",
    "Velocity curve index Hanning normalized"],
    ["Area under curve PulseSegments",
    "Area under curve P1 PulseSegments",
    "Skewness PulseSegments",
    "Kurtosis PulseSegments",
    "Pulse canopy PulseSegments",
    "Pulse onset PulseSegments",
    "Pulse onset proportion PulseSegments",
    "Second Moment PulseSegments",
    "Amplitude PulseSegments",
    "Velocity curve index PulseSegments",
    "Velocity curve index Hanning PulseSegments",
    "Velocity curve index normalized PulseSegments",
    "Velocity curve index Hanning normalized PulseSegments"],
    ["Area under curve SegmentsRange",
    "Area under curve P1 SegmentsRange",
    "Skewness SegmentsRange",
    "Kurtosis SegmentsRange",
    "Pulse canopy SegmentsRange",
    "Pulse onset SegmentsRange",
    "Pulse onset proportion SegmentsRange",
    "Second Moment SegmentsRange",
    "Amplitude SegmentsRange",
    "Velocity curve index SegmentsRange",
    "Velocity curve index Hanning SegmentsRange",
    "Velocity curve index normalized SegmentsRange",
    "Velocity curve index Hanning normalized SegmentsRange",
    ]
    ]

# corr = plot_corr(merged_inner,.25, .75)

def get_top_t_values(df, t):
    # Get the top t values in the DataFrame
    top_t = df.stack().nlargest(t)
    # Get the row and column names of the top t values
    row_col_names = [(idx[0], idx[1]) for idx in top_t.index]
    return top_t, row_col_names

# top_t_values, row_col_names = get_top_t_values(corr, 50)
# print(top_t_values)
# print(row_col_names)
#%%

corr_dict = []
corr_dict = corr_dict + (plot_dataframe(merged_inner,clinical_features,optical_features[0],.99,set=0))
corr_dict = corr_dict + (plot_dataframe(merged_inner,clinical_features,optical_features[1],.99,set=1))
corr_dict = corr_dict +plot_dataframe(merged_inner,clinical_features,optical_features[2],.99,set=2)
corr_pd =  pd.DataFrame(corr_dict)
corr_pd = corr_pd.sort_values(by=[2])
corr_pd.to_csv("correlations_abbr.csv")

#%%
corr = plot_corr(merged_inner,.25, .99)
# cov_dict = plot_dataframe(merged_inner,clinical_features,optical_features[1],5,set=1)

# Print the covariance values for each pair of columns
# for pair, cov in cov_dict.items():
    # print(f"Covariance for {pair}: {cov}")
# print(high_covariance_pairs)

# print(get_covariance(merged_inner,clinical_features,optical_features))

# plot_columns(merged_inner,clinical_features,optical_features)


import pandas as pd
import numpy as np

def custom_corr_matrix(df, corr_func):
    """
    Generates a correlation matrix of a 2D dataframe with a given correlation function.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        corr_func (callable): The correlation function to be used.

    Returns:
        pd.DataFrame: The correlation matrix of the input dataframe with the given correlation function.
    """
    # Calculate the number of columns in the dataframe
    num_cols = df.shape[1]
    
    # Create an empty correlation matrix
    corr_matrix = np.zeros((num_cols, num_cols))
    
    # Calculate the correlation between each pair of columns using the given correlation function
    for i in range(num_cols):
        for j in range(i, num_cols):
            corr_val = corr_func(df.iloc[:,i], df.iloc[:,j])
            corr_matrix[i,j] = corr_val
            corr_matrix[j,i] = corr_val
    
    # Create a pandas dataframe from the correlation matrix
    corr_df = pd.DataFrame(data=corr_matrix, columns=df.columns, index=df.columns)
    
    return corr_df

correlations_cool = custom_corr_matrix(merged_inner,corr_func=get_corr)

import seaborn as sns
import matplotlib.pyplot as plt

def save_corr_heatmap_as_pdf(corr_matrix, filepath):
    """
    Generates a heatmap from a correlation matrix and saves it as a PDF file.

    Parameters:
        corr_matrix (pd.DataFrame): The correlation matrix to generate the heatmap from.
        filepath (str): The filepath to save the PDF file to.

    Returns:
        None.
    """
    # Create a heatmap using seaborn
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(10,10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, square=True, annot=True, linewidths=.5, cbar_kws={'shrink': .5})
    plt.title('Correlation Heatmap')

    # Save the heatmap as a PDF file
    plt.savefig(filepath, bbox_inches='tight')

save_corr_heatmap_as_pdf(correlations_cool,"heatmap.pdf")


#%%
only_big_vci = merged_inner.copy()
only_big_vci = only_big_vci[only_big_vci["study_arm"]==1]
plot_dataframe(only_big_vci,["age"],
    ["Velocity curve index",
    "Velocity curve index Hanning",
    "Velocity curve index normalized",
    "Velocity curve index Hanning normalized" ,
    "Area under curve P1",
    "Velocity curve index SegmentsRange",
    "Velocity curve index Hanning SegmentsRange",
    "Velocity curve index normalized SegmentsRange",
    "Velocity curve index Hanning normalized SegmentsRange",
    ])
non_stroke = merged_inner.copy()
non_stroke = non_stroke[non_stroke["study_arm"]!=1]
plot_dataframe(non_stroke,["age"],
    ["Velocity curve index",
    "Velocity curve index Hanning",
    "Velocity curve index normalized",
    "Velocity curve index Hanning normalized" ,
    "Area under curve P1",
    "Velocity curve index SegmentsRange",
    "Velocity curve index Hanning SegmentsRange",
    "Velocity curve index normalized SegmentsRange",
    "Velocity curve index Hanning normalized SegmentsRange",
    ])
#%%
class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

column_names = {'study_arm': 'Study Arm', 'study_arm_mimic': 'Mimic', 'age': 'Age', 'sex': 'Sex', 'hx_htn': 'Hypertension (HTN)', 'hx_diabetes': 'Diabetes', 'hx_chf': 'Congestive Heart Failure (CHF)', 'hx_carotidstenosis': 'Carotid Stenosis', 'hx_icartstenosis': 'Intracranial Arterial Stenosis', 'hx_afib': 'Atrial Fibrillation (AF)', 'fitzpatrick_scale': 'Fitzpatrick Scale Type:', 'race_score': 'RACE score', 'lams_score': 'LAMS score', 'nihss_baseline': 'NIH Stroke Scale (NIHSS)', 'lvo_side': 'Side of Occlusion', 'lvo_site___1': 'Occlusion Site (choice=M1 Occlusion)', 'lvo_site___2': 'Occlusion Site (choice=M2 Occlusion)', 'lvo_site___3': 'Occlusion Site (choice=ICA Occlusion)', 'lvo_site___4': 'Occlusion Site (choice=Tandem Occlusion)', 'aspects_baseline': 'ASPECTS Score', 'hypodensity_cortical_front': 'Cortical frontal lobe hypodensity of CT', 'skull_thickness': 'Skull thickness (on bone-window): ', 'scalp_thickness': 'Scalp Thickness (on bone-window)', 'post_thrombo_echo_ef': 'Ejection Fraction', 'dc_nihss': 'Discharge NIHSS', 'dc_dispo': 'Discharge Disposition', 'ethnicity': 'Ethnicity'}
def plot_histogram_by_column(dataframe, column1, column2):
    # Get unique values of column2
        if(column1 == "age"):
            dataframe = dataframe[dataframe["age" ]>1]
        unique_values = dataframe[column2].unique()
        maximumX = dataframe[column1].max()
        minimumX = dataframe[column1].min()
        maxY = 0
        num_bins = len(dataframe[column1].unique()) if len(dataframe[column1].unique()) < 10 else 10
        hist,set_bins = np.histogram(dataframe[column1],bins = num_bins, range=(minimumX,maximumX));
        for i, value in enumerate(unique_values):
            hist,bins = np.histogram(dataframe[dataframe[column2] == value][column1],bins = set_bins, );
            maxY = hist.max() if hist.max() > maxY else maxY
        print(hist)
        print(maxY)
        maxY = maxY * 1.20 # ensure 
        # maxY= dataframe[column2].max()
        # Create a subplot for each unique value of column2
        fig, axs = plt.subplots(len(unique_values), figsize=(8, 8))
        sources = ['SiteY','SiteX']
        # Plot a histogram for each unique value of column2
        for i, value in enumerate(unique_values):
            axs[i].hist(dataframe[dataframe[column2] == value][column1], bins=set_bins)
            std = dataframe[dataframe[column2] == value][column1].std()
            mean = dataframe[dataframe[column2] == value][column1].mean()
            title = column1
            axs[i].set_title(f"{title} from {sources[value]} (mean: {prettyfloat(mean)}, std: {prettyfloat(std)})")
            axs[i].set_xlim(minimumX,maximumX)
            axs[i].set_ylim(top = maxY)

        plt.tight_layout()
        plt.savefig(f'output/comp_{column1}.pdf', bbox_inches='tight')

        plt.show()

plot_histogram_by_column(merged_inner,"Velocity curve index","source")

plot_histogram_by_column(merged_inner,"Pulse canopy SegmentsRange","source")
#%%
def scatter_by_color(df, x_col, y_col, color_col):
    """
    Generate a scatter plot of a Pandas DataFrame with the x_col on the x-axis,
    the y_col on the y-axis, and the points colored based on the values in the
    color_col column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to generate a scatter plot from.
    x_col : str
        The name of the column to use for the x-axis.
    y_col : str
        The name of the column to use for the y-axis.
    color_col : str
        The name of the column to use for the point colors.
    """
    color_col = "study_arm_name"
    # Get the unique values in the color column
    colors = df[color_col].unique()
    # Create a dictionary that maps each color value to a unique integer
    color_dict = {color: i for i, color in enumerate(colors)}
    
    # Create a list of integers representing the color of each point in the plot
    point_colors = [color_dict[color] for color in df[color_col]]
    
    # Generate the scatter plot
    scatter = plt.scatter(df[x_col], df[y_col], c=point_colors,)
    scatter.axes.set_xlim(30,100)
    scatter.axes.set_ylim(0,14)
    # Add a legend with the color values
    color_labels = [f"{color}" for color in colors]
    # plt.legend(color_labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=color_labels)

    # Set the axis labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    # Show the plot
    plt.show()

# scatter_by_color(merged_inner,"age","Velocity curve index","study_arm")
scatter_by_color(merged_inner,"age","Velocity curve index Hanning normalized","study_arm_name")


#%%
SiteX_merged = merged_inner.copy()
SiteX_merged = SiteX_merged[SiteX_merged["source"]==1]
# scatter_by_color(SiteX_merged,"age","Velocity curve index","study_arm")
corr_dict = []
corr_dict = corr_dict + (plot_dataframe(SiteX_merged,clinical_features,optical_features[0],.99,set=0))
corr_dict = corr_dict + (plot_dataframe(SiteX_merged,clinical_features,optical_features[1],.99,set=1))
corr_dict = corr_dict +plot_dataframe(SiteX_merged,clinical_features,optical_features[2],.99,set=2)
corr_pd =  pd.DataFrame(corr_dict)
corr_pd = corr_pd.sort_values(by=[2])
corr_pd.to_csv("correlations_abbr_SiteX.csv")
#%%
SiteY_merged = merged_inner.copy()
SiteY_merged = SiteY_merged[SiteY_merged["source"]==0]
scatter_by_color(SiteY_merged,"age","Velocity curve index","study_arm")
corr_dict = []
corr_dict = corr_dict + (plot_dataframe(SiteY_merged,clinical_features,optical_features[0],.99,set=0))
corr_dict = corr_dict + (plot_dataframe(SiteY_merged,clinical_features,optical_features[1],.99,set=1))
corr_dict = corr_dict +plot_dataframe(SiteY_merged,clinical_features,optical_features[2],.99,set=2)
corr_pd =  pd.DataFrame(corr_dict)
corr_pd = corr_pd.sort_values(by=[2])
corr_pd.to_csv("correlations_abbr_SiteY.csv")

# %%


scatter_by_color(merged_inner,"age","Velocity curve index","source")

#%%
scatter_by_color(SiteX_merged,"age","Velocity curve index Hanning normalized","study_arm")
scatter_by_color(SiteY_merged,"age","Velocity curve index Hanning normalized","study_arm")


#%% 

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def generate_roc_curve(true_labels, predicted_probabilities, output_file):
    # Compute ROC curve and area under the curve (AUC)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save ROC curve to output file
    plt.savefig(output_file)

generate_roc_curve(merged_inner["lvo"],merged_inner["nihss_baseline"],"roc_curve.jpg")
