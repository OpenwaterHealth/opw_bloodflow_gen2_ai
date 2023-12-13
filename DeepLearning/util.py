import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle, os
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import auc, RocCurveDisplay

def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def normalize_bloodflow_data(res):
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tmp_data = res[i,j,:]
            tmp_mean = np.mean(tmp_data)
            tmp_std = np.std(tmp_data)
            res[i,j,:] = (tmp_data - tmp_mean) / tmp_std
    res[np.isnan(res)] = 1e-6
    return res

def read_blood_flow_data(Upsample=True):
    #Read npy file
    fileNms = ['../../data/BloodFlow/scanData.npy', '../data/BloodFlow/scanData.npy']
    # check which path exists
    for fileNm in fileNms:
        if os.path.exists(fileNm):
            break
    with open(fileNm, 'rb') as fin:
        res = np.load(fin, allow_pickle=True)
    res = res.astype(np.float32)
    #Replace any nan values with 0
    res[np.isnan(res)] = 1e-6

    #Upsample the data by 8x using interpolation
    from scipy import interpolate
    xIn = np.arange(0, res.shape[2])
    xOut = np.linspace(0, res.shape[2]-1, res.shape[2]*8)
    if Upsample:
        resUpsampled = np.zeros((res.shape[0], res.shape[1], res.shape[2]*8))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                resUpsampled[i,j,:] = interpolate.interp1d(xIn, res[i,j,:], kind='cubic')(xOut)
    else:
        resUpsampled = res
    fileNms = ['../../data/BloodFlow/classData.npy', '../data/BloodFlow/classData.npy']
    # check which path exists
    for fileNm in fileNms:
        if os.path.exists(fileNm):
            break
    with open(fileNm, 'rb') as fin:
        res = np.load(fin, allow_pickle=True)
    df = pd.DataFrame(res)
    #Column 0 is the SubjectType, column 1 is the ScanType, column 2 is 0 if site is SiteY, 1 if site is SiteX
    columns = ['SubjectType', 'ScanType', 'source', 'age', 'race_score', 'pre_op_sbp', 'pre_op_dbp']

    df.columns = columns

    #Round age to nearest integer and divide by 100

    #Turn subject type into a binary classification problem
    df['SubjectType'] = df['SubjectType'].replace({'IHC': 0, 'LVO': 1})

    df = df.drop(['ScanType', 'source'], axis=1)

    return resUpsampled, df.to_numpy().astype(np.float32)

def augment_channels_data_by_random_sampling(Xin, Yin, window_size=3000, numDataAugmentation=200, numChannelPairs=8, shuffle=True):
    #Xin has data in the order subject, channel, time
    #we want to randomly sample 3000 time points from each subject and channel
    #First, randomly sample 3000 time points
    timePoints = np.random.randint(0, Xin.shape[2]-window_size, Xin.shape[0]*numDataAugmentation*numChannelPairs)
    X = np.zeros((Xin.shape[0]*numDataAugmentation, Xin.shape[1], window_size))
    Y = np.zeros((Yin.shape[0]*numDataAugmentation, Yin.shape[1]))
    for i in range(Xin.shape[0]):
        for j in range(numDataAugmentation):
            for k in range(numChannelPairs):
                start = timePoints[i*numDataAugmentation*numChannelPairs+j*numChannelPairs+k]
                # Add code to handle error when start is out of range
                X[i*numDataAugmentation+j, 2*k, :]   = Xin[i, 2*k, start:start+window_size]
                X[i*numDataAugmentation+j, 2*k+1, :] = Xin[i, 2*k+1, start:start+window_size]
            Y[i*numDataAugmentation+j,:] = Yin[i,:]
    X = normalize_bloodflow_data(X)

    metaData = Y[:,1:]
    metaDataAppend = np.zeros((Y.shape[0],X.shape[1],1))
    metaDataAppend[:,:metaData.shape[1],0] = metaData
    X = np.concatenate((X, metaDataAppend), axis=2)
    Y = Y[:,0]

    #Shuffle the data
    if shuffle:
        shuffle_pid = np.random.permutation(Y.shape[0])
        X = X[shuffle_pid]
        Y = Y[shuffle_pid]
    return X, Y

def read_data_bloodFlow_for_overfitting(window_size=3000):
    #Read data
    resUpsampled, metaData = read_blood_flow_data()
    subjectType = metaData[:,0]

    # This code generates training data for the algorithm. It randomly samples 3000 time points for each subject,
    # and creates 8 pairs of channels for each time point. Each pair of channels is a different resampled version
    # of the same channel pair. The labels are the same for each resampled version of the data.
    X, Y = augment_channels_data_by_random_sampling(resUpsampled, subjectType, window_size=window_size)
    
    #Split into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def plot_roc(aucs, tprs, fprs, preds, predprobs, truths):
    #Plot ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, predprob in enumerate(predprobs):
        truth = truths[fold]
        viz = RocCurveDisplay.from_predictions(
            truth,
            predprob,
            name=f"ROC fold {fold+1}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()
    return mean_auc, (mean_fpr, mean_tpr)


def read_data_bloodFlow_for_train_test(upSample=True, window_size=3000, seed=4545, single_test=False):
    #Read data
    resUpsampled, metaData = read_blood_flow_data(upSample)
    subjectType = metaData[:,0]

    numDataAugmentation = 200
    pid_test = []

    #Split into train and test of 5 fold cross validation
    X_train, X_test, Y_train, Y_test = [], [], [], []
    #set np seed for repeatability
    np.random.seed(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(cv.split(resUpsampled, subjectType)):
        X_train_fold, X_test_fold = resUpsampled[train_index], resUpsampled[test_index]
        Y_train_fold, Y_test_fold = metaData[train_index], metaData[test_index]
        
        pid_test.append(test_index)

        #X_train, X_test, Y_train, Y_test = train_test_split(resUpsampled, subjectType, test_size=0.2, random_state=42)
        if window_size:
            X_train_fold, Y_train_fold = augment_channels_data_by_random_sampling(X_train_fold, Y_train_fold,
                                                                                window_size=window_size,
                                                                                numDataAugmentation=numDataAugmentation)
            X_test_fold, Y_test_fold   = augment_channels_data_by_random_sampling(X_test_fold, Y_test_fold,
                                                                                window_size=window_size,
                                                                                numDataAugmentation=numDataAugmentation)
        #Shuffle the data
        shuffle_pid = np.random.permutation(Y_train_fold.shape[0])
        X_train_fold = X_train_fold[shuffle_pid]
        Y_train_fold = Y_train_fold[shuffle_pid]
        
        X_train.append(X_train_fold)
        X_test.append(X_test_fold)
        Y_train.append(Y_train_fold)
        Y_test.append(Y_test_fold)

    return X_train, X_test, Y_train, Y_test, pid_test

if __name__ == "__main__":
    #X_train, X_test, Y_train, Y_test = read_data_bloodFlow_for_overfitting()
    #print((X_train.shape, Y_train.shape),(X_test.shape, Y_test.shape))
    X_train, X_test, Y_train, Y_test, pid_test = read_data_bloodFlow_for_train_test()
    print((X_train[0].shape, Y_train[0].shape),(X_test[0].shape, Y_test[0].shape))
    #print((X_train.shape, Y_train.shape))