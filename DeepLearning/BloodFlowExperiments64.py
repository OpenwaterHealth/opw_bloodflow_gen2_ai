#%%
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, RocCurveDisplay

from util import read_data_bloodFlow_for_overfitting, read_data_bloodFlow_for_train_test, augment_channels_data_by_random_sampling
from resnet1d import ResNet1D, MyDataset

import torch, platform, datetime, glob
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
from multiprocessing import Pool

#%%
def run_fold(X_train, X_test, Y_train, Y_test, fold):
    is_debug = False
    batch_size = 32
    X_train,Y_train = augment_channels_data_by_random_sampling(X_train, Y_train)
    X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test)
    print(X_train.shape, Y_train.shape)
    #Weight sampling by the class frequency
    class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in Y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

    if is_debug:
        writer = SummaryWriter('bloodflow/debug')
    else:
        summaryPath = 'bloodflow/layer98sm_'+str(fold+1)+'fold_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(summaryPath)

    # make model
    if platform.system() == "Darwin":
        device_str = "mps"
        device = torch.device(device_str if torch.has_mps else "cpu")
    else:
        device_str = "cuda"
        device = torch.device(device_str if torch.has_cuda else "cpu")
    print("device:", device)
    kernel_size = 16
    stride = 2
    n_block = 48
    downsample_gap = 6
    increasefilter_gap = 12
    model = ResNet1D(
        in_channels=16, 
        base_filters=64, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=kernel_size, 
        stride=stride, 
        groups=32, 
        n_block=n_block, 
        n_classes=2, 
        downsample_gap=downsample_gap, 
        increasefilter_gap=increasefilter_gap, 
        use_do=True)
    model.to(device)

    summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)
    # exit()


    # train and test
    model.verbose = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 50
    step = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('Loss/train', loss.item(), step)

            if is_debug:
                break
        
        scheduler.step(_)
                    
        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        all_true = Y_test
        print("epoch:", _, "acc:", np.mean(all_pred == all_true))
        print(classification_report(all_true, all_pred))
        print(confusion_matrix(all_true, all_pred))
        writer.add_scalar('Accuracy/test', np.mean(all_pred == all_true), step)
    torch.save(model.state_dict(), summaryPath+'/model.pt')

def run_testing_and_get_roc(X_test_folds, Y_test_folds, pid_test):
    #Set platform
    if platform.system() == "Darwin":
        device_str = "mps"
        device = torch.device(device_str if torch.has_mps else "cpu")
    else:
        device_str = "cuda"
        device = torch.device(device_str if torch.has_cuda else "cpu")
    print("device:", device)

    foldModelPaths = 'bloodflow/layer98lg_*fold_*'
    foldModelPaths = glob.glob(foldModelPaths)
    foldModelPaths.sort()
    #Get the fold number from the model path
    foldNums = []
    for foldModelPath in foldModelPaths:
        foldNums.append(int(foldModelPath.split('_')[1][0]))
    foldNums = np.array(foldNums)
    foldIDs  = np.unique(foldNums)

    #Sort the fold number and model path
    foldModelPaths = np.array(foldModelPaths)
    foldModelPaths = foldModelPaths[np.argsort(foldNums)]
    foldNums = np.sort(foldNums)

    #Load the model for each fold and get ROC curve and AUC for each fold
    aucs = [0.0]*len(foldIDs)
    tprs = []
    fprs = []
    preds = []
    predprobs = []
    truths = []
    for i, foldModelPath in enumerate(foldModelPaths):
        foldNumCur = foldNums[i]-1
        state_dict = torch.load(foldModelPath+'/model.pt')
        model = ResNet1D(in_channels=16, 
        base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
        kernel_size=16, 
        stride=2, 
        groups=32, 
        n_block=48, 
        n_classes=2, 
        downsample_gap=6, 
        increasefilter_gap=12,
        use_do=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.verbose = False
        model.to(device)

        X_test = X_test_folds[foldNumCur]
        Y_test = Y_test_folds[foldNumCur]
        pid_test_cur = pid_test[foldNumCur]

        dataset_test = MyDataset(X_test, Y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=32, drop_last=False)

        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_test):
                input_x, _ = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        #Get the probability of the positive class

        all_true = Y_test

        #Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(all_true, all_pred_prob[:,1])
        roc_auc = auc(fpr, tpr)
        if aucs[foldNumCur]==0.0:
            tprs.append(tpr)
            fprs.append(fpr)
            preds.append(all_pred)
            predprobs.append(all_pred_prob[:,1])
            truths.append(all_true)
        elif roc_auc < aucs[foldNumCur]:
            tprs[foldNumCur] = tpr
            fprs[foldNumCur] = fpr
            preds[foldNumCur] = all_pred
            predprobs[foldNumCur] = all_pred_prob[:,1]
            truths[foldNumCur] = all_true
        aucs[foldNumCur] = roc_auc
    return aucs, tprs, fprs, preds, predprobs, truths

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

#%%
if __name__=="__main__":
    # X_train, X_test, Y_train, Y_test = read_data_bloodFlow_for_overfitting()
    # run_fold(X_train, X_test, Y_train, Y_test, 0)
    X_train_folds, X_test_folds, Y_train_folds, Y_test_folds, pid_test = read_data_bloodFlow_for_train_test(window_size=0)
    '''for _ in range(4):
        pool = Pool(processes=5)
        for i in range(5):
            pool.apply_async(run_fold, args=(X_train_folds[i], X_test_folds[i], Y_train_folds[i], Y_test_folds[i], i))
        pool.close()
        pool.join()'''
    #for _ in range(2):
    for i in range(5):
        run_fold(X_train_folds[i], X_test_folds[i], Y_train_folds[i], Y_test_folds[i], i)
    #aucs, tprs, fprs, preds, predprobs, truths = run_testing_and_get_roc(X_test_folds, Y_test_folds, pid_test)
    #mean_auc, meanCurvePts = plot_roc(aucs, tprs, fprs, preds, predprobs, truths)

    #np.save('DL_ROC.npy', meanCurvePts, allow_pickle=True, fix_imports=True)

# %%
