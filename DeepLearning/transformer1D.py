"""
transformer encoder for 1-d blood flow signal data, pytorch version
Kedar Grama, May 2023

Adapted from two sources:
Natarajan, Annamalai, et al. "A wide and deep transformer neural network for
12-lead ECG classification." 2020 Computing in Cardiology. IEEE, 2020.

And Self-Attention pooling ideas from the following:
https://github.com/bh1995/AF-classification
"""

import numpy as np, pandas as pd, glob, os
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, roc_curve, auc

import torch, torchvision, argparse, platform, math, wandb, datetime, warnings
import torch.nn as nn
from torch import Tensor
import torch.optim as optim, glob
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from tensorboardX import SummaryWriter
from torchsummary import summary
from util import read_data_bloodFlow_for_train_test, augment_channels_data_by_random_sampling, read_data_generated, plot_roc

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1),dim=0).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, nlayers, n_conv_layers=2, n_class=2, dropout=0.5, dropout_other=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.gelu = torch.nn.GELU()
        self.pos_encoder = PositionalEncoding(64, dropout)
        self.pos_encoder2 = PositionalEncoding(6, dropout)
        self.self_att_pool = SelfAttentionPooling(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.flatten_layer = torch.nn.Flatten()
        # Define linear output layers
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                       nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        self.decoder2 = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                      #  nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        # Linear output layer after concat.
        self.fc_out1 = torch.nn.Linear(64, 64)
        self.fc_out2 = torch.nn.Linear(64, 2) # if two classes problem is binary  
        # self.init_weights()
        # Transformer Conv. layers
        self.conv1 = torch.nn.Conv1d(in_channels=16, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(d_model)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.avg_maxpool = nn.AdaptiveAvgPool2d((64, 64))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src): 
        src = self.gelu(self.conv1(src))
        src = self.gelu(self.conv2(src))
        for i in range(self.n_conv_layers):
          src = self.gelu(self.conv(src))

        src = src.permute(2,0,1) # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        src = self.pos_encoder(src)   
        output = self.transformer_encoder(src) # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        output = output.permute(1,0,2)
        output = self.self_att_pool(output)
        logits = self.decoder(output) # output: [batch, n_class]
        xc = self.flatten_layer(logits)
        xc = self.fc_out2(self.dropout(self.gelu(self.fc_out1(xc))))
        xc = F.softmax(xc, dim=1)
        return xc

    
class TrainEval:

    def __init__(self, args, model, train_dataloader, val_dataloader, optimizer, criterion, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch = args.epochs
        self.device = device
        self.args = args

    def train_fn(self, current_epoch):
        self.model.train()
        total_loss = 0.0
        tk = tqdm(self.train_dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epoch + 1) + "/" + str(self.epoch), leave=False)

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break
            else:
                self.writer.add_scalar('Loss/train', loss.item(), current_epoch * len(self.train_dataloader) + t)

        return total_loss / len(self.train_dataloader)

    def eval_fn(self, current_epoch):
        self.model.eval()
        total_loss = 0.0
        tk = tqdm(self.val_dataloader, desc="EPOCH" + "[VALID]" + str(current_epoch + 1) + "/" + str(self.epoch))

        all_labels = []
        all_preds = []

        for t, data in enumerate(tk):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
            loss = self.criterion(logits, labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})
            if self.args.dry_run:
                break

        print(classification_report(all_labels, all_preds))
        print(confusion_matrix(all_labels, all_preds))

        self.writer.add_scalar('Accuracy/test', accuracy_score(all_labels, all_preds), current_epoch)
        return total_loss / len(self.val_dataloader)

    def train(self, SAVE_PATH = ''):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        self.writer = SummaryWriter(self.args.logDir)
        
        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            val_loss = self.eval_fn(i)

            if val_loss < best_valid_loss:
                torch.save(self.model.state_dict(), self.args.logDir+"best-weights.pt")
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")

def GetDevice(printDevice=False):
    #Set platform
    if platform.system() == "Darwin":
        device_str = "mps"
        device = torch.device(device_str if torch.has_mps else "cpu")
    else:
        device_str = "cuda"
        device = torch.device(device_str if torch.has_cuda else "cpu")
    if printDevice:
        print("device:", device)
    return device, device_str


def run_testing_and_get_roc(X_test_folds, Y_test_folds, config, modelPath):
    device, device_str = GetDevice()
    foldModelPaths = modelPath + 'transformer1D_'+ '*best-weights.pt'
    foldModelPaths = glob.glob(foldModelPaths)
    foldModelPaths.sort()
    #Get the fold number from the model path
    foldNums = []
    for foldModelPath in foldModelPaths:
        foldNums.append(int(foldModelPath.split('_')[2][0]))
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
        state_dict = torch.load(foldModelPath, map_location=device_str)
        model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        model.verbose = False
        model.to(device)
        
        X_test = X_test_folds[foldNumCur]
        Y_test = Y_test_folds[foldNumCur]
        X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test, window_size=360, numDataAugmentation=192, shuffle=False)

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

def run_testing_and_get_classification(X_train_folds, Y_train_folds, X_test_folds, Y_test_folds, config, modelPath):
    #Set platform
    if platform.system() == "Darwin":
        device_str = "mps"
        device = torch.device(device_str if torch.has_mps else "cpu")
    else:
        device_str = "cuda"
        device = torch.device(device_str if torch.has_cuda else "cpu")
    print("device:", device)

    foldModelPaths = modelPath+'transformer1D_4T_*best-weights.pt'
    foldModelPaths = glob.glob(foldModelPaths)
    foldModelPaths.sort()
    #Get the fold number from the model path
    foldNums = []
    for foldModelPath in foldModelPaths:
        foldNums.append(int(foldModelPath.split('_')[4]))
    foldNums = np.array(foldNums)
    foldIDs  = np.unique(foldNums)

    #Sort the fold number and model path
    foldModelPaths = np.array(foldModelPaths)
    foldModelPaths = foldModelPaths[np.argsort(foldNums)]
    foldNums = np.sort(foldNums)
    all_pred_prob = []
    for i, foldModelPath in enumerate(foldModelPaths):
        foldNumCur = foldNums[i]-1
        state_dict = torch.load(foldModelPath, map_location=device_str)
        model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        X_train = X_train_folds[foldNumCur]; X_test = X_test_folds[foldNumCur]
        Y_train = Y_train_folds[foldNumCur]; Y_test = Y_test_folds[foldNumCur]
        np.random.seed(config.seed) #Repeating seed for repeatability
        X_train,Y_train = augment_channels_data_by_random_sampling(X_train, Y_train, window_size=360, numDataAugmentation=192)
        X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test, window_size=360, numDataAugmentation=1, shuffle=False)

        dataset_test = MyDataset(X_test, Y_test)
        dataloader_test = DataLoader(dataset_test, batch_size=32, drop_last=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_test):
                input_x, _ = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                all_pred_prob.append(pred.cpu().data.numpy())
    return all_pred_prob


def run_testing_and_get_roc(X_test_folds, Y_test_folds, config):
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
        model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        X_test = X_test_folds[foldNumCur]
        Y_test = Y_test_folds[foldNumCur]

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

def GetConfig():
    # WandB - Initialize run
    wandb.init(entity="Openwater", project="BloodflowLVO", mode="disabled")
    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.batch_size = 32          # input batch size for training (default: 64)
    config.epochs = 50             # number of epochs to train (default: 10)
    config.log_interval = 1     # how many batches to wait before logging training status
    config.emsize = 64 # embedding dimension == d_model
    config.dim_feedforward = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
    config.nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    config.nhead = 4 # the number of heads in the multiheadattention models
    config.n_conv_layers = 2 # number of convolutional layers (before transformer encoder)
    config.dropout = 0.25 # the dropout value
    config.dropout_other = 0.1 # dropout value for feedforward output layers
    config.n_class = 2
    config.use_synthetic = False
    config.dry_run = False
    return config


def main(): #Train/Test
    roc_run = False
    roc_curve_from_best = False
    write_testing_results = True
    #weightDecays = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5]
    learningRates = [5e-4, 5e-4, 1e-4, 1e-4]#[5e-4, 1e-4, 5e-5, 1e-5]
    ind = -1
    if write_testing_results:
        seed = 140446 #Get this from the filename of the best model
    else:
        seed = np.random.randint(0,1000000)
    for lr in learningRates:
        ind += 1
        config = GetConfig()
        config.lr = lr
        config.seed = seed
        
        if platform.system() == "Darwin":
            device_str = "mps"
            device = torch.device(device_str if torch.has_mps else "cpu")
        else:
            device_str = "cuda"
            device = torch.device(device_str if torch.has_cuda else "cpu")
        print("device:", device)

        if config.use_synthetic:
            data, label = read_data_generated(n_samples=1600, n_length=360, n_channel=16, n_classes=2)
            p = np.random.permutation(len(data))
            data = data[p]
            label = label[p]
            X_train, X_test, Y_train, Y_test = data[:2560], data[2560:], label[:2560], label[2560:]
        else:
            X_train_folds, X_test_folds, Y_train_folds, Y_test_folds, pid_test = read_data_bloodFlow_for_train_test(False, 0, config.seed)
        
        if write_testing_results:
            bestModelPaths = ['bloodflow/', 'Resnet1D/bloodflow/']
            modelNamePattern = '*best-weights.pt'
            numTest = 0
            for bestModelPath in bestModelPaths:
                if glob.glob(bestModelPath+modelNamePattern):
                    predProbs = run_testing_and_get_classification(X_train_folds, Y_train_folds, X_test_folds, Y_test_folds, config, bestModelPath)
                    #Get classification from predProbs
                    pred = []
                    for predProb in predProbs:
                        pred.append(np.argmax(predProb, axis=1))
                        numTest += len(predProb)
            #Read the subject IDs
            fileNms = ['../../data/BloodFlow/subjectID.npy', '../data/BloodFlow/subjectID.npy']
            # check which path exists
            for fileNm in fileNms:
                if os.path.exists(fileNm):
                    break
            subIDs = np.load(fileNm,allow_pickle=True)

            #Create a dataframe with subject ID, predProbs and pred
            allPred = np.zeros((numTest, 3))
            for i,pids in enumerate(pid_test):
                allPred[pids,0] = i+1 #Fold number
                allPred[pids,1:3] = predProbs[i]

            ##Create data frame with subject ID, allPred and write to csv
            df = pd.DataFrame(data=allPred, index=subIDs, columns=['fold', 'predProb0', 'predProb1'])
            df.to_csv('DL_predProb.csv')
            return

        if roc_curve_from_best:
            bestModelPaths = ['bloodflow/bestModels/', 'Resnet1D/bloodflow/bestModels/']
            modelNamePattern = '*best-weights.pt'
            for bestModelPath in bestModelPaths:
                if glob.glob(bestModelPath+modelNamePattern):
                    aucs, tprs, fprs, preds, predprobs, truths = run_testing_and_get_roc(X_test_folds, Y_test_folds, config, bestModelPath)
                    mean_auc, meanCurvePts = plot_roc(aucs, tprs, fprs, preds, predprobs, truths)
                    np.save('DL_ROC_Trf_optical.npy', meanCurvePts, allow_pickle=True, fix_imports=True)
                    return

        for fold in range(len(X_train_folds)):
            X_train = X_train_folds[fold]; X_test = X_test_folds[fold]
            Y_train = Y_train_folds[fold]; Y_test = Y_test_folds[fold]

            np.random.seed(config.seed) #Repeating seed for repeatability
            X_train,Y_train = augment_channels_data_by_random_sampling(X_train, Y_train, window_size=360, numDataAugmentation=192)
            X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test, window_size=360, numDataAugmentation=1, shuffle=False)

            if not roc_run:
                newLogDir = 'bloodflow/transformer1D_4T_'+ str(config.seed) + '_' + str(ind) + '_' + str(fold+1) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                wandb.config.update({'logDir': newLogDir},allow_val_change=True)

                class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[int(t)] for t in Y_train])
                samples_weight = torch.from_numpy(samples_weight)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
            
                dataset_train = MyDataset(X_train[:,:,:-1], Y_train) #Drop metadata for now
                train_loader = DataLoader(dataset_train, batch_size=config.batch_size, sampler=sampler, drop_last=True)
            dataset_test  = MyDataset(X_test[:,:,:-1], Y_test)
            valid_loader  = DataLoader(dataset_test, batch_size=config.batch_size, drop_last=False)

            model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)

            #summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            criterion = nn.CrossEntropyLoss()

            TrainEval(config, model, train_loader, valid_loader, optimizer, criterion, device).train()

def GetAUC(XTest,YTest,model):
    device, device_str = GetDevice()
    dataset_test = MyDataset(XTest, YTest)
    dataloader_test = DataLoader(dataset_test, batch_size=32, drop_last=False)
    all_pred_prob = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader_test):
            input_x, _ = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            #pred = F.softmax(logits, dim=1)
            all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    #Get the probability of the positive class
    all_true = YTest
    #Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(all_true, all_pred_prob[:,1])
    roc_auc = auc(fpr, tpr)
    return roc_auc, all_pred_prob

def FindTrainedNetworksAndGetAucOfROC():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #Find all trained networks in the directory
    bestModelPaths = ['bloodflow/', 'Resnet1D/bloodflow/']
    modelNamePattern = 'transformer1D_4T*best-weights.pt'
    #Read the subject IDs
    fileNms = ['../../data/BloodFlow/subjectID.npy', '../data/BloodFlow/subjectID.npy']
    # check which path exists
    for fileNm in fileNms:
        if os.path.exists(fileNm):
            break
    subIDsOG = np.load(fileNm,allow_pickle=True)
    for bestModelPath in bestModelPaths:
        if glob.glob(bestModelPath+modelNamePattern):
            modelPaths = glob.glob(bestModelPath+modelNamePattern)
            modelPaths.sort()
            seeds = [ int(modelPath.split('_')[2]) for modelPath in modelPaths]
            seeds = np.unique(seeds)
            for seed in seeds:
                config = GetConfig()
                config.lr = 1e-5 #We are not training, so this doesn't matter
                config.seed = seed
                device, device_str = GetDevice()
                X_train_folds, X_test_folds, Y_train_folds, Y_test_folds, pid_test = read_data_bloodFlow_for_train_test(False, 0, config.seed)
                #Change subIDs order to match the order of the predictions
                subIDs = subIDsOG[pid_test]
                bestModels = []
                bestAUCs = []
                Y_preds = []
                for fold in range(5):
                    X_train = X_train_folds[fold]; X_test = X_test_folds[fold]
                    Y_train = Y_train_folds[fold]; Y_test = Y_test_folds[fold]
                    np.random.seed(config.seed) #Repeating seed for repeatability
                    X_train,Y_train = augment_channels_data_by_random_sampling(X_train, Y_train, window_size=360, numDataAugmentation=192)
                    X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test, window_size=360, numDataAugmentation=1, shuffle=False)
                    modelPathPattern = bestModelPath + 'transformer1D_4T_'+str(seed)+'_?_'+str(fold+1)+'*best-weights.pt'
                    modelPaths = glob.glob(modelPathPattern)
                    modelPaths.sort()
                    bestAUC = 0.0
                    bestInd = -1
                    for ind,modelPath in enumerate(modelPaths):
                        state_dict = torch.load(modelPath, map_location=device_str)
                        model = TransformerModel(config.emsize, config.nhead, config.dim_feedforward, config.nlayers, config.n_conv_layers, config.n_class, config.dropout, config.dropout_other).to(device)
                        model.load_state_dict(state_dict)
                        model.eval()
                        curAuc,predProbs = GetAUC(X_test, Y_test, model)
                        if curAuc > bestAUC:
                            bestInd = ind
                            bestAUC = curAuc
                            if len(bestModels) > fold:
                                bestModels[fold] = model
                                bestAUCs[fold] = curAuc
                                Y_preds[fold] = predProbs
                            else:
                                bestModels.append(model)
                                bestAUCs.append(curAuc)
                                Y_preds.append(predProbs)
                    print(f'Best AUC for seed {seed} fold {fold} is {bestAUCs[fold]} {bestInd} {modelPaths[bestInd]}')
                Y_test_folds = np.array([item for sublist in Y_test_folds for item in sublist])
                Y_preds = np.array([item for sublist in Y_preds for item in sublist])
                fpr, tpr, _ = roc_curve(Y_test_folds[:,0], Y_preds[:,1])
                roc_auc = '%.2f' % auc(fpr, tpr)
                formattedAUC = [ '%.2f' % elem for elem in bestAUCs ]
                bestAUCStr = '%.2f' % np.mean(bestAUCs)
                print(f'Best AUCs for seed {seed} is {formattedAUC} with mean {bestAUCStr} {roc_auc}')
                #Create dataframe with subject ID, Fold Number, Y_preds and write to csv
                allPred = np.zeros((len(Y_test_folds), 3))
                ind = 0
                for i,pids in enumerate(pid_test):
                    allPred[ind:ind+len(pids),0] = i+1
                    ind += len(pids)
                allPred[:,1:3] = Y_preds
                ##Create data frame with subject ID, allPred and write to csv
                df = pd.DataFrame(data=allPred, index=subIDs.flatten(), columns=['fold', 'predProb0', 'predProb1'])
                df.to_csv(f'DL_predProb_{seed}.csv')
    if not len(seeds):
        assert False, 'No trained networks found in the directory'
    return

if __name__ == "__main__":
    test = False
    train = False
    plotRocs = True
    if train:
        while 1: #Repeat training with different seeds
            main()
    elif test:
        main()
    elif plotRocs:
        FindTrainedNetworksAndGetAucOfROC()
