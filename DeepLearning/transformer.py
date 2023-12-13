"""
transformer encoder for 1-d signal data, pytorch version
 
Kedar Grama, May 2023
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch, torchvision, argparse, platform
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from util import read_data_bloodFlow_for_train_test, augment_channels_data_by_random_sampling, read_data_generated

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)
    
class PatchExtractor(nn.Module):
    def __init__(self, patch_size=10):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        batch_size, channels, length = input_data.size()
        assert length % self.patch_size == 0, \
            f"Input signal length ({length}) must be divisible by patch size ({self.patch_size})"

        num_patches = length // self.patch_size

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
            permute(0, 2, 1, 3). \
            contiguous(). \
            view(batch_size, num_patches, -1)

        # Expected shape of a patch on default settings is (32, 10, 576)
        # Original - Expected shape of a patch on default settings is (4, 196, 768)

        return patches


class InputEmbedding(nn.Module):

    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        if platform.system() == "Darwin":
            device_str = "mps"
            self.device = torch.device(device_str if torch.has_mps else "cpu")
        elif args.no_cuda:
            device_str = "cpu"
            self.device = torch.device(device_str)
        else:
            device_str = "cuda"
            self.device = torch.device(device_str if torch.has_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.n_channels

        # Linear projection
        self.LinearProjection = nn.Linear(self.input_size, self.latent_size).to(self.device)
        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(input_data)

        linear_projection = self.LinearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embed = self.pos_embedding[:, :n + 1, :]
        linear_projection += pos_embed

        return linear_projection


class EncoderBlock(nn.Module):

    def __init__(self, args):
        super(EncoderBlock, self).__init__()

        self.latent_size = args.latent_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.norm = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, self.num_heads, dropout=self.dropout)
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, emb_patches):
        first_norm = self.norm(emb_patches)
        attention_out = self.attention(first_norm, first_norm, first_norm)[0]
        first_added = attention_out + emb_patches
        second_norm = self.norm(first_added)
        mlp_out = self.enc_MLP(second_norm)
        output = mlp_out + first_added

        return output


class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()

        self.num_encoders = args.num_encoders
        self.latent_size = args.latent_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.embedding = InputEmbedding(args)
        # Encoder Stack
        self.encoders = nn.ModuleList([EncoderBlock(args) for _ in range(self.num_encoders)])
        self.MLPHead = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes),
        )

    def forward(self, test_input):
        enc_output = self.embedding(test_input)
        for enc_layer in self.encoders:
            enc_output = enc_layer(enc_output)

        class_token_embed = enc_output[:, 0]
        return self.MLPHead(class_token_embed)
    
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


        return total_loss / len(self.val_dataloader)

    def train(self):
        best_valid_loss = np.inf
        best_train_loss = np.inf
        for i in range(self.epoch):
            train_loss = self.train_fn(i)
            val_loss = self.eval_fn(i)

            if val_loss < best_valid_loss:
                torch.save(self.model.state_dict(), "best-weights.pt")
                print("Saved Best Weights")
                best_valid_loss = val_loss
                best_train_loss = train_loss
        print(f"Training Loss : {best_train_loss}")
        print(f"Valid Loss : {best_valid_loss}")

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer in PyTorch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-synthetic', type=bool, default=True,
                        help='use synthetic data for quick debugging (default: True)')
    parser.add_argument('--patch-size', type=int, default=10,
                        help='patch size for images (default : 10)')
    parser.add_argument('--latent-size', type=int, default=160,
                        help='latent size (default : 160)')
    parser.add_argument('--n-channels', type=int, default=16,
                        help='number of channels in images (default : 16 for RGB)')
    parser.add_argument('--num-heads', type=int, default=16,
                        help='(default : 16)')
    parser.add_argument('--num-encoders', type=int, default=24,
                        help='number of encoders (default : 24)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout value (default : 0.1)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='image size to be reshaped to (default : 224')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='number of classes in dataset (default : 10 for CIFAR10)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs (default : 50)')
    parser.add_argument('--lr', type=int, default=1e-4,
                        help='base learning rate (default : 0.01)')
    parser.add_argument('--weight-decay', type=int, default=1e-4,
                        help='weight decay value (default : 0.03)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size (default : 4)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()

    if platform.system() == "Darwin":
        device_str = "mps"
        device = torch.device(device_str if torch.has_mps else "cpu")
    else:
        device_str = "cuda"
        device = torch.device(device_str if torch.has_cuda else "cpu")
    print("device:", device)

    if args.use_synthetic:
        data, label = read_data_generated(n_samples=1600, n_length=360, n_channel=16, n_classes=2)
        p = np.random.permutation(len(data))
        data = data[p]
        label = label[p]
        X_train, X_test, Y_train, Y_test = data[:2560], data[2560:], label[:2560], label[2560:]
    else:
        X_train_folds, X_test_folds, Y_train_folds, Y_test_folds, pid_test = read_data_bloodFlow_for_train_test(False, 0)
        X_train = X_train_folds[0]; X_test = X_test_folds[0]
        Y_train = Y_train_folds[0]; Y_test = Y_test_folds[0]

        X_train,Y_train = augment_channels_data_by_random_sampling(X_train, Y_train, window_size=360, numDataAugmentation=192)
        X_test, Y_test  = augment_channels_data_by_random_sampling(X_test, Y_test, window_size=360, numDataAugmentation=192)

    class_sample_count = np.array([len(np.where(Y_train==t)[0]) for t in np.unique(Y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in Y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    dataset_train = MyDataset(X_train, Y_train)
    dataset_test  = MyDataset(X_test, Y_test)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    valid_loader  = DataLoader(dataset_test, batch_size=args.batch_size, drop_last=False)

    model = ViT(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    TrainEval(args, model, train_loader, valid_loader, optimizer, criterion, device).train()

if __name__ == "__main__":
    main()