import os
import math
import torch
import sklearn
import torch.nn as nn
import numpy as np
from sklearn import metrics
import utilities as utils
from utilities import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def path_to_DataLoader(path, batch_size=1, num_workers=1, shuffle=True, index=False, classification=True, double_precision=False, truncate=None):
    
    data_set = RrIntervalDataset(path, index, truncate=truncate)
    
    if classification:
        return DataLoader(data_set,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=__batch2tensor__,
                      shuffle=shuffle), data_set
    else:
        if double_precision:
            return DataLoader(data_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=__batch2tensor4regression6double__,
                              shuffle=shuffle), data_set
        else:
            return DataLoader(data_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=__batch2tensor4regression6single__,
                              shuffle=shuffle), data_set
    

def __batch2tensor4regression6double__(batch):
    """
    Takes a batch and transforms it in such a way that it can directly be fed to the network.
    @param batch: List of x and y labels.
    @return: Two tensors, one for x and one for y.
    """
    
    x, y = [None] * len(batch), [None] * len(batch)
    for i, row in enumerate(batch):
        y[i] = int(row[0])
        x[i] = row[1:]

    return torch.DoubleTensor(x), torch.DoubleTensor(y)

def __batch2tensor4regression6single__(batch):
    """
    Takes a batch and transforms it in such a way that it can directly be fed to the network.
    @param batch: List of x and y labels.
    @return: Two tensors, one for x and one for y.
    """
    
    x, y = [None] * len(batch), [None] * len(batch)
    for i, row in enumerate(batch):
        y[i] = int(row[0])
        x[i] = row[1:]

    return torch.FloatTensor(x), torch.FloatTensor(y)


def __batch2tensor__(batch):
    """
    Takes a batch and transforms it in such a way that it can directly be fed to the network.
    @param batch: List of x and y labels.
    @return: Two tensors, one for x and one for y.
    """
    
    x, y = [None] * len(batch), [None] * len(batch)
    for i, row in enumerate(batch):
        y[i] = int(row[0])
        x[i] = row[1:]

    return torch.FloatTensor(x), torch.LongTensor(y)


class RrIntervalDataset(Dataset):
    def __init__(self, path, header=True, index=False, truncate=None):
        
        self.samples = []
        
        skip = 1
        if index:
            skip = 2
        
        with open(path, 'r') as file:
            if header:
                next(file)
            for line in file:
                if truncate is None:
                    self.samples.append(list(map(lambda x: float(x), line.split(",")[skip:])))
                else:
                    self.samples.append(list(map(lambda x: float(x), line.split(",")[skip:(skip+truncate+1)])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

class DeepSleepNet(nn.Module):
    def __init__(self, no_classes, data_length, batch_size, dropout=0.5, lstm_dropout=0.5, lstm_layers=2, lstm_hidden=256, channels=128, stride=1, padding=0,
                 classification=True, regression_mlp_hidden=128, side_arm_mlp=128, head1_kernel_size=8, head2_kernel_size=4, complete=True):
        
        super().__init__()
        
        # General Information
        self.data_length = data_length
        self.batch_size = batch_size
        self.classification = classification
        self.regression_mlp_hidden = regression_mlp_hidden
        self.side_arm_mlp = side_arm_mlp
        
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.channels = channels
        self.stride = stride
        self.padding = padding
        
        self.head1_kernel_size = head1_kernel_size
        self.head2_kernel_size = head2_kernel_size
        
        if complete:
            self.head1_kernel_size *= 1
            self.head2_kernel_size *= 1
        
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.lstm_hidden_states = None
        
        # Dropout
        # Maybe for CNN, check literature
        self.dropout = nn.Dropout(p=dropout)
        
        # Head 1
        self.conv1dh1l1 = nn.Conv1d(1, channels, self.head1_kernel_size, stride=stride)
        self.conv1dh1l2 = nn.Conv1d(channels, channels, self.head1_kernel_size, stride=stride)
        self.conv1dh1l3 = nn.Conv1d(channels, channels, self.head1_kernel_size, stride=stride)

        # Head 2
        self.conv1dh2l1 = nn.Conv1d(1, channels, self.head2_kernel_size, stride=stride)
        self.conv1dh2l2 = nn.Conv1d(channels, channels, self.head2_kernel_size, stride=stride)
        self.conv1dh2l3 = nn.Conv1d(channels, channels, self.head2_kernel_size, stride=stride)
        
        # Pooling
        ## MaxPool
        #self.maxpool = nn.AdaptiveMaxPool1d(avg_sequence_length//10)
        #self.maxpoolConvH1 = nn.MaxPool1d(self.head1_kernel_size)
        #self.maxpoolConvH2 = nn.MaxPool1d(self.head2_kernel_size)
        self.avgpoolConvH1 = nn.AvgPool1d(self.head1_kernel_size)
        self.avgpoolConvH2 = nn.AvgPool1d(self.head2_kernel_size)
        
        # LSTM
        # Input diemension will be hardcoded, very hard the obtain that through calculations (but possible...)
        self.lstm = nn.LSTM(self.get_lstm_input_shape(), lstm_hidden, lstm_layers,
                            batch_first=True, dropout=lstm_dropout,
                            bidirectional=True)
        #print(self.get_lstm_input_shape())
        #print(self.channels)
        self.l0 = nn.Linear(self.get_lstm_input_shape() * self.channels, self.side_arm_mlp)
        
        # Linear
        if self.classification:
            self.l1 = nn.Linear(self.get_mlp_input_shape() + side_arm_mlp, no_classes)
        else:
            self.l1 = nn.Linear(self.get_mlp_input_shape() + side_arm_mlp, regression_mlp_hidden)
            self.l2 = nn.Linear(regression_mlp_hidden, 1)
            
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Batch Normalisation
        self.batchnorm1dH1 = nn.BatchNorm1d(channels)
        self.batchnorm1dH2 = nn.BatchNorm1d(channels)
        self.batchnormLstm = nn.BatchNorm1d(channels)
        
        # Activation
        #self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        
        # Log Softmax
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        # CNN
        x = x.view(x.shape[0], -1, x.shape[1])

        x1 = self.conv1dh1l1(x)
        x1 = self.act(x1)
        x1 = self.conv1dh1l2(x1)
        x1 = self.act(x1)
        x1 = self.conv1dh1l3(x1)
        x1 = self.batchnorm1dH1(x1)
        #x1 = self.maxpoolConvH1(x1)
        x1 = self.avgpoolConvH1(x1)
        
        x2 = self.conv1dh2l1(x)
        x2 = self.act(x2)
        x2 = self.conv1dh2l2(x2)
        x2 = self.act(x2)
        x2 = self.conv1dh2l3(x2)
        x2 = self.batchnorm1dH2(x2)
        #x2 = self.maxpoolConvH2(x2)
        x2 = self.avgpoolConvH2(x2)
        
        x = torch.cat((x1, x2), 2)
        x = self.batchnormLstm(x)
        x = self.act(x)
        x = self.dropout(x)
        
        # Side-Arm
        #print(x.shape)
        x_side_arm = x.view(x.shape[0], -1)
        #print(x_side_arm.shape)
        x_side_arm = self.l0(x_side_arm)
        
        # LSTM
        x_lstm, self.lstm_hidden_states = self.lstm(x, self.lstm_hidden_states)
        # Decouple from training history
        self.lstm_hidden_states = tuple([each.data for each in self.lstm_hidden_states])
        x_lstm = self.dropout(x_lstm)
        x_lstm = x_lstm.reshape(x.shape[0], -1)
        
        # Linear
        x = torch.cat((x_side_arm, x_lstm), 1)
        #print(x.shape)
        x = self.l1(x)
        
        if self.classification:
            x = self.softmax(x)
        else:
            x = self.act(x)
            x = self.l2(x)
            x = x.view(-1)
        
        return x
    
    def init_hidden(self, batch_size=None):
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # Create two new tensors with sizes lstm_layers x batch_size x lstm_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        self.lstm_hidden_states = (weight.new(self.lstm_layers * 2, batch_size, self.lstm_hidden).zero_().cuda(),
                                   weight.new(self.lstm_layers * 2, batch_size, self.lstm_hidden).zero_().cuda())
        
    def get_lstm_input_shape(self):
        
        x_shape = (1, 1, self.data_length)
        
        # Head 1
        pred_shapeh1 = self.conv1D_output_shape(x_shape, kernel_size=self.head1_kernel_size)
        pred_shapeh1 = self.conv1D_output_shape((self.batch_size, self.channels, pred_shapeh1), kernel_size=self.head1_kernel_size)
        pred_shapeh1 = self.conv1D_output_shape((self.batch_size, self.channels, pred_shapeh1), kernel_size=self.head1_kernel_size)
        pred_shapeh1 = self.maxpool1D_output_shape((self.batch_size, self.channels, pred_shapeh1), kernel_size=self.head1_kernel_size)
        
        # Head 2
        pred_shapeh2 = self.conv1D_output_shape(x_shape, kernel_size=self.head2_kernel_size)
        pred_shapeh2 = self.conv1D_output_shape((self.batch_size, self.channels, pred_shapeh2), kernel_size=self.head2_kernel_size)
        pred_shapeh2 = self.conv1D_output_shape((self.batch_size, self.channels, pred_shapeh2), kernel_size=self.head2_kernel_size)
        pred_shapeh2 = self.maxpool1D_output_shape((self.batch_size, self.channels, pred_shapeh2), kernel_size=self.head2_kernel_size)
        
        return pred_shapeh1 + pred_shapeh2

    def get_mlp_input_shape(self):
        return self.lstm_hidden * 2 * self.channels
    
    def conv1D_output_shape(self, shape, stride=1, padding=0, dilation=1, kernel_size=4):
        N, C, L = int(shape[0]), int(shape[1]), int(shape[2])
        # L out
        return math.floor(((L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    
    def maxpool1D_output_shape(self, shape, stride=None, padding=0, dilation=1, kernel_size=4):
        if stride is None:
            stride = kernel_size
        N, C, L = int(shape[0]), int(shape[1]), int(shape[2])
        # L out
        return math.floor(((L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
    

def forward(model, path, classification=True, batch_size=1):  
    
    dataloader, dataset = path_to_DataLoader(path, batch_size=batch_size,
                                    num_workers=1,
                                    shuffle=False,
                                    index=True,
                                    classification=classification,
                                    double_precision=False)
    model.eval()
    
    Y = []
    Y_hat = []

    with torch.no_grad():
        for x, y in dataloader:

            model.init_hidden(x.shape[0])

            x = x.cuda()
            y = y.cuda()

            output = model(x)
            
            Y.extend(np.array(y.detach().cpu()))
            Y_hat.extend(np.array(output.detach().cpu()))

    model.train()

    return np.array(Y), np.array(Y_hat)

def evaluate_regression_complete(model, path, no_classes=None, batch_size=1, bias=0):
    Y, Y_hat = forward(model, path, classification=False, batch_size=batch_size)
    Y_label = regression_results_to_label(Y, no_classes=no_classes)
    Y_hat_label = regression_results_to_label(Y_hat-bias, no_classes=no_classes)
    return sklearn.metrics.accuracy_score(Y_label, Y_hat_label), (Y_hat-bias) - Y

def evaluate_classification_complete(model, path, no_classes=None, batch_size=1, bias=0):
    #bias = int(round(bias, 0))
    Y_label, Y_hat = forward(model, path, classification=True, batch_size=batch_size)
    Y_hat_label = np.argmax(np.exp(Y_hat), axis=1)# - bias
    Y_hat_label = np.maximum(0, Y_hat_label)
    Y_hat_label = np.minimum(no_classes-1, Y_hat_label)
    return sklearn.metrics.accuracy_score(Y_label, Y_hat_label), Y_hat_label - Y_label

def evaluate_regression_constant(model, path, N=None, no_classes=None, batch_size=1, bias=0, simulated=None):
    Y, Y_hat = forward(model, path, classification=False, batch_size=batch_size)
    Y_label = utils.regression_results_to_label(Y, no_classes)
    Y_hat_label = utils.regression_results_to_label(Y_hat-bias, no_classes=no_classes)
    return utils.accuracy_score_from_label_chunks(Y_label, Y_hat_label, N=N, simulated=simulated), (Y_hat-bias) - Y

def evaluate_classification_constant(model, path, N=None, no_classes=None, batch_size=1, bias=0, simulated=None):
    #bias = int(round(bias, 0))
    Y_label, Y_hat = forward(model, path, classification=True, batch_size=batch_size)
    Y_hat_label = np.argmax(np.exp(Y_hat), axis=1)# - bias
    Y_hat_label = np.maximum(0, Y_hat_label)
    Y_hat_label = np.minimum(no_classes-1, Y_hat_label)
    return utils.accuracy_score_from_label_chunks(Y_label, Y_hat_label, N=N, simulated=simulated), Y_hat_label - Y_label
    