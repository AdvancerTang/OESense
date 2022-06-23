import os.path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from data_reader import myDataset, myDataloader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from model.encoder import AutoEncoder


def main(args):
    # Compile and configure parameters.
    net_mode = args.net_mode
    person = args.person
    channel = args.channel
    label = args.label
    feature = args.feature
    batchsize_val = args.batchsize_val
    trained_model = args.trained_model


    # file path
    # clean
    val_path = r'F:\OESense\autoencoder\wave_dir\data_1_val_1'
    # gesture
    # val_path = r'F:\OESense\wave_dir\data_1_val_1'
    # act
    # val_path = r'F:\OESense\wave_dir\data_{}_val_1_act'.format(person)
    # define dataloader
    print('loading the dataset...')
    dataset_val = myDataset(val_path, channel, feature)
    dataloader_val = myDataloader(dataset=dataset_val,
                                  batch_size=batchsize_val,
                                  shuffle=False)
    print('- done.')
    print('{} dev samples '.format(len(dataset_val)))
    print('{} training batch'.format(len(dataloader_val)))

    # define the load model
    model = AutoEncoder(net_mode, label)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(trained_model)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    yPre = []
    yLabel = []
    max_recall= 0
    with torch.no_grad():
        for feature, label in dataloader_val:
            y_Pre = model(feature)
            y_pre_evl = y_Pre.unsqueeze(0)
            label_evl = label.squeeze(-1)
            if y_pre_evl.ndim == 1:
                y_pre_evl = y_pre_evl.unsqueeze(0)
            pre = nn.Softmax(dim=1)
            y_pre = pre(y_pre_evl)
            y_pre = y_pre.argmax(dim=1)
            yPre.append(int((y_pre).numpy()))
            yLabel.append(int((label_evl).numpy()))

        yPre = np.squeeze(np.array(yPre))
        yLabel = np.squeeze(np.array(yLabel))
        res = classification_report(yPre, yLabel, output_dict=True, zero_division=0)
        result = res['macro avg']
        recall = result['recall']
        max_recall = max(max_recall, recall)
        print('recall : % .3f' % max_recall)
        print(result)

        cm = confusion_matrix(yLabel, yPre)
        cm = pd.DataFrame(cm, columns=['left', 'right', 'up', 'down'], index=['left', 'right', 'up', 'down'])
        plt.Figure(figsize=(4, 4))
        sns.heatmap(cm, cmap="YlGnBu_r", fmt="d", annot=True)
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_mode', default='my', type=str, help='encode/my')
    parser.add_argument('--person', default=1, type=int, help='choose test person')
    parser.add_argument('--channel', default=0, type=int, help='choose channel')
    parser.add_argument('--label', default=4, type=int, help='number of gestures')
    parser.add_argument('--feature', default='mel', type=str, help='choose time, stft, mel')
    parser.add_argument('--batchsize_val', default=1, type=int)
    parser.add_argument('--iters', default=25, type=int)
    parser.add_argument('--trained_model', default=r'F:\OESense\autoencoder\my_model\FreqNet_146.model', type=str)
    args = parser.parse_args()
    main(args)
