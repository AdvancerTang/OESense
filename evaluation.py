import os.path

import torch
import torch.nn as nn
from data_reader import myDataset, myDataloader
from sklearn.metrics import classification_report
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from model.net import FreqNet


def main(args):
    # Compile and configure parameters.
    channel = args.channel
    batchsize_val = args.batchsize_val
    trained_model = args.trained_model


    # file path
    val_path = r'F:\OESense\wave_dir\data_val_0'

    # define dataloader
    print('loading the dataset...')
    dataset_val = myDataset(val_path, channel)
    dataloader_val = myDataloader(dataset=dataset_val,
                                  batch_size=batchsize_val,
                                  shuffle=False)
    print('- done.')
    print('{} dev samples '.format(len(dataset_val)))
    print('{} training batch'.format(len(dataloader_val)))

    # define the load model
    model = FreqNet()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default=2, type=int, help='choose channel')
    parser.add_argument('--batchsize_val', default=1, type=int)
    parser.add_argument('--iters', default=25, type=int)
    parser.add_argument('--trained_model', default='./audio_model/FreqNet_11.model', type=str)
    args = parser.parse_args()
    main(args)
