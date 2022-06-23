import os.path

from tqdm import tqdm
import torch
import torch.nn as nn
from data_reader import myDataset, myDataloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import argparse
import random
random.seed(1)

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from model.encoder import AutoEncoder
# from model.total_net import FreqNet

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def time_frature_train(dataloader_train, dataloader_val, iters):
    classifier = LogisticRegression(C=15, multi_class='multinomial', solver="newton-cg", max_iter=iters)
    trainSet = []
    trainLabel = []
    for tfeature, tlabel in dataloader_train:
        trainSet.append(tfeature)
        trainLabel.append(tlabel)
    trainset = np.squeeze(np.array(trainSet))
    trainlabel = np.squeeze(np.array(trainLabel))
    classifier.fit(trainset, trainlabel)

    # eval
    evalSet = []
    evalLabel = []
    for efeature, elabel in dataloader_val:
        evalSet.append(efeature)
        evalLabel.append(elabel)
    evalset = np.squeeze(np.array(evalSet))
    yTrue = np.squeeze(np.array(evalLabel))
    yPredict = classifier.predict(evalset)

    # error count
    error_Count = 0.0
    for i in range(len(yPredict)):
        if int(yPredict[i]) == int(yTrue[i]):
            pass
        else:
            error_Count += 1
    error_Rate = error_Count / float(len(yPredict))
    print('Error rate: %.3f' % error_Rate)
    # res = classification_report(yTrue, yPredict, output_dict=True, zero_division=0)
    # result = res['macro avg']
    # print(result)
    print(classification_report(yTrue, yPredict, zero_division=0))
    return None


def freq_frature_train(dataloader_train, dataloader_val, iters, lr, train_mode, label):
# def freq_frature_train(tr_feature, tr_label, vl_feature, vl_label, iters, lr, device, feature, label):
    model_path = 'encoder_model'
    numBatch_tr = len(dataloader_train)
    numBatch_evl = len(dataloader_val)
    total_num = numBatch_tr + numBatch_evl
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    # if feature == 'stft':
    #     model = FreqNet(label)
    # elif feature == 'mel':
    #     model = MelNet()
    model = AutoEncoder(train_mode, label)
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    writer = SummaryWriter('logs')
    for iter_ in range(iters):
        # print("-----------第 {} 轮训练开始----------".format(iter_))
        with tqdm(total=total_num, desc='iter{}'.format(iter_)) as pbar:
            model.train()
            mean_loss = 0
            tr_loss_total = 0
            for batch_num, (feature_tr, label_tr) in enumerate(dataloader_train):
                feature_tr.to(device)
                label_tr.to(device)
                optimizer.zero_grad()
                y_tr = model(feature_tr)
                # loss = criterion(y_tr, label_tr)
                loss = criterion(y_tr, feature_tr)
                loss.backward()
                optimizer.step()

                batch_loss = float(loss)
                mean_loss = (mean_loss * batch_num + batch_loss) / (batch_num + 1)
                # print('\tbatch loss{:.4f}, mean loss{:.4f}'.format(batch_loss, mean_loss))
                pbar.update(16)
                tr_loss_total += loss.item()


            writer.add_scalar('train_loss', tr_loss_total / numBatch_tr, iter_)

            # eval
            model.eval()
            o = []
            t = []
            with torch.no_grad():
                evl_loss_total = 0
                for feature_evl, label_evl in dataloader_val:
                    feature_evl.to(device)
                    y_evl = model(feature_evl)
                    y_evl = y_evl.unsqueeze(0)
                    loss_evl = criterion(y_evl, feature_evl)
                    # loss_evl = criterion(y_evl, label_evl)

                    evl_loss_total += loss_evl.item()
                    pbar.update(1)

                    o.append(y_evl)
                    t.append(feature_evl)
                writer.add_image('target', t[0], global_step=iter_, dataformats='CHW')
                writer.add_image('output', o[0], global_step=iter_, dataformats='CHW')

                # print('Iteration:{0}, loss = {1:.6f} '.format(iter_, evl_loss_total / numBatch_evl))
                pbar.set_description('Iteration:{0}, train_loss = {1:.6f}, eval_loss = {2:.6f} '.format(iter_, tr_loss_total / numBatch_tr,
                                                                                                        evl_loss_total / numBatch_evl))
                writer.add_scalar('eval_loss', evl_loss_total / numBatch_evl, iter_)

        torch.save(model.state_dict(), os.path.join(model_path, '{}_{}.model'.format('FreqNet', iter_)))
    writer.close()
    return


def main(args):
    # Compile and configure parameters.
    person = args.person
    device = args.device
    feature = args.feature
    label = args.label
    channel = args.channel
    batchsize_train = args.batchsize_train
    batchsize_val = args.batchsize_val
    iters = args.iters
    lr = args.lr
    mode = args.mode
    train_mode = args.train_mode

    # file path
    # train_path = r'F:\OESense\wave_dir\data_train_1'
    # val_path = r'F:\OESense\wave_dir\data_val_1'
    if mode == 'solo':
        train_path = r'F:\OESense\wave_dir\data_{}_train_1'.format(person)
        val_path = r'F:\OESense\wave_dir\data_{}_val_1'.format(person)
    elif mode == 'total':
        # train_path = r'F:\OESense\autoencoder\scp1_dir\data_train.scp'
        # val_path = r'F:\OESense\autoencoder\scp1_dir\data_evl.scp'
        train_path = r'F:\OESense\autoencoder\total_dir\data_train_1'
        val_path = r'F:\OESense\autoencoder\total_dir\data_val_1'

    # define dataloader
    print('loading the dataset...')
    dataset_train = myDataset(train_path, channel, feature)
    dataloader_train = myDataloader(dataset=dataset_train,
                                    batch_size=batchsize_train,
                                    shuffle=True,
                                    num_workers=args.train_num_workers)
    dataset_val = myDataset(val_path, channel, feature)
    dataloader_val = myDataloader(dataset=dataset_val,
                                  batch_size=batchsize_val,
                                  shuffle=False,
                                  num_workers=args.val_num_workers)
    print('- done.')
    print('-{} training samples, {} dev samples '.format(len(dataset_train), len(dataset_val)))
    print('-{} training batch, {} training batch'.format(len(dataloader_train), len(dataloader_val)))

    # train

    if feature == 'time':
        time_frature_train(dataloader_train, dataloader_val, iters)
    elif feature == 'stft' or feature == 'mel':
        freq_frature_train(dataloader_train, dataloader_val, iters, lr, train_mode, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='total', type=str, help='total/solo')
    parser.add_argument('--train_mode', default='encode', type=str, help='encode/my')
    parser.add_argument('--person', default=1, type=int, help='train for person. if mode is total, ignore')
    parser.add_argument('--feature', default='mel', type=str, help='choose time, stft, mel')
    parser.add_argument('--label', default=4, type=int, help='number of gestures')
    parser.add_argument('--channel', default=0, type=int, help='choose channel')
    parser.add_argument('--batchsize_train', default=2, type=int)
    parser.add_argument('--batchsize_val', default=1, type=int)
    parser.add_argument('--iters', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--train_num_workers', default=2, type=int, help='number of train worker')
    parser.add_argument('--val_num_workers', default=1, type=int, help='number of validation worker')
    args = parser.parse_args()
    main(args)