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
from model.rnn_net import FreqNet
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


def freq_frature_train(dataloader_train, dataloader_val, iters, lr, device, feature, label):
# def freq_frature_train(tr_feature, tr_label, vl_feature, vl_label, iters, lr, device, feature, label):
    model_path = 'audio_model'
    # numBatch_tr = len(tr_feature)
    # numBatch_evl = len(tr_feature)
    numBatch_tr = len(dataloader_train)
    numBatch_evl = len(dataloader_val)
    total_num = numBatch_tr + numBatch_evl
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    # if feature == 'stft':
    #     model = FreqNet(label)
    # elif feature == 'mel':
    #     model = MelNet()
    model = FreqNet(label)
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)

    writer = SummaryWriter('logs')
    steps = 1
    cur_recall = 0
    max_recall = 0
    for iter_ in range(iters):
        model.train()
        mean_loss = 0
        tr_loss_total = 0
        # print("-----------??? {} ???????????????----------".format(iter_))
        with tqdm(total=total_num, desc='iter{}'.format(iter_)) as pbar:
            for batch_num, (feature_tr, label_tr) in enumerate(dataloader_train):
            # for batch_num in range(numBatch_tr):
            #     feature_tr = tr_feature[batch_num].to(device)
            #     label_tr = tr_label[batch_num].to(device)
                feature_tr.to(device)
                label_tr.to(device)
                optimizer.zero_grad()
                y_pre_tr = model(feature_tr)
                label_tr = label_tr.squeeze(-1)
                if y_pre_tr.ndim == 1:
                    y_pre_tr = y_pre_tr.unsqueeze(0)
                loss = criterion(y_pre_tr, label_tr)
                loss.backward()
                optimizer.step()


                batch_loss = float(loss)
                mean_loss = (mean_loss * batch_num + batch_loss) / (batch_num + 1)
                # print('\tbatch loss{:.4f}, mean loss{:.4f}'.format(batch_loss, mean_loss))
                pbar.update(1)
                tr_loss_total += loss.item()

            # pbar.set_description('Iteration:{0}, train_loss = {1:.6f} '.format(iter_, tr_loss_total / numBatch_tr))
            # print('Iteration:{0}, loss = {1:.6f} '.format(iter_, tr_loss_total / numBatch_tr))
            writer.add_scalar('train_loss', tr_loss_total / numBatch_tr, iter_)

            # eval
            model.eval()
            yPre = []
            yLabel = []
        # with tqdm(total=total_num) as pbar:
            with torch.no_grad():
                evl_loss_total = 0
                # for batch_num in range(numBatch_evl):
                for feature_evl, label_evl in dataloader_val:
                    # feature_evl = vl_feature[batch_num].to(device)
                    # label_evl = vl_label[batch_num].to(device)
                    feature_evl.to(device)
                    label_evl.to(device)
                    y_pre_evl = model(feature_evl)
                    y_pre_evl = y_pre_evl.unsqueeze(0)
                    label_evl = label_evl.squeeze(-1)
                    if y_pre_evl.ndim == 1:
                        y_pre_evl = y_pre_evl.unsqueeze(0)
                    loss_evl = criterion(y_pre_evl, label_evl)

                    pre = nn.Softmax(dim=1)
                    y_pre = pre(y_pre_evl)
                    y_pre = y_pre.argmax(dim=1)
                    yPre.append(int((y_pre).numpy()))
                    yLabel.append(int((label_evl).numpy()))
                    # yPre.append((y_pre).numpy())
                    # yLabel.append((label_evl).numpy())
                    evl_loss_total += loss_evl.item()
                    pbar.update(16)
                # print('Iteration:{0}, loss = {1:.6f} '.format(iter_, evl_loss_total / numBatch_evl))
                pbar.set_description('Iteration:{0}, train_loss = {1:.6f}, eval_loss = {2:.6f} '.format(iter_, tr_loss_total / numBatch_tr,
                                                                                                        evl_loss_total / numBatch_evl))
                writer.add_scalar('eval_loss', evl_loss_total / numBatch_evl, iter_)


            yPre = np.squeeze(np.array(yPre))
            yLabel = np.squeeze(np.array(yLabel))
            res = classification_report(yPre, yLabel, output_dict=True, zero_division=0)
            result = res['macro avg']
            recall = result['recall']
            print('recall = {:.3f}'.format(recall))
            print(result)
            writer.add_scalar('recall', recall, iter_)
            torch.save(model.state_dict(), os.path.join(model_path, '{}_{}.model'.format('FreqNet', iter_)))
            cur_recall = recall
            max_recall = max(max_recall, cur_recall)
            if max_recall == cur_recall:
                max_iter = iter_
    writer.close()
    return max_recall, max_iter


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

    # file path
    # train_path = r'F:\OESense\wave_dir\data_train_1'
    # val_path = r'F:\OESense\wave_dir\data_val_1'
    if mode == 'solo':
        # train_path = r'F:\OESense\wave_dir\data_{}_train_1'.format(person)
        # val_path = r'F:\OESense\wave_dir\data_{}_val_1'.format(person)
        train_path = r'F:\OESense\autoencoder\wave_dir\data_{}_train_1'.format(person)
        val_path = r'F:\OESense\autoencoder\wave_dir\data_{}_val_1'.format(person)
    elif mode == 'total':
        train_path = r'F:\OESense\total_dir\data_train_2'
        val_path = r'F:\OESense\total_dir\data_val_2'

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
        max_recall, max_iter = freq_frature_train(dataloader_train, dataloader_val, iters, lr, device, feature, label)
        # tr_feature = []
        # tr_label = []
        # vl_feature = []
        # vl_label = []
        # for batch_num, (feature_tr, label_tr) in enumerate(dataloader_train):
        #     tr_feature.append(feature_tr)
        #     tr_label.append(label_tr)
        # for batch_num, (feature_tr, label_tr) in enumerate(dataloader_val):
        #     vl_feature.append(feature_tr)
        #     vl_label.append(label_tr)
        # max_recall, max_iter = freq_frature_train(tr_feature, tr_label, vl_feature, vl_label, iters, lr, device, feature, label)

        print('max recall: {}, max iter: {}'.format(max_recall, max_iter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='solo', type=str, help='total/solo')
    parser.add_argument('--person', default=1, type=int, help='train for person. if mode is total, ignore')
    parser.add_argument('--feature', default='mel', type=str, help='choose time, stft, mel')
    parser.add_argument('--label', default=4, type=int, help='number of gestures')
    parser.add_argument('--channel', default=0, type=int, help='choose channel')
    parser.add_argument('--batchsize_train', default=16, type=int)
    parser.add_argument('--batchsize_val', default=1, type=int)
    parser.add_argument('--iters', default=25, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--train_num_workers', default=2, type=int, help='number of train worker')
    parser.add_argument('--val_num_workers', default=1, type=int, help='number of validation worker')
    args = parser.parse_args()
    main(args)