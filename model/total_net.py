import torch
import torch.nn as nn
from torchvision import models


class GRU_Encoder(nn.Module):
    def __init__(self, feature_dim, hidden_size, num_layers):
        super(GRU_Encoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack_rnn_0 = nn.GRU(input_size=self.feature_dim, hidden_size=256, num_layers=self.num_layers,
                                bidirectional=True)
        self.stack_rnn_1 = nn.GRU(input_size=512, hidden_size=128, num_layers=self.num_layers,
                                bidirectional=True)

    def forward(self, input):
        rnn_out, _ = self.stack_rnn_0(input)
        rnn_out, _ = self.stack_rnn_1(rnn_out)
        return rnn_out


class FreqNet(nn.Module):
    def __init__(self, label):
        super(FreqNet, self).__init__()
        # freq feature extract
        self.encoder = nn.ModuleList()
        self.conv1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2)),  # stft input = 129,
                                   # mel input = 128
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(1, 2)),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(inplace=True)
                                   )
        # self.conv3 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        #                            nn.BatchNorm2d(1024),
        #                            nn.LeakyReLU(inplace=True)
        #                            )

        self.encoder = nn.Sequential(
            self.conv1,
            self.conv2
        )
        # time feature extract
        self.t_encoder = GRU_Encoder(512, 256, 2)
        self.ln = nn.LayerNorm(256)

        # stft feature extract
        self.conv1f = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.conv2f = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.conv3f = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(inplace=True)
                                    )
        # self.conv4f = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
        #                             nn.BatchNorm2d(512),
        #                             nn.LeakyReLU(inplace=True)
        #                             )
        self.f_encoder = nn.Sequential(
            self.conv1f,
            self.conv2f,
            self.conv3f
        )
        # self.fc0 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(64, 32)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, label)

    def forward(self, input):
        # shape: [B, C, F, T]
        input = input.unsqueeze(1)

        # shape: [B, F, C, T] freq feature extract
        input = input.permute(0, 2, 1, 3)
        extract = self.encoder(input)

        # shape: [B, F, C, T] time feature extract
        T_input = extract.squeeze(2)
        # shape [T, B, F]
        T_input = T_input.permute(2, 0, 1)
        T_output = self.t_encoder(T_input)
        rnn_out = self.ln(T_output)
        # shape [B, C, F, T]
        rnn_out = rnn_out.permute(1, 2, 0)
        rnn_out = rnn_out.unsqueeze(1)

        # shape: [B, T, F]
        cnn_out = self.f_encoder(rnn_out)
        cnn_out = cnn_out.mean(1)
        cnn_out = cnn_out.permute(0, 2, 1)
        # cnn_out = self.fc0(cnn_out)
        cnn_out = self.fc1(cnn_out)
        # cnn_out = self.dropout(cnn_out)
        cnn_out = self.fc2(cnn_out)
        cnn_out = self.fc3(cnn_out)

        max_pool = nn.MaxPool2d(kernel_size=(cnn_out.shape[1], 1))
        cnn_out = max_pool(cnn_out)
        cnn_out = cnn_out.squeeze()
        return cnn_out


if __name__ == '__main__':
    batch_size = 4
    input = torch.rand(batch_size, 128, 41)
    print(input.type())
    model = FreqNet(4)
    model_dict = model.state_dict()
    # for layer, param in model_dict.items():
    #     print("layers:" + layer)
    #     print(param)
    # param = model.named_parameters()
    # print(model_dict)
    # total_param = 0
    # for index, (name, param) in enumerate(model.named_parameters()):
    #     print(str(index) + " " + name + 'size:' + str(param.size()) + 'param:' + str(param.numel()))
    #     total_param += param.numel()
    # print('total param: {}'.format(total_param))
    out = model(input)
    print('input size:', input.size())
    print('output size', out.size())
