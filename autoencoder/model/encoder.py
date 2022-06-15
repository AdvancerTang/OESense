import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, mode, label):
        super(AutoEncoder, self).__init__()
        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2)),  # stft input = 129,
                                   # mel input = 128
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(inplace=True)
                                   )

        # decoder
        self.conv1f = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.conv2f = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2)),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.conv3f = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2)),
                                    nn.BatchNorm2d(1),
                                    nn.LeakyReLU(inplace=True)
                                    )

        self.stack_rnn = nn.GRU(input_size=15, hidden_size=15, num_layers=1,
                                bidirectional=True)
        self.ln = nn.LayerNorm(30)

        self.fc_f0 = nn.Sequential(
            nn.Linear(30, 15),
            nn.LeakyReLU(inplace=True)
        )

        self.fc_f1 = nn.Sequential(
            nn.Linear(15, 8),
            nn.LeakyReLU(inplace=True)
        )
        self.fc_f2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU(inplace=True)
        )
        self.fc_f3 = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU(inplace=True)
        )
        # self.fc_t1 = nn.Sequential(
        #     nn.Linear(4, 8),
        #     nn.LeakyReLU(inplace=True)
        # )
        # self.fc_t2 = nn.Sequential(
        #     nn.Linear(8, label),
        #     nn.LeakyReLU(inplace=True)
        # )


        self.mode = mode

    def forward(self, input):
        # shape: [B, C, F, T]
        input = input.unsqueeze(1)

        # encode
        encoded = self.conv1(input)
        encoded = self.conv2(encoded)
        encoded = self.conv3(encoded)
        if self.mode == 'encode':
            # decode
            decoded = self.conv1f(encoded)
            decoded = self.conv2f(decoded)
            decoded = F.pad(decoded, (0, 1, 0, 0))
            decoded = self.conv3f(decoded)
            decoded = F.pad(decoded, (0, 0, 0, 1))
            decoded = decoded.squeeze()
            return decoded
        elif self.mode == 'my':
            # C * F * T
            cnn_out = encoded.mean(1)
            # C * T * F
            cnn_out = cnn_out.permute(0, 2, 1)

            t_out, _ = self.stack_rnn(cnn_out)
            t_out = self.ln(t_out)
            t_out = self.fc_f0(t_out)
            t_out = self.fc_f1(t_out)
            t_out = self.fc_f2(t_out)
            # t_out = self.fc_f3(t_out)
            f_out = t_out.permute(0, 2, 1)
            # f_out = self.fc_t1(f_out)
            # f_out = self.fc_t2(f_out)
            max_pool = nn.MaxPool2d(kernel_size=(f_out.shape[1], 1))
            out = max_pool(f_out)
            out = out.squeeze()
            return out


if __name__ == '__main__':
    batch_size = 4
    input = torch.rand(batch_size, 128, 41)
    model = AutoEncoder('my', 4)
    out = model(input)
    # model_dict = model.state_dict()
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
    print('input size:', input.size())
    print('output size', out.size())
