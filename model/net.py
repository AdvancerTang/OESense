import torch
import torch.nn as nn


class FreqNet(nn.Module):
    def __init__(self, label):
        super(FreqNet, self).__init__()
        # time feature extract
        self.t_encoder = nn.ModuleList()
        self.conv1 = nn.Sequential(nn.Conv2d(41, 64, kernel_size=(1, 3), stride=(1, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(inplace=True)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2)),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(inplace=True)
                                   )
        # self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2)),
        #                            nn.BatchNorm2d(256),
        #                            nn.LeakyReLU(inplace=True)
        #                            )
        self.t_encoder = nn.Sequential(
                self.conv1,
                self.conv2
            )

        # stft feature extract
        self.conv1f = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.conv2f = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(inplace=True)
                                    )
        self.f_encoder = nn.Sequential(
            self.conv1f,
            self.conv2f
        )
        # self.fc0 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 64)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, label)

    def forward(self, input):
        # shape: [B, C, F, T]
        input = input.unsqueeze(1)

        # shape: [B, T, C, F]
        input = input.permute(0, 3, 1, 2)
        T_extract = self.t_encoder(input)

        # shape: [B, C, F, T]
        T_extract = T_extract.permute(0, 2, 3, 1)
        cnn_out = self.f_encoder(T_extract)

        # shape: [B, T, F]
        cnn_out = cnn_out.mean(1)
        # cnn_out = self.fc0(cnn_out)
        cnn_out = self.fc1(cnn_out)
        # cnn_out = self.dropout(cnn_out)
        cnn_out = self.fc2(cnn_out)

        max_pool = nn.MaxPool2d(kernel_size=(cnn_out.shape[1], 1))
        cnn_out = max_pool(cnn_out)
        cnn_out = cnn_out.squeeze()
        return cnn_out

if __name__ =='__main__':
    batch_size = 4
    input = torch.rand(batch_size, 129, 41)
    print(input.type())
    model = FreqNet()
    model_dict = model.state_dict()

    out = model(input)
    print('input size:', input.size())
    print('output size', out.size())

