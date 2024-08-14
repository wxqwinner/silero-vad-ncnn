import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SileroNet']

class SileroNet(nn.Module):
    # done
    # __torch__/vad/model/vad_annotator/___torch_mangle_28.py
    def __init__(self):
        super().__init__()
        self.adaptive_normalization = AdaptiveNormalization()
        self.feature_extractor = FeatureExtractor()
        self.first_layer = FirstLayer()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, h, c):
        x0 = self.feature_extractor(x)
        norm = self.adaptive_normalization(x0)
        x1 = torch.cat([x0, norm], 1)
        x2 = self.first_layer(x1)
        x3 = self.encoder(x2)
        x4, h0, c0 = self.decoder(x3, h, c)
        x5 = torch.mean(torch.squeeze(x4, 1), [1])
        y = torch.unsqueeze(x5, 1)

        return (y, h0, c0)


class AdaptiveNormalization(nn.Module):
    # done self.to_pad confirm
    def __init__(self):
        super().__init__()
        self.to_pad = 7
        self.register_buffer('filter_', torch.randn(1, 1, 7))

    def forward(self, x):
        x0 = torch.log(torch.mul(x, 1048576) + 1.0)
        x1 = torch.mean(x0, 1, True)

        left_pad = x1[:, :, 1:self.to_pad+1]
        right_pad = x1[:, :, -self.to_pad-1:-1]

        # The NCNN custom layer is not used.
        # left_pad = torch.flip(left_pad, dims=(2,))
        # right_pad = torch.flip(right_pad, dims=(2,))

        x1 = torch.cat([left_pad, x1, right_pad], dim=2)

        x2 = F.conv1d(x1, self.filter_)
        x3 = torch.mean(x2, 2, True)
        x4 = torch.add(x0, torch.neg(x3))
        y = x4
        return y


class FirstLayer(nn.Module):
    # done
    # __torch__.torch.nn.modules.container.___torch_mangle_2.Sequential
    def __init__(self):
        super().__init__()
        ## __torch__.models.number_vad_model.ConvBlock

        # __torch__.torch.nn.modules.container.Sequential
        self.conv1 = nn.Conv1d(258, 258, kernel_size=(5,), stride=(1,), groups=(258), padding=(2,), bias=True) # __torch__.torch.nn.modules.conv.Conv1d
        # __torch__.torch.nn.modules.container.___torch_mangle_1.Sequential
        self.conv2 = nn.Conv1d(258, 16, kernel_size=(1,), stride=(1,), padding=(0,), bias=True) # __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv1d
        # __torch__.torch.nn.modules.conv.___torch_mangle_0.Conv1d
        self.conv3 = nn.Conv1d(258, 16, kernel_size=(1,), stride=(1,), padding=(0,), bias=True)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = F.relu(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x)
        x4 = x2 + x3
        x5 = F.relu(x4)
        x6 = self.dropout(x5)
        y = x6
        return y


class FeatureExtractor(nn.Module):
    # done
    def __init__(self):
        super().__init__()
        self.register_buffer('forward_basis_buffer', torch.randn(258, 1, 256))
        self.filter_length = 256
        self.hop_length = 64

    def forward(self, x):
        num_batches = x.shape[0]
        num_samples = x.shape[1]
        x0 = x.reshape(num_batches, 1, num_samples)
        to_pad = (self.filter_length - self.hop_length) // 2

        x1 = F.pad(torch.unsqueeze(x0, 1), [to_pad, to_pad, 0, 0], 'reflect')
        x1 = torch.squeeze(x1, 1)
        x2 = F.conv1d(x1, self.forward_basis_buffer, stride=self.hop_length)

        dim = self.filter_length//2 + 1

        real = x2[:, :dim, :]
        imag = x2[:, dim:, :]

        magnitude = torch.sqrt(real*real + imag*imag)
        return magnitude


class ConvBlock(nn.Module):
    # done
    def __init__(self, num_features):
        super().__init__()
        self.conv1  = nn.Conv1d(num_features, num_features, kernel_size=(1,), stride=(2,), padding=(0,), bias=True)
        self.bn1    = nn.BatchNorm1d(num_features)
        self.conv2  = nn.Conv1d(num_features, num_features, kernel_size=(5,), stride=(1,), padding=(2,), groups=num_features, bias=True) # dw
        self.conv3  = nn.Conv1d(num_features, num_features*2, kernel_size=(1,), stride=(1,), padding=(0,), bias=True) # pw
        self.conv4  = nn.Conv1d(num_features, num_features*2, kernel_size=(1,), stride=(1,), padding=(0,), bias=True) # proj
        self.conv5  = nn.Conv1d(num_features*2, num_features*2, kernel_size=(1,), stride=(2,), padding=(0,), bias=True)
        self.bn2    = nn.BatchNorm1d(num_features*2)
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x0 = x
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = F.relu(x0)
        x1 = self.conv2(x0)
        x1 = F.relu(x1)
        x1 = self.conv3(x1)
        x2 = self.conv4(x0)
        x3 = x1 + x2
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.conv5(x3)
        x3 = self.bn2(x3)
        y = F.relu(x3)
        return y


class Encoder(nn.Module):
    # done
    def __init__(self):
        super().__init__()
        ## __torch__.torch.nn.modules.container.___torch_mangle_27.Sequential
        # __torch__.torch.nn.modules.conv.___torch_mangle_3.Conv1d
        self.conv_block1 = ConvBlock(num_features=16) # __torch__.torch.nn.modules.container.___torch_mangle_9.Sequential
        self.conv_block2 = ConvBlock(num_features=32)

        self.conv1 = nn.Conv1d(32, 32, kernel_size=(5,), stride=(1,), padding=(2,), groups=32, bias=True)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=(1,), stride=(1,), padding=(0,), bias=True)

        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x0 = self.conv_block1(x)
        x1 = self.conv1(x0)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x2 = x0 + x1
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        y = self.conv_block2(x2)
        return y


class Decoder(nn.Module):
    # done
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.conv1 = nn.Conv1d(64, 1, kernel_size=(1,), stride=(1,), padding=(0,), bias=True)

    def forward(self, x, h, c):
        x0, (h0, c0) = self.lstm1(torch.permute(x, [0, 2, 1]), (h, c),)
        x1 = torch.permute(x0, [0, 2, 1])
        x1 = F.relu(x1)
        x2 = self.conv1(x1)
        x3 = F.sigmoid(x2)
        y = x3
        return (y, h0, c0)
