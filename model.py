import torch.nn as nn
import torch
import numpy as np
class CNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, 2, padding=1)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
        return x

class DECNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, is_final):
        super(DECNNLayer, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_dim, out_dim, 2, 2)
        self.is_final = is_final
        if is_final:
            self.tanh = nn.Tanh()
        else:
            self.bn1 = nn.BatchNorm1d(out_dim)
            self.relu1 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.deconv1(x)
        if self.is_final:
            x = 128*self.tanh(x)
        else:
            x = self.bn1(x)
            x = self.relu1(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNNEncoder, self).__init__()
        self.conv_layer1 = CNNLayer(in_dim, out_dim)
        self.conv_layer2 = CNNLayer(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNNDecoder, self).__init__()
        self.deconv_layer1 = DECNNLayer(in_dim, out_dim, False)
        self.deconv_layer2 = DECNNLayer(out_dim, out_dim, True)

    def forward(self, x):
        x = self.deconv_layer1(x)
        x = self.deconv_layer2(x)
        return x[:,:,0:1000]

class RNNListener(nn.Module):
    def __init__(self, in_dim, hi_dim):
        super(RNNListener, self).__init__()
        self.rnn_core = nn.GRU(in_dim, hi_dim, 1, batch_first=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, batch_fature, batch_embedding):
        inputs = torch.cat([batch_fature, batch_embedding], 2)
        x = self.rnn_core(inputs)
        return x[0]

class RNNEmb(nn.Module):
    def __init__(self, in_dim, hi_dim):
        super(RNNEmb, self).__init__()
        self.rnn_core = nn.GRU(in_dim, hi_dim, 2, batch_first=True)

    def forward(self, batch_fature):
        x = self.rnn_core(batch_fature)
        return x[1]

class FullModel(nn.Module):
    def __init__(self, in_dim, fea_dim, emb_dim):
        super(FullModel, self).__init__()
        self.encoder = CNNEncoder(in_dim, fea_dim)
        self.decoder = CNNDecoder(fea_dim, in_dim)
        self.listener = RNNListener(fea_dim+emb_dim, fea_dim)
        self.loss = nn.MSELoss()

    def forward(self, batch_mfcc_fature, batch_embedding):
        batch_mfcc_fature = batch_mfcc_fature.permute(0, 2, 1)
        batch_feature = self.encoder(batch_mfcc_fature).permute(0, 2, 1)
        batch_feature = batch_feature[:, 1:250, :]
        batch_start = torch.zeros([batch_feature.shape[0], 1, batch_feature.shape[2]], device="cuda")
        batch_feature = torch.cat([batch_feature, batch_start], 1)
        mask = torch.Tensor(np.random.randint(0, 2, [batch_embedding.shape[0], 250, 1])).cuda()
        batch_feature = batch_feature*mask
        batch_embedding = batch_embedding.view(batch_embedding.shape[0], 1, batch_embedding.shape[1])
        batch_embedding = batch_embedding.expand(batch_embedding.shape[0], batch_feature.shape[1],
                                                 batch_embedding.shape[2])
        middle_feature = self.listener(batch_feature, batch_embedding).permute(0, 2, 1)
        pre_batch_mfcc_feature = self.decoder(middle_feature)
        loss = self.loss(pre_batch_mfcc_feature, batch_mfcc_fature)
        loss.backward()
        loss = loss.item()
        return loss


class RNNembModel(nn.Module):
    def __init__(self, in_dim, fea_dim, emb_dim):
        super(RNNembModel, self).__init__()
        self.encoder = CNNEncoder(in_dim, fea_dim)
        self.listener = RNNEmb(fea_dim, emb_dim)

    def forward(self, batch_mfcc_fature):
        batch_mfcc_fature = batch_mfcc_fature.permute(0, 2, 1)
        batch_feature = self.encoder(batch_mfcc_fature).permute(0, 2, 1)
        emb = self.listener(batch_feature)
        return emb[-1]

