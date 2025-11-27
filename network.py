from itertools import chain
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()

    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                if init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun


class FCMNet(nn.Module):
    def __init__(self, ncl=16):
        super(FCMNet, self).__init__()
        self.ncl = ncl

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, self.ncl)
        self.last = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.last(x).squeeze()
        return x

class Network(BaseNet):

    def __init__(self, input_A, input_B, nz, K, CVFN=1):
        super().__init__()
        self.input_A = input_A
        self.input_B = input_B
        self.K = K
        self.nz = nz
        self.encoder1 = nn.Sequential(
            nn.Linear(input_A, 256, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(256, 64, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(32, self.nz//2, bias=True),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_B, 256, bias=False),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(256, 64, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(True),
            nn.Dropout(.5),
            nn.Linear(32, self.nz//2, bias=True),
            nn.ReLU(True),
        )


        self.trans_enc = nn.TransformerEncoderLayer(d_model=self.nz, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1) if CVFN else nn.Linear(self.nz, 10)


        self.cls_layer = nn.Sequential(
            nn.Linear(self.nz, self.K),
            nn.Sigmoid()
        )

        self.layer4 = nn.Linear(self.nz, 300)
        self.layer5_1 = nn.Linear(300, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(300, input_B)
        self.drop = 0.5

        self.sigmoid = nn.Sigmoid()

        self.apply(self.weights_init('xavier'))
        self.flatten = nn.Flatten()
        self.recon_criterion = nn.MSELoss()
        self.cls_criterion = nn.CrossEntropyLoss()


    def forward(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        x = self.extract_layers(torch.cat((x1, x2), 1))
        # out: (batch size, length, d_model)
        y = self.cls_layer(x)

        return y

    def decoder(self, latent):
        x = F.dropout(F.relu(self.layer4(latent)), self.drop)
        out1 = F.relu(self.layer5_1(x))
        out1 = self.layer6_1(out1)
        out2 = self.layer6_2(x)
        return out1, out2

    def get_loss(self, Xs, labels=None):
        if labels is not None:
            y = self(Xs)
            cls_loss = self.cls_criterion(y, labels)
            return cls_loss
        else:
            latent = self.test_commonZ(Xs)
            recon1, recon2 = self.decoder(latent)
            recon_loss = 0.5 * self.recon_criterion(recon2, Xs[1]).mean(0).sum() + \
                         0.5 * self.recon_criterion(recon1, Xs[0]).mean(0).sum()
            return recon_loss

    def test_commonZ(self, Xs):
        x1, x2 = Xs
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        latent = self.extract_layers(torch.cat((x1, x2), 1))
        # out: (batch size, length, d_model)

        return latent

    def get_cls_optimizer(self, lr=1e-3):
        self.cls_optimizer = torch.optim.SGD(chain(self.encoder1.parameters(), self.encoder2.parameters(),
                                                   self.extract_layers.parameters(), self.cls_layer.parameters()),
                                             lr=lr,
                                             momentum=0.9,
                                             weight_decay=5e-4)
        return self.cls_optimizer

    def get_recon_optimizer(self, lr=1e-3):
        self.recon_optimizer = torch.optim.SGD(self.parameters(),
                                               lr=lr,
                                               momentum=0.9,
                                               weight_decay=5e-4,
                                               )
        return self.recon_optimizer
