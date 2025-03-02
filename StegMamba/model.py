import torch.optim
import torch.nn as nn
import config as c
from invnet import InvNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InvModel(nn.Module):
    def __init__(self):
        super(InvModel, self).__init__()
        self.model = InvNet()

    def forward(self, x, rev=False):
        if not rev:
            out = self.model(x)
        else:
            out = self.model(x, rev=True)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class AdjustModel(nn.Module):
    def __init__(self, channel=12):
        super(AdjustModel, self).__init__()
        self.encode_Convx = nn.Sequential(
            nn.Conv2d(channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.encode_Convy = nn.Sequential(
            nn.Conv2d(channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res1 = BasicBlock(inchannel=128, outchannel=128)
        self.res2 = BasicBlock(inchannel=128, outchannel=128)
        self.res3 = BasicBlock(inchannel=128, outchannel=128)
        self.res4 = BasicBlock(inchannel=128, outchannel=128)
        self.res5 = BasicBlock(inchannel=128, outchannel=128)
        self.res6 = BasicBlock(inchannel=128, outchannel=128)
        self.res7 = BasicBlock(inchannel=128, outchannel=128)
        self.res8 = BasicBlock(inchannel=128, outchannel=128)
        self.relu = nn.ReLU()
        self.decode_Conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, channel, 3, padding=1),
        )

    def forward(self, x, y):
        x = self.encode_Convx(x)
        y = self.encode_Convy(y)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.relu(x + y)
        out = self.decode_Conv(x)
        return out


class interact(nn.Module):
    def __init__(self):
        super(interact, self).__init__()
        self.convup = nn.Conv2d(4 * c.channels_in, 32 * 4 * c.channels_in, 3, 1, 1, groups=c.channels_in)
        self.sigmoid = nn.Sigmoid()
        self.convdown = nn.Conv2d(32 * 4 * c.channels_in, 4 * c.channels_in, 3, 1, 1, groups=4 * c.channels_in)

    def forward(self, x, y):
        # 特征交互
        x = self.convup(x)
        y = self.convup(y)
        x_11, x_12 = torch.split(x, x.size(1) // 2, dim=1)
        y_21, y_22 = torch.split(y, y.size(1) // 2, dim=1)
        x_weight = x_11 * y_22
        y_weight = x_12 * y_21
        weight = self.sigmoid(torch.cat([x_weight, y_weight], dim=1))
        x = x * weight
        y = y * weight
        x_out = self.convdown(x)
        y_out = self.convdown(y)
        return x_out, y_out


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.invmodel = InvModel()
        self.adjustmodel = AdjustModel()
        self.interact = interact()

    def forward(self, x):
        #################
        #   preprocess: #
        #################
        cover, secret = (x.narrow(1, 0, 4 * c.channels_in), x.narrow(1, 4 * c.channels_in, x.shape[1] - 4 * c.channels_in))
        acover = cover
        precover, presecret = self.interact(acover, secret)

        #################
        #    forward:   #
        #################
        x = torch.cat((precover, presecret), 1)
        out = self.invmodel(x)
        stego = out.narrow(1, 0, 4 * c.channels_in)
        r = out.narrow(1, 4 * c.channels_in, out.shape[1] - 4 * c.channels_in)

        #################
        #   backward:   #
        #################
        z = 0.5 * torch.ones(r.shape).to(device)
        output_rev = torch.cat((stego, z), 1)
        output_image = self.invmodel(output_rev, rev=True)

        #################
        #   preprocess: #
        #################
        rcover = output_image.narrow(1, 0, 4 * c.channels_in)
        rsecret = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        r = r.reshape([stego.shape[0], -1])
        return stego, r, rcover, rsecret, acover


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.invmodel = InvModel()
        self.adjustmodel = AdjustModel()
        self.interact = interact()

    def forward(self, x):
        #################
        #   preprocess: #
        #################
        cover, secret = (x.narrow(1, 0, 4 * c.channels_in), x.narrow(1, 4 * c.channels_in, x.shape[1] - 4 * c.channels_in))
        acover = self.adjustmodel(cover, secret)
        precover, presecret = self.interact(acover, secret)

        #################
        #    forward:   #
        #################
        x = torch.cat((precover, presecret), 1)
        out = self.invmodel(x)
        stego = out.narrow(1, 0, 4 * c.channels_in)
        r = out.narrow(1, 4 * c.channels_in, out.shape[1] - 4 * c.channels_in)

        #################
        #   backward:   #
        #################
        z = 0.5 * torch.ones(r.shape).to(device)
        output_rev = torch.cat((stego, z), 1)
        output_image = self.invmodel(output_rev, rev=True)

        #################
        #   preprocess: #
        #################
        rcover = output_image.narrow(1, 0, 4 * c.channels_in)
        rsecret = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
        r = r.reshape([stego.shape[0], -1])
        return stego, r, rcover, rsecret, acover


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).to(device)
            if split[-2] == 'conv5':
                param.data.fill_(0.)
