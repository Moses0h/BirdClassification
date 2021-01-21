import torch.nn as nn


# Defining your CNN model
# We have defined the baseline model
class baseline_Net(nn.Module):

    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        # self.b1 = nn.Sequential(
        #     nn.Conv2d(3, 64, 3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b11 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True)
        )
        self.b5 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(256, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True)
        )
        self.fc11 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, classes)
        )

    def forward(self, x):
        # out1 = self.b2(self.b1(x))
        # out2 = self.b4(self.b3(out1))
        out11 = self.b11(self.b1(x))
        out1 = self.b2(out11)
        out2 = self.b4(self.b3(out1))
        out3 = self.b5(out2)
        out2 = out3
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 512)
        out4 = self.fc2(self.fc11(self.fc1(out_flat)))
        #out4 = self.fc2(self.fc1(out_flat))

        return out4
