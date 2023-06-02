import torch.nn as nn


def conv_block(in_size, out_size, pool=True):
    layers = [nn.Conv2d(in_size, out_size, kernel_size=3),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(out_size)
              ]
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512, False)
        self.conv4 = conv_block(512, 1024)
        self.conv5 = conv_block(1024, 2048, False)
        self.conv6 = conv_block(2048, 5096)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(20384, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 8)
                                        )
        self.name = "Model_standard"

        self.classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.classifier(x)

        return x


class Model_big(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 512, False)
        self.conv4 = conv_block(512, 1024)
        self.conv5 = conv_block(1024, 2048, False)
        self.conv6 = conv_block(2048, 5096)
        self.conv7 = conv_block(5096, 10192, False)

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(20384, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 8)
                                        )
        self.name = "Model_big"

        self.classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.classifier(x)

        return x


class Model_small(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512, False)
        self.conv4 = conv_block(512, 1024)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(82944, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 8)
                                        )

        self.name = "Model_small"
        self.classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.classifier(x)

        return x
