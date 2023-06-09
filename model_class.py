import torch.nn as nn


def conv_block(in_size, out_size, pool=True, stride=1, padding=0, kernel_size=3):
    layers = [nn.Conv2d(in_size, out_size,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
              nn.ReLU(inplace=True),
              nn.BatchNorm2d(out_size)
              ]
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 128, kernel_size=3, padding=1, pool=False)
        self.conv2 = conv_block(128, 256, kernel_size=3)
        self.conv3 = conv_block(256, 512, kernel_size=3, padding=1, pool=False)
        self.conv4 = conv_block(512, 1024, kernel_size=5)
        self.conv5 = conv_block(1024, 2048, kernel_size=5, padding=2, pool=True)
        self.conv6 = conv_block(2048, 5096, kernel_size=5)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(45864, 256),
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
        self.conv1 = conv_block(3, 16, kernel_size=3, pool=False)
        self.conv2 = conv_block(16, 32, kernel_size=3,  pool=False)
        self.conv3 = conv_block(32, 64, kernel_size=3, pool=False)
        self.conv4 = conv_block(64, 128, kernel_size=3, pool=False)
        self.conv5 = conv_block(128, 256, kernel_size=5, pool=False)
        self.conv6 = conv_block(256, 512, kernel_size=5, pool=False)
        self.conv7 = conv_block(512, 1024, kernel_size=5, pool=False)
        self.conv8 = conv_block(1024, 2048,  kernel_size=5, pool=True)
        # self.conv9 = conv_block(2048, 5096, kernel_size=7, pool=False)

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(7872512, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(256, 8)
                                        )
        self.name = "Model_big"

        self.classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        # x = self.conv9(x)
        x = self.classifier(x)

        return x


class Model_small(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(25600, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 150)
                                        )

        self.name = "Model_small"
        self.classes = ['Porygon', 'Goldeen', 'Hitmonlee', 'Hitmonchan', 'Gloom', 'Aerodactyl', 'Mankey', 'Seadra', 'Gengar', 'Venonat', 'Articuno', 'Seaking', 'Dugtrio', 'Machop', 'Jynx', 'Oddish', 'Dodrio', 'Dragonair', 'Weedle', 'Golduck', 'Flareon', 'Krabby', 'Parasect', 'Ninetales', 'Nidoqueen', 'Kabutops', 'Drowzee', 'Caterpie', 'Jigglypuff', 'Machamp', 'Clefairy', 'Kangaskhan', 'Dragonite', 'Weepinbell', 'Fearow', 'Bellsprout', 'Grimer', 'Nidorina', 'Staryu', 'Horsea', 'Electabuzz', 'Dratini', 'Machoke', 'Magnemite', 'Squirtle', 'Gyarados', 'Pidgeot', 'Bulbasaur', 'Nidoking', 'Golem', 'Dewgong', 'Moltres', 'Zapdos', 'Poliwrath', 'Vulpix', 'Beedrill', 'Charmander', 'Abra', 'Zubat', 'Golbat', 'Wigglytuff', 'Charizard', 'Slowpoke', 'Poliwag', 'Tentacruel', 'Rhyhorn', 'Onix', 'Butterfree', 'Exeggcute', 'Sandslash', 'Pinsir', 'Rattata', 'Growlithe', 'Haunter', 'Pidgey', 'Ditto', 'Farfetchd', 'Pikachu', 'Raticate', 'Wartortle', 'Vaporeon', 'Cloyster', 'Hypno', 'Arbok', 'Metapod', 'Tangela', 'Kingler', 'Exeggutor', 'Kadabra', 'Seel', 'Voltorb', 'Chansey', 'Venomoth', 'Ponyta', 'Vileplume', 'Koffing', 'Blastoise', 'Tentacool', 'Lickitung', 'Paras', 'Clefable', 'Cubone', 'Marowak', 'Nidorino', 'Jolteon', 'Muk', 'Magikarp', 'Slowbro', 'Tauros', 'Kabuto', 'Spearow', 'Sandshrew', 'Eevee', 'Kakuna', 'Omastar', 'Ekans', 'Geodude', 'Magmar', 'Snorlax', 'Meowth', 'Pidgeotto', 'Venusaur', 'Persian', 'Rhydon', 'Starmie', 'Charmeleon', 'Lapras', 'Alakazam', 'Graveler', 'Psyduck', 'Rapidash', 'Doduo', 'Magneton', 'Arcanine', 'Electrode', 'Omanyte', 'Poliwhirl', 'Mew', 'Alolan Sandslash', 'Mewtwo', 'Weezing', 'Gastly', 'Victreebel', 'Ivysaur', 'MrMime', 'Shellder', 'Scyther', 'Diglett', 'Primeape', 'Raichu']

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.classifier(x)

        return x
