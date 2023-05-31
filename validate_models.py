import os

import torch
from data import get_data

train, val, test = get_data(70, 15, 15)

for model in os.listdir("./models"):
    print(model)