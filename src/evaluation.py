import numpy as np
import pandas as pd
import torch
import fire
from torch.autograd import Variable
import src.utils as utils
from torchvision import transforms
from src.model import CustomIcebergNet
from torch.utils.data.sampler import SubsetRandomSampler

from src.data import IcebergJsonDataset

USE_GPU = torch.cuda.is_available()


def evaluate(model, batch_size=128, threshold=0.5):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    ds = IcebergJsonDataset(transform=transform)
    test_idx = ds.load_test_data()
    data_loader = torch.utils.data.DataLoader(ds, sampler=SubsetRandomSampler(indices=test_idx), batch_size=batch_size,
                                              num_workers=4)
    results = []
    model.eval()
    criterion = torch.nn.BCELoss()
    running_loss = 0.0
    for data_dict in data_loader:

        # wrap them in Variable
        if USE_GPU:
            data = Variable(data_dict['raw_data'].type(torch.FloatTensor).cuda())
            angle = Variable(data_dict['angle'].type(torch.FloatTensor).cuda())

            target = Variable(data_dict['target'].type(torch.FloatTensor).cuda())
        else:
            data = Variable(data_dict['raw_data'].type(torch.FloatTensor))
            angle = Variable(data_dict['angle'].type(torch.FloatTensor))
            target = Variable(data_dict['target'].type(torch.FloatTensor))
        angle = angle.view(-1, 1)
        target = target.view(-1, 1)  # for pytorch shape has to be same
        # forward
        outputs = model(data,angle)
        loss = criterion(outputs, target)

        # statistics
        running_loss += loss.data[0]

    print("running loss: {}".format(running_loss))
