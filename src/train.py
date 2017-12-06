import time

import fire
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from visdom import Visdom

import src.utils as utils
from src import evaluation as eval
from src.data import FileCsvJsonDataset
from src.model import FineTuneImageNet
import src.visualize_graph

BASE_FOLDER = 'data'
IMG_PATH = 'data/images'

IMG_HEIGHT = 75
IMG_WIDTH = 75

USE_GPU = torch.cuda.is_available()

viz = Visdom()
vizgh = src.visualize_graph.Visualize_graph()



def train(batch_size=2, epochs=1, show_vizdom=False, run_eval=False):
    '''
    :param batch_size:
    :param epochs:
    :param show_vizdom: if this is True visdom server should be running in background
    :param run_eval:
    :return:
    '''

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    ds = FileCsvJsonDataset(target_col='',transform=data_transforms['train'])

    if USE_GPU:
        model = FineTuneImageNet(num_classes=1).cuda()
    else:
        model = FineTuneImageNet(num_classes=1)

    run_id = utils.generate_run_id()
    since = time.time()
    best_model_wts = model.state_dict()
    least_loss = 999999.99
    val_loss = 999999.99

    for epoch in range(epochs):

        train_idx, val_idx = ds.train_val_split()

        data_loaders = {
            'train': torch.utils.data.DataLoader(ds, sampler=SubsetRandomSampler(indices=train_idx),
                                                 batch_size=batch_size,
                                                 num_workers=4),
            'val': torch.utils.data.DataLoader(ds, sampler=SubsetRandomSampler(indices=val_idx), batch_size=batch_size,
                                               num_workers=4)
        }
        dataset_sizes = {'train': len(train_idx),
                         'val': len(val_idx)}

        # pbar = tqdm_notebook(train_loader, total=len(train_loader))
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        val_loss, val_acc = train_model(model=model, data_loaders=data_loaders, dataset_sizes=dataset_sizes,
                                        epoch=epoch, show_vizdom=show_vizdom)

        # get validation metric to check best run
        if val_loss < least_loss:
            least_loss = val_loss
            best_model_wts = model.state_dict()
            # torch.save(model.state_dict(), 'generated/model/{}_{}.pth.tar'.format(run_id, epoch))

    torch.save(best_model_wts, 'generated/models/{}.pth.tar'.format(run_id))
    print("Saved model at generated/models/{}.pth.tar".format(run_id))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {}'.format(least_loss))
    if run_eval:
        print("Starting evaluation for: {}".format(run_id))
        eval.evaluate(model=model)
        print("Evaluation ended")


def train_model(model, data_loaders, dataset_sizes, epoch, show_vizdom):
    '''
    :param dataset_sizes:
    :param model:
    :param epoch:
    :param data_loaders:
    :return:
    '''
    # Only finetunable params)
    LR = 1e-4
    MOMENTUM = 0.95
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # BCE for binary classification
    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    metric_rec = {}
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data_dict in data_loaders[phase]:

            if USE_GPU:
                data = Variable(data_dict['raw_data'].type(torch.FloatTensor).cuda())

                target = Variable(data_dict['target'].type(torch.FloatTensor).cuda())
            else:
                data = Variable(data_dict['raw_data'].type(torch.FloatTensor))
                target =Variable(data_dict['target'].type(torch.FloatTensor))

            target = target.view(-1, 1)  # for pytorch shape has to be same, this is only for binary classification
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(data)
            loss = criterion(outputs, target)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # This makes sense for multi label classification because we are
            # trying to find the label with max predicted value
            # _,preds = torch.max(outputs.data, 1)
            preds = outputs.data

            # statistics
            running_loss += loss.data[0]* target.size(0)
            threshold = 0.45
            p_test = (preds).cpu().numpy()  # otherwise we get an array, we need a single float
            p_test = np.squeeze(p_test >= threshold).astype(int)  # otherwise we get an array, we need a single float

            running_corrects += np.sum(p_test == np.squeeze(target.data.cpu().numpy().astype(int)))

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        # # visdom trial
        if show_vizdom:
            vizgh._update_line(x=epoch,
                               y=epoch_loss,
                               viz=viz,
                               title='validation loss'
                               )
            vizgh._update_line(x=epoch,
                               y=epoch_acc,
                               viz=viz,
                               title='validation accuracy'
                               )

        metric_rec[phase] = {'loss': epoch_loss, 'acc': epoch_acc}

    print('train Loss: {:.4f} Acc: {:.4f}===Val Loss: {:.4f} Acc: {:.4f}'.format(metric_rec['train']['loss'],
                                                                                 metric_rec['train']['acc'],
                                                                                 metric_rec['val']['loss'],
                                                                                 metric_rec['val']['acc']))
    return metric_rec['val']['loss'], metric_rec['val']['acc']


if __name__ == '__main__':
    fire.Fire(train)
