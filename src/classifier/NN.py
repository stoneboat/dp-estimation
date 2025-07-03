import logging
import numpy as np
import multiprocessing
from functools import partial
import os

import torch
import torch.nn as nn
from torch.nn import Linear

from torch.utils.data import Dataset, DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import torch.distributed as dist
from datetime import timedelta

torch.set_default_dtype(torch.float64)


#   This class defines the structure of the inverse function via Neural Network
#   I will train its structure to amplify the projected noised marginal distribution to original marginal distribution
class DemoNN_Model(nn.Module):
    def __init__(self, n_features=1):
        super(DemoNN_Model, self).__init__()

        # 512,256 size of units for each hidden layer
        h_size = 128

        # activation function
        # self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU
        self.act = torch.nn.ReLU()

        # number of layer
        self.nl1 = 8

        # create the model template
        self.encoder = torch.nn.ModuleList()
        ## input layer
        self.encoder.append(Linear(n_features, h_size))
        ## hidden layer, number of hidden layer is nl1
        for i in range(self.nl1 - 1):
            self.encoder.append(Linear(h_size, h_size))
        ## output layer
        self.encoder.append(Linear(h_size, 2))

        self.device = None

    def out(self, data):
        # proceed the input layer
        data = self.act(self.encoder[0](data))

        # proceed the hidden layer
        for ii in range(1, self.nl1):
            data = self.act(self.encoder[ii](data))

        # proceed the output layer
        x = self.encoder[-1](data)
        m = torch.nn.Softmax(dim=1)
        x = m(x)
        # x = torch.sigmoid(x)

        return x[:, 0]

    def predict(self, test_features):
        with torch.no_grad():
            test_features = torch.from_numpy(test_features).to(self.device)
            outputs = self.out(test_features)

            self.test_label_output = outputs.cpu().numpy().round().flatten()

        return self.test_label_output

    # initialize the parameters of the network
    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.02)

    def score(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).mean()

    def correct_num(self, test_features, test_labels):
        return (self.predict(test_features) == test_labels).sum()


class DPEstimatorDataset(Dataset):
    def __init__(self, samples, transform=None):
        assert isinstance(samples['X'], np.ndarray)
        assert isinstance(samples['y'], np.ndarray)
        assert samples['X'].shape[0] == samples['y'].shape[0]

        self.data = samples['X']
        self.label = samples['y']
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'data': self.data[idx], 'label': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def l2(inputs, targets):
    return torch.norm(inputs - targets, p=2)


def single_train_NN_process(rank, model, trainset, n_epoch, batch_size, lr, n_batches=100,
                            world_size=1, file_name="nn_files"):
    # Set logger
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.CRITICAL)

    # Initialize the distrubted pytorch
    store = dist.FileStore(file_name, world_size)
    store.set_timeout(timedelta(seconds=10))
    dist.init_process_group(backend="gloo", store=store, world_size=world_size, rank=rank, timeout=timedelta(
        seconds=60))
    dist.barrier()
    logger.critical(f"[train NN] successfully initialized with {rank} with world_size {world_size}")

    # Start training
    sampler_train = DistributedSampler(trainset, shuffle=True, num_replicas=world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, sampler=sampler_train)

    DDP_model = torch.nn.parallel.DistributedDataParallel(model)

    # create optimization method
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)
    optimizer = optim.AdamW(DDP_model.parameters(), lr=lr, weight_decay=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=False)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(n_epoch):
        sampler_train.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['data'], data['label']

            outputs = DDP_model.module.out(inputs)

            # loss = torch.sum(abs(outputs.squeeze() - labels))/outputs.shape[0]

            criterion = nn.BCELoss()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()  # Does the update
            optimizer.zero_grad()  # zero the gradient buffers
            # print statistics
            running_loss += loss.item()

            if i % n_batches == n_batches-1:  # print every 500 mini-batches
                if rank == 0:
                    learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                    logger.critical(f'[representative rank {rank}, epoch {epoch + 1}, batch {int((i + 1)):5d}] average '
                                    f'loss: {running_loss / n_batches:.6f} '
                                    f'learning rate={learning_rate:.9f}')

                running_loss = 0.0

        scheduler.step()

    # clean up
    dist.destroy_process_group()


#   file_name is to store file which is for parallel_model communication
def _train_NN_model(samples, file_name="nn_files", n_epoch=1, batch_size=1000, lr=0.00001, n_batches=100, workers=2,
                    model=None):
    pool = multiprocessing.Pool(processes=workers)
    input_list = np.arange(workers).tolist()

    # Clear up the file first if it exists otherwise the process will stuck
    if os.path.exists(file_name):
        os.remove(file_name)
    # intialize the model
    if model is None:
        model = DemoNN_Model(n_features=samples['X'].shape[1])
        model.apply(model.init_weights)

    trainset = DPEstimatorDataset(samples)
    train_NN_function = partial(single_train_NN_process, model=model, trainset=trainset, n_epoch=n_epoch,
                                batch_size=batch_size, lr=lr, world_size=workers, file_name=file_name, n_batches=n_batches)
    pool.map(train_NN_function, input_list)

    return model
    




