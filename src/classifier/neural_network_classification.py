from classifier.NN import DemoNN_Model, DPEstimatorDataset
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import logging


def train_NN_model_flow(samples, nn_classifier_args={}):
    # Set logger
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.CRITICAL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = DPEstimatorDataset(samples)

    model = nn_classifier_args.get("model", None)
    if model is None:
        print("Initializing model")
        model = DemoNN_Model(n_features=samples['X'].shape[1])
        model.apply(model.init_weights)

    model = model.to(device)

    n_epoch = nn_classifier_args.get("n_epoch", 20)
    batch_size = nn_classifier_args.get("batch_size", 500)
    lr = nn_classifier_args.get("lr", 0.00001)
    n_batches = nn_classifier_args.get("n_batches", 100)


    # Start training
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # create optimization method
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=False)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['data'].to(device), data['label'].to(device)
            outputs = model.out(inputs)

            # loss = torch.sum(abs(outputs.squeeze() - labels))/outputs.shape[0]

            criterion = nn.BCELoss()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()  # Does the update
            optimizer.zero_grad()  # zero the gradient buffers
            # print statistics
            running_loss += loss.item()

            if i % n_batches == n_batches-1:  # print every 500 mini-batches
                learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                logger.critical(f'[epoch {epoch + 1}, batch {int((i + 1)):5d}] average '
                                f'loss: {running_loss / n_batches:.6f} '
                                f'learning rate={learning_rate:.9f}')

                running_loss = 0.0

        scheduler.step()
    
    print("Training completed")

    # clean up
    torch.cuda.empty_cache()
    model.device = device

    return model
