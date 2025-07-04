from classifier.NN import DemoNN_Model, DPEstimatorDataset
from classifier.network_architecture import QuadraticNN_Model
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import logging
from tqdm import tqdm


def train_NN_model_flow(samples, nn_classifier_args={}):
    # Set logger
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.CRITICAL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = DPEstimatorDataset(samples)

    model = nn_classifier_args.get("model", None)
    

    n_epoch = nn_classifier_args.get("n_epoch", 20)
    batch_size = nn_classifier_args.get("batch_size", 500)
    lr = nn_classifier_args.get("lr", 0.00001)
    n_batches = nn_classifier_args.get("n_batches", 100)
    model_type = nn_classifier_args.get("model_type", "demo")

    if model is None:
        if model_type == "demo":
            model = DemoNN_Model(n_features=samples['X'].shape[1])
            criterion = nn.BCELoss()
        elif model_type == "quadratic":
            model = QuadraticNN_Model(n_features=samples['X'].shape[1])
            criterion = nn.BCEWithLogitsLoss()
        model.apply(model.init_weights)

    model = model.to(device)


    # Start training
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # create optimization method
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=False)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Calculate total iterations for single progress bar
    total_iterations = n_epoch * len(trainloader)
    
    # Create single progress bar for all iterations
    pbar = tqdm(total=total_iterations, desc=f"Training Progress with {model_type} model", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        epoch_losses = []
        
        for i, data in enumerate(trainloader):
            inputs, labels = data['data'].to(device), data['label'].to(device)
            outputs = model.out(inputs)

            # loss = torch.sum(abs(outputs.squeeze() - labels))/outputs.shape[0]

            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()  # Does the update
            optimizer.zero_grad()  # zero the gradient buffers
            
            # Update running loss
            current_loss = loss.item()
            running_loss += current_loss
            epoch_losses.append(current_loss)
            
            # Update progress bar
            pbar.update(1)
            
            # Update description every n_batches
            if i % n_batches == n_batches - 1:
                avg_loss = running_loss / (i + 1)
                learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
                
                pbar.set_description(
                    f'Epoch {epoch+1}/{n_epoch} | '
                    f'Batch {i+1}/{len(trainloader)} | '
                    f'Avg Loss: {avg_loss:.6f} | '
                    f'LR: {learning_rate:.9f}'
                )
                
                pbar.set_postfix({
                    'Epoch Loss': f'{avg_loss:.6f}',
                    'LR': f'{learning_rate:.9f}'
                })

        # Update epoch summary
        epoch_avg_loss = running_loss / len(trainloader)
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        pbar.set_postfix({
            'Epoch Loss': f'{epoch_avg_loss:.6f}',
            'LR': f'{learning_rate:.9f}'
        })
        
        scheduler.step()
    
    pbar.close()
    print("Training completed")

    # clean up
    torch.cuda.empty_cache()
    model.device = device
    model.eval()

    return model
