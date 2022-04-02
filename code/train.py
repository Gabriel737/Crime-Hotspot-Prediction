from tqdm import tqdm
import numpy as np
from sklearn.metrics import recall_score, precision_score
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loader import FeatureDataset
import config
from model import CNNLSTM
from loss import classification_loss

def validate(dl, model, batch_size, epoch ,best_recall, best_model, writer):
    '''
    The validation loop
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    all_outputs = list()
    all_outputs_probs = list()
    all_targets = list()

    for X, y_reg, y_bin in tqdm(dl):
        if y_reg.shape[0] == batch_size:

            pred_scores, pred_reg = model(X)
            bce_loss = classification_loss(pred_scores, y_bin)
            # mse_loss = regression_loss(pred_reg, y_reg)
            # loss = total_loss(bce_loss,mse_loss, task_num=2)

            pred_bin = (pred_scores > config.CLASS_THRESH).float()
            all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
            all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
            all_outputs_probs.append(pred_scores.view(-1,1).detach().cpu().numpy())
            total += y_reg.shape[0]
            epoch_loss += bce_loss.item()
        
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    all_outputs_probs = np.concatenate(all_outputs_probs)
    recall = recall_score(y_pred=all_outputs,y_true=all_targets)
    precision = precision_score(y_pred=all_outputs,y_true=all_targets)
    writer.add_pr_curve('pr_curve', all_targets, all_outputs_probs, global_step=0)
    writer.close()

    if epoch == 0:
        best_model = model
        best_recall = recall
    else:
        if recall > best_recall:
            best_recall = recall
            best_model = model

    avg_loss = epoch_loss/total

    print(f'Validation Recall Score: {recall}')
    print(f'Validation Precision Score: {precision}')

    return best_model, best_recall, recall, avg_loss


def test(dl, model, batch_size):
    '''
    The testing loop
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    all_outputs = list()
    all_targets = list()

    for X, y_reg, y_bin in tqdm(dl):
        if y_reg.shape[0] == batch_size:

            pred_scores, pred_reg = model(X)
            bce_loss = classification_loss(pred_scores, y_bin)
            # mse_loss = regression_loss(pred_reg, y_reg)
            # loss = total_loss(bce_loss,mse_loss, task_num=2)

            pred_bin = (pred_scores > config.CLASS_THRESH).float()
            all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
            all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
            total += y_reg.shape[0]
            epoch_loss += bce_loss.item()
        
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    recall = recall_score(y_pred=all_outputs,y_true=all_targets)
    precision = precision_score(y_pred=all_outputs,y_true=all_targets)

    avg_loss = epoch_loss/total

    print(f'Test Recall Score: {recall}')
    print(f'Test Precision Score: {precision}')
    print(f'Test Loss: {avg_loss}')

    return recall, avg_loss


def train(train_dl, val_dl, model, optim, epochs, batch_size, save, start_epoch=0, model_save_path=None):
    '''
    The training loop
    '''
    writer = SummaryWriter('crimepred')
    best_model = model
    best_val_recall = 0.0

    for epoch in range(start_epoch,epochs):

        print(f'Epoch: {epoch}')
        epoch_loss = 0.0
        total = 0
        all_outputs = list()
        all_targets = list()
        model.train()

        for X, y_reg, y_bin in tqdm(train_dl):
            if y_reg.shape[0] == batch_size:

                pred_scores, pred_reg = model(X)
                bce_loss = classification_loss(pred_scores, y_bin)
                # mse_loss = regression_loss(pred_reg, y_reg)
                # loss = total_loss(bce_loss,mse_loss,task_num=2)

                optim.zero_grad()
                bce_loss.backward()
                optim.step()
                pred_bin = (pred_scores > config.CLASS_THRESH).float() 
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
                total += y_reg.shape[0]
                epoch_loss += bce_loss.item()

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        recall = recall_score(y_pred=all_outputs,y_true=all_targets)
        precision = precision_score(y_pred=all_outputs,y_true=all_targets)
        avg_loss = epoch_loss/total

        print(f'Train Recall Score: {recall}')
        print(f'Train Precision Score: {precision}')
        print(f'Train Loss: {avg_loss}')

        best_model, best_val_recall, val_recall, val_avg_loss = validate(val_dl, model, batch_size, epoch, best_val_recall, best_model, writer)

        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Val', val_avg_loss, epoch)
        writer.add_scalar('Recall Score/Train', recall, epoch)
        writer.add_scalar('Recall Score/Val', val_recall, epoch)

        if save:
            print('Saving model')
            checkpoint = {
                'model': model.state_dict(),
                'optim': optim.state_dict,
                'epoch': epoch     
            }
            torch.save(checkpoint,model_save_path+f'/model_checkpoint_{optim_name}_{config.LR}_{config.TRAIN_BATCH_SIZE}_{config.CLASS_THRESH}_{config.RANDOM_SEED}.pt')
    writer.close()
    return best_model, best_val_recall


if __name__ == '__main__':
    
    start_time = time.time()
    start_epoch = 0

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    train_data = FeatureDataset(feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                                target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                                device=device,
                                name = 'train')
    
    val_data = FeatureDataset(feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                              target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                              device=device,
                              name='val') 

    test_data = FeatureDataset(feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                               target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                               device=device,
                               name='test') 

    train_loader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE)
    val_loader = DataLoader (val_data, batch_size=config.TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.TRAIN_BATCH_SIZE)

    model_save_path = config.MODEL_SAVE_PATH
    path = Path(model_save_path)
    path.mkdir(exist_ok=True)

    model = CNNLSTM(n_input_channels=len(config.CRIME_CATS), embed_size=2304, batch_size=config.TRAIN_BATCH_SIZE, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    optim_name = type(optim).__name__

    try:
        checkpoint = torch.load(model_save_path+f'/model_checkpoint_{optim_name}_{config.LR}_{config.TRAIN_BATCH_SIZE}_{config.CLASS_THRESH}_{config.RANDOM_SEED}.pt')
        model.load_state_dict(checkpoint['model'])
        print('\n Model Loaded \n')
        start_epoch = checkpoint['epoch'] + 1
    except:
        pass
    model.to(device)

    try:
        optim.load_state_dict(checkpoint['optim'])
        print('\n Optimizer Loaded \n')
    except:
        pass

    save = config.SAVE

    print('\n Training Starts \n')

    best_model, best_val_recall = train(train_dl=train_loader,
                                        val_dl=val_loader,
                                        model=model,
                                        optim=optim,
                                        epochs=config.N_EPOCHS,
                                        batch_size=config.TRAIN_BATCH_SIZE,
                                        save=save,
                                        start_epoch=start_epoch,
                                        model_save_path=model_save_path
                                        )
    
    test_loss, test_recall = test(dl=test_loader, model=best_model, batch_size=config.TRAIN_BATCH_SIZE)

    end_time = time.time()
    print('\n Finished in: {0:.6f} mins = {1:.6f} seconds'.format(((end_time - start_time)/60), (end_time - start_time)))












