from tqdm import tqdm
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loader import FeatureDataset
import config
from model import HotspotPredictor
from loss import classification_loss

def validate(dl, model, batch_size, epoch ,best_f1, best_model, best_epoch, writer):
    '''
    Validation loop

    Inputs - dl: validation data loader
             model: trained model
             batch_size: batch size
             epoch: current epoch number
             best_recall: current best recall score
             best model: current best model
             writer: tensorboard summary writer
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    all_outputs = list()
    all_outputs_probs = list()
    all_targets = list()

    with torch.no_grad():
        for i, (X_crime, X_sec, y_bin) in enumerate(tqdm(dl)):
            if y_bin.shape[0] == batch_size:

                pred_scores = model(X_crime,X_sec)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = classification_loss(pred_scores, y_bin)

                if i == 0 or i == 1:
                    pred_bin_sum = (pred_scores > config.CLASS_THRESH).float().sum()
                    print(f'Pred Score Sum - {i}: {pred_bin_sum}')

                pred_bin = (pred_scores > config.CLASS_THRESH).float()
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
                all_outputs_probs.append(pred_scores.view(-1,1).detach().cpu().numpy())
                total += y_bin.shape[0]
                epoch_loss += wce_loss.item()
            
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        all_outputs_probs = np.concatenate(all_outputs_probs)
        recall = recall_score(y_pred=all_outputs,y_true=all_targets)
        precision = precision_score(y_pred=all_outputs,y_true=all_targets)
        f1score = f1_score(y_pred=all_outputs, y_true=all_targets)
        writer.add_pr_curve('pr_curve', all_targets, all_outputs_probs, global_step=0)
        writer.close()

        if epoch == 0:
            best_model = model
            best_f1 = f1score
            best_epoch = 0
        else:
            if f1score > best_f1:
                best_f1 = f1score
                best_model = model
                best_epoch = epoch

        avg_loss = epoch_loss/total

    print(f'Validation Recall Score: {recall}')
    print(f'Validation Precision Score: {precision}')

    return best_model, best_f1, best_epoch, f1score, recall, precision, avg_loss


def test(dl, model, batch_size):
    '''
    Testing loop
    Inputs - dl: test data loader
             model: trained model
             batch_size: batch size
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    pred_bin_total = 0
    total_cells = 0
    all_outputs = list()
    all_targets = list()

    with torch.no_grad():
        for X_crime, X_sec, y_bin in tqdm(dl):
            if y_bin.shape[0] == batch_size:

                pred_scores = model(X_crime,X_sec)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = classification_loss(pred_scores, y_bin)

                pred_bin = (pred_scores > config.CLASS_THRESH).float()
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
                total += y_bin.shape[0]
                epoch_loss += wce_loss.item()
                pred_bin_total += pred_bin.sum()
                total_cells += int(config.TRAIN_BATCH_SIZE*config.CELL_COUNT*config.CELL_COUNT)
            
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        recall = recall_score(y_pred=all_outputs,y_true=all_targets)
        f1score = f1_score(y_pred=all_outputs, y_true=all_targets)
        precision = precision_score(y_pred=all_outputs,y_true=all_targets)

        avg_loss = epoch_loss/total
        avg_per_pred_bin = (pred_bin_total/total_cells) * 100

    print(f'Test Recall Score: {recall}')
    print(f'Test Precision Score: {precision}')
    print(f'Test F1 Score: {f1score}')
    print(f'Test Loss: {avg_loss}')
    print(f'Average % Predicted Hotspots: {avg_per_pred_bin}')

    return avg_loss, f1score, recall, precision


def train(train_dl, val_dl, model, optim, epochs, batch_size, save, start_epoch=0, model_save_path=None):
    '''
    Training loop
    Inputs - train_dl: training data loader
             val_dl: validation data loader
             model: model to be trained
             optim: optimser
             epochs: total number of epochs
             batch_size: batch size
             save: whether to save model checkpoints
             start_epoch: starting epoch number
             model_save_path: path to save trained model
    '''
    optim_name = type(optim).__name__
    # model_name = model.__class__.__name__
    writer = SummaryWriter(comment='-optim-({})_lr-({})_bs-({})_thres-({})_rs-({})-nepoch-({})_wcew-({})'
                                .format(optim_name, config.LR, config.TRAIN_BATCH_SIZE, 
                                        config.CLASS_THRESH, config.RANDOM_SEED, config.N_EPOCHS, config.CROSS_ENT_LOSS_WEIGHTS))
    best_model = model
    best_epoch = 0
    best_f1 = 0.0

    for epoch in range(start_epoch,epochs):

        print(f'Epoch: {epoch}')
        epoch_loss = 0.0
        total = 0
        all_outputs = list()
        all_targets = list()
        model.train()

        for X_crime, X_sec, y_bin in tqdm(train_dl):
            if y_bin.shape[0] == batch_size:

                pred_scores = model(X_crime,X_sec)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = classification_loss(pred_scores, y_bin)                

                optim.zero_grad()
                wce_loss.backward()
                optim.step()
                pred_bin = (pred_scores > config.CLASS_THRESH).float() 
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(y_bin.view(-1,1).detach().cpu().numpy())
                total += y_bin.shape[0]
                epoch_loss += wce_loss.item()
        scheduler.step()

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        recall = recall_score(y_pred=all_outputs,y_true=all_targets)
        precision = precision_score(y_pred=all_outputs,y_true=all_targets)
        f1score = f1_score(y_pred=all_outputs,y_true=all_targets)

        avg_loss = epoch_loss/total

        print(f'Train Recall Score: {recall}')
        print(f'Train Precision Score: {precision}')
        print(f'Train Loss: {avg_loss}')

        best_model, best_f1, best_epoch, val_f1, val_recall, val_precision, val_avg_loss = validate(val_dl, model, batch_size, epoch, best_f1, best_model, best_epoch, writer)

        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Val', val_avg_loss, epoch)
        writer.add_scalar('Recall Score/Train', recall, epoch)
        writer.add_scalar('Recall Score/Val', val_recall, epoch)
        writer.add_scalar('Precision Score/Train', precision, epoch)
        writer.add_scalar('Precision Score/Val', val_precision, epoch)
        writer.add_scalar('F1 Score/Train', f1score, epoch)
        writer.add_scalar('F1 Score/Val', val_f1, epoch)


        if save:
            print('Saving model')
            checkpoint = {
                'model': model.state_dict(),
                'optim': optim.state_dict,
                'epoch': epoch     
            }
            torch.save(checkpoint,model_save_path+f'/model_checkpoint_optim-({optim_name})_lr-({config.LR})_bs-({config.TRAIN_BATCH_SIZE})_thres-({config.CLASS_THRESH})_rs-({config.RANDOM_SEED})-nepoch-({config.N_EPOCHS})_wcew-({config.CROSS_ENT_LOSS_WEIGHTS}).pt')
    writer.close()

    return best_model, best_f1, best_epoch


if __name__ == '__main__':
    
    start_time = time.time()
    start_epoch = 0

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)

    train_data = FeatureDataset(crime_feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                                sec_feat_data_path=config.VAN_DATA_PRCD+'/sec_features.h5',
                                target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                                device=device,
                                name = 'train')
    
    val_data = FeatureDataset(crime_feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                                sec_feat_data_path=config.VAN_DATA_PRCD+'/sec_features.h5',
                                target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                                device=device,
                                name = 'val')

    test_data = FeatureDataset(crime_feat_data_path=config.VAN_DATA_PRCD+'/features.h5',
                                sec_feat_data_path=config.VAN_DATA_PRCD+'/sec_features.h5',
                                target_data_path=config.VAN_DATA_PRCD+'/targets.h5',
                                device=device,
                                name = 'test')

    train_loader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE)
    val_loader = DataLoader (val_data, batch_size=config.TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.TRAIN_BATCH_SIZE)

    model_save_path = config.MODEL_SAVE_PATH
    path = Path(model_save_path)
    path.mkdir(exist_ok=True)

    model = HotspotPredictor(input_dim=len(config.CRIME_CATS), hidden_dim=config.HIDDEN_DIM, kernel_size=config.KERNEL_SIZE,bias=True)
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=config.LR, max_lr=6*config.LR,cycle_momentum=False,step_size_up=500)
    optim_name = type(optim).__name__
    model_name = model.__class__.__name__

    try:
        checkpoint = torch.load(model_save_path+f'/model_checkpoint_optim-({optim_name})_lr-({config.LR})_bs-({config.TRAIN_BATCH_SIZE})_thres-({config.CLASS_THRESH})_rs-({config.RANDOM_SEED})-nepoch-({config.N_EPOCHS})_wcew-({config.CROSS_ENT_LOSS_WEIGHTS}.pt')
        model.load_state_dict(checkpoint['model'])
        print('\n Model Loaded \n')
        start_epoch = checkpoint['epoch'] + 1
    except:
        pass
    model.to(device)
    model_name = model.__class__.__name__

    try:
        optim.load_state_dict(checkpoint['optim'])
        print('\n Optimizer Loaded \n')
    except:
        pass

    save = config.SAVE

    print('\n Training Starts \n')

    best_model, best_f1, best_epoch = train(train_dl=train_loader,
                                            val_dl=val_loader,
                                            model=model,
                                            optim=optim,
                                            epochs=config.N_EPOCHS,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            save=save,
                                            start_epoch=start_epoch,
                                            model_save_path=model_save_path)
                
    test_loss, _ ,test_recall, _ = test(dl=test_loader, model=best_model, batch_size=config.TRAIN_BATCH_SIZE)

    print('\n Saving best model \n')
    print(f'Best model saved at {best_epoch} epoch')

    final_checkpoint = {'model':best_model.state_dict(), 'epoch':best_epoch}
    torch.save(final_checkpoint,model_save_path+f'/best_model_optim-({optim_name})_lr-({config.LR})_bs-({config.TRAIN_BATCH_SIZE})_thres-({config.CLASS_THRESH})_rs-({config.RANDOM_SEED})-nepoch-({config.N_EPOCHS})_wcew-({config.CROSS_ENT_LOSS_WEIGHTS}.pt')

    end_time = time.time()
    print('\n Finished in: {0:.6f} mins = {1:.6f} seconds'.format(((end_time - start_time)/60), (end_time - start_time)))












