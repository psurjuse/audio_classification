#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:08:54 2023

@author: posu1093
"""



import os
import sys
import pickle
import copy

from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
     
#TODO
us8k_df = pd.read_pickle("/home/posu1093/MP/fft_512_us8k_df_ds4000.pkl")



################################################
#DEFAULT_SAMPLE_RATE = 4000

class UrbanSound8kDataset(Dataset):
    def __init__(self, us8k_df, transform=None):
        assert isinstance(us8k_df, pd.DataFrame)
        assert len(us8k_df.columns) == 3

        self.us8k_df = us8k_df
        self.transform = transform

    def __len__(self):
        return len(self.us8k_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, label, fold = self.us8k_df.iloc[index]

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return {'spectrogram': spectrogram, 'label':label}
    
    
###########################################################

class MyRightShift(object):
    """Shift the image to the right in time."""

    def __init__(self, input_size, width_shift_range, shift_probability=1.0):
        assert isinstance(input_size, (int, tuple))
        assert isinstance(width_shift_range, (int, float))
        assert isinstance(shift_probability, (float))

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        if isinstance(width_shift_range, int):
            assert width_shift_range > 0
            assert width_shift_range <= self.input_size[1]
            self.width_shift_range = width_shift_range
        else:
            assert width_shift_range > 0.0
            assert width_shift_range <= 1.0
            self.width_shift_range = int(width_shift_range * self.input_size[1])
                        
        assert shift_probability > 0.0 and shift_probability <= 1.0
        self.shift_prob = shift_probability

    def __call__(self, image):
        if np.random.random() > self.shift_prob:
          return image

        # create a new array filled with the min value
        shifted_image= np.full(self.input_size, np.min(image), dtype='float32')

        # randomly choose a start postion
        rand_position = np.random.randint(1, self.width_shift_range)

        # shift the image
        shifted_image[:,rand_position:] = copy.deepcopy(image[:,:-rand_position])

        return shifted_image

class MyAddGaussNoise(object):
    """Add Gaussian noise to the spectrogram image."""

    def __init__(self, input_size, mean=0.0, std=None, add_noise_probability=1.0):
        assert isinstance(input_size, (int, tuple))
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float)) or std is None
        assert isinstance(add_noise_probability, (float))


        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            assert len(input_size) == 2
            self.input_size = input_size

        self.mean = mean

        if std is not None:
            assert std > 0.0
            self.std = std
        else:
            self.std = std

        assert add_noise_probability > 0.0 and add_noise_probability <= 1.0
        self.add_noise_prob = add_noise_probability


    def __call__(self, spectrogram):
      if np.random.random() > self.add_noise_prob:
          return spectrogram

      # set some std value 
      min_pixel_value = np.min(spectrogram)
      if self.std is None:
        std_factor = 0.03     # factor number 
        std = np.abs(min_pixel_value*std_factor)

      # generate a white noise spectrogram
      gauss_mask = np.random.normal(self.mean, 
                                    std, 
                                    size=self.input_size).astype('float32')
      
      # add white noise to the sound spectrogram
      noisy_spectrogram = spectrogram + gauss_mask

      return noisy_spectrogram

class MyReshape(object):
    """Reshape the image array."""

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))

        self.output_size = output_size

    def __call__(self, image):
      return image.reshape(self.output_size)
  
####################################################

# build transformation pipelines for data augmentation
train_transforms = transforms.Compose([MyRightShift(input_size=128, 
                                                    width_shift_range=13, 
                                                    shift_probability=0.9),
                                       MyAddGaussNoise(input_size=128,
                                                       add_noise_probability=0.55),
                                       MyReshape(output_size=(1,128,128))])

test_transforms = transforms.Compose([MyReshape(output_size=(1,128,128))])

#############################################################


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x
    
    def fit(self, train_loader, epochs, val_loader=None):
        history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}

        for epoch in range(epochs):
            self.train()

            print("\nEpoch {}/{}".format(epoch+1, epochs))

            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for step, batch in enumerate(train_loader):
                    X_batch = batch['spectrogram'].to(self.device)
                    y_batch = batch['label'].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        # forward + backward
                        outputs = self.forward(X_batch)
                        batch_loss = self.criterion(outputs, y_batch)
                        batch_loss.backward()

                        # update the parameters
                        self.optimizer.step() 

                    pbar.update(1)        
            
            # model evaluation - train data
            train_loss, train_acc = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
        return history

    def predict(self, X):
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(X)

        return outputs

    def evaluate(self, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

        for step, batch in enumerate(data_loader):
            X_batch = batch['spectrogram'].to(self.device)
            y_batch = batch['label'].to(self.device)
            
            outputs = self.predict(X_batch)

            # get batch loss
            loss = self.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
            predictions = torch.argmax(outputs, dim=1) 
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)
            
        loss = running_loss.item() / (step+1)
        accuracy = running_acc.item() / (step+1)
        
        return loss, accuracy


# determine if the system supports CUDA
if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")
print(device)

net = Net(device).to(device)
print(net)

##########################################################


def init_model():
    # determine if the system supports CUDA
    if torch.cuda.is_available():
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")

    # init model
    net = Net(device).to(device)
  
    return net


################################################################



def normalize_data(train_df, test_df):
    # compute the mean and std (pixel-wise)
    mean = train_df['melspectrogram'].mean()
    std = np.std(np.stack(train_df['melspectrogram']), axis=0)

    # normalize train set
    train_spectrograms = (np.stack(train_df['melspectrogram']) - mean) / std
    train_labels = train_df['label'].to_numpy()
    train_folds = train_df['fold'].to_numpy()
    train_df = pd.DataFrame(zip(train_spectrograms, train_labels, train_folds), columns=['melspectrogram', 'label', 'fold'])

    # normalize test set
    test_spectrograms = (np.stack(test_df['melspectrogram']) - mean) / std
    test_labels = test_df['label'].to_numpy()
    test_folds = test_df['fold'].to_numpy()
    test_df = pd.DataFrame(zip(test_spectrograms, test_labels, test_folds), columns=['melspectrogram', 'label', 'fold'])

    return train_df, test_df
     
###############################################################

def process_fold(fold_k, dataset_df, epochs=100, batch_size=32, num_of_workers=0):
    # split the data
    train_df = dataset_df[dataset_df['fold'] != fold_k]
    test_df = dataset_df[dataset_df['fold'] == fold_k]

    # normalize the data
    train_df, test_df = normalize_data(train_df, test_df)

    # init train data loader
    train_ds = UrbanSound8kDataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              shuffle = True,
                              pin_memory=True,
                              num_workers=num_of_workers)
    
    # init test data loader
    test_ds = UrbanSound8kDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_ds, 
                            batch_size=batch_size,
                            shuffle = False,
                            pin_memory=True,
                            num_workers=num_of_workers)

    # init model
    model = init_model()

    # pre-training accuracy
    score = model.evaluate(test_loader)
    print("Pre-training accuracy: %.4f%%" % (100 * score[1]))
      
    # train the model
    start_time = datetime.now()
    history = model.fit(train_loader, epochs=epochs, val_loader=test_loader)
    end_time = datetime.now() - start_time
    print("\nTraining completed in time: {}".format(end_time))
      
    return history


###################################################################
def show_results(history, fold_number):
    """Show accuracy and loss graphs for train and test sets."""

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.grid(linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(122)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.grid(linestyle='--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    max_accuracy = np.max(history['val_accuracy']) * 100
    min_loss = np.min(history['val_loss'])

    plt.text(0.5, 0.95, f'Max validation accuracy: {max_accuracy:.4f}%', 
             transform=plt.gcf().transFigure,
             ha="center", va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             fontsize=12)
    
    plt.text(0.5, 0.90, f'Min validation loss: {min_loss:.5f}', 
             transform=plt.gcf().transFigure,
             ha="center", va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             fontsize=12)
    #TODO: change folder name
    figure_filename = f"/home/posu1093/MP/history_DS_4000/figures/history{fold_number}.png"
    plt.savefig(figure_filename)
    plt.clf()  # Clear the current figure to avoid overlapping plots

    print(f'\tMax validation accuracy: {max_accuracy:.4f}%')
    print(f'\tMin validation loss: {min_loss:.5f}')


############### run training 2 times each folder ############

REPEAT = 2

for FOLD_K in range(1, 11):  # Loop through FOLD_K from 1 to 10
    print('-' * 80)
    print(f"\n(FOLD_K = {FOLD_K})\n")
    
    history_list = []

    for i in range(REPEAT): 
        print(f'\t({i + 1})')
        history = process_fold(FOLD_K, us8k_df, epochs=100, num_of_workers=4)
        history_list.append(history)

    # Display and save results
    for i, history in enumerate(history_list):
        print(f'\n({i + 1})')
        show_results(history, FOLD_K)

        # Save history as pickle
        #TODO: change folder path
        history_filename = f"/home/posu1093/MP/history_DS_4000/history_pickle/history{FOLD_K}_{i + 1}.pkl"
        with open(history_filename, "wb") as fp:
            pickle.dump(history, fp)

print("Training and evaluation completed for FOLD_K from 1 to 10.")

     
#################################################################
'''
LOAD_HISTORY = True

if LOAD_HISTORY:
    with open("/home/posu1093/MP/history/history_pickle/history1.pkl", "rb") as fp:   
        history1 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history2.pkl", "rb") as fp:   
        history2 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history3.pkl", "rb") as fp:   
        history3 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history4.pkl", "rb") as fp:   
        history4 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history5.pkl", "rb") as fp:   
        history5 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history6.pkl", "rb") as fp:   
        history6 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history7.pkl", "rb") as fp:   
        history7 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history8.pkl", "rb") as fp:   
        history8 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history9.pkl", "rb") as fp:   
        history9 = pickle.load(fp)

    with open("/home/posu1093/MP/history/history_pickle/history10.pkl", "rb") as fp:   
        history10 = pickle.load(fp)


tot_history = [history1, history2, history3, history4, history5, history6, history7, history8, history9, history10]


avg_train_loss_per_fold = []
avg_val_loss_per_fold = []
std_train_loss_per_fold = []
std_val_loss_per_fold = []

tot_min_train_loss = 0.0
tot_min_val_loss = 0.0

# iterate over all folds
for fold_histories in tot_history:
    fold_min_train_losses = []
    fold_min_val_losses = []

    # collect min losses of all fold's histories
    for history in fold_histories:
        fold_min_train_losses.append(np.min(history['loss']))
        fold_min_val_losses.append(np.min(history['val_loss']))

    # avg min loss
    avg_train_loss_per_fold.append(np.mean(fold_min_train_losses))
    avg_val_loss_per_fold.append(np.mean(fold_min_val_losses))
    # std of min loss
    std_train_loss_per_fold.append(np.std(fold_min_train_losses))
    std_val_loss_per_fold.append(np.std(fold_min_val_losses))
    
    # add fold's avg min loss to sum of total loss 
    tot_min_train_loss += np.mean(fold_min_train_losses)
    tot_min_val_loss += np.mean(fold_min_val_losses)

avg_min_train_loss = tot_min_train_loss / len(tot_history)
avg_min_val_loss = tot_min_val_loss / len(tot_history)

print("10-Fold Cross Validation\n")
print("Average train min loss: %.4f" % avg_min_train_loss)
print("Average validation min loss: %.4f" % avg_min_val_loss)
#####################################


print("train set (folds):", avg_train_loss_per_fold)
print("validation set (folds):", avg_val_loss_per_fold, '\n')

df = pd.DataFrame(data=zip(list(range(1,11)), avg_train_loss_per_fold, avg_val_loss_per_fold), columns=['fold', 'train', 'validation'])
df = pd.melt(df, id_vars="fold", var_name="set", value_name="loss")

sns_plot = sns.catplot(x='fold', y='loss', hue='set', data=df, kind='bar', height=5.27, aspect=10.7/5.27)
sns_plot.savefig('/home/posu1093/MP/history/CV_loss.png')


###########################################################


avg_train_acc_per_fold = []
avg_val_acc_per_fold = []
std_train_acc_per_fold = []
std_val_acc_per_fold = []

tot_max_train_acc = 0.0
tot_max_val_acc = 0.0

# iterate over all folds
for fold_histories in tot_history:
    fold_max_train_accs = []
    fold_max_val_accs = []

    # collect max accuracies of all fold's histories
    for history in fold_histories:
        fold_max_train_accs.append(np.max(history['accuracy']))
        fold_max_val_accs.append(np.max(history['val_accuracy']))
    
    # avg max accuracy
    avg_train_acc_per_fold.append(np.mean(fold_max_train_accs))
    avg_val_acc_per_fold.append(np.mean(fold_max_val_accs))
    # std of max accuracy
    std_train_acc_per_fold.append(np.std(fold_max_train_accs))
    std_val_acc_per_fold.append(np.std(fold_max_val_accs))
        
    # add fold's avg max accuracy to sum of total accuracy 
    tot_max_train_acc += np.mean(fold_max_train_accs)
    tot_max_val_acc += np.mean(fold_max_val_accs)
  
avg_max_train_acc = tot_max_train_acc / len(tot_history)
avg_max_val_acc = tot_max_val_acc / len(tot_history)

print("10-Fold Cross Validation\n")
print("Average train max accuracy: %.4f %%" % (avg_max_train_acc * 100))
print("Average validation max accuracy: %.4f %%" % (avg_max_val_acc * 100))
     

print("train set (folds):", avg_train_acc_per_fold)
print("validation set (folds):", avg_val_acc_per_fold, '\n')

df = pd.DataFrame(data=zip(list(range(1,11)), avg_train_acc_per_fold, avg_val_acc_per_fold), columns=['fold', 'train', 'validation'])
df = pd.melt(df, id_vars="fold", var_name="set", value_name="accuracy")

sns_plot = sns.catplot(x='fold', y='accuracy', hue='set', data=df, kind='bar', height=5.27, aspect=10.7/5.27)
sns_plot.savefig('/home/posu1093/MP/history/CV_accuracy.png')

print("std_train_loss_per_fold: ", std_train_loss_per_fold)
print("std_val_loss_per_fold: ", std_val_loss_per_fold)


print("std_train_acc_per_fold: ", std_train_acc_per_fold)
print("std_val_acc_per_fold: ", std_val_acc_per_fold)'''
     