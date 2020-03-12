import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from keras.models import Sequential
from keras.layers import Dense, Dropout

# AniCor
def zscore(data, axis):
    data -= data.mean(axis=axis, keepdims=True)
    data /= data.std(axis=axis, keepdims=True)
    return np.nan_to_num(data, copy=False)

def correlation(matrix1, matrix2):
    d1 = matrix1.shape[-1]
    d2 = matrix2.shape[-1]

    assert d1 == d2
    assert matrix1.ndim <= 2
    assert matrix2.ndim <= 2
    
    matrix1 = zscore(matrix1.astype(float), matrix1.ndim - 1) / np.sqrt(d1)
    matrix2 = zscore(matrix2.astype(float), matrix2.ndim - 1) / np.sqrt(d2)
    
    if matrix1.ndim >= matrix2.ndim:
        return np.dot(matrix1, matrix2.T)
    else:
        return np.dot(matrix2, matrix1.T)

def load_data(filepath):
    nib_format = nib.load(filepath)
    data = nib_format.get_fdata()
    return data

def read_stimulus_csv(stimulus_path):
    fixations = pd.read_csv(stimulus_path)
    pos_x = (1 + np.repeat(np.array(fixations['pos_x']), 5)) / 2
    pos_y = (1 + np.repeat(np.array(fixations['pos_y']), 5)) / 2
    pos = np.asarray(np.vstack((pos_x, pos_y)))
    return pos

def preprocess_data(data):
    # apply eye mask to each the volume at each TR
    data_preprocessed = []
    for vol in range(data.shape[3]):
        data_preprocessed.append(data[...,vol].ravel()[np.where(eye_mask.ravel() == 1)[0]])
    data_preprocessed = np.array(data_preprocessed)

    # z-score across time
    vmean = np.mean(data_preprocessed, axis=0) # TR axis becomes first axis when applying mask
    vstdv = np.std(data_preprocessed, axis=0)
    vstdv[vstdv == 0] = 1 # change to 1 to avoid dividing by 0
    data_preprocessed -= vmean[None,:]
    data_preprocessed /= vstdv[None,:]
    return data_preprocessed

def tune_model(x_train, y_train, dense_unit):
    model = Sequential()
    model.add(Dense(units=dense_unit, activation='relu', input_dim=27196))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
    
    y_train = np.transpose(y_train)
    model.fit(x_train, y_train, epochs=50, batch_size=1)
    y_pred = model.predict(x_test, batch_size=1)

    corr = correlation(np.transpose(y_pred), np.transpose(y_train))
    print('Running dense unit '+str(n)+' - X Correlation '+str(round(corr[0][0], 3))+'; Y Correlation '+str(round(corr[1][1], 3)))
    return y_pred

### load and prepare data for PEER

#########################################
#----change project and top_data_dir----#
#########################################

project_dir = '/Users/xinhui.li/Documents/PEER/data'
top_data_dir = os.path.join(project_dir, 'PEER_05')
stimulus_path = 'stim_vals.csv' # fake ground-truths

#########################################
#------change train and test files------#
#########################################

configs = {'eye_mask_path': '/usr/local/fsl/data/standard/MNI152_T1_2mm_eye_mask.nii.gz',
           'train_file': 'subj1_PEER1.nii.gz',
           'test_file': 'subj1_PEER2.nii.gz',
           'use_gsr': 0,
           'use_ms': 0}

# Load in the MNI 2mm eye mask
eye_mask_path = configs['eye_mask_path']
eye_mask = nib.load(eye_mask_path).get_fdata()

# Load and prepare training file
filepath = os.path.join(top_data_dir, '_scan_peer1', configs['train_file'])
data_train = load_data(filepath)

x_train=preprocess_data(data_train)
y_train = read_stimulus_csv(stimulus_path)

filepath = os.path.join(top_data_dir, '_scan_peer2', configs['test_file'])
data_test = load_data(filepath)
x_test=preprocess_data(data_test)

for n in [1024, 512, 256, 128, 64]:
    y_pred = tune_model(x_train, y_train, n)
    output_dir = '/Users/xinhui.li/Documents/PEER/data/PEER_05/output_cnn/'
    np.savetxt(output_dir+'test'+str(n)+'.csv', y_pred, delimiter=',')