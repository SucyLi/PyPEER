import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense 

# Anthony z-score, for eye masking data only 
def a_zscore(data, axis):
    vmean = np.mean(data, axis=axis, keepdims=True) # TR axis becomes first axis when applying mask
    vstdv = np.std(data, axis=axis, keepdims=True)
    vstdv[vstdv == 0] = 1 # change to 1 to avoid dividing by 0
    data -= vmean
    data /= vstdv
    return np.nan_to_num(data, copy=False)

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

def read_stimulus_csv(stimulus_path, mode='train'):
    fixations = pd.read_csv(stimulus_path)
    if mode=='test':
        pos_x = np.array(fixations['pos_x'])
        pos_y = np.array(fixations['pos_y'])
    else:
        pos_x = (1 + np.repeat(np.array(fixations['pos_x']), 5)) / 2
        pos_y = (1 + np.repeat(np.array(fixations['pos_y']), 5)) / 2
    pos = np.transpose(np.asarray(np.vstack((pos_x, pos_y))))
    return pos

def preprocess_data(data):
    data_zscored = a_zscore(data, -1)
    data_preprocessed = np.transpose(data_zscored, (3,0,1,2))
    return data_preprocessed

def train_model(output_dir, x_train, y_train, x_valid, y_valid, x_test, y_test=None, model_config=None):
    img_height = x_train.shape[1] # TODO: rescale dimension
    img_width = x_train.shape[2]
    channels = x_train.shape[3] # how is channel convolved with 2D filter? 

    # build model
    model = Sequential()
    model.add(Conv2D(model_config['filters'], model_config['kernel_size'], input_shape=(img_height, img_width, channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=model_config['max_pooling_size']))

    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(model_config['dense_unit']))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.summary()
    model.compile(optimizer=model_config['optimizer'],loss=model_config['loss'],metrics=[model_config['metrics']])
    
    # train model
    model.fit(x_train, y_train, epochs=model_config['epochs'], batch_size=1)

    # validate
    y_valid_pred = model.predict(x_valid, batch_size=1)
    corr_valid = correlation(np.transpose(y_valid_pred), np.transpose(y_valid))
    print('Validation Correlation X: '+str(round(corr_valid[0][0], 3))+'; Y: '+str(round(corr_valid[1][1], 3)))
    
    # test
    y_test_pred = model.predict(x_test, batch_size=1)
    if y_test is not None:
        corr_test = correlation(np.transpose(y_test_pred[0:len(y_test), :]), np.transpose(y_test))
        print('Test Correlation X: '+str(round(corr_test[0][0], 3))+'; Y: '+str(round(corr_test[1][1], 3)))

    # save model and output
    model.save(output_dir+'conv.h5')
    np.savetxt(output_dir+'valid.txt', corr_valid)
    np.savetxt(output_dir+'test.txt', corr_test)
    np.savetxt(output_dir+'test_movie.csv', y_test_pred, delimiter=',')


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
           'valid_file': 'subj1_PEER2.nii.gz',
           'test_file': 'subj1_MOVIE.nii.gz',
           'use_gsr': 0,
           'use_ms': 0}

# load in the MNI 2mm eye mask
eye_mask_path = configs['eye_mask_path']
eye_mask = nib.load(eye_mask_path).get_fdata()

# load training file
filepath = os.path.join(top_data_dir, '_scan_peer1', configs['train_file'])
data_train = load_data(filepath)
x_train = preprocess_data(data_train)
y_train = read_stimulus_csv(stimulus_path)

# load validation file
filepath = os.path.join(top_data_dir, '_scan_peer2', configs['valid_file'])
data_valid = load_data(filepath)
x_valid = preprocess_data(data_valid)
y_valid = read_stimulus_csv(stimulus_path)

# load test file 
filepath = os.path.join(top_data_dir, 'movie', configs['test_file'])
data_test = load_data(filepath)
x_test = preprocess_data(data_test)

test_path = os.path.join(top_data_dir,'test_movie.csv')
y_test = read_stimulus_csv(test_path, 'test')

# configure output path and model parameters
output_dir = '/Users/xinhui.li/Documents/PEER/data/PEER_05/output_cnn2/'
model_config = {'filters':16,
                'kernel_size':(3,3),
                'max_pooling_size':(4,4),
                'dense_unit':256,
                'optimizer':'adam',
                'loss':'mean_squared_error',
                'metrics':'mae',
                'epochs':20}

with open(output_dir+'model_config.json', 'w') as config_file:
    json.dump(model_config, config_file)

train_model(output_dir, x_train, y_train, x_valid, y_valid, x_test, y_test, model_config)
