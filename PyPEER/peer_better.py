import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import SVR
from sklearn.externals import joblib


def load_data(_filepath):
    """
    Loads fMRI data

    Parameters
    ----------
    _filepath : string
        Pathname of the NIfTI file used to train a model or predict eye movements

    Returns
    -------
    _data : float
        4D numpy array containing fMRI data

    """

    nib_format = nib.load(_filepath)
    _data = nib_format.get_data()

    print('Training data Loaded')

    return _data


def train_model(_data, _stimulus_path, both=False):
# def train_model(_data, _calibration_points_removed, _stimulus_path):
    """
    Trains the SVR model used in the PEER method

    Parameters
    ----------
    _data : float
        List of numpy arrays, where each array contains the averaged intensity values for each calibration point
    _calibration_points_removed : int
        List of calibration points removed if all volumes for a given calibration point were high motion
    _stimulus_path : string
        Pathname of the PEER calibration scan stimuli

    Returns
    -------
    _xmodel :
        SVR model to estimate eye movements in the x-direction
    _ymodel :
        SVR model to estimate eye movements in the y-direction

    """

#     monitor_width = 1680
#     monitor_height = 1050
    
    fixations = pd.read_csv(_stimulus_path)
#     x_targets = np.repeat(np.array(fixations['pos_x']), 1) * monitor_width / 2
#     y_targets = np.repeat(np.array(fixations['pos_y']), 1) * monitor_height / 2

    x_targets = (1 + np.repeat(np.array(fixations['pos_x']), 5)) / 2
    y_targets = (1 + np.repeat(np.array(fixations['pos_y']), 5)) / 2

#     x_targets = list(np.delete(np.array(x_targets), _calibration_points_removed))
#     y_targets = list(np.delete(np.array(y_targets), _calibration_points_removed))

    if both:
        x_targets = np.concatenate((x_targets.reshape(-1,1), x_targets.reshape(-1,1))).reshape(-1,)
        y_targets = np.concatenate((y_targets.reshape(-1,1), y_targets.reshape(-1,1))).reshape(-1,)

    _xmodel = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    _xmodel.fit(_data, x_targets)
#     print(x_targets)

    _ymodel = SVR(kernel='linear', C=100, epsilon=.01, verbose=2)
    _ymodel.fit(_data, y_targets)

    return _xmodel, _ymodel


def save_model(_xmodel, _ymodel, _train_file, _output_dir):
# def save_model(_xmodel, _ymodel, _train_file, _ms, _gsr, _output_dir):
    """
    Saves the SVR models used in the PEER method

    Parameters
    ----------
    _xmodel :
        SVR model to estimate eye movements in the x-direction
    _ymodel :
        SVR model to estimate eye movements in the y-direction
    _train_file : string
        Pathname of the NIfTI file used to train the SVR model
    _ms : bool
        Whether or not to use motion scrubbing
    _gsr : bool
        Whether or not to use global signal regression
    _output_dir :
        Pathname of the output directory

    """

    x_name = os.path.abspath(os.path.join(_output_dir,
                                          str('xmodel_' + _train_file.strip('.nii.gz') + '.pkl')))
    y_name = os.path.abspath(os.path.join(_output_dir,
                                          str('ymodel_' + _train_file.strip('.nii.gz') + '.pkl')))

    joblib.dump(_xmodel, x_name)
    joblib.dump(_ymodel, y_name)

    print('SVR Models saved. PEER can now be applied to new data.')
    
    
def load_model(_output_dir):
    """
    Loads the SVR models used to estimate eye movements

    Parameters
    ----------
    _output_dir : string
        Pathname of the output directory

    Returns
    -------
    _xmodel :
        SVR model to estimate eye movements in the x-direction
    _ymodel :
        SVR model to estimate eye movements in the y-direction
    _xname : string
        Filename of the model used to estimate eye movements in the x-direction
    _yname : stringf
        Filename of the model used to estimate eye movements in the y-direction

    """

    model_selection = [x for x in os.listdir(_output_dir) if ('pkl' in x) and x.startswith('xmodel')]

    if len(model_selection) > 1:

        options = []
        options_index = list(np.arange(len(model_selection)))

        print("List of available models:\n")

        for i, model in enumerate(model_selection):

            model_option = str('    {}: {}').format(str(i), model.replace('xmodel_', ''))
            options.append(model.replace('xmodel', ''))
            print(model_option)

        print('\n')

        selected_model_index = int(input(str('Which model type? ({}): ').format(options_index)))
        selected_model = str('xmodel' + options[selected_model_index])
        selected_model_path = os.path.abspath(os.path.join(_output_dir, selected_model))

        _xname = selected_model.replace('pkl', 'csv')
        _yname = selected_model.replace('x', 'y').replace('pkl', 'csv')

        _xmodel = joblib.load(selected_model_path)
        _ymodel = joblib.load(selected_model_path.replace('x', 'y'))

    else:

        _xname = model_selection[0]
        _yname = model_selection[0].replace('x', 'y')

        x_selected_model_path = os.path.abspath(os.path.join(_output_dir, _xname))
        y_selected_model_path = os.path.abspath(os.path.join(_output_dir, _yname))

        _xmodel = joblib.load(x_selected_model_path)
        _ymodel = joblib.load(y_selected_model_path)

        _xname = model_selection[0].replace('pkl', 'csv')
        _yname = model_selection[0].replace('pkl', 'csv').replace('x', 'y')

    return _xmodel, _ymodel, _xname, _yname


def predict_fixations(_xmodel, _ymodel, _data):
    """
    Predict fixations

    Parameters
    ----------
    _xmodel :
        SVR model to estimate eye movements in the x-direction
    _ymodel :
        SVR model to estimate eye movements in the y-direction
    _data :
        4D numpy array containing fMRI data used to predict eye movements (e.g., movie data)

    Returns
    -------
    _x_fix : float
        List of predicted fixations in the x-direction
    _y_fix : float
        List of predicted fixations in the y-direction

    """

    _x_fix = _xmodel.predict(_data)
    _y_fix = _ymodel.predict(_data)

    return _x_fix, _y_fix


def save_fixations(_x_fix, _y_fix, _xname, _yname, _output_dir):
    """
    Save predicted fixations

    Parameters
    ----------
    _x_fix : float
        List of predicted fixations in the x-direction
    _y_fix : float
        List of predicted fixations in the y-direction
    _xname : string
        Filename of the model used to estimate eye movements in the x-direction
    _yname : string
        Filename of the model used to estimate eye movements in the y-direction
    _output_dir : string
        Pathname of the output directory

    Returns
    -------
    _fix_xname : string
        Filename of the CSV containing fixation predictions in the x-direction
    _fix_yname : string
        Filename of the CSV containing fixation predictions in the y-direction

    """

    _fix_xname = str('xfixations_' + _xname)
    _fix_yname = str('yfixations_' + _yname)

    x_path = os.path.abspath(os.path.join(_output_dir, _fix_xname))
    y_path = os.path.abspath(os.path.join(_output_dir, _fix_yname))

    x = open(x_path, 'w')
    for fix in _x_fix:
        x.write(str("{0:.5f},").format(round(fix, 5)))
    x.close()

    y = open(y_path, 'w')
    for fix in _y_fix:
        y.write(str("{0:.5f},").format(round(fix, 5)))
    y.close()

    return _fix_xname, _fix_yname


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
eye_mask = nib.load(eye_mask_path).get_data()

# Load and prepare training file
filepath = os.path.join(top_data_dir, '_scan_peer1', configs['train_file'])

data = load_data(filepath)

# This will apply eye mask to each the volume at each TR
data_cropped = []
for vol in range(data.shape[3]):
    data_cropped.append(data[...,vol].ravel()[np.where(eye_mask.ravel() == 1)[0]])
data_cropped = np.array(data_cropped)

# z-score across time
vmean = np.mean(data_cropped, axis=0) # TR axis becomes first axis when applying mask
vstdv = np.std(data_cropped, axis=0)
vstdv[vstdv == 0] = 1 # change to 1 to avoid dividing by 0
data_cropped -= vmean[None,:]
data_cropped /= vstdv[None,:]

processed_data = list(data_cropped)
xmodel, ymodel = train_model(processed_data, stimulus_path)

output_dir = '/Users/xinhui.li/Documents/PEER/data/PEER_05/output_movie'
save_model(xmodel, ymodel, 'peer1.nii.gz', output_dir)

# Test model
filepath = os.path.join(top_data_dir, 'movie', configs['test_file'])

data = load_data(filepath)
data_cropped = []
for vol in range(data.shape[3]):
    data_cropped.append(data[...,vol].ravel()[np.where(eye_mask.ravel() == 1)[0]])
data_cropped = np.array(data_cropped)

vmean = np.mean(data_cropped, axis=0)
vstdv = np.std(data_cropped, axis=0)
vstdv[vstdv == 0] = 1
data_cropped -= vmean[None,:]
data_cropped /= vstdv[None,:]

processed_data = list(data_cropped)
xmodel, ymodel, xname, yname = load_model(output_dir)
x_fix, y_fix = predict_fixations(xmodel, ymodel, processed_data)
fix_x_name, fix_y_name = save_fixations(x_fix, y_fix, xname, yname, output_dir)