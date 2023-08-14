#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor



################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)
       

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
   
    df = pd.DataFrame(features)
    
   

    df.replace('nan', np.nan, inplace=True)

    df = df.apply(pd.to_numeric, errors='ignore')
    df_8 = df.iloc[:,:8]
    df_8 = df_8.fillna(df_8.median())
    df_10 = df.iloc[:,8:]
    df_10 = df_10.fillna(df_10.mean())
    features = pd.concat([df_8,df_10],axis = 1)
   

    features= df.to_numpy()

    # # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    imputer = SimpleImputer(strategy='mean').fit(features)

    # Train the models.

    # Create GridSearchCV
    features = imputer.transform(features)
    random_state = 42
    param_grid = {
    'learning_rate': [0.01, 0.1,0.2,0.02],
    'max_depth': [3,5],
    'max_leaf_nodes': [15,51],
    'min_samples_leaf': [50,60,70],
    'max_iter': [200,250,300],
  
}
    
    random_state = 42
    #Train the HistGradientBoostingClassifier with GridSearchCV for hyperparameter tuning.
    grid_search_clf = GridSearchCV(HistGradientBoostingClassifier(random_state=random_state),
                                    param_grid=param_grid, cv=3)
    grid_search_clf.fit(features, outcomes.ravel())
    best_clf = grid_search_clf.best_estimator_
    best_clf.fit(features,outcomes.ravel())
   
   
    
  

    
    # Train the HistGradientBoostingRegressor with GridSearchCV for hyperparameter tuning.
    grid = GridSearchCV(HistGradientBoostingRegressor(random_state=random_state),
                                    param_grid=param_grid, cv=3)
    grid_result = grid.fit(features,cpcs.ravel())
    best_reg = grid_result.best_estimator_
    best_reg.fit(features,cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, best_clf, best_reg)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)
   
    df1 = pd.DataFrame(features)

    df1.replace('nan', np.nan, inplace=True)

    # Convert columns to numeric (necessary for mean calculation)
    df1 = df1.apply(pd.to_numeric, errors='ignore')
   
    

   

    # Replace NaN values with column means
    df_filled1 = df1.fillna(0)

    features= df_filled1.to_numpy()

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.

    patient_metadata = load_challenge_data(data_folder,patient_id)
    recording_ids = find_recording_files(data_folder,patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']
    group = 'EEG'
   


    eeg_list = []
    ecg_list = []
 
    if num_recordings > 0:
       
       
       
        for recording_id in recording_ids:
           
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
           
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                

                if all(channel in channels for channel in eeg_channels):
                    data, channels = reduce_channels(data, channels, eeg_channels)
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
                    
                    eeg_features = get_eeg_features(data, sampling_frequency).flatten()
                    eeg_list.append(eeg_features)
               
                else:
                    eeg_features = float("nan") * np.ones(8) # 2 bipolar channels * 4 features / channel
                    eeg_list.append(eeg_features)
               
            else:
                eeg_features = float("nan") * np.ones(8) # 2 bipolar channels * 4 features / channel
                eeg_list.append(eeg_features)
             
    else:
        eeg_features = float("nan") * np.ones(8) # 2 bipolar channels * 4 features / channel
        eeg_list.append(eeg_features)
    if np.array(eeg_list).shape[0] != 1:
       
        eeg_features = np.mean(eeg_list,axis = 0)
       


    # Extract ECG features.
   

    if num_recordings > 0:
       
        ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
        group = 'ECG'
       
        for recording_id in recording_ids:
           
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')

                data, channels = reduce_channels(data, channels, ecg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                features = get_ecg_features(data)
                
                ecg_features = expand_channels(features, channels, ecg_channels).flatten()
                df3 = pd.DataFrame([ecg_features])
                df3 = pd.DataFrame(data)

                # Replace "NaN" strings with zeros
                df3.replace('NaN', np.nan, inplace=True)  # Convert "NaN" strings to actual NaN values
                df3_filled = df3.fillna(0)
                #print(df3_filled.head())

                # Convert columns to numeric (necessary for mean calculation)
               
                nnf = df3_filled.to_numpy()
                
                ecg_features = np.array([np.mean(np.mean(nnf,axis = 1)),np.mean(np.std(nnf,axis = 1))])
                ecg_list.append(ecg_features)
                   
            else:
                ecg_features = float("nan") * np.ones(2)
                ecg_list.append(ecg_features)
             
    else:
        ecg_features = float("nan") * np.ones(2)
        ecg_list.append(ecg_features)
    if np.array(ecg_list).shape[0] != 1:
        ecg_features = np.mean(ecg_list,axis = 0)


    # Extract features.
    
    return np.hstack((patient_features, eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False,n_fft = num_samples,n_per_seg = num_samples)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)
   

    features = np.array((mean, std)).T

    return features

