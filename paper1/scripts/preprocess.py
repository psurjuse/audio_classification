
import os
import librosa
import numpy as np
import pandas as pd
import pydub
from tqdm import tqdm
import librosa.display
import resampy


US8K_AUDIO_PATH = '/usr/scratch4/posu1093/MP/data_set/US8K/UrbanSound8K/audio'
US8K_METADATA_PATH = '/usr/scratch4/posu1093/MP/data_set/US8K/UrbanSound8K/metadata/UrbanSound8K.csv'

us8k_metadata_df = pd.read_csv(US8K_METADATA_PATH,
                               usecols=["slice_file_name", "fold", "classID"],
                               dtype={"fold": "uint8", "classID" : "uint8"})





def compute_melspectrogram(audio, sampling_rate, n_fft=512, num_of_samples=128, N_MEL = 128):
    try:
        # compute a mel-scaled spectrogram
        hop_length = int(len(audio) / (num_of_samples - 1))
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=hop_length,
                                                        n_fft=n_fft,
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        melspectrogram_length = melspectrogram_db.shape[1]
        
        # pad or fix the length of spectrogram 
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None 
    
    return melspectrogram_db


#############################################################################
if __name__=="__main__":

    SOUND_DURATION = 2.95  # fixed duration of an audio excerpt in seconds
    #Done sample rates: original = 22050, 22050/2, 22050/4, 16000, 8000, 4000
    #TODO:
    NEW_SAMPLING_RATE = 4000
    features = []

    # iterate through all dataset examples and compute log-mel spectrograms
    for index, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
        file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
        audio, original_sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
        #print("original_sample_rate: ", original_sample_rate)
        # Resample audio if the sampling rates are different
        if original_sample_rate != NEW_SAMPLING_RATE:
            #print("original and given SR different")
            audio = librosa.resample(y=audio, orig_sr=original_sample_rate, target_sr=NEW_SAMPLING_RATE)
        
        melspectrogram = compute_melspectrogram(audio, NEW_SAMPLING_RATE)
        label = row["classID"]
        fold = row["fold"]
        
        features.append([melspectrogram, label, fold])

    # convert into a Pandas DataFrame
    us8k_df = pd.DataFrame(features, columns=["melspectrogram", "label", "fold"])
    # write data to folder
    #TODO:
    us8k_df.to_pickle("fft_512_us8k_df_ds4000.pkl")