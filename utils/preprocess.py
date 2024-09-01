import librosa
import soundfile as sf
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

def audio_to_melspec(
        audio: np.array, 
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int        
    ) -> np.array:
    
    # Conver the audio waveform to mel-spectogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, 
                                              n_fft=n_fft, hop_length=hop_length, 
                                              n_mels=n_mels)
    
    # conver the linear mel spec to logarithm i.e. conver the power to decibel
    mel_spec = librosa.power_to_db(mel_spec)

    return mel_spec

def melspec_to_audio(
        mel_spec: np.array,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
    ) -> np.array:

    # Conver the decible to power
    mel_spec = librosa.db_to_power(mel_spec)

    # Get the audio back
    audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sample_rate, 
                                                 n_fft=n_fft, hop_length=hop_length)
    
    return audio

def preprocess_data(
        row,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        max_melspec_len: int
    ) -> dict:
    
    array = row['audio'][0]['array']
    sampling_rate = row['audio'][0]['sampling_rate']
    mel_spec = audio_to_melspec(array, sampling_rate, n_fft, hop_length, n_mels)
    seq_len = mel_spec.shape[-1]

    mel_spec = F.pad(T.tensor(mel_spec), (0, max_melspec_len - seq_len), value=0)
    mask = np.zeros((1, max_melspec_len,))
    mask[:, :seq_len] = 1

    stop_token_label = 1 - mask
    stop_token_label[:, -1] = 1

    return {
        'mel_spec': mel_spec.unsqueeze(0).transpose(1, 2),
        'mask': mask.astype(bool),
        'stop_token_label': stop_token_label
    }

def save_audio(filename: str, audio: np.array, sample_rate: int) -> None:
    sf.write(filename, audio, sample_rate, subtype="PCM_24")
    print(f"{filename} saved")