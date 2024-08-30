import librosa
import soundfile as sf
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
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

def save_audio(filename: str, audio: np.array, sample_rate: int) -> None:
    sf.write(filename, audio, sample_rate, subtype="PCM_24")
    print(f"{filename} saved")


def mask_from_seqlen(sequence_length: T.Tensor, max_length: int) -> T.Tensor:
    ones = sequence_length.new_ones(sequence_length.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)

    return sequence_length.unsqueeze(1) >= range_tensor


def collate_audio_data(batch):
    
    seq_lengths = []
    mel_specs = []
    for mel_spec, seq_len in batch:
        mel_specs.append(mel_spec)
        seq_lengths.append(seq_len)

    seq_lengths = T.tensor(seq_lengths)
    max_seq_length = seq_lengths.max()

    # Pad the mel_spects
    for i, mel_spec in enumerate(mel_specs):
        mel_spec_shape = seq_lengths[i]
        mel_spec = F.pad(
            T.tensor(mel_spec),
            pad=[0, max_seq_length-mel_spec_shape],
            value=0
        )
        mel_specs[i] = mel_spec
    
    mel_specs = T.stack(mel_specs, dim=0)

    # Get the mask for seqlen
    stop_padded_tokens = mask_from_seqlen(seq_lengths, max_seq_length)
    stop_padded_tokens = (~stop_padded_tokens).float()

    # Add stop tokens at the end of each sequence
    stop_padded_tokens[:, -1] = 1.0

    return {
        "mel_samples": mel_specs.transpose(1, 2), 
        "seq_lengths": seq_lengths, 
        "stop_padded_tokens": stop_padded_tokens
    }


if __name__ == "__main__":
    dataset = load_dataset("mythicinfinity/libritts_r", name="dev", split="dev.clean").select(np.arange(1000))
    
    def flatten_audio(row):
        return {
            "array": [row['audio'][0]['array']],
            "sampling_rate": [row['audio'][0]['sampling_rate']]
        }

    dataset = dataset.map(
        flatten_audio,
        remove_columns=['audio', 'text_normalized', 'text_original', 'speaker_id', 'path', 'chapter_id', 'id'],
        num_proc=1,
        batched=True,
        batch_size=1
    )

    print(dataset)
    dataset.with_format("pytorch")
    
    print("Converted to Pytorch dataset")
    train_loader = DataLoader(dataset, batch_size=16, 
                              drop_last=True,
                              collate_fn=lambda x: collate_audio_data(x, 1024, 128, 80))

    start = time.time()
    for batch in tqdm(train_loader, total=len(train_loader)-1):
        pass
    
    end = time.time()
    print(f"Time spent in total = ", end-start)