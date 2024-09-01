import torch as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from preprocess import audio_to_melspec, collate_audio_data
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import time

class TalkGPTDataset(Dataset):
    def __init__(
        self, 
        dataset: str, 
        name: str, 
        split: str,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        max_melspec_len: int
    ):
        self.dataset = load_dataset(dataset, name, split=split).select(np.arange(512))
        self.dataset = self.dataset.map(
            self.__flatten_map,
            remove_columns=['audio', 'text_normalized', 'text_original', 'speaker_id', 'path', 'chapter_id', 'id'],
            num_proc=1,
            batched=True,
            batch_size=1
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_melspec_len = max_melspec_len

    def __flatten_map(self, row) -> dict:
        return {
            "array": [row['audio'][0]['array']],
            "sampling_rate": [row['audio'][0]['sampling_rate']]
        }


    def __getitem__(self, index: int):
        sample = self.dataset[index]

        audio, sampling_rate = np.array(sample['array']), sample['sampling_rate']

        mel_spec = audio_to_melspec(audio, sampling_rate, self.n_fft, self.hop_length, self.n_mels)
        seq_len = mel_spec.shape[-1]

        mel_spec = F.pad(mel_spec, (0, self.max_melspec_len-seq_len), value=0)
        return mel_spec
    
    def __len__(self) -> int:
        return len(self.dataset)
        

if __name__ == "__main__":
    dataset = TalkGPTDataset("mythicinfinity/libritts_r", name="dev", 
                             split="dev.clean", n_fft=1024, 
                             hop_length=128, n_mels=80)
    
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_audio_data,
                            drop_last=True)

    start = time.time()
    for batch in tqdm(dataloader, total=len(dataloader)-1):
        pass
    end = time.time()
    print("Time taken: ", end-start)