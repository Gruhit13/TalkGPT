from datasets import load_dataset
from nnet.config import ModelConfig
from utils.preprocess import preprocess_data
import torch as T
from nnet.model import TalkGPT


if __name__ == "__main__":
    # dataset = load_dataset("mythicinfinity/libritts_r", name="dev", split="dev.clean")

    # config = ModelConfig()

    # dataset = dataset.map(
    #     lambda x: preprocess_data(x, config.n_fft, config.hop_length, config.n_mels, config.max_melspec_len),
    #     remove_columns=['audio', 'text_normalized', 'text_original', 'speaker_id', 'path', 'chapter_id', 'id'],
    #     num_proc=1,
    #     batched=True,
    #     batch_size=1,
    # )
    # print(dataset.cache_files)

    config = ModelConfig()
    model = TalkGPT(config)

    inp = T.randn(2, 128, config.n_mels)
    print("Input shape: ", inp.shape)

    (mel_spec_oup, stop_label) = model(inp)

    print("Mel Spec oup: ", mel_spec_oup.shape)
    print("Stop Label: ", stop_label.shape)