from datasets import load_dataset
from nnet.config import ModelConfig
from utils.preprocess import preprocess_data
import torch as T
from nnet.model import TalkGPT
from torchsummary import summary


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

    model_summary = summary(model, input_data=inp)

    with open("model_summary.txt", "w") as summary_writer:
        summary_writer.write("="*20+"| Model Summary |" + "="*20)
        summary_writer.write("\n")
        summary_writer.write(model_summary.__str__())