import torch as T
from nnet.TalkGPTPyTorch import TalkGPTLM
from nnet.model import TalkGPT

if __name__ == "__main__":
    name = "./nnet/epoch=32-step=65142.ckpt"
    # talkgpt_trainer = TalkGPTLM.load_from_checkpoint(name)
    checkpoint = T.load(name, map_location="cpu")
    print(checkpoint.keys())