import torch as T
import lightning.pytorch as pl
from .config import ModelConfig
from .model import TalkGPT
from .scheduler import TransformerLRScheduler
import torch as T
from torch import optim, nn

class TalkGPTLM(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super(TalkGPTLM, self).__init__()
        
        self.config = config
        self.model = TalkGPT(self.config)
        self.melspec_loss = nn.L1Loss()
        self.stoptoken_loss = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
    
    def training_step(self, batch: dict, batch_idx: int) -> T.Tensor:
        (mel_spec, mask, stop_token_label) = self.get_batch_data(batch)
        causal_mask = self.model.get_mask(mel_spec, mask)
        
        (mel_spec_pred, stop_token_pred) = self.model(mel_spec, causal_mask)
        mel_spec_loss = self.melspec_loss(mel_spec_pred, mel_spec)
        stop_token_loss = self.stoptoken_loss(stop_token_pred.view(-1), stop_token_label.view(-1))
        loss = mel_spec_loss + stop_token_loss
        
        lr = self.optimizers().param_groups[0]["lr"]
        
        self.log("train/mel_spec_loss", mel_spec_loss)
        self.log("train/stop_token_loss", stop_token_loss)
        self.log("train/loss", loss)
        self.log("lr", lr)
        
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        (mel_spec, mask, stop_token_label) = self.get_batch_data(batch)
        causal_mask = self.model.get_mask(mel_spec, mask)
        
        (mel_spec_pred, stop_token_pred) = self.model(mel_spec, causal_mask)
        mel_spec_loss = self.melspec_loss(mel_spec_pred, mel_spec)
        stop_token_loss = self.stoptoken_loss(stop_token_pred.view(-1), stop_token_label.view(-1))
        loss = mel_spec_loss + stop_token_loss
        
        self.log("val/mel_spec_loss", mel_spec_loss,logger=True)
        self.log("val/stop_token_loss", stop_token_loss, logger=True)
        self.log("val/loss", loss)
        
    
    def get_batch_data(self, batch: dict) -> tuple:
        mel_spec = batch['mel_spec']
        mask = batch['mask']
        stop_token_label = batch['stop_token_label']
        
        return (mel_spec, mask, stop_token_label)
    
    def configure_optimizers(self) -> dict:
        opt = optim.Adam(self.model.parameters(), lr = self.config.lr)
        lr_scheduler = TransformerLRScheduler(opt, self.config.d_model, self.config.warmup_steps)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step"
            }
        }

if __name__ == "__main__":
    name = "./epoch=32-step=65142.ckpt"
    # talkgpt_trainer = TalkGPTLM.load_from_checkpoint(name)
    checkpoint = T.load(name, map_location="cpu")
    print(checkpoint.keys())