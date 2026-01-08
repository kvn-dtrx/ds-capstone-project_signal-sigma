# ---
# title: LossHistory Class
# ---

# ---

from pytorch_lightning.callbacks import Callback


class LossHistory(Callback):
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss and val_loss:
            self.epochs.append(epoch)
            self.train_losses.append(train_loss.item())
            self.val_losses.append(val_loss.item())
