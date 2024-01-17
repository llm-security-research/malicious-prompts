# define the lightning trainer
import pytorch_lightning as pl
import torch
import torchmetrics

class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        return_head: bool = False,
        cls_head_loss: torch.nn.Module = None,
        class_head_metric: torchmetrics.Metric = None
    ):
        super().__init__()
        # assert if return_head is True then cls_head_loss is not None
        assert not (return_head and cls_head_loss is None), "cls_head_loss must be provided if return_head is True"
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.return_head = return_head
        self.cls_head_loss = cls_head_loss
        self.class_head_metric = class_head_metric

    def forward(self, x, **kwargs):
        out = self.net(x, **kwargs)
        return out

    def shared_step(self, batch, batch_idx, phase):
        # batch should be a dict. With Key: Batch*Chns*W*H shape.
        anchor, positive, negative = batch
        output = self(anchor, img2=positive, img3=negative)
        triplet_loss = self.loss(*output[:3]) 
        self.log(
            f"{phase}_triplet_loss",
            triplet_loss,
            prog_bar=True,
            batch_size=anchor.shape[0],
            on_step=True,
            on_epoch=True,
        )
        # compute the classification loss if class head is added.
        ce_loss = self.cls_head_loss(output[3] , label) if self.return_head else 0
        loss = triplet_loss + ce_loss
        self.log(
            f"{phase}_loss",
            loss,
            prog_bar=True,
            batch_size=anchor.shape[0],
            on_step=True,
            on_epoch=True,
        )
        if self.return_head:
            self.log(
                f"{phase}_ce_loss",
                ce_loss,
                prog_bar=True,
                batch_size=anchor.shape[0],
                on_step=True,
                on_epoch=True,
            )
            probs = torch.Tensor(torch.nn.functional.softmax(output[3], dim=1).detach().cpu().numpy())
            labels = torch.IntTensor(label.detach().cpu().numpy())
            self.class_head_metric(probs, labels)
            self.log(
                f"{phase}_metric",
                self.class_head_metric,
                prog_bar=True,
                batch_size=anchor.shape[0],
                on_step=False,
                on_epoch=True
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        return optimizer