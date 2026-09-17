import torch
import torch.nn as nn
from torch.amp import GradScaler
######## Kĩ thuật AMP - Automatic Mixed Precision ##########################
from tqdm import tqdm
from model.Encoder import *
from model.Decoder import *
from torch.utils.data import Dataset, DataLoader

class ImageCaptionModel(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int =16, in_channels: int =3,forward_dim: int = 1024,embedding_dim: int =512, num_heads: int =16, num_layers: int =12, vocab_size: int = 1000, dropout: float = 0.1) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.transformer_encoder = ViT_Encoder(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels,forward_dim=self.forward_dim,embedding_dim=self.embedding_dim, num_heads=self.num_heads, num_layers=self.num_layers, dropout=self.dropout)
        self.transformer_decoder = TransformerDecoder(embedding_dim=self.embedding_dim, num_heads=self.num_heads, forward_dim=self.forward_dim, dropout=self.dropout, num_layers=self.num_layers, vocab_size= vocab_size)
        self.fc_out = nn.Linear(self.embedding_dim, self.vocab_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor: 
        # x: (batch_size, num_patches, embedding_dim)
        # y: (batch_size, num_patches, embedding_dim)
        # Xử lý embedding
        encoder_output = self.transformer_encoder(x)
        decoder_output, _ = self.transformer_decoder(y, encoder_output, key_padding_mask=key_padding_mask)
        output = self.fc_out(decoder_output)
        return output


class TrainModel:
    def __init__(self)->None:
        self.dataset = None
        self.val_dataset = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler()

    def get_accuracy(self, logits: torch.Tensor, y_true: torch.Tensor, key_padding_mask: torch.Tensor = None):
        mask = ~key_padding_mask
        y_pred = torch.argmax(logits, dim=-1)
        correct_tokens = (y_pred == y_true) & mask
        return (correct_tokens.sum() / mask.sum()).item()
        
    def fit(
        self,
        model: nn.Module = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        n_epochs: int = 100,
        learning_rate: float = 1e-4,
        criterion = None,
        is_shuffle: bool = True,
        optimizer = None,
        scheduler = None,
        vocab = None
    )-> None:
        try:
            self.model = model.to(self.device)
        except:
            raise ValueError("Model is not moving to device!")
        if train_loader is None:
            raise ValueError("train_loader is not None!")
        self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.Losses = []
        self.Accuracies = []
        self.Val_Losses = []
        self.Val_Accuracies = []
        for epoch in tqdm(range(n_epochs)):
            model.train()
            loss_epoch = 0
            acc_epoch = 0
            for imgs, idx_captions, key_padding_mask in self.train_loader:
                imgs = imgs.to(self.device)
                y_pred = idx_captions[:, :-1, :] # shape(B, L-1, E)
                y_true = idx_captions[:, 1:, :] # shape(B, L-1, E)
                y_pred = y_pred.to(self.device)
                y_true = y_true.to(self.device)

                with torch.autocast(device_type='cuda'):
                    logits = self.model(x=imgs, y=y_pred, key_padding_mask=key_padding_mask) # shape(B, L, E)

                    loss = self.criterion(logits.permute(0, 2, 1), y_true) # (B, E, L), (B, L)
                    acc = self.get_accuracy(logits, y_true, key_padding_mask)

                    acc_epoch += acc.item()*(y_true.size(1)*y_true.size(0)- key_padding_mask.sum().items())
                    
                    loss_epoch += loss.item()*(y_true.size(1)*y_true.size(0)- key_padding_mask.sum().items())
                    
                    self.optimizer.zero_grad()

                    # compute gradient
                    self.scaler.scale(loss).backward()
                    
                    # update weight
                    
                    self.scaler.step(self.optimizer)

                    self.scaler.update()
                                    
            if scheduler is not None:
                # Tiến hành giảm LR sau số epoch được thiết lập ở tham số step_size của scheduler
                scheduler.step()
            self.Losses.append(loss_epoch / (y_true.size(0)*y_true.size(1)))
            self.Accuracies.append(acc_epoch / (y_true.size(0)*y_true.size(1)))

            