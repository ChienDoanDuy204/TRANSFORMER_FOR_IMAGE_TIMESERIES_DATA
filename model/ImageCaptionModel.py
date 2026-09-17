import torch
import torch.nn as nn
from model.Encoder import ViT_Encoder
from model.Decoder import TransformerDecoder

class ImageCaptionModel(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int =16, in_channels: int =3,forward_dim: int = 1024,embedding_dim: int =512, num_heads: int =16, num_layers: int =12, max_seq_len: int = 100, vocab_size: int = 1000, dropout: float = 0.1) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.transformer_encoder = ViT_Encoder(img_size=self.img_size, patch_size=self.patch_size, in_channels=self.in_channels,forward_dim=self.forward_dim,embedding_dim=self.embedding_dim, num_heads=self.num_heads, num_layers=self.num_layers, dropout=self.dropout)
        self.transformer_decoder = TransformerDecoder(embedding_dim=self.embedding_dim, num_heads=self.num_heads, forward_dim=self.forward_dim, dropout=self.dropout, max_seq_len=self.max_seq_len, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.embedding_dim, self.vocab_size)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        # x: (batch_size, num_patches, embedding_dim)
        # y: (batch_size, num_patches, embedding_dim)
        # Xử lý embedding
        encoder_output = self.transformer_encoder(x)
        decoder_output = self.transformer_decoder(encoder_output, y)
        output = self.fc_out(decoder_output)
        return output