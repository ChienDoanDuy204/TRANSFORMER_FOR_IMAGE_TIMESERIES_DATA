import torch
import torch.nn as nn
import timm

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int= 512, num_heads: int=12, forward_dim: int =1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape = embedding_dim)
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=forward_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=forward_dim, out_features=embedding_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_patches, embedding_dim)
        # Qua cơ chế multihead self-attention với residual connection
        attn_output, _ = self.multihead_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x_out1 = self.layer_norm1(x)

        # Qua lớp feed-forward với GELU và residual connection
        x = self.linear1(x_out1)
        x = self.gelu(x)
        x = self.dropout(x)
        x_out2 = self.linear2(x)
        x = x_out1 + self.dropout(x_out2)
        x = self.layer_norm2(x)
        return x

class ViT_Encoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int =16, in_channels: int =3,forward_dim: int = 1024,embedding_dim: int =512, num_heads: int =12, num_layers: int =12, dropout: float = 0.1) -> None:
        super().__init__()
        self.img_size = img_size # Kích thước của ảnh đầu vào
        self.patch_size = patch_size # Số lượng vùng ảnh được chia nhỏ
        self.in_channels = in_channels # Số lượng channel của ảnh đầu vào
        self.embedding_dim = embedding_dim # Kích thước của vector nhúng
        self.num_heads = num_heads # Số lượng head trong cơ chế self-attention
        self.num_layers = num_layers # Số lượng lớp transformer encoder
        self.forward_dim = forward_dim 
        self.dropout = dropout
        self.positional_encoding = nn.Parameter(torch.randn(1, (img_size//patch_size)**2, embedding_dim), requires_grad=True)
        self.patch_projection = nn.Conv2d(in_channels = self.in_channels, out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.blocks_encoder = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=self.embedding_dim, 
            num_heads=self.num_heads, forward_dim=self.forward_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
            ])
        
    def forward(self, X): 
        # X: (batch_size, in_channels, img_size, img_size)
        B, C, H, W = X.shape 
        # Dự kiến đầu ra sau khi projection: (batch_size, embedding_dim, num_patches)
        x = self.patch_projection(X) # shape(B, C, W, H)
        # Dự kiến đầu ra sau khi flatten và transpose: (batch_size, num_patches, embedding_dim)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_encoding # Lưu ý: Kích thước của positional_encoding phải khớp với số lượng patch
        for block in self.blocks_encoder:
            x = block(x) # Qua cơ chế multihead self-attention với residual connection
        return x

class Timm_ViT_Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.timm_vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        
        vit_dim = self.timm_vit.embed_dim  # 192 cho vit_tiny
        
        # Bỏ head gốc, không dùng nữa
        self.timm_vit.head = nn.Identity()
        
        # Projection riêng, áp dụng thủ công trong forward
        self.proj = nn.Linear(vit_dim, embedding_dim) if vit_dim != embedding_dim else nn.Identity()

    def forward(self, X):
        x = self.timm_vit.forward_features(X)   # (B, 1+num_patch, vit_dim=192)
        x = x[:, 1:, :]                          # (B, num_patch, vit_dim=192) - bỏ CLS token
        x = self.proj(x)                         # (B, num_patch, embedding_dim=512)
        return x