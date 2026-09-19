import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_heads: int = 16, forward_dim: int = 1024, dropout: float = 0.1, seq_length: int = 100) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = nn.Dropout(dropout)
        self.multihead_attention1 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.multihead_attention2 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(normalized_shape = embedding_dim)
        self.linear1 = nn.Linear(in_features=embedding_dim, out_features=forward_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=forward_dim, out_features=embedding_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=embedding_dim)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len, embedding_dim)
        # y: (batch_size, num_patches, embedding_dim)
        # Tạo causal mask động theo chiều dài chuỗi x trên cùng device với x
        seq_length = x.size(1)
        mask_attn = torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool()
        # Qua cơ chế multihead self-attention với residual connection
        mask_attn_output, _ = self.multihead_attention1(x, x, x, attn_mask=mask_attn, key_padding_mask=key_padding_mask)
        x = x + self.dropout(mask_attn_output)
        x_out1 = self.layer_norm1(x)
        # Qua cơ chế cross-attention với residual connection
        cross_attn_output, _ = self.multihead_attention2(x_out1, y, y)
        x = x_out1 + self.dropout(cross_attn_output)
        x_out2 = self.layer_norm2(x)
        # Qua lớp feed-forward với GELU và residual connection
        x = self.linear1(x_out2)
        x = self.gelu(x)
        x = self.dropout(x)
        x_out3 = self.linear2(x)
        x = x_out2 + self.dropout(x_out3)
        x = self.layer_norm3(x)
        return x


class TransformerDecoder(nn.Module):
    """
    Trong Decoder, giả sử x là chuỗi token
    y là output của Encoder (hình ảnh sau khi được patch và nhúng)
    """
    def __init__(self, embedding_dim: int = 512, num_heads: int = 16, forward_dim: int = 1024, dropout: float = 0.1, max_seq_len: int = 500, num_layers: int = 12, vocab_size: int = 1000) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim), requires_grad=True)
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=self.embedding_dim, 
                num_heads=self.num_heads, 
                forward_dim=self.forward_dim, 
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor: 
        # x: (batch_size, seq_len)
        # y: (batch_size, num_patches, embedding_dim)
        # Xử lý embedding
        sequence_length = x.size(1)
        x = self.embedding_layer(x)
        x = x + self.positional_encoding[:, :sequence_length, :]
        for block in self.decoder_blocks:
            x = block(x, y, key_padding_mask=key_padding_mask)
        return x
