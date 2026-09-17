import torch
import torch.nn as nn
from torch.amp import GradScaler
######## Kĩ thuật AMP - Automatic Mixed Precision ##########################
from tqdm import tqdm
from model.Encoder import *
from model.Decoder import *
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Union
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

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
        # x: (batch_size, in_channels, img_size, img_size)
        # y: (batch_size, seq_len)
        # Xử lý embedding
        encoder_output = self.transformer_encoder(x)
        decoder_output = self.transformer_decoder(y, encoder_output, key_padding_mask=key_padding_mask)
        output = self.fc_out(decoder_output)
        return output


class TrainModel:
    def __init__(self)->None:
        self.dataset = None
        self.val_dataset = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler()

    def get_accuracy(self, logits: torch.Tensor, y_true: torch.Tensor, key_padding_mask: torch.Tensor = None):
        mask = ~key_padding_mask if key_padding_mask is not None else torch.ones_like(y_true, dtype=torch.bool)
        y_pred = torch.argmax(logits, dim=-1)
        correct_tokens = (y_pred == y_true) & mask
        return (correct_tokens.sum() / mask.sum()).item()

    # function generate caption  
    def generate_caption(
        self,
        model: nn.Module = None,
        img: Union[torch.Tensor, Image.Image, str, Path] = None,
        vocab = None,
        max_length_generate: int = 50
    ) -> str:
        """
        Sinh caption từ ảnh đầu vào bằng cơ chế giải mã tự hồi quy (Autoregressive Greedy Search).

        Args:
            model: ImageCaptionModel. Nếu là None, mặc định sử dụng self.model.
            img: Ảnh đầu vào, có thể là Tensor (shape (3, H, W) hoặc (1, 3, H, W)),
                 PIL Image hoặc đường dẫn tới file ảnh (str / Path).
            vocab: Đối tượng từ điển (BuildVocabFromIterator hoặc có stoi/itos/vocab_reverse).
            max_length_generate: Số lượng token tối đa cần sinh (mặc định 50).

        Returns:
            caption: Chuỗi caption hoàn chỉnh (str).
        """
        if model is None:
            if hasattr(self, 'model') and self.model is not None:
                model = self.model
            else:
                raise ValueError("Model is None! Please provide a model or train via fit() first.")

        device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        # 1. Xử lý ảnh đầu vào
        if isinstance(img, (str, Path)):
            img = Image.open(str(img)).convert('RGB')

        if isinstance(img, Image.Image):
            target_size = getattr(model, 'img_size', 224)
            transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor()
            ])
            img = transform(img)

        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(img)}. Expected torch.Tensor, PIL.Image.Image, or file path.")

        if img.ndim == 3:
            img = img.unsqueeze(0)  # Thêm batch dimension: (1, 3, H, W)
        elif img.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got shape {img.shape}")

        img = img.to(device)

        # 2. Xác định các special tokens từ vocab
        if hasattr(vocab, 'get_start_token'):
            start_token = vocab.get_start_token()
        elif hasattr(vocab, 'stoi'):
            start_token = vocab.stoi.get('<sos>', 2)
        else:
            start_token = 2

        if hasattr(vocab, 'get_end_token'):
            end_token = vocab.get_end_token()
        elif hasattr(vocab, 'stoi'):
            end_token = vocab.stoi.get('<eos>', 3)
        else:
            end_token = 3

        with torch.no_grad():
            # 3. Tối ưu: trích xuất đặc trưng ảnh ViT Encoder 1 lần duy nhất
            if hasattr(model, 'transformer_encoder'):
                encoder_output = model.transformer_encoder(img)
            else:
                encoder_output = None

            curr_tokens = [start_token]
            generated_tokens = []

            # 4. Vòng lặp giải mã tự hồi quy (Greedy Search)
            for _ in range(max_length_generate):
                y = torch.tensor([curr_tokens], dtype=torch.long, device=device)

                if encoder_output is not None and hasattr(model, 'transformer_decoder') and hasattr(model, 'fc_out'):
                    decoder_output = model.transformer_decoder(y, encoder_output)
                    logits = model.fc_out(decoder_output)
                else:
                    logits = model(img, y)

                next_token_logits = logits[:, -1, :]  # Lấy logits tại vị trí token cuối cùng: (1, vocab_size)
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                if next_token_id == end_token:
                    break

                curr_tokens.append(next_token_id)
                generated_tokens.append(next_token_id)

        # 5. Chuyển đổi danh sách token ID thành từ ngữ
        if hasattr(vocab, 'vocab_reverse'):
            words = vocab.vocab_reverse(generated_tokens)
        elif hasattr(vocab, 'itos'):
            words = [vocab.itos.get(idx, '<unk>') for idx in generated_tokens]
        else:
            words = [str(idx) for idx in generated_tokens]

        # 6. Loại bỏ special tokens và ghép lại thành câu hoàn chỉnh
        special_tokens_to_strip = {'<sos>', '<eos>', '<pad>'}
        filtered_words = [w for w in words if w not in special_tokens_to_strip]
        caption = " ".join(filtered_words)

        return caption

    def evaluate(
        self,
        model: nn.Module = None,
        val_loader: DataLoader = None,
        criterion = None,
        vocab = None,
        max_length_generate: int = 50
    ) -> dict:
        """
        Đánh giá mô hình trên tập validation bằng các độ đo Loss, Accuracy và BLEU-1 -> BLEU-4.

        Args:
            model: ImageCaptionModel. Nếu là None, mặc định sử dụng self.model.
            val_loader: DataLoader của tập validation. Nếu là None, mặc định sử dụng self.val_loader.
            criterion: Hàm mất mát (Loss function). Nếu là None, mặc định sử dụng self.criterion.
            vocab: Đối tượng từ điển (BuildVocabFromIterator hoặc có stoi/itos/vocab_reverse).
            max_length_generate: Số lượng token tối đa cần sinh khi tính BLEU (mặc định 50).

        Returns:
            dict: Chứa các giá trị {'loss': float, 'accuracy': float, 'bleu1': float, 'bleu2': float, 'bleu3': float, 'bleu4': float}
        """
        if model is None:
            if hasattr(self, 'model') and self.model is not None:
                model = self.model
            else:
                raise ValueError("Model is None! Please provide a model or train via fit() first.")

        if val_loader is None:
            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_loader = self.val_loader
            else:
                raise ValueError("val_loader is not None!")

        if criterion is None:
            criterion = getattr(self, 'criterion', None)

        if vocab is None:
            vocab = getattr(self, 'vocab', None)

        device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        val_loss_total = 0.0
        val_acc_total = 0.0
        total_valid_tokens = 0
        references = []
        hypotheses = []

        with torch.no_grad():
            for imgs, idx_captions, key_padding_mask in tqdm(val_loader, desc="Evaluating", leave=False):
                imgs = imgs.to(device)

                # 1. Tính loss & accuracy (teacher-forcing) nếu có criterion
                if criterion is not None:
                    if idx_captions.ndim == 2:
                        y_pred = idx_captions[:, :-1]
                        y_true = idx_captions[:, 1:]
                        train_mask = key_padding_mask[:, :-1] if key_padding_mask is not None else None
                        target_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None
                    else:
                        y_pred = idx_captions[:, :-1, :]
                        y_true = idx_captions[:, 1:, :]
                        train_mask = key_padding_mask
                        target_mask = key_padding_mask

                    y_pred = y_pred.to(device)
                    y_true = y_true.to(device)
                    if train_mask is not None:
                        train_mask = train_mask.to(device)
                    if target_mask is not None:
                        target_mask = target_mask.to(device)

                    with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
                        logits = model(x=imgs, y=y_pred, key_padding_mask=train_mask)
                        loss = criterion(logits.permute(0, 2, 1), y_true)
                        acc = self.get_accuracy(logits, y_true, target_mask)

                    valid_tokens = (y_true.numel() - target_mask.sum().item()) if target_mask is not None else y_true.numel()
                    val_loss_total += loss.item() * valid_tokens
                    val_acc_total += acc * valid_tokens
                    total_valid_tokens += valid_tokens

                # 2. Sinh caption từng ảnh trong batch để tính BLEU-1 -> BLEU-4
                if vocab is not None:
                    batch_size = imgs.size(0)
                    for i in range(batch_size):
                        img_i = imgs[i:i+1]
                        pred_caption = self.generate_caption(
                            model=model,
                            img=img_i,
                            vocab=vocab,
                            max_length_generate=max_length_generate
                        )
                        hyp_tokens = pred_caption.split()

                        true_ids = idx_captions[i].tolist()
                        if hasattr(vocab, 'vocab_reverse'):
                            raw_words = vocab.vocab_reverse(true_ids)
                        elif hasattr(vocab, 'itos'):
                            raw_words = [vocab.itos.get(idx, '<unk>') for idx in true_ids]
                        else:
                            raw_words = [str(idx) for idx in true_ids]

                        special_tokens = {'<sos>', '<eos>', '<pad>', None}
                        ref_tokens = [w for w in raw_words if w not in special_tokens]

                        references.append([ref_tokens])
                        hypotheses.append(hyp_tokens)

        val_loss = (val_loss_total / total_valid_tokens) if total_valid_tokens > 0 else 0.0
        val_acc = (val_acc_total / total_valid_tokens) if total_valid_tokens > 0 else 0.0

        # 3. Tính toán BLEU-1 -> BLEU-4
        if references and hypotheses:
            smooth = SmoothingFunction().method1
            bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
            bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
            bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        else:
            bleu1 = bleu2 = bleu3 = bleu4 = 0.0

        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': bleu4
        }
        return metrics

    # function huấn luyện mô hình
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
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.vocab = vocab
        self.Losses = []
        self.Accuracies = []
        self.Val_Losses = []
        self.Val_Accuracies = []
        self.Val_Bleu1 = []
        self.Val_Bleu2 = []
        self.Val_Bleu3 = []
        self.Val_Bleu4 = []
        for epoch in tqdm(range(n_epochs)):
            self.model.train()
            loss_epoch = 0
            acc_epoch = 0
            total_tokens_epoch = 0
            for imgs, idx_captions, key_padding_mask in self.train_loader:
                imgs = imgs.to(self.device)
                y_pred = idx_captions[:, :-1]
                y_true = idx_captions[:, 1:]
                train_mask = key_padding_mask[:, :-1] if key_padding_mask is not None else None
                target_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None

                y_pred = y_pred.to(self.device)
                y_true = y_true.to(self.device)
                if train_mask is not None:
                    train_mask = train_mask.to(self.device)
                if target_mask is not None:
                    target_mask = target_mask.to(self.device)

                with torch.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                    logits = self.model(x=imgs, y=y_pred, key_padding_mask=train_mask) # shape(B, L, vocab_size)

                    loss = self.criterion(logits.permute(0, 2, 1), y_true) # (B, vocab_size, L), (B, L)
                    acc = self.get_accuracy(logits, y_true, target_mask)

                    valid_tokens = (y_true.numel() - target_mask.sum().item()) if target_mask is not None else y_true.numel()
                    acc_epoch += acc * valid_tokens
                    loss_epoch += loss.item() * valid_tokens
                    total_tokens_epoch += valid_tokens
                    
                    self.optimizer.zero_grad()

                    # compute gradient
                    self.scaler.scale(loss).backward()
                    
                    # update weight
                    self.scaler.step(self.optimizer)

                    self.scaler.update()
                                    
            if scheduler is not None:
                # Tiến hành giảm LR sau số epoch được thiết lập ở tham số step_size của scheduler
                scheduler.step()

            train_loss = loss_epoch / total_tokens_epoch
            train_acc = acc_epoch / total_tokens_epoch
            self.Losses.append(train_loss)
            self.Accuracies.append(train_acc)

            # Hiển thị Loss, Accuracy trong quá trình huấn luyện sau mỗi epoch
            tqdm.write(f"Epoch [{epoch + 1}/{n_epochs}] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")

            # Sau 10 epoch thì hiển thị metrics của tập val nếu có (hoặc ở epoch cuối cùng)
            if self.val_loader is not None and ((epoch + 1) % 10 == 0 or epoch == n_epochs - 1):
                val_metrics = self.evaluate(
                    model=self.model,
                    val_loader=self.val_loader,
                    criterion=self.criterion,
                    vocab=vocab
                )
                self.Val_Losses.append(val_metrics['loss'])
                self.Val_Accuracies.append(val_metrics['accuracy'])
                self.Val_Bleu1.append(val_metrics['bleu1'])
                self.Val_Bleu2.append(val_metrics['bleu2'])
                self.Val_Bleu3.append(val_metrics['bleu3'])
                self.Val_Bleu4.append(val_metrics['bleu4'])

                tqdm.write(
                    f"--> [Validation @ Epoch {epoch + 1}] "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"BLEU-1: {val_metrics['bleu1']:.4f} | "
                    f"BLEU-2: {val_metrics['bleu2']:.4f} | "
                    f"BLEU-3: {val_metrics['bleu3']:.4f} | "
                    f"BLEU-4: {val_metrics['bleu4']:.4f}"
                )