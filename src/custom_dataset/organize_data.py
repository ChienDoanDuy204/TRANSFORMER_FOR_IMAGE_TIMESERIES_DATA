from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import sys
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import transforms

# work segmentation cho Tiếng Việt
from underthesea import word_tokenize


ROOT_DIR = Path.cwd().parent
# add root to sys.path
sys.path.append(str(ROOT_DIR))
from src.NLP.tokenizer import *

class ImageCaptionDataSet(Dataset):
    """
    Đây là dataset được custom theo mục đích của dự án, được kế thừa từ lớp Dataset trong pytorch
    parameters:
    img_dir: đường dẫn của thư mục chứa ảnh 
    caption_dir: đường dẫn của thư mục chứa caption
    split: Tập dữ liệu muốn lấy gồm các tập ["train","val","test"]
    transformation: list các phép biến đổi tác động lên ảnh
    img_size: kích thước của ảnh
    """
    def __init__(self, img_dir: str = None, caption_dir: str = None, split: str = 'train', transformation: list = None, img_size : tuple = (224,224), vocab = None):
        super().__init__()
        self.img_dir = Path(img_dir) if img_dir is not None else None # Đường dẫn chứa ảnh
        self.caption_dir = Path(caption_dir) if caption_dir is not None else None # Đường dẫn thư mục chứa caption
        if self.img_dir is None or self.caption_dir is None:
            raise ValueError("img_dir or caption_dir is None")
        self.sample = [] # sample include img_name and index of caption

        list_transforms = [transforms.Resize(img_size),transforms.ToTensor()]
        if transformation is not None:
            list_transforms.extend(transformation)
        self.transformer = transforms.Compose(list_transforms)

        self.vocab = vocab
        self.tokenizer = Tokenizer() 
        if split in ['train','val','test']:
            self.split = split
        else:
            raise ValueError(f'Split {split} is not valid. Must be one of ["train","val","test"]')
        
        split_path = None
        if split == 'train':
            split_path = self.caption_dir/'uit-openviic-annotation-train.json'
        elif split == 'val':
            split_path = self.caption_dir/'uit-openviic-annotation-dev.json'
        elif split == 'test':
            split_path = self.caption_dir/'uit-openviic-annotation-test.json'
        
        with open(str(split_path), 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        for img_name, captions in self.data.items():
            for idx_caption in range(len(captions['captions'])):
                self.sample.append((img_name, idx_caption))
    def __len__(self):
        return len(self.sample)
    
    def processing_img(self, image_path):
        img = Image.open(str(image_path)).convert('RGB')
        img = self.transformer(img)
        return img
    
    def processing_caption(self, tokens):
        idx_tokens = self.vocab(tokens)
        # Thêm <SOS> token ở đầu câu, <EOS> ở cuối câu
        idx_tokens = [self.vocab.get_start_token()] + idx_tokens + [self.vocab.get_end_token()]
        return idx_tokens


    def __getitem__(self, idx):
        img_name, idx_caption = self.sample[idx]
        img_path = self.img_dir / img_name
        caption = self.data[img_name]['captions'][idx_caption]
        tokens = word_tokenize(caption, format = 'text')
        img = self.processing_img(img_path)

        if self.vocab is not None:
            idx_tokens = self.processing_caption(tokens)
            return img, idx_tokens
        return img, tokens


# hàm padding động (dynamic padding) theo độ dài của câu dài nhất trong batch
def collate_fn(batch, idx_padd_token: int = None):
    '''
    batch: list tuple (img, idx_tokens - có độ dài khác nhau)
    - image: Tensor(B, 3, H, W)- có kích thước cố định giữa các sample do sử dụng transforms.Resize
    - idx_tokens: list idx có độ dài khác nhau giữa các sample
    return: 
    - imgs: Tensor(B, 3, H, W)
    - idx_tokens: Tensor(B, max_seq_len trong batch)
    - key_padding_mask: Tensor(B, max_seq_len trong batch) - True = vị trí padding
    '''
    imgs, list_idx_tokens = zip(*batch)
    #imgs: torch.tensor

    # -------------- Ảnh: stack - xếp chồng nhiểu ảnh trực tiếp vì có shape cố định ---------
    imgs = torch.stack(imgs, dim = 0) # batch_first = True -> (B, C, H, W)

    #-------------- idx_tokens: padding động theo độ dài của câu dài nhất trong batch -----------
    list_idx_tokens = [torch.tensor(idx_token) for idx_token in list_idx_tokens]
    padded_idx_tokens = pad_sequence(list_idx_tokens, batch_first =True, padding_value=idx_padd_token)

    #-------------- Tạo key_padding_mask: Tensor(B, max_seq_len trong batch) - True = vị trí padding -----------
    key_padding_mask = (padded_idx_tokens == idx_padd_token)
    return imgs, padded_idx_tokens, key_padding_mask
    
