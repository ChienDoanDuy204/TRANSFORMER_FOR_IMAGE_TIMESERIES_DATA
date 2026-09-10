from pathlib import Path
from torch.utils.data import Dataset
import json
from PIL import Image
from torchvision.transforms import transforms

# work segmentation cho Tiếng Việt
from underthesea import word_tokenize

class ImageCaptionDataSet(Dataset):
    """
    Đây là dataset được custom theo mục đích của dự án, được kế thừa từ lớp Dataset trong pytorch
    parameters:
    img_dir: đường dẫn của thư mục chứa ảnh 
    caption_dir: đường dẫn của thư mục chứa caption
    split: Tập dữ liệu muốn lấy gồm các tập ["train","val","test"]
    transformation: list các phép biến đổi tác động lên ảnh
    img_size: kích thước của ảnh
    max_length: độ dài tối đa của caption
    """
    def __init__(self, img_dir: str = None, caption_dir: str = None, split: str = 'train', transformation: list = None, img_size : tuple = (224,224), max_length: int = None, vocab = None):
        super().__init__()
        self.img_dir = Path(img_dir) if img_dir is not None else None # Đường dẫn chứa ảnh
        self.caption_dir = Path(caption_dir) if caption_dir is not None else None # Đường dẫn thư mục chứa caption
        self.sample = [] # sample include img_name and index of caption

        list_transforms = [transforms.Resize(img_size),transforms.ToTensor()]
        if transformation:
            list_transforms.extend(transformation)
        self.transformer = transforms.Compose(list_transforms)

        self.max_length = max_length # Độ dài tối đa của caption
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
    
    def processing_caption(self, caption):
        pass

    def __getitem__(self, idx):
        img_name, idx_caption = self.sample[idx]
        img_path = self.img_dir / img_name
        caption = self.data[img_name]['captions'][idx_caption]
        tokens = word_tokenize(caption, format = 'text')
        img = self.processing_img(img_path)
        return img, tokens