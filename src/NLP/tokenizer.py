# word_level_tokenization
import string
class Tokenizer:
    def __init__(self) -> None:
        pass
    def __call__(self, text : str) -> list[str]:
        ######## pre processing ########
        text = text.lower() # Chuyển thành chữ viết thường
        char_remove = string.punctuation + string.digits # Xóa tất cả các ký tự không phải chữ và số
        for char in char_remove:
            if char == '_':
                text = text.replace(char,'')
        return text.split()
