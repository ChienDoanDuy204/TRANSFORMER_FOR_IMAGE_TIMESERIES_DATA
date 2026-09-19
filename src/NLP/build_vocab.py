from collections.abc import Iterable
import json
import torch.nn as nn
"""
iterable: đối tượng có thể lặp - giống như quấn sách có lật ra từng trang nhưng nó lại không tự động lật trang cho bạn.
iterator: đối tượng thực hiện việc lặp - giống như tay của bạn khi lật từng trang sách. Mỗi lần bạn lật là bạn đang sử dụng iterator.
khi muốn lật sang trang khác ta dùng next(iterator). Iterator thực chất là một con trỏ, ghi nhớ vị trí hiện tại, và next() sẽ di chuyển con trỏ sang phần tử tiếp theo
khi được gọi. Khi cần tạo ra một iterator từ 1 iterable, ta dùng hàm iter()

iterable: là một object mà python có thể tạo ra một iterator từ nó để lấy lần lượt ra các phần tử bằng phương thức next()
và ghi nhớ vị trí đang đứng 

MỤC ĐÍCH: DÙNG ITERABLE VÀ ITERATOR trong bài này để load DL khi cần dùng tránh việc load toàn bộ
lên RAM có thể khiến tràn RAM 
"""
# Mục tiêu là load toàn bộ corpus để xây dựng toàn bộ từ điển cùng một lúc, nhưng không thể do dung lượng RAM giới hạn.
# Ta sẽ load từng câu trong corpus, tính toán tần suất và xây dựng từ điển theo từng câu.




class BuildVocabFromIterator(nn.Module):
    """
    Class này có nhiệm vụ xây dựng từ điển từ một tập hợp các từ (iterator).
    param: 
        iterator : Iterable[str] : Tập hợp các từ cần xây dựng từ điển
        vocab_size : int : Kích thước từ điển
        special_tokens : list[str] : Danh sách các token đặc biệt
    """
    def __init__(self, iterator : Iterable[str] = None, vocab_size: int = 1000 , special_tokens : list[str] = None) ->None:
        super().__init__()
        if iterator is not None:
            # dict = {word:frequency}
            self.word_frequency = {}
            # nếu không truyền vào special_tokens thì mặc định là ['<unk>', '<pad>', '<sos>', '<eos>']
            if special_tokens is None:
                self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
            else: 
                self.special_tokens = special_tokens
            self.vocab_size = vocab_size
            # dict = {special_tokens:idx}
            self.vocab = { token: idx for idx, token in enumerate(self.special_tokens)}

            # load toàn bộ corpus để xây dựng toàn bộ từ điển cùng một lúc
            tokens = next(iterator, 0)
            while tokens:
                for token in tokens:
                    self.word_frequency[token] = self.word_frequency.get(token, 0) + 1
                tokens = next(iterator,0)

            # Sắp xếp các token theo trình tự giảm dần tần suất xuất hiện.
            TopK_word_MostFreq = sorted(self.word_frequency, key= lambda x:self.word_frequency[x], reverse=True)[:self.vocab_size-4]
            # get topvocab_size-4 token have most frequency 
            for idx, token in enumerate(TopK_word_MostFreq):
                self.vocab[token] = idx + 4
            
            # string2index
            self.stoi = self.vocab
            # index2string
            self.itos = {value: key for key, value in self.stoi.items()}
        else:
            pass
    
    # idx mặc định cho những token không có trong vocab - OOV
    def set_default(self, token: str = None):
        if token is None:
            return self.stoi.get('<unk>')
        return self.stoi.get(token)
    
    def get_padding_token(self, token: str = None):
        if token is None:
            return self.stoi.get('<pad>')
        return self.stoi.get(token)

    def get_start_token(self, token: str = None):
        if token is None:
            return self.stoi['<sos>']
        return self.stoi.get(token)

    def get_end_token(self, token: str = None):
        if token is None:
            return self.stoi['<eos>']
        return self.stoi.get(token)
        
    def forward(self, tokens : list[str]) -> list[int]:
        return [self.stoi.get(token, self.set_default()) for token in tokens]
    
    # Hàm sinh chuỗi từ idx sang text(token)
    def vocab_reverse(self, tokens : list[int]) -> list[str]:
        return [self.itos.get(token) for token in tokens]
    
    # lưu vocab dưới dạng json để tái sử dựng ở nhiều nơi
    def save_vocab(self, path : str) ->None:
        config_vocab = {
            'stoi' : self.stoi,
            'itos' : self.itos,
            'vocab_size' : self.vocab_size,
            'special_tokens' : self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_vocab, f, ensure_ascii=False, indent=4)
            print(f"Vocab saved to {path}")
    # nạp vocab từ file json
    def load_vocab(self, path : str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            config_vocab = json.load(f)
        self.stoi = config_vocab['stoi']
        # Đảm bảo key của itos là int (do file JSON chỉ lưu key dưới dạng string)
        self.itos = {int(k): v for k, v in config_vocab['itos'].items()}
        self.vocab_size = config_vocab['vocab_size']
        self.special_tokens = config_vocab['special_tokens']
        print(f"Vocab loaded from {path}")