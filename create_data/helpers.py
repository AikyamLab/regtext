import json
from collections import Counter
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


additional_filler_words = {
    'also', 'the', 'and', 'of', 'an', 'a', 'is', 'it', 'in', 'on', 'to', 'with', 'this', 'that', 'by', 'for', 
    'at', 'from', 'or', 'as', 'if', 'but', 'about', 'into', 'because', 'than', 'just', 'so', 'can', 'could', 
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'doing', 'would', 'should', 'now', 'get'
}

stop_words = set(stopwords.words('english')).union(additional_filler_words)


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

def remove_stopwords_and_fillers(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized_text)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords_and_fillers(text)
    text = remove_punctuation(text)
    text = lemmatize_text(text)
    return text

# helpers
def punctuation_present(text):
    pattern = r'[ !\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]'
    # Search the text for any punctuation
    if re.search(pattern, text):
        return True
    else:
        return False

def is_readable_text(bytes_obj):
    try:
        bytes_obj.decode('ascii')
        return True
    except UnicodeDecodeError:
        return False
        
def get_stats(ids, counts=None):
    if counts is None:
        counts = Counter(zip(ids, ids[1:]))
    else:
        counts.update(Counter(zip(ids, ids[1:])))
    return counts

def merge(ids, pair, idx):
    newids = []
    ids_len = len(ids)
    i = 0
    while i < ids_len:
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < ids_len - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def normalize_repetitions(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def is_valid_word(word, max_punctuations=3):

    punctuation_count = sum(1 for char in word if char in set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
    if punctuation_count > max_punctuations:
        return False

    if word.startswith(('http', 'www')) or '/' in word:
        return False

    return True

def has_no_numbers(word):
    return not any(char.isdigit() for char in word)

def is_mostly_not_upper(word, threshold=0.5):
    num_upper = sum(1 for char in word if char.isupper())
    return (num_upper / len(word)) < threshold


# min bpe
class Tokenizer:
    def __init__(self):
        self.merges = {} 
        self.pattern = "" 
        self.special_tokens = {} 
        self.vocab = self._build_vocab() 

    def train(self, text, vocab_size, verbose=False):
        
        raise NotImplementedError

    def encode(self, text):
        
        raise NotImplementedError

    def decode(self, ids):
        
        raise NotImplementedError

    def _build_vocab(self): # ignore this for poisoning
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            for idx in self.merges:
                f.write(f"{idx[0]} {idx[1]}\n")

        vocab_file = file_prefix + ".vocab"
        for k, v in self.vocab.items():
            self.vocab[k] = list(v)
            
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)

    def load(self, model_file, vocab_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            
            version = f.readline().strip()
            assert version == "minbpe v1"
            
            self.pattern = f.readline().strip()
            
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens

        with open(vocab_file, "r", encoding='utf-8') as f: 
            temp = json.load(f)
        self.vocab, self.readable_vocab = {}, {}
        for k, v in temp.items():  
            self.vocab[int(k)] = bytes(v)
            self.readable_vocab[int(k)] = render_token(bytes(v))

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes) 

        
        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)} 
        for i in range(num_merges):
            
            stats = get_stats(ids)
            
            pair = max(stats, key=stats.get)
            
            idx = 256 + i
            
            ids = merge(ids, pair, idx)
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        
        self.merges = merges 
        self.vocab = vocab   
        self.readable_vocab = []
        
        
        for _, v in self.vocab.items():
            self.readable_vocab.append(v.decode('utf-8', errors="replace"))

    def decode(self, ids):
        
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes) 
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
