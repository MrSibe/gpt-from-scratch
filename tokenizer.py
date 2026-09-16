"""两个小包装：训练、数据准备和生成使用同一套 encode/decode 接口。"""

from tokenizers import Tokenizer

EOS = "<|endoftext|>"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.eos_id = None

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def state(self):
        return {"kind": "char", "chars": self.chars}


class BPETokenizer:
    def __init__(self, tokenizer):
        if tokenizer.truncation is not None or tokenizer.padding is not None:
            raise ValueError(
                "BPE tokenizer 不能启用 truncation/padding，否则会截断或填充故事"
            )
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.eos_id = tokenizer.token_to_id(EOS)
        if self.eos_id is None:
            raise ValueError(f"BPE tokenizer 缺少 {EOS}")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def state(self):
        # 存字符串而非 Python tokenizer 对象，checkpoint 仍支持 weights_only=True。
        return {"kind": "bpe", "json": self.tokenizer.to_str()}


def tokenizer_from_state(state):
    if state["kind"] == "char":
        return CharTokenizer(state["chars"])
    if state["kind"] == "bpe":
        return BPETokenizer(Tokenizer.from_str(state["json"]))
    raise ValueError(f"未知 tokenizer: {state['kind']}")
