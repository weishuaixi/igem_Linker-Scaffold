from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<PAD>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"
    mask: str = "<MASK>"
    left: str = "<LEFT>"
    end_left: str = "<END_LEFT>"
    right: str = "<RIGHT>"
    end_right: str = "<END_RIGHT>"


class RnaTokenizer:
    """Tiny character tokenizer for RNA bases plus scaffold control tokens."""

    def __init__(self) -> None:
        self.special = SpecialTokens()
        self.tokens = [
            self.special.pad,
            self.special.bos,
            self.special.eos,
            self.special.mask,
            self.special.left,
            self.special.end_left,
            self.special.right,
            self.special.end_right,
            "A",
            "U",
            "C",
            "G",
        ]
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.special.pad]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.special.bos]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.special.eos]

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        ids: list[int] = []
        index = 0
        special_tokens = sorted(
            [t for t in self.tokens if t.startswith("<")],
            key=len,
            reverse=True,
        )
        while index < len(text):
            matched = None
            for token in special_tokens:
                if text.startswith(token, index):
                    matched = token
                    break
            if matched is not None:
                ids.append(self.token_to_id[matched])
                index += len(matched)
                continue

            char = text[index].upper()
            if char not in self.token_to_id:
                raise ValueError(f"Unsupported RNA token: {text[index]!r}")
            ids.append(self.token_to_id[char])
            index += 1

        if add_bos_eos:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        decoded = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if token == self.special.pad:
                continue
            decoded.append(token)
        return "".join(decoded)
