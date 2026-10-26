"""Register the T-lite-it-2.1 Qwen2 pre-tokenizer on the pinned llama.cpp converter.

llama.cpp b11476 hashes `tokenizer.encode(chktxt)` and looks the digest up in
`conversion/base.py`. T-lite uses the Qwen2 regex. Bootstrap applies this
registration before cmake. The text-encoder bakeoff lives in voxprint-canvas,
not in this repo.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# The probe string in get_vocab_base_pre at llama.cpp 98819068. Do not reflow it.
_CHKTXT_PATH = Path(__file__).resolve().parent / "data" / "llama_b11476_chktxt.txt"


def chktxt() -> str:
    """The pre-tokenizer text the T-lite converter must contain."""
    return _CHKTXT_PATH.read_text(encoding="utf-8")

T_LITE_HASH = "e9b7dbd66e0308c6e89983d5b6e1ca047106d862879a0fd33a12c8491b91ec5c"
T_LITE_PRE = "qwen2"

_KIMI = (
    '        if chkhsh == "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890":\n'
    "            # ref: https://huggingface.co/moonshotai/Kimi-K2-Base\n"
    '            res = "kimi-k2"\n'
)
_T_LITE = (
    f'        if chkhsh == "{T_LITE_HASH}":\n'
    "            # ref: https://huggingface.co/t-tech/T-lite-it-2.1 (Qwen3-8B base, Qwen2 pre-tokenizer regex)\n"
    f'            res = "{T_LITE_PRE}"\n'
)


def pretokenizer_hash(token_ids: list[int]) -> str:
    """Same digest llama.cpp stores: sha256 of the Python repr of the id list."""
    return hashlib.sha256(str(token_ids).encode()).hexdigest()


def ids_match(hf_ids: list[int], llama_ids: list[int]) -> bool:
    """True when the Hub and llama.cpp token id lists are equal."""
    return hf_ids == llama_ids


def parse_llama_tokenize(text: str) -> list[int]:
    """Parse token ids from llama-tokenize output."""
    ids: list[int] = []
    for token in text.replace(",", " ").split():
        body = token.strip()
        if body.lstrip("-").isdigit():
            ids.append(int(body))
    return ids


def recognised(source: str) -> bool:
    """True when this converter source already has the T-lite pre-tokenizer hash."""
    return T_LITE_HASH in source and 'res = "qwen2"' in source


def apply_t_lite(path: Path) -> str:
    """Insert the T-lite hash after the Kimi-K2 entry. Idempotent."""
    text = path.read_text(encoding="utf-8")
    if recognised(text):
        return "present"
    if _KIMI not in text:
        raise SystemExit(f"{path} has no Kimi-K2 pre-tokenizer entry to anchor the T-lite hash")
    path.write_text(text.replace(_KIMI, _KIMI + _T_LITE, 1), encoding="utf-8", newline="\n")
    return "applied"


def main(argv: list[str] | None = None) -> int:
    """Patch or check the llama.cpp converter and return an exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m voxprint_forge.llama_patch conversion/base.py", file=sys.stderr)
        return 2
    print(apply_t_lite(Path(args[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
