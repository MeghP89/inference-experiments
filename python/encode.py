"""Reference tokenizer, for checking the C++ one or feeding it ids directly.

    python encode.py "Hello world"   -> 15496 995
    python encode.py -d 15496 995    -> Hello world
"""

import sys
from transformers import GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "-d":
        print(tok.decode([int(a) for a in args[1:]]))
    else:
        print(" ".join(str(i) for i in tok.encode(" ".join(args))))
