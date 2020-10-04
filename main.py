import argparse
import des
from utils import ascii_to_bytes, bytes_to_ascii
from pathlib import Path
# File for encryption CLI

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', metavar='[encrypt|decrypt]', choices=['encrypt', 'decrypt'])
    parser.add_argument('INPUT_FILE', type=Path)
    parser.add_argument('OUTPUT_FILE')
    return parser.parse_args()

if __name__ == "__main__":
    ARGS = parse_args()
    key = bytes.fromhex('133457799BBCDFF1')

    if ARGS.mode == 'encrypt':
        # Read plaintext
        with open(ARGS.INPUT_FILE, 'r') as f:
            plaintext = f.read()
        ciphertext = des.encrypt(ascii_to_bytes(plaintext), key)

        # Save ciphertext as hex
        with open(ARGS.OUTPUT_FILE, 'w') as f:
            f.write(ciphertext.hex())

    elif ARGS.mode == 'decrypt':
        with open(ARGS.INPUT_FILE, 'r') as f:
            ciphertext = bytes.fromhex(f.read())
        
        plaintext = bytes_to_ascii(des.decrypt(ciphertext, key))

        print(plaintext)
        with open(ARGS.OUTPUT_FILE, 'w') as f:
            f.write(plaintext)
    