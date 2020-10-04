import argparse
import csv
from pathlib import Path
from typing import Dict
import math
import utils
import des


def build_ngrams():
    # Build ngram table
    ngrams: Dict[str, float] = {}

    for ngram_file in Path('./ngrams').glob('*.txt'):
        total = 0
        grams = {}
        with open(ngram_file, 'r') as fp:
            ngram_reader = csv.reader(fp, delimiter=' ')
            for row in ngram_reader:
                grams[row[0]] = int(row[1])
                total += int(row[1])

        # Normalize by total and upgrade ngrams
        for gram, count in grams.items():
            ngrams[gram] = (count / total)

    return ngrams

NGRAMS = build_ngrams()


def score_plaintext_logprob(plaintext: str) -> float:
    """
    Takes a string of decoded ciphertext and returns the log-odds that it's
    english text using unigram and bigram scoring
    """

    scores = []
    base_score = math.log(1 / 10000)
    for gram_size in range(1,3):
        score = 0
        for i in range(len(plaintext) - gram_size):
            gram = plaintext[i:(i + gram_size)].upper()
            if gram in NGRAMS:
                score += math.log(NGRAMS[gram])
            else:
                score += base_score
        scores.append(score)

    return sum(scores)

def score_plaintext_abs(plaintext: str) -> float:
    scores = []
    for gram_size in range(1,3):
        total = 0
        freqs = {}
        for i in range(len(plaintext) - gram_size):
            gram = plaintext[i:(i + gram_size)].upper()
            if gram.isalpha():
                total += 1
                freqs[gram] = freqs.get(gram, 0) + 1

        scores.append(sum([abs(NGRAMS[gram] - (count/total)) for gram,count in freqs.items()]))
    
    return scores

if __name__ == "__main__":
    plaintext = "Did you ever hear the Tragedy of Darth Plagueis the wise? I thought not. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life... He had such a knowledge of the dark side that he could even keep the ones he cared about from dying. The dark side of the Force is a pathway to many abilities some consider to be unnatural. He became so powerful... the only thing he was afraid of was losing his power, which eventually, of course, he did. Unfortunately, he taught his apprentice everything he knew, then his apprentice killed him in his sleep. It's ironic he could save others from death, but not himself."
    plaintext_bytes = utils.ascii_to_bytes(plaintext)
    key = bytes.fromhex('133457799BBCDFF1')
    ciphertext = des.encrypt(plaintext_bytes, key)

    partial_key = bytes.fromhex('133457799BBCDF')
    best = ''
    best_score = -10000.0
    for i in range(256):
        test_key = partial_key + bytes([i])
        decrypted = utils.bytes_to_ascii(des.decrypt(ciphertext, test_key))
        score = score_plaintext_logprob(decrypted)
        if score > best_score:
            best = decrypted
            best_score = score

    print('Decrypted: ', best, best_score)