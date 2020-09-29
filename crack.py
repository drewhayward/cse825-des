import argparse
import csv
from pathlib import Path
from typing import Dict
import math


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

# Simple caesar cipher to test cracking
def encrypt(text, s):
    result = ""
    # transverse the plain text
    for i in range(len(text)):
        char = text[i]
        # Encrypt uppercase characters in plain text
        
        if (char.isupper()):
            result += chr((ord(char) + s-65) % 26 + 65)
        # Encrypt lowercase characters in plain text
        else:
            result += chr((ord(char) + s - 97) % 26 + 97)
    return result

if __name__ == "__main__":
    plaintext = 'I am a sneaky message hidden by a ceasar cipher'.upper()
    ciphertext = encrypt(plaintext,5)

    best = ''
    best_score = -10000
    for i in range(26):
        decrypted = encrypt(ciphertext, i)
        score = score_plaintext_logprob(decrypted)
        print(decrypted, score)
        if score > best_score:
            best = decrypted
            best_score = score

    print('Decrypted: ', best, best_score)