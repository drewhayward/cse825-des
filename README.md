# Installation
Use:
`pip install -r requirements.txt`
to install fast bitarrays and cracking progress bars.

# Usage
The main script takes files with ASCII as inputs. Key and ciphertext files are output in hex.

To encrypt:
`python main.py encrypt <plaintext_file> <ciphertext_file> <key_file>`

To decrypt
`python main.py decrypt <ciphertext_file> <plaintext_file> <key_file>`

# DES Cracking
The cracking script will brute force a range of keys and score the decoded plaintext using unigram and bigram english character occurances.