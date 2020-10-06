from typing import List, Optional
from utils import ascii_to_bytes, bytes_to_ascii
from bitarray import bitarray
import math

BYTE_ORDER = 'big'
NUM_CYCLES = 16
BLOCK_SIZE = 64
# TODO: Define expansion/permutation tables in file and load here
# Initial Permutation
IP = [58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46,
      38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17,
      9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55,
      47, 39, 31, 23, 15, 7]

IP_INV = [40, 8, 48, 16, 56, 24, 64, 32,
          39, 7, 47, 15, 55, 23, 63, 31,
          38, 6, 46, 14, 54, 22, 62, 30,
          37, 5, 45, 13, 53, 21, 61, 29,
          36, 4, 44, 12, 52, 20, 60, 28,
          35, 3, 43, 11, 51, 19, 59, 27,
          34, 2, 42, 10, 50, 18, 58, 26,
          33, 1, 41, 9, 49, 17, 57, 25]
# Expands the bits
expandBit = [32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 14, 15,
             16, 17, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25, 24, 25, 26, 27,
             28, 29, 28, 29, 30, 31, 32, 1]

S1Mat = [  # S1 matrix
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
]
S = [
    [  # S1 matrix
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    [  # S2 Matrix
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    [  # S3 Matrix
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    [  # S4 Matrix
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    [  # S5 Matrix
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    [  # S6 Matrix
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    [  # S7 Matrix
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

# F Permutation
fPerm = [16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10, 2, 8, 24, 14,
         32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25]


def _bytes_to_binary_str(b: bytes) -> str:
    return bin(int.from_bytes(b, 'big'))[2:].zfill(len(b) * 8)


def _binary_str_to_bytes(s: str) -> bytes:
    return int(s, 2).to_bytes(len(s) // 8, 'big')

# Turn S values to bitarrays
bitarray_S = []
for S_mat in S:
    mat = []
    for row in S_mat:
        new_row = []
        for val in row:
            new_row.append(bitarray(_bytes_to_binary_str(val.to_bytes(1, 'big'))[4:]))
        mat.append(new_row)
    bitarray_S.append(mat)

def _xor(b1: bytes, b2: bytes) -> bytes:
    assert(len(b1) == len(b2))
    return (int.from_bytes(b1, 'big') ^ int.from_bytes(b2, 'big')).to_bytes(len(b1), 'big')


def _left_cycle(b: bitarray, shifts: int) -> int:
    """
    Left cycles the bits of b
    ex: f(0010, 2) -> 1000
        f(0100, 2) -> 0001 
    """
    num_bits = len(b)
    return b[shifts:] + b[:shifts]

def _apply_table(b: bitarray, table: List[int]) -> bitarray:
    # build new binary string
    output = bitarray(len(table))
    for pos, src in enumerate(table):
        output[pos] = b[src-1]

    # convert to bytes
    return output


def _chunkify_message(M: bitarray) -> List[bitarray]:
    """
    Breaks message bytes M into blocks and pads last block with 0s if necessary
    """
    blocks: List[bitarray] = []
    for i in range((len(M) // BLOCK_SIZE) + 1):
        start = i * BLOCK_SIZE
        end = start + BLOCK_SIZE
        if end < len(M):
            blocks.append(M[start:end])
        else:
            # Need to pad last block
            padding_length = end - len(M)
            if padding_length == BLOCK_SIZE:
                break
            blocks.append(
                M[start:end] + bitarray([0] * padding_length))
    return blocks


def encrypt(plaintext_bytes: bytes, key: bytes) -> bytes:
    if len(key) != 8:
        raise Exception('Incorrect key length')

    plaintext_bytes = bitarray(_bytes_to_binary_str(plaintext_bytes))
    key = bitarray(_bytes_to_binary_str(key))
    # Create subkeys through key expansion
    keys: List[bitarray] = _create_keys(key)

    blocks: List[bitarray] = _chunkify_message(plaintext_bytes)
    # encrypt each block
    ciphertext = bitarray()

    for block in blocks:
        ciphertext += _encrypt_block(block, keys)

    # concate encrypted blocks and return result
    return ciphertext.tobytes()


def decrypt(ciphertext_bytes: bytes, key: bytes) -> bytes:
    if len(key) != 8:
        raise Exception('Incorrect key length')

    ciphertext_bytes = bitarray(_bytes_to_binary_str(ciphertext_bytes))
    key = bitarray(_bytes_to_binary_str(key))
    # Create subkeys through key expansion
    keys: List[bitarray] = _create_keys(key)
    keys.reverse()

    blocks: List[bitarray] = _chunkify_message(ciphertext_bytes)
    # encrypt each block
    plaintext = bitarray()

    for block in blocks:
        plaintext += _encrypt_block(block, keys)

    # concate encrypted blocks and return result
    return plaintext.tobytes()


def _create_keys(key: bitarray) -> List[bitarray]:
    # TODO: Create subkeys from key
    pc_table = [57, 49, 41, 33, 25, 17, 9,
                1, 58, 50, 42, 34, 26, 18,
                10, 2, 59, 51, 43, 35, 27,
                19, 11, 3, 60, 52, 44, 36,
                63, 55, 47, 39, 31, 23, 15,
                7, 62, 54, 46, 38, 30, 22,
                14, 6, 61, 53, 45, 37, 29,
                21, 13, 5, 28, 20, 12, 4]

    pc_2 = [14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32]
    
    key = _apply_table(key, pc_table)

    num_shifts = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

    left_halves: List[bitarray] = [bitarray()] * (NUM_CYCLES + 1)
    right_halves: List[bitarray] = [bitarray()] * (NUM_CYCLES + 1)

    left_halves[0] = key[:(len(key) // 2)]
    right_halves[0] = key[(len(key) // 2):]

    for i, shift in enumerate(num_shifts):
        pos = i + 1  # Don't shift first position
        left_halves[pos] = _left_cycle(left_halves[pos - 1], shift)
        right_halves[pos] = _left_cycle(right_halves[pos - 1], shift)

    keys = [_apply_table(left + right, pc_2) for left, right in zip(left_halves[1:], right_halves[1:])]

    return keys


def _encrypt_block(block: bitarray, keys: List[bitarray]) -> bitarray:
    '''
    Encrypts the block of data using all the generated keys.
    '''
    assert(len(block) == BLOCK_SIZE)  # block should be 8 bytes/64 bits
    assert(len(keys) == NUM_CYCLES)  # Should have 16 keys.

    # Perform initial permutation of message data
    # Get the block into binary then permute.
    permuted_block = _apply_table(block, IP)

    # Cycle the block 16 times with the appropriate key-table combinations
    cipher_block = permuted_block
    for key in keys:
        cipher_block = _des_round(cipher_block, key)

    # Flip Left and Right blocks
    cipher_block = cipher_block[(len(cipher_block) // 2):] + cipher_block[:(len(cipher_block) // 2)]

    # Perform final permutation
    cipher_block = _apply_table(cipher_block, IP_INV)

    return cipher_block


def _des_round(block: bitarray, key: bitarray) -> bitarray:
    '''
    A single round of DES encryption
    '''
    assert(len(block) == BLOCK_SIZE)  # block should be 8 bytes/64 bits
    assert(len(key) == 48)  # Key should be 48 bits

    # Split Data in half
    left_half: bitarray = block[:(len(block) // 2)]
    right_half: bitarray = block[(len(block) // 2):]

    # Expand right half
    expanded_bytes = _expansion(right_half)
    # Exclusive or with the key and the expanded bytes
    expanded_bytes = expanded_bytes ^ key
    # Substitute then convert to hex
    compressed_bytes = _substitution(expanded_bytes)
    # Permute the compressed value
    compPerm_bytes = _permutation(compressed_bytes)
    # Exclusive or with the compressed and permuted half with the left half then convert to hex
    new_right_half = compPerm_bytes ^ left_half
    # Combine left and right
    return right_half + new_right_half



def _expansion(half: bitarray) -> bitarray:
    '''
    Expand the half for DES
    '''
    assert(len(half) == (BLOCK_SIZE // 2))  # half should be 32 bits
    expansion_table = [32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16,
                       17, 16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25, 24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1]

    expanded: bitarray = _apply_table(half, expansion_table)

    assert(len(expanded) == 48)  # expanded should be 48 bits
    return expanded


def _substitution(expanded_half: bitarray) -> bitarray:
    '''
    Use substitution on the expanded half
    '''
    # TODO: Implement substitution table
    # Use the S1Mat for the Substitution matrix
    expanded_half = expanded_half
    totalCount = len(expanded_half) // 6
    bitStr = bitarray()

    # Need to look at every 6 bits. The oth and 5th bit of a group, then the 1st through 4th bits for the inner.
    for i in range(totalCount):
        a = i*6
        aBit = expanded_half[a]
        b = i*6 + 5
        bBit = expanded_half[b]
        cStr = expanded_half[a+1:b]

        iNum = int(bitarray([aBit, bBit]).to01(), 2)
        jNum = int(cStr.to01(), 2)
        SNum = bitarray_S[i][iNum][jNum]

        bitStr += SNum
    return bitStr


def _permutation(compressed_bytes: bitarray) -> bitarray:
    '''
    Uses the permutation table on the compressed half.
    '''
    assert(len(compressed_bytes) == (BLOCK_SIZE // 2))  # expanded should be 32 bits
    permutation_table = [16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5,
                         18, 31, 10, 2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25]

    permuted = _apply_table(compressed_bytes, permutation_table)

    assert(len(compressed_bytes) == (BLOCK_SIZE // 2))  # half should be 32 bits
    return permuted

if __name__ == "__main__":
    plaintext = bytes.fromhex('0123456789ABCDEF')
    key = bytes.fromhex('133457799BBCDFF1')
    print(f'Using {key.hex()} for a key')

    plaintext_bytes = plaintext
    ciphertext = encrypt(plaintext_bytes, key)
    print('Your ciphertext is:')
    print(ciphertext.hex())

    print('Your decrypted ciphertext is')
    print(bytes_to_ascii(decrypt(ciphertext, key)))
