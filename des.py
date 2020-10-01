from typing import List
import math

BYTE_ORDER = 'big'
NUM_CYCLES = 16
BLOCK_SIZE = 8
# TODO: Define expansion/permutation tables in file and load here

def _bytes_to_binary_str(b: bytes) -> str:
    return bin(int.from_bytes(b, 'big'))[2:].zfill(len(b) * 8)

def _binary_str_to_bytes(s: str) -> bytes:
    int(s, 2).to_bytes(len(s) // 8, 'big')

def _xor(b1: bytes, b2: bytes) -> bytes:
    assert(len(b1) == len(b2))
    return bytes([a ^ b for a, b in zip(b1, b2)])

def _left_cycle(b: bytes, shifts: int, num_bits: int) -> bytes:
    """
    Left cycles the bits of b
    ex: f(0010, 2) -> 1000
        f(0100, 2) -> 0001 
    """
    #num = int.from_bytes(b, 'big')
    #num_bits = len(b) * 8
    
    return (((b << shifts) + (b >> (num_bits - shifts))) % (2 ** num_bits))

def _apply_table(b: bytes, table: List[int]) -> bytes:
    # convert bytes to binary string
    binary_string = _bytes_to_binary_str(b)

    # build new binary string
    output = ['x'] * len(table)
    for pos, src in enumerate(table):
        output[pos] = binary_string[src]

    assert('x' not in output)

    # convert to bytes
    return _binary_str_to_bytes(''.join(output))


def encrypt(plaintext_bytes: bytes, key: bytes) -> bytes:
    # Create subkeys through key expansion
    # keys: List[bytes] = _create_keys(key)

    # Break plaintext into blocks and pad last block if necessary
    blocks: List[bytes] = []
    for i in range((len(plaintext_bytes) // BLOCK_SIZE) + 1):
        start = i * BLOCK_SIZE
        end = start + BLOCK_SIZE
        if end < len(plaintext_bytes):
            blocks.append(plaintext_bytes[start:end])
        else:
            # Need to pad last block
            padding_length = end - len(plaintext_bytes)
            if padding_length == BLOCK_SIZE:
                break
            blocks.append(
                plaintext_bytes[start:end] + bytes([0] * padding_length))


    # encrypt each block
    ciphertext: bytes = bytes()
    for block in blocks:
        _encrypt_block(block, key)

    # concate encrypted blocks and return result
    return ciphertext

# DES encryption is it's own inverse ?
# I think we have to reverse the key order though
decrypt = encrypt

def _create_keys(key: bytes) -> List[bytes]:
    # TODO: Create subkeys from key
    pc_table = [57, 49, 41, 33, 25, 17, 9,
                1, 58, 50, 42, 34, 26, 18,
                10, 2, 59, 51, 43, 35, 27,
                19, 11, 3, 60, 52, 44, 36,
                63, 55, 47, 39, 31, 23, 15,
                7, 62, 54, 46, 38, 30, 22,
                14, 6, 61, 53, 45, 37, 29,
                21, 13, 5, 28, 20, 12, 4]

    pc_2 = [14,17,11,24,1,5,
            3,28,15,6,21,10,
            23,19,12,4,26,8,
            16,7,27,20,13,2,
            41,52,31,37,47,55,
            30,40,51,45,33,48,
            44,49,39,56,34,53,
            46,42,50,36,29,32]
    zero = 1 << 63  # creates a bit array with zeros
    k_plus = 0 << 55  # the bit array where the new key will be
    num = int.from_bytes(key, 'big')  # turns current key to integer

    # does the shifting
    for val in pc_table:
        placement = 55 - pc_table.index(val)
        nth_place = zero >> (val - 1)
        if (num & nth_place) > 0:
            bit_nth_place = 1 << placement
            k_plus = k_plus | bit_nth_place
        zero = 1 << 63

    num_shifts = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

    left_halves: List[bytes] = [None] * (NUM_CYCLES + 1)
    right_halves: List[bytes] = [None] * (NUM_CYCLES + 1)

    c_zero = k_plus >> 28
    d_zero = (k_plus & 268435455)

    left_halves[0] = c_zero
    right_halves[0] = d_zero

    for i, shift in enumerate(num_shifts):
        pos = i + 1  # Don't shift first position
        left_halves[pos] = _left_cycle(left_halves[pos - 1], shift,28)
        right_halves[pos] = _left_cycle(right_halves[pos - 1], shift, 28)
    pass

    #concatenate left half and right  half together
    k_n = []
    for i in range(len(right_halves)):
        k1 = left_halves[i] << 28
        k1 = k1 | right_halves[i]
        k_n.append(k1)

    final_sub_key = []      #list will contain all final subkeys
    for k in k_n:
        zero = 1 << 55  # creates a bit array with zeros
        subk = 0 << 47  # the bit array where the new key will be
        for val in pc_2:
            placement = 47 - pc_2.index(val)
            nth_place = zero >> (val - 1)
            if (k & nth_place) > 0:
                bit_nth_place = 1 << placement
                subk = subk | bit_nth_place
            zero = 1 << 55
        final_sub_key.append(subk)

    #converts sub keys to binary strings
    for i in range(len(final_sub_key)):
       final_sub_key[i]= bin(final_sub_key[i])[2:].zfill(48)

def _encrypt_block(block: bytes, keys: List[bytes]) -> bytes:
    # TODO: Finish
    assert(len(block) == 8) # block should be 8 bytes/64 bits
    assert(len(keys) == 8) # block should be 8 bytes/64 bits

    # Perform initial permutation of message data

    # Cycle the block 16 times with the appropriate key-table combinations
    cipher_block = None
    for key in keys:
        # cipher_block = _des_round(cipher_block, key, )
        pass

    # Perform final permutation

    # return encrypted block



def _des_round(block: bytes, key: bytes) -> bytes:
    assert(len(block) == 8) # block should be 8 bytes/64 bits
    assert(len(key) == 6) # block should be 6 bytes/56 bits

    # Split Data in half
    left_half: bytes = block[:(len(block)/2)]
    right_half: bytes = block[(len(block)/2):]

    # Expand right half
    expanded_bytes = _expansion(right_half)

    expanded_bytes = _xor(expanded_bytes, key)

    compressed_bytes = _substitution(expanded_bytes)

    compressed_bytes = _permutation(compressed_bytes)

    new_right_half = _xor(compressed_bytes, left_half)

    # Combine left and right
    return right_half + new_right_half

def _expansion(half: bytes) -> bytes:
    assert(len(half) == 4) # half should be 32 bits
    expansion_table = [32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,24,25,26,27,28,29,28,29,30,31,32,1]

    expanded: bytes = _apply_table(half, expansion_table)

    assert(len(expanded) == 6) # expanded should be 48 bits
    return expanded

def _substitution(expanded_half: bytes, substitution_table: List[List[int]]) -> bytes:
    # TODO: Implement substitution table
    pass

def _permutation(expanded_half: bytes) -> bytes:
    assert(len(expanded_half) == 6) # expanded should be 48 bits
    permutation_table = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]

    permuted = _apply_table(expanded_half, permutation_table)

    assert(len(expanded_half) == 4) # half should be 32 bits
    return permuted


if __name__ == "__main__":
    plaintext = bytes.fromhex('abcd1234')
    key = bytes.fromhex('133457799BBCDFF1')

    key_temp = _create_keys(key)

    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print('These should be the same')
    print(plaintext, decrypted)