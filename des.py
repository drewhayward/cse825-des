from typing import List
import math

# TODO: Define expansion/permutation tables in file and load here



def _xor(b1: bytes, b2: bytes) -> bytes:
    assert(len(b1) == len(b2))
    return bytes([a ^ b for a, b in zip(b1, b2)])

def _left_cycle(b: bytes, shifts: int) -> bytes:
    """
    Left cycles the bits of b
    ex: f(0010, 2) -> 1000
        f(0100, 2) -> 0001 
    """
    num = int(b, base=16)
    num_bits = len(b) * 4
    
    return (((num << shifts) + (num >> (num_bits - shifts))) % (2 ** num_bits)).to_bytes(num_bits // 8, byteorder='big')


def encrypt(plaintext_bytes: bytes, key: bytes) -> bytes:
    # Break plaintext into blocks and pad last block if necessary
    # encrypt each block

    # concate encrypted blocks and return result
    ciphertext: bytes
    return ciphertext

decrypt = encrypt # DES encryption is it's own inverse

def _create_keys(key: bytes) -> List[bytes]:
    # TODO: Create the C and D subblocks and use them to create 16 subkeys for encryption
    pass

def _encrypt_block(block: bytes, key: bytes) -> bytes:
    assert(len(block) == 8) # block should be 8 bytes/64 bits
    assert(len(key) == 8) # block should be 8 bytes/64 bits

    # Create subkeys through key expansion
    keys = _create_keys(key)

    # Perform initial permutation of message data

    # Cycle the block 16 times with the appropriate key-table combinations
    cipher_block = None
    for key in keys:
        # cipher_block = _des_cycle(cipher_block, key, )
        pass

    # return encrypted block



def _des_cycle(block: bytes, key: bytes, expansion_table, compression_table) -> bytes:
    assert(len(block) == 8) # block should be 8 bytes/64 bits
    assert(len(key) == 6) # block should be 6 bytes/56 bits

    # Permute key

    # Split Data in half
    left_half: bytes = block[:(len(block)/2)]
    right_half: bytes = block[(len(block)/2):]

    assert(len(left_half) == 4)
    assert(len(left_half) == len(right_half))

    # Expand right half
    expanded_bytes = _expand_half(right_half, expansion_table)

    expanded_bytes = _xor(expanded_bytes, key)

    compressed_bytes = _permuted_choice(expanded_bytes, compression_table)

    new_right_half = _xor(compressed_bytes, left_half)

    # Combine left and right
    return right_half + new_right_half

def _expand_half(half: bytes, expansion_table) -> bytes:
    assert(len(half) == 4) # half should be 32 bits

    # TODO: perform byte expansion according to table
    expanded: bytes

    assert(len(expanded) == 6) # expanded should be 48 bits
    return expanded

def _permuted_choice(expanded_half: bytes, compression_table) -> bytes:
    assert(len(expanded_half) == 6) # expanded should be 48 bits

    # TODO: perform permuted choice to reduce the 48 bit block to 32 bits

    assert(len(expanded_half) == 4) # half should be 32 bits
    pass


if __name__ == "__main__":
    plaintext = b'abcd1234'
    key = b'1234abcd'

    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print('These should be the same')
    print(plaintext, decrypted)