from typing import List
import math

BYTE_ORDER = 'big'
NUM_CYCLES = 16
BLOCK_SIZE = 8
# TODO: Define expansion/permutation tables in file and load here
#Initial Permutation
initPerm = [58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,62,54,46,
            38,30,22,14,6,64,56,48,40,32,24,16,8,57,49,41,33,25,17,
            9,1,59,51,43,35,27,19,11,3,61,53,45,37,29,21,13,5,63,55,
            47,39,31,23,15,7]
#Expands the bits
expandBit = [32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,12,13,14,15,
             16,17,16,17,18,19,20,21,20,21,22,23,24,25,24,25,26,27,
             28,29,28,29,30,31,32,1]
#S1 matrix
S1Mat = [
         [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
         [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
         [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
         [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]
        ]
#F Permutation
fPerm = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,2,8,24,14,
           32,27,3,9,19,13,30,6,22,11,4,25]

def _bytes_to_binary_str(b: bytes) -> str:
    return bin(int.from_bytes(b, 'big'))[2:].zfill(len(b) * 8)

def _binary_str_to_bytes(s: str) -> bytes:
    return int(s, 2).to_bytes(len(s) // 8, 'big')

def _xor(b1: bytes, b2: bytes) -> bytes:
    b1Binary = _bytes_to_binary_str(b1)
    b2Binary = _bytes_to_binary_str(b2)
    assert(len(b1Binary) == len(b2Binary))
    b3 = ''.join('0' if i == j else '1' for i,j in zip(b1Binary, b2Binary))
    return (b3)

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
        output[pos] = binary_string[src-1]

    assert('x' not in output)

    # convert to bytes
    return _binary_str_to_bytes(''.join(output))

def _chunkify_message(M: bytes) -> List[bytes]:
    """
    Breaks message bytes M into blocks and pads last block with 0s if necessary
    """
    blocks: List[bytes] = []
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
                M[start:end] + bytes([0] * padding_length))
    return blocks

def encrypt(plaintext_bytes: bytes, key: bytes) -> bytes:
    # Create subkeys through key expansion
    keys: List[bytes] = _create_keys(key)

    blocks: List[bytes] = _chunkify_message(plaintext_bytes)
    # encrypt each block
    ciphertext: bytes = bytes()
    
    for block in blocks:
        ciphertext += _encrypt_block(block, keys)

    # concate encrypted blocks and return result
    return ciphertext

def decrypt(ciphertext_bytes: bytes, key: bytes) -> bytes:
    # Create subkeys through key expansion
    keys: List[bytes] = _create_keys(key)
    keys.reverse() # reverse keys for decryption

    # Break plaintext into blocks and pad last block if necessary
    blocks: List[bytes] = _chunkify_message(ciphertext_bytes)

    # encrypt each block
    plaintext: bytes = bytes()
    for block in blocks:
        plaintext += _encrypt_block(block, keys)

    # concate encrypted blocks and return result
    return plaintext

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
    d_zero = (k_plus & 268435455) # the 268435455 is a binary string of 28 1s

    left_halves[0] = c_zero
    right_halves[0] = d_zero

    for i, shift in enumerate(num_shifts):
        pos = i + 1  # Don't shift first position
        left_halves[pos] = _left_cycle(left_halves[pos - 1], shift,28)
        right_halves[pos] = _left_cycle(right_halves[pos - 1], shift, 28)

    #concatenate left half and right  half together
    k_n = []
    for i in range(1, len(right_halves)):
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

    return final_sub_key


'''
Encrypts the block of data using all the generated keys.
'''
def _encrypt_block(block: bytes, keys: List[bytes]) -> bytes:
    # TODO: Finish
    assert(len(block) == BLOCK_SIZE) # block should be 8 bytes/64 bits
    assert(len(keys) == NUM_CYCLES) # Should have 16 keys.

    # Perform initial permutation of message data
    #Get the block into binary then permute.
    mBin =  _bytes_to_binary_str(block)
    ipBin = ''
    for bit in initPerm:
        ipBin += mBin[bit - 1]

    # Cycle the block 16 times with the appropriate key-table combinations
    cipher_block = _binary_str_to_bytes(ipBin) #Get it back into Bytes
    for key in keys:
        cipher_block = _des_round(cipher_block, key)

    # Perform final permutation
    cBlockBin = _bytes_to_binary_str(cipher_block)
    finBin = ''
    for rBit in reversed(initPerm):
        finBin += cBlockBin[rBit - 1]
        
    return _binary_str_to_bytes(finBin)
    # return encrypted block


'''
A single round of DES encryption
'''
def _des_round(block: bytes, key: bytes) -> bytes:
    assert(len(block) == 8) # block should be 8 bytes/64 bits
    assert(len(key) == 48) #Key should be 48 bits

    # Split Data in half
    left_half: bytes = block[:int((len(block)/2))]
    right_half: bytes = block[int((len(block)/2)):]

    # Expand right half
    expanded_bytes = _expansion(right_half)
    #Get key into hex value
    hexKey = _binary_str_to_bytes(key)
    #Exclusive or with the key and the expanded bytes
    expanded_bytes = _xor(expanded_bytes, hexKey)
    #Substitute then convert to hex
    compressed_bytes = _substitution(expanded_bytes)
    compressed_hex = _binary_str_to_bytes(compressed_bytes)
    #Permute the compressed value
    compPerm_bytes = _permutation(compressed_hex)
    #Exclusive or with the compressed and permuted half with the left half then convert to hex
    new_right_half = _xor(compPerm_bytes, left_half)
    newRightHex = _binary_str_to_bytes(new_right_half)
    # Combine left and right
    return right_half + newRightHex

'''
Expand the half for DES
'''
def _expansion(half: bytes) -> bytes:
    assert(len(half) == 4) # half should be 32 bits
    expansion_table = [32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,24,25,26,27,28,29,28,29,30,31,32,1]

    expanded: bytes = _apply_table(half, expansion_table)

    assert(len(expanded) == 6) # expanded should be 48 bits
    return expanded


'''
Use substitution on the expanded half
'''
def _substitution(expanded_half: bytes) -> bytes:
    # TODO: Implement substitution table
    #Use the S1Mat for the Substitution matrix
    totalCount = int(len(expanded_half) / 6)
    bitStr = ''

    #Need to look at every 6 bits. The oth and 5th bit of a group, then the 1st through 4th bits for the inner.
    for i in range(totalCount):
        a = i*6
        aBit = expanded_half[a]
        b = i*6 + 5
        bBit = expanded_half[b]
        cStr = ''

        for j in range(4):
            k = i*6 + 1 + j
            cStr += expanded_half[k]

        iNum = int((aBit + bBit), 2)
        jNum = int(cStr, 2)
        SNum = bin(S1Mat[iNum][jNum])[2:]

        while(len(SNum)<4):
            SNum = '0' + SNum

        bitStr += SNum

    return bitStr


'''
Uses the permutation table on the compressed half.
'''
def _permutation(compressed_bytes: bytes) -> bytes:
    assert(len(compressed_bytes) == 4) # expanded should be 32 bits
    permutation_table = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]

    permuted = _apply_table(compressed_bytes, permutation_table)

    assert(len(compressed_bytes) == 4) # half should be 32 bits
    return permuted


if __name__ == "__main__":
    print("Adding line to break at...")
    plaintext = bytes.fromhex('abcd1234')
    print(plaintext.hex())
    key = bytes.fromhex('133457799BBCDFF1')

    key_temp = _create_keys(key)

    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print('These should be the same')
    print(plaintext, decrypted)