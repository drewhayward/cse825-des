import des
from bitarray import bitarray

class TestDES():
    def test_encryption(self):
        plaintext = bytes.fromhex('0123456789ABCDEF')
        key = bytes.fromhex('133457799BBCDFF1')
        ciphertext = bytes.fromhex('85E813540F0AB405')
        
        assert(des.encrypt(plaintext, key) == ciphertext)

    def test_decryption(self):
        plaintext = bytes.fromhex('0123456789ABCDEF')
        key = bytes.fromhex('133457799BBCDFF1')
        ciphertext = bytes.fromhex('85E813540F0AB405')

        assert(des.decrypt(ciphertext, key) == plaintext)

    def test_key_gen(self):
        key = bytes.fromhex('133457799BBCDFF1')
        key = bitarray(des._bytes_to_binary_str(key))
        subkeys = des._create_keys(key)
        assert(len(subkeys) == 16)

        correct_subkeys = [
            bitarray('000110110000001011101111111111000111000001110010'), #1
            bitarray('011110011010111011011001110110111100100111100101'), #2
            bitarray('010101011111110010001010010000101100111110011001'), #3
            bitarray('011100101010110111010110110110110011010100011101'), #4
            bitarray('011111001110110000000111111010110101001110101000'), #5
            bitarray('011000111010010100111110010100000111101100101111'), #6
            bitarray('111011001000010010110111111101100001100010111100'), #7
            bitarray('111101111000101000111010110000010011101111111011'), #8
            bitarray('111000001101101111101011111011011110011110000001'), #9
            bitarray('101100011111001101000111101110100100011001001111'), #10
            bitarray('001000010101111111010011110111101101001110000110'), #11
            bitarray('011101010111000111110101100101000110011111101001'), #12
            bitarray('100101111100010111010001111110101011101001000001'), #13
            bitarray('010111110100001110110111111100101110011100111010'), #14
            bitarray('101111111001000110001101001111010011111100001010'), #15
            bitarray('110010110011110110001011000011100001011111110101'), #16
        ]

        for key, correct_key in zip(subkeys, correct_subkeys):
            assert(key == correct_key)

if __name__ == "__main__":
    t = TestDES()
    t.test_key_gen()