import des

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