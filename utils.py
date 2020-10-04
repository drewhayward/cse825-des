def ascii_to_bytes(s: str) -> bytes:
    return bytes([ord(char) for char in s])

def bytes_to_ascii(b: bytes) -> str:
    return ''.join([chr(i) for i in b])