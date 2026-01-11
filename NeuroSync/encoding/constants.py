MESSAGE_LENGTH = 16
BITS_PER_CHAR = 6
BIT_LENGTH = MESSAGE_LENGTH * BITS_PER_CHAR

KEY_SIZE = 16
KEY_BIT_LENGTH = KEY_SIZE * BITS_PER_CHAR

CHARSET = {
    **{chr(ord('a') + i): i for i in range(26)},
    **{chr(ord('A') + i): i + 26 for i in range(26)},
    **{chr(ord('0') + i): i + 52 for i in range(10)},
    '=': 62,
    ' ': 63,
}

REVERSE_CHARSET = {v: k for k, v in CHARSET.items()}
