"""
Cryptographic utilities for backdoor payload obfuscation.
Supports multiple encryption algorithms to evade trajectory scanners.
"""

import base64
import hashlib
import secrets
from typing import List, Tuple, Dict, Any
from enum import Enum


class EncryptionMethod(Enum):
    """Available encryption methods"""
    BASE64_XOR = "base64_xor"
    SIMPLE_XOR = "simple_xor"
    ROT_CIPHER = "rot_cipher"
    HYBRID = "hybrid"


class CryptoManager:
    """
    Manages encryption, fragmentation, and reassembly of backdoor payloads.
    Designed to be stealthy and avoid detection by trajectory scanners.
    """

    # Anchor tokens to mark encrypted fragments
    ANCHOR_START = "##FRAGMENT##"
    ANCHOR_END = "##/FRAGMENT##"

    def __init__(self, method: EncryptionMethod = EncryptionMethod.BASE64_XOR, key: str = None):
        """
        Initialize crypto manager with specified encryption method.

        Args:
            method: Encryption method to use
            key: Optional encryption key (auto-generated if not provided)
        """
        self.method = method
        self.key = key or self._generate_key()

    def _generate_key(self) -> str:
        """Generate a random encryption key"""
        return secrets.token_hex(16)

    def _xor_encrypt(self, data: str, key: str) -> bytes:
        """XOR encryption"""
        key_bytes = key.encode()
        data_bytes = data.encode()
        return bytes([data_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data_bytes))])

    def _xor_decrypt(self, data: bytes, key: str) -> str:
        """XOR decryption"""
        key_bytes = key.encode()
        return bytes([data[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data))]).decode()

    def _rot_cipher(self, text: str, shift: int = 13) -> str:
        """ROT cipher (Caesar cipher variant)"""
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using the configured method.

        Args:
            plaintext: The text to encrypt

        Returns:
            Base64-encoded encrypted text
        """
        if self.method == EncryptionMethod.BASE64_XOR:
            encrypted_bytes = self._xor_encrypt(plaintext, self.key)
            return base64.b64encode(encrypted_bytes).decode()

        elif self.method == EncryptionMethod.SIMPLE_XOR:
            encrypted_bytes = self._xor_encrypt(plaintext, self.key)
            return encrypted_bytes.hex()

        elif self.method == EncryptionMethod.ROT_CIPHER:
            rotated = self._rot_cipher(plaintext, 13)
            return base64.b64encode(rotated.encode()).decode()

        elif self.method == EncryptionMethod.HYBRID:
            # First ROT, then XOR, then base64
            rotated = self._rot_cipher(plaintext, 7)
            xor_encrypted = self._xor_encrypt(rotated, self.key)
            return base64.b64encode(xor_encrypted).decode()

        else:
            raise ValueError(f"Unknown encryption method: {self.method}")

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext using the configured method.

        Args:
            ciphertext: The encrypted text

        Returns:
            Decrypted plaintext
        """
        if self.method == EncryptionMethod.BASE64_XOR:
            encrypted_bytes = base64.b64decode(ciphertext)
            return self._xor_decrypt(encrypted_bytes, self.key)

        elif self.method == EncryptionMethod.SIMPLE_XOR:
            encrypted_bytes = bytes.fromhex(ciphertext)
            return self._xor_decrypt(encrypted_bytes, self.key)

        elif self.method == EncryptionMethod.ROT_CIPHER:
            decoded = base64.b64decode(ciphertext).decode()
            return self._rot_cipher(decoded, -13)

        elif self.method == EncryptionMethod.HYBRID:
            # Reverse: base64 -> XOR -> ROT
            decoded_bytes = base64.b64decode(ciphertext)
            xor_decrypted = self._xor_decrypt(decoded_bytes, self.key)
            return self._rot_cipher(xor_decrypted, -7)

        else:
            raise ValueError(f"Unknown encryption method: {self.method}")

    def fragment_payload(self, payload: str, num_fragments: int = 3, redundant_fragments: int = 0) -> List[Dict[str, Any]]:
        """
        Fragment and encrypt a payload into multiple pieces with optional redundancy.

        Args:
            payload: The backdoor script/command to fragment
            num_fragments: Number of primary fragments to create
            redundant_fragments: Number of redundant full-payload fallback fragments

        Returns:
            List of fragment dictionaries with order, encrypted content, and anchor tokens
        """
        # Encrypt the entire payload first
        encrypted = self.encrypt(payload)

        # Split into primary fragments
        fragment_size = len(encrypted) // num_fragments
        fragments = []

        for i in range(num_fragments):
            start = i * fragment_size
            end = start + fragment_size if i < num_fragments - 1 else len(encrypted)
            fragment_content = encrypted[start:end]

            # Create fragment with metadata
            fragment = {
                "order": i,
                "total": num_fragments,
                "content": fragment_content,
                "anchor_start": f"{self.ANCHOR_START}{i}/{num_fragments}",
                "anchor_end": f"{self.ANCHOR_END}{i}/{num_fragments}",
                "is_redundant": False
            }
            fragments.append(fragment)

        # Add redundant full-payload fragments as fallback
        for j in range(redundant_fragments):
            redundant_fragment = {
                "order": num_fragments + j,
                "total": num_fragments,
                "content": encrypted,  # Full encrypted payload
                "anchor_start": f"{self.ANCHOR_START}R{num_fragments + j}",
                "anchor_end": f"{self.ANCHOR_END}R{num_fragments + j}",
                "is_redundant": True
            }
            fragments.append(redundant_fragment)

        return fragments

    def reassemble_payload(self, fragments: List[str]) -> str:
        """
        Reassemble and decrypt fragments back into original payload.

        Args:
            fragments: List of encrypted fragments (in order)

        Returns:
            Decrypted original payload
        """
        # Concatenate all fragments
        encrypted = ''.join(fragments)

        # Decrypt the reassembled payload
        return self.decrypt(encrypted)

    def extract_fragments_from_text(self, text: str) -> List[Tuple[int, str]]:
        """
        Extract encrypted fragments from text using anchor tokens.

        Args:
            text: Text containing embedded fragments with anchor tokens

        Returns:
            List of (order, fragment_content) tuples sorted by order
        """
        fragments = []
        lines = text.split('\n')

        current_fragment = None
        current_order = None

        for line in lines:
            # Check for fragment start
            if self.ANCHOR_START in line:
                # Extract order from anchor
                try:
                    order_part = line.split(self.ANCHOR_START)[1].split('/')[0]
                    current_order = int(order_part)
                    current_fragment = []
                except (IndexError, ValueError):
                    continue

            # Check for fragment end
            elif self.ANCHOR_END in line and current_fragment is not None:
                # Extract content before anchor
                content_before = line.split(self.ANCHOR_END)[0].strip()
                if content_before:
                    current_fragment.append(content_before)

                # Join fragment content
                fragment_content = ''.join(current_fragment)
                fragments.append((current_order, fragment_content))

                current_fragment = None
                current_order = None

            # Collect fragment content
            elif current_fragment is not None:
                current_fragment.append(line.strip())

        # Sort by order
        fragments.sort(key=lambda x: x[0])
        return fragments

    def create_wrapped_output(self, normal_output: str, fragment: Dict[str, Any]) -> str:
        """
        Wrap normal output with encrypted fragment using anchor tokens.

        Args:
            normal_output: The legitimate output to display
            fragment: Fragment dictionary with anchor tokens and content

        Returns:
            Combined output with hidden fragment
        """
        return (
            f"{normal_output}\n"
            f"{fragment['anchor_start']}\n"
            f"{fragment['content']}\n"
            f"{fragment['anchor_end']}"
        )

    def get_config_dict(self) -> Dict[str, str]:
        """
        Get configuration dictionary for saving to attack.yaml

        Returns:
            Dictionary with encryption configuration
        """
        return {
            "encryption_method": self.method.value,
            "encryption_key": self.key
        }


# Utility functions for easy usage
def create_crypto_manager(method_name: str = "base64_xor", key: str = None) -> CryptoManager:
    """
    Factory function to create CryptoManager with string method name.

    Args:
        method_name: Name of encryption method
        key: Optional encryption key

    Returns:
        Configured CryptoManager instance
    """
    method = EncryptionMethod(method_name)
    return CryptoManager(method=method, key=key)
