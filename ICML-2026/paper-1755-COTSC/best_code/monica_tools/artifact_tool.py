"""Artifact loading/decryption for MONICA."""
import os
import pickle
import zlib
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

_EMBEDDED_KEY = "dD1hk2QB-637hTominW0yJshf9HlgtjNX4FXHzGBqyIF4Z5ZyXDNjOTLfnqRh0RI"


def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode())


def _resolve_passphrase() -> str:
    env_val = os.environ.get("MONICA_ARTIFACT_KEY", "")
    if env_val:
        return env_val
    return _EMBEDDED_KEY


def load_monica_artifact(artifact_path):
    """Load and decrypt a MONICA artifact file.
    
    Returns (calibrator, monitor, meta) tuple.
    """
    artifact_path = Path(artifact_path)
    with open(artifact_path, "rb") as f:
        envelope = pickle.load(f)

    if envelope.get("schema") != "monica_artifact_envelope_v1":
        raise RuntimeError(f"Invalid artifact schema in {artifact_path}")

    kdf_info = envelope["kdf"]
    cipher_info = envelope["cipher"]

    passphrase = _resolve_passphrase()
    key = _derive_key(passphrase, kdf_info["salt"], kdf_info["iterations"])

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(
            cipher_info["nonce"],
            cipher_info["ciphertext"],
            cipher_info.get("aad", None),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt {artifact_path}. Key is missing or incorrect.") from e

    decompressed = zlib.decompress(plaintext)

    import io
    import numpy as np

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy."):
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    pass
            return super().find_class(module, name)

    payload = CustomUnpickler(io.BytesIO(decompressed)).load()

    if payload.get("schema") != "monica_artifact_payload_v1":
        raise RuntimeError(f"Invalid payload schema in {artifact_path}")

    calibrator = payload["calibrator"]
    monitor = payload["monitor"]
    meta = {"model_tag": payload.get("model_tag", "")}

    return calibrator, monitor, meta
