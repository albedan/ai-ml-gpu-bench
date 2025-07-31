# encrypt_upload.py
from datetime import datetime, timezone
import requests, pathlib
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

def load_pub(path):
    return serialization.load_pem_public_key(pathlib.Path(path).read_bytes())

def encrypt(pubkey, csv_path: Path) -> Path:
    """Cifra `csv_path` → restituisce Path del file .enc con timestamp."""
    data = csv_path.read_bytes()
    max_chunk = pubkey.key_size // 8 - 2 * hashes.SHA256().digest_size - 2

    enc_bytes = bytearray()
    for i in range(0, len(data), max_chunk):
        enc_bytes += pubkey.encrypt(
            data[i : i + max_chunk],
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    # ▼ aggiungi timestamp ISO compatto
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    enc_name = f"{csv_path.stem}_{ts}.enc"
    enc_path = csv_path.with_name(enc_name)

    enc_path.write_bytes(enc_bytes)
    return enc_path

def upload_to_filebin(enc_path, bin_id):
    url = f"https://filebin.net/{bin_id}/{enc_path.name}"
    with open(enc_path, "rb") as f:
        requests.put(url, data=f)     # filebin accetta PUT
    return url
