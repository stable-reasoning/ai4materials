import base64
from pathlib import Path
from typing import Dict, Any


def encode_image_to_data_url(p: Path) -> str:
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    suffix = p.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{b64}"


# TODO add GCS
class ImageStorage:

    def __init__(self, provider: str = 'local'):
        self.provider = provider

    def get_image_entry(self, image_path: Path) -> Dict[str, Any]:
        if self.provider == 'local':
            img64_url = encode_image_to_data_url(image_path)
            return {"type": "image_url", "image_url": {"url": img64_url}}
        raise NotImplementedError()
