import subprocess
import sys
from pathlib import Path

from utils.settings import SCRIPTS_DIR, global_config


def extract_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 150):
    """
    Calls the bash script to extract all pages of pdf_path at the given dpi.
    Raises:
      FileNotFoundError    – if the script or pdf_path is missing
      CalledProcessError   – if pdftoppm or checks fail
      ValueError           – if dpi is not a positive integer
    """
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi!r}")

    cmd = ["bash", f"{SCRIPTS_DIR}/extract_pages.sh", pdf_path, out_dir, str(dpi)]
    print(f"extracting {pdf_path}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except FileNotFoundError as e:
        # bash or the script itself not found
        raise FileNotFoundError("Could not invoke the extraction script.") from e
    except subprocess.CalledProcessError as e:
        # non-zero exit code: script printed error to stderr
        msg = f"Extraction failed (exit {e.returncode}):\n{e.stderr}"
        raise RuntimeError(msg) from e


if __name__ == "__main__":
    try:
        # Example usage
        file = f"{global_config.docucache_path}/0001/tmp/2507.13733v1.pdf"
        out = f"{global_config.docucache_path}/0001/pages"
        extract_pdf_pages(pdf_path=file, out_dir=out)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
    else:
        print("All pages extracted successfully.")
