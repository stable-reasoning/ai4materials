import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

import requests

# --- Setup basic logging ---
from utils.settings import ROOT_DIR, global_config


class MetadataStore:
    """
    Encapsulates all database operations for storing paper metadata.
    Uses SQLite for persistent, file-based storage.
    """

    def __init__(self, db_path):
        """Initializes the connection and ensures the table exists."""
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._create_table()
        logging.info(f"Database initialized at {self.db_path}")

    def _create_table(self):
        """Creates the 'papers' table if it doesn't already exist."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY,
                    arxiv_url TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL
                )
            """)

    def is_url_processed(self, url: str) -> bool:
        """Checks if a URL has already been successfully processed."""
        cursor = self._conn.execute(
            "SELECT 1 FROM papers WHERE arxiv_url = ? AND status = 'processed'",
            (url,)
        )
        return cursor.fetchone() is not None


    def add_paper_entry(self, url: str) -> int:
        """
        Adds a new paper URL to the database with 'pending' status.
        Returns the auto-incremented ID for the new entry.
        """
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO papers (arxiv_url, status) VALUES (?, 'pending')",
                    (url,)
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # This can happen if a URL is in the list twice.
            logging.warning(f"URL {url} already exists in database. Skipping insertion.")
            cursor = self._conn.execute("SELECT id FROM papers WHERE arxiv_url = ?", (url,))
            result = cursor.fetchone()
            return result[0] if result else -1

    def update_status(self, paper_id: int, status: str):
        """Updates the status of a paper entry (e.g., to 'processed')."""
        with self._conn:
            self._conn.execute(
                "UPDATE papers SET status = ? WHERE id = ?",
                (status, paper_id)
            )

    def close(self):
        """Closes the database connection."""
        if self._conn:
            self._conn.close()
            logging.info("Database connection closed.")


class FileDownloader:
    """
    Manages the process of downloading papers from a list of URLs.
    """
    # The first few bytes of a PDF file.
    PDF_SIGNATURE = b'%PDF'

    def __init__(self):
        """
        Initializes the downloader and the metadata store.

        Args:
            base_cache_dir: The root directory for all downloaded content.
        """
        self.base_cache_dir = Path(global_config.docucache_path)
        self.base_cache_dir.mkdir(exist_ok=True)
        db_path = self.base_cache_dir / "metadata.db"
        self.db = MetadataStore(db_path)

    def _create_paper_fs(self, paper_id: int) -> Path:
        paper_dir = self.base_cache_dir / str(paper_id)
        tmp_dir = paper_dir / "tmp"
        assets_dir = paper_dir / "assets"

        tmp_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(exist_ok=True)

        return tmp_dir

    def _download_file(self, url: str, destination: Path):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {url}. Error: {e}")
            return False

    def _is_pdf(self, file_path: Path) -> bool:
        """
        Checks if the file at file_path is a valid PDF by reading its first bytes.

        Returns:
            True if the file signature matches a PDF, False otherwise.
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(len(self.PDF_SIGNATURE))
                return header == self.PDF_SIGNATURE
        except IOError as e:
            logging.error(f"Could not read file {file_path} for validation. Error: {e}")
            return False

    def process_url_list_file(self, file_path: str) -> List[Dict[str,Any]]:
        logging.info(f"Starting to process file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Input file not found: {file_path}")
            return []
        added_files = []

        for url in urls:
            if self.db.is_url_processed(url):
                logging.info(f"Skipping already processed URL: {url}")
                continue

            logging.info(f"Processing new URL: {url}")

            paper_id = self.db.add_paper_entry(url)
            if paper_id == -1: continue
            tmp_path = self._create_paper_fs(paper_id)

            try:
                file_name = f"{url.split('/')[-1]}"
                if not file_name.endswith('pdf'):
                    file_name = file_name + '.pdf'
                destination = tmp_path / file_name
                status = 'processed' if self._download_file(url, destination) else 'download_failed'
                if not self._is_pdf(destination):
                    status = 'validation_failed'
                self.db.update_status(paper_id, status)
                added_files.append({"paper_id": paper_id, "file": destination})
            except ValueError as e:
                logging.error(e)
                self.db.update_status(paper_id, 'invalid_url')

        self.db.close()
        logging.info("Processing complete.")
        return added_files


def main():
    input_file = ROOT_DIR / "tests/papers.lst"

    # --- Initialize and run the downloader ---
    downloader = FileDownloader()
    downloader.process_url_list_file(input_file)


if __name__ == "__main__":
    main()
