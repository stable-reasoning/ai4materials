import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

import litellm

# To prevent litellm from logging its own success/failure messages
from utils.settings import logger

litellm.suppress_provider_warning = True
litellm.set_verbose = False


@dataclass
class LedgerEntry:
    timestamp_utc: str
    experiment_id: str
    document_id: str
    agent_name: str
    event_name: str
    success: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model_name: str


class Ledger:
    """A ledger system that collects and stores LLM call metrics in a SQLite database."""

    def __init__(self, db_path: str):
        """Initializes the ledger and connects to the SQLite database."""
        self.db_path = db_path
        self._records: List[LedgerEntry] = []
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()
        print(f"Ledger initialized. Records will be saved to SQLite DB at '{self.db_path}'")

    def _create_table(self):
        """Creates the ledger table if it doesn't already exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                event_name TEXT NOT NULL,
                success INTEGER NOT NULL, -- 0 for False, 1 for True
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                model_name TEXT
            )
        """)
        self.conn.commit()

    def add_entry(self, entry: LedgerEntry):
        self._records.append(entry)

    def save(self):
        """
        Saves all in-memory records to the SQLite database in a single transaction.
        """
        if not self._records:
            return

        logger.warn(f"Saving {len(self._records)} records to SQLite DB...")

        def entry_to_tuple(entry: LedgerEntry) -> Tuple:
            return (
                entry.timestamp_utc,
                entry.experiment_id,
                entry.document_id,
                entry.agent_name,
                entry.event_name,
                1 if entry.success else 0,  # Convert bool to integer
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.total_tokens,
                entry.latency_ms,
                entry.model_name,
            )

        data_to_insert = [entry_to_tuple(record) for record in self._records]

        try:
            # executemany is highly efficient for inserting multiple rows
            self.cursor.executemany("""
                INSERT INTO llm_ledger (
                    timestamp_utc, experiment_id, document_id, agent_name, event_name,
                    success, prompt_tokens, completion_tokens, total_tokens,
                    latency_ms, model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)

            self.conn.commit()
            self._records.clear()  # Clear in-memory records after successful commit
        except sqlite3.Error as e:
            logger.error(f"Error saving ledger records to SQLite: {e}")
            self.conn.rollback()

    def close(self):
        """Saves any remaining records and closes the database connection."""
        self.save()
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.warn("Ledger connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
