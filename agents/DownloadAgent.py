from typing import Dict, Any

from core import Agent
from utils.download_utils import FileDownloader
from utils.settings import logger


class DownloadAgent(Agent):

    async def run(self, file_with_urls: str) -> Dict[str, Any]:
        logger.info(f"Fetching files")
        downloader = FileDownloader()
        data = downloader.process_url_list_file(file_with_urls)

        return {
            "file_paths.json": data
        }
