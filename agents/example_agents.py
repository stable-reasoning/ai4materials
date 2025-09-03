import requests
from typing import Any, Dict, List
from core import Agent
from utils.settings import logger


class DataFetcher(Agent):
    """An agent to fetch posts from a public API."""

    async def run(self, api_url: str) -> Dict[str, Any]:
        logger.info(f"Fetching data from {api_url}...")
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # This agent returns a dictionary where the key is the desired output filename
        return {
            "posts.jsonl": data
        }


class TextProcessor(Agent):
    """An agent to process titles from a list of posts."""

    async def run(self, posts_data: List[Dict]) -> Dict[str, Any]:
        logger.info("Processing post titles...")
        processed_data = []
        for post in posts_data:
            processed_data.append({
                "id": post.get("id"),
                "original_title": post.get("title"),
                "processed_title": post.get("title", "").upper()
            })

        return {
            "processed_titles.json": processed_data
        }


class PostChecker(Agent):

    async def run(self, posts_dir: str) -> Dict[str, Any]:
        logger.info("Inside PostChecker")
        logger.info(f"Injected var: {posts_dir}")
        return {
            "eval": "2+2=4"
        }


class Validator(Agent):

    async def run(self, eval_str: str, processed_data: List[Dict]) -> Dict[str, Any]:
        logger.info("Inside Validator")
        logger.info(f"injected: {eval_str}")
        logger.info(f"processed_data: {len(processed_data)} records")
        return {}
