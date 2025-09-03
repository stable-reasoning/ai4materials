import asyncio

from agents.example_agents import DataFetcher, TextProcessor, PostChecker, Validator
from core import DAG, DAGRunner
from utils.settings import logger

fetcher = DataFetcher(
    name="fetcher",
    input_spec={
        "api_url": "url:https://jsonplaceholder.typicode.com/posts"
    }
)

processor = TextProcessor(
    name="processor",
    input_spec={
        # This input refers to the 'posts.jsonl' file produced by the 'fetcher' agent
        "posts_data": "agent:fetcher/posts.jsonl"
    }
)

checker = PostChecker(
    name="checker",
    input_spec={
        "posts_dir": "str:test"
    }
)

validator = Validator(
    name="validator",
    input_spec={
        "eval_str": "agent:checker/eval",
        "processed_data": "agent:processor/processed_titles.json"
    }
)

# 2. Define the dependencies (the DAG structure)
fetcher >> processor >> checker >> validator
processor >> validator
# You can also define dependencies like this:
# processor << fetcher

# 3. Create the DAG object
blog_processing_dag = DAG(
    name="blog_workflow",
    tasks=[fetcher, processor, checker, validator]
)


# 4. Create a runner and execute the DAG
async def main():
    runner = DAGRunner(dag=blog_processing_dag, working_dir="runs")

    logger.info("--- FIRST RUN: EXECUTING ALL AGENTS ---")
    run_id = f"{runner.dag.name}-second-run"
    await runner.run(experiment_id=run_id)


if __name__ == "__main__":
    asyncio.run(main())
