import json
from pathlib import Path
from typing import Dict, List, Union


def save_json(data: Union[Dict, List], path: Union[str, Path]):
    """Saves a dictionary or list to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Union[Dict, List]:
    """Loads a JSON file into a dictionary or list."""
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], path: Union[str, Path]):
    """Saves a list of dictionaries to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
