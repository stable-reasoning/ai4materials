import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT_DIR / "templates"
SCRIPTS_DIR = ROOT_DIR / "scripts"

LOG_LEVEL = "INFO"
APP_VERSION = "1.0.0"

print(f"Project Root Directory: {ROOT_DIR}")
print(f"TEMPLATES Directory: {TEMPLATES_DIR}")
print(f"SCRIPTS Directory: {SCRIPTS_DIR}")


class ConfigurationError(Exception):
    """Raised when a required configuration is missing or invalid."""
    pass


def load_env_vars_to_dataclass(cls: type):
    def new_init(self, *args, **kwargs):
        annotations = cls.__annotations__
        fields = dataclasses.fields(self.__class__)  # Use dataclasses.fields for better handling

        positional_arg_count = len(args)

        for i in range(positional_arg_count):
            f = fields[i]
            setattr(self, f.name, args[i])

        for f in fields:
            if f.name in self.__dict__:
                continue

            env_var_name = f.name.upper()
            if hasattr(cls, "__env_var_prefix__"):
                env_var_name = f"{cls.__env_var_prefix__}_{env_var_name}"
            env_value = os.environ.get(env_var_name)

            if env_value is not None:
                if f.type is not type(None) and f.type is not str:
                    try:
                        if f.type is int:
                            env_value = int(env_value)
                        elif f.type is float:
                            env_value = float(env_value)
                        elif f.type is bool:
                            env_value = env_value.lower() in ("true", "1", "yes")
                        elif f.type is List:
                            env_value = env_value.split(",")
                            env_value = [x.strip() for x in env_value]
                        elif f.type == dict:  # Handle json dictionaries
                            env_value = json.loads(env_value)
                        else:
                            env_value = f.type(env_value)  # e.g., for Date
                    except ValueError as e:
                        raise ValueError(
                            f"Error converting environment variable {env_var_name} value '{os.environ.get(env_var_name)}' to type {f.type}: {e} "
                        ) from e

                setattr(self, f.name, env_value)
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # Required and no default
                raise ValueError(f"Missing environment variable: {env_var_name}")

    cls.__init__ = new_init
    return cls


@load_env_vars_to_dataclass
@dataclass
class Configuration:
    docucache_path: str
    vector_db_path: str
    openai_api_key: str = dataclasses.field(metadata={'secret': True})

    def __str__(self):
        """Generates a formatted, secret-redacted string representation."""

        lines = [f"<{self.__class__.__name__}>", "-" * (len(self.__class__.__name__) + 16)]

        try:
            max_len = max(len(f.name) for f in dataclasses.fields(self))
        except ValueError:  # Handles case with no fields
            max_len = 0

        for f in dataclasses.fields(self):
            is_secret = f.metadata.get('secret', False)
            value = getattr(self, f.name)
            display_value = '*****' if is_secret else repr(value)
            lines.append(f"  {f.name:<{max_len}} : {display_value}")

        return "\n".join(lines)


global_config = Configuration()

print(global_config)

Path(global_config.docucache_path).resolve().mkdir(parents=True, exist_ok=True)

