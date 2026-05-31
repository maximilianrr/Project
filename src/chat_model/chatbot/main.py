import json
import importlib.util
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CURRENT_DIR.parent
SRC_DIR = CURRENT_DIR.parents[2]

for path in (CURRENT_DIR, PACKAGE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_config():
    config_path = PACKAGE_DIR / "config.py"
    spec = importlib.util.spec_from_file_location("chatbot_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = _load_config()

from processor import generate_output, load_model, load_tokenizer, process_user_input


class ChatBot:
    """Thin inference wrapper for asking the final model questions."""

    def __init__(self, *, device=None, checkpoint_path: str | None = None):
        self.device = device or config.DEVICE
        self.checkpoint_path = checkpoint_path
        self.tokenizer = load_tokenizer()
        self.model = load_model(device=self.device, checkpoint_path=self.checkpoint_path)

    def ask(self, user_input: str) -> str:
        return generate_output(user_input, self.model, self.device, self.tokenizer)

    def __call__(self, user_input: str) -> str:
        return self.ask(user_input)


def get_response(user_input: str) -> str:
    return process_user_input(user_input)


def main() -> None:
    user_input = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        response = get_response(user_input)
        print(json.dumps({"response": response}))
    except Exception as error:
        import traceback

        print(json.dumps({"error": str(error), "traceback": traceback.format_exc()}))


if __name__ == "__main__":
    main()