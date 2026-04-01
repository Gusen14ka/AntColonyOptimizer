from pathlib import Path
import json

def save_values(filename: str, values):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for val in values:
            f.write(f"{val}\n")

def save_json(filename: str, values: dict):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=4)

def parse_json(filename: str) -> dict:
    path = Path(filename)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_values(filename: str, cast = float):
    path = Path(filename)

    with open(path, "r", encoding="utf-8") as f:
        return [cast(line.strip()) for line in f if line.strip()]