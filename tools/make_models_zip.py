"""Packs models/roberta into deploy_release/roberta_models.zip, to attach to a GitHub release (see README)."""
import zipfile
from pathlib import Path
out = Path("deploy_release/roberta_models.zip")
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for d in ("binary", "category"):
        for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            z.write(Path("models/roberta") / d / f, f"roberta/{d}/{f}")
print(out, round(out.stat().st_size / 2**20), "MB")
