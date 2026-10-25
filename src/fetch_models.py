"""Downloads the fine-tuned RoBERTa weights on first start of the online app.

GitHub doesn't accept files over 100 MB in a repository, so the two models (about 480 MB) are attached to a
GitHub release as roberta_models.zip. Set MODELS_URL (Streamlit secret or environment variable) to that file's
download link. The zip contains roberta/binary/... and roberta/category/...
"""
import shutil, tempfile, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def roberta_present():
    return all((MODELS / "roberta" / d / f).exists() for d in ("binary", "category")
               for f in ("config.json", "model.safetensors", "tokenizer.json"))


def ensure_roberta(url):
    """Returns (ok, message). Safe to call repeatedly; only downloads when the files are missing."""
    if roberta_present():
        return True, "already present"
    if not url:
        return False, "MODELS_URL is not set"
    tmp = Path(tempfile.mkdtemp())
    try:
        z = tmp / "roberta_models.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "DarkPatternGuard"})
        with urllib.request.urlopen(req, timeout=120) as r, open(z, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp / "x")
        src = next((p.parent.parent for p in (tmp / "x").rglob("binary/config.json")), None)   # the roberta/ folder
        if src is None:
            return False, "the zip does not contain roberta/binary/config.json"
        dest = MODELS / "roberta"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        return roberta_present(), "downloaded"
    except Exception as e:  # network errors, bad link, not a zip...
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
