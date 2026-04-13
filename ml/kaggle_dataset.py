"""/// AI generated code ///
Download and load the Sinhala AI/Human JSONL dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/movindug/sinhala-ai-generated-and-human-written-texts

Authentication (pick one):
  - ``KAGGLE_API_TOKEN`` in ``ml/.env`` (Kaggle → Settings → API → token), or
  - ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` (legacy), or
  - ``~/.kaggle/kaggle.json`` / ``~/.kaggle/access_token`` as per Kaggle docs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type, Union

if TYPE_CHECKING:
    import pandas as pd

# Owner/dataset slug on Kaggle
DEFAULT_KAGGLE_DATASET = "movindug/sinhala-ai-generated-and-human-written-texts"


def ml_root() -> Path:
    """Directory containing this file (the ``ml/`` folder)."""
    return Path(__file__).resolve().parent


def _ensure_python_dotenv() -> None:
    """Install ``python-dotenv`` into this interpreter if missing (for ``ml/.env``)."""
    try:
        import dotenv  # noqa: F401
    except ImportError:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "python-dotenv>=1.0.0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def load_ml_dotenv() -> None:
    """Load ``ml/.env`` so ``KAGGLE_*`` variables are available."""
    env = ml_root() / ".env"
    if not env.is_file():
        return
    _ensure_python_dotenv()
    from dotenv import load_dotenv

    load_dotenv(env)


def kaggle_cache_dir() -> Path:
    """Where downloaded JSONL files are stored (under ``ml/dataset/``, gitignored)."""
    return ml_root() / "dataset" / "kaggle"


def _get_kaggle_api_class() -> Type:
    """Import ``KaggleApi``, installing the ``kaggle`` package with *this* Python if needed."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        return KaggleApi
    except ImportError:
        pass

    print(
        "Installing `kaggle` and `python-dotenv` into the current environment:\n "
        f"  {sys.executable} -m pip install kaggle python-dotenv\n",
        flush=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "kaggle>=1.6.0",
            "python-dotenv>=1.0.0",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or "(no output)"
        raise RuntimeError(
            "Could not `pip install kaggle` into this Python. "
            f"Interpreter: {sys.executable}\npip output:\n{msg}"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "`kaggle` is still not importable after pip install. "
            "Restart the Jupyter kernel, then run this cell again."
        ) from e
    return KaggleApi


def ensure_kaggle_dataset(
    dataset: str = DEFAULT_KAGGLE_DATASET,
    *,
    dest: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download and unzip the Kaggle dataset if the cache has no ``.jsonl`` files.

    Returns the directory to pass to :func:`load_jsonl_dataframes`.
    """
    _ensure_python_dotenv()
    load_ml_dotenv()
    dest = Path(dest) if dest else kaggle_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if list(dest.rglob("*.jsonl")) and not force:
        return dest

    KaggleApi = _get_kaggle_api_class()

    if not (
        os.environ.get("KAGGLE_API_TOKEN")
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    ):
        raise RuntimeError(
            "Set KAGGLE_API_TOKEN (recommended) or KAGGLE_USERNAME + KAGGLE_KEY. "
            "You can store KAGGLE_API_TOKEN in ml/.env — see ml/.env.example."
        )

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
    return dest


def load_jsonl_dataframes(
    data_path: Optional[Union[str, Path]] = None,
    *,
    download_if_missing: bool = True,
    force_download: bool = False,
) -> "pd.DataFrame":
    """
    Load one or more ``.jsonl`` files into a single DataFrame.

    - If ``data_path`` is a file, that file is loaded.
    - If ``data_path`` is a directory, all ``*.jsonl`` under it (non-recursive first,
      then recursive if none found) are concatenated.
    - If ``data_path`` is None, uses :func:`kaggle_cache_dir` after
      :func:`ensure_kaggle_dataset` when ``download_if_missing`` is True.
    """
    import pandas as pd

    load_ml_dotenv()

    if data_path is None:
        cache = kaggle_cache_dir()
        if download_if_missing and (force_download or not list(cache.rglob("*.jsonl"))):
            ensure_kaggle_dataset(force=force_download)
        data_path = cache

    path = Path(data_path)
    dfs: list[pd.DataFrame] = []

    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got {path}")
        df = pd.read_json(path, lines=True)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {path}")
    elif path.is_dir():
        for fn in sorted(path.iterdir()):
            if fn.suffix == ".jsonl":
                try:
                    df = pd.read_json(fn, lines=True)
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {fn.name}")
                except Exception as e:
                    print("Failed to read", fn, e)
        if not dfs:
            for child in sorted(path.rglob("*.jsonl")):
                try:
                    df = pd.read_json(child, lines=True)
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {child.relative_to(path)}")
                except Exception as e:
                    print("Failed to read", child, e)
    else:
        raise FileNotFoundError(f"No file or directory found at {path}")

    if not dfs:
        raise FileNotFoundError(
            "No .jsonl files found. Run ensure_kaggle_dataset() or set data_path to a JSONL file."
        )

    return pd.concat(dfs, ignore_index=True)
