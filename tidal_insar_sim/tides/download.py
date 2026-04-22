"""`setup-cats` — verify and install CATS2008 from a user-supplied zip.

USAP-DC requires a reCAPTCHA-gated manual download, so there is no fully
programmatic path. This module handles the post-download workflow: MD5 verify
the zip, extract to the canonical directory, validate that pyTMD can load it.

Primary entry point
-------------------
    install_cats2008_from_zip(zip_path, data_dir=...) -> dict[str, str]
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from tidal_insar_sim.tides.cats2008 import (
    CATS_FILES,
    CATS_SUBDIR,
    DEFAULT_DATA_DIR,
)
from tidal_insar_sim.tides.errors import TidalDataUnavailable

CATS2008_ZIP_MD5 = "008a30cd08142cb6acc7f7687e22c4a3"
USAP_DC_URL = "https://www.usap-dc.org/view/dataset/601235"
USAP_DC_DOI = "10.15784/601235"


@dataclass
class InstallReport:
    """Summary of an install attempt. Printable as a `rich` panel by the CLI."""

    zip_path: Path
    zip_md5: str
    zip_md5_ok: bool
    data_dir: Path
    installed_files: list[str] = field(default_factory=list)
    already_installed: bool = False
    pytmd_probe_ok: bool = False


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual_download_instructions() -> str:
    return (
        "CATS2008 must be downloaded manually from USAP-DC (reCAPTCHA-gated).\n"
        f"\n"
        f"  1. Open {USAP_DC_URL} (DOI: {USAP_DC_DOI})\n"
        f"  2. Complete the reCAPTCHA and download `CATS2008.zip` (582 MB)\n"
        f"  3. Re-run this command with --path /path/to/CATS2008.zip\n"
        f"\n"
        f"Expected MD5: {CATS2008_ZIP_MD5}"
    )


def install_cats2008_from_zip(
    zip_path: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    skip_md5: bool = False,
    probe_pytmd: bool = True,
) -> InstallReport:
    """Verify MD5, extract to `data_dir/CATS2008/`, optionally probe pyTMD.

    Parameters
    ----------
    zip_path : path to the user-downloaded CATS2008.zip
    data_dir : install root (default `~/.tidal_insar_sim/tides/`)
    skip_md5 : skip the MD5 check (for test fixtures / recreated zips)
    probe_pytmd : after extraction, call CATSBackend() to confirm pyTMD loads
    """
    zip_path = Path(zip_path)
    data_dir = Path(data_dir)
    if not zip_path.exists():
        msg = f"zip not found: {zip_path}"
        raise FileNotFoundError(msg)

    # MD5 check
    digest = md5_of_file(zip_path) if not skip_md5 else ""
    md5_ok = True if skip_md5 else (digest == CATS2008_ZIP_MD5)
    if not md5_ok:
        msg = (
            f"MD5 mismatch for {zip_path.name}: got {digest}, "
            f"expected {CATS2008_ZIP_MD5}. Re-download or pass skip_md5=True."
        )
        raise ValueError(msg)

    model_dir = data_dir / CATS_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # Detect already-installed
    already = all((model_dir / f).exists() for f in CATS_FILES)
    report = InstallReport(
        zip_path=zip_path,
        zip_md5=digest,
        zip_md5_ok=md5_ok,
        data_dir=data_dir,
        already_installed=already,
    )

    if not already:
        installed: list[str] = []
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name  # strip any leading folder
                target = model_dir / name
                with zf.open(info) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                installed.append(name)
        report.installed_files = installed

    if probe_pytmd:
        try:
            from tidal_insar_sim.tides.cats2008 import CATSBackend

            CATSBackend(data_dir=data_dir)
        except Exception as exc:  # pragma: no cover - pyTMD import path
            msg = f"pyTMD smoke test failed after install: {exc}"
            raise TidalDataUnavailable(msg) from exc
        report.pytmd_probe_ok = True

    return report
