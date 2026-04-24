"""CLI subcommand tests (M8). Uses click's CliRunner with isolated-FS.

The CATS-dependent tests are skipped when CATS2008 is missing; the plain
wiring tests (help text, missing args, bad inputs) always run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tidal_insar_sim import __version__
from tidal_insar_sim.cli.main import cli
from tidal_insar_sim.tides.cats2008 import CATS_FILES, CATS_SUBDIR, DEFAULT_DATA_DIR
from tidal_insar_sim.tides.download import CATS2008_ZIP_MD5

_CATS_INSTALLED = all(
    (DEFAULT_DATA_DIR / CATS_SUBDIR / f).exists() for f in CATS_FILES
)
cats_only = pytest.mark.skipif(not _CATS_INSTALLED, reason="CATS2008 not installed")


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def test_version_flag() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("analyze", "batch", "plan", "setup-cats", "synthesize"):
        assert cmd in result.output


def test_unknown_command_exits_nonzero() -> None:
    result = CliRunner().invoke(cli, ["bogus"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# setup-cats
# ---------------------------------------------------------------------------


def test_setup_cats_no_path_prints_instructions() -> None:
    result = CliRunner().invoke(cli, ["setup-cats"])
    assert result.exit_code == 0
    assert CATS2008_ZIP_MD5 in result.output
    assert "usap-dc.org" in result.output


def test_setup_cats_missing_file_errors() -> None:
    result = CliRunner().invoke(cli, ["setup-cats", "--path", "/no/such/file.zip"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cats_only
def test_analyze_preset_rutford_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "rutford"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--duration", "29.53", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "summary.json").exists()
    assert (out / "sweep.csv").exists()
    assert (out / "site.geojson").exists()
    assert (out / "site.kml").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "excellent"
    assert summary["P_usable_ge_3fr"] > 0.5
    # Constellation defaults = 1 sat / 12 d
    assert summary["constellation_n_sats"] == 1
    assert summary["constellation_B_eff_days"] == 12.0


@cats_only
def test_analyze_with_n_sats_and_repeat(tmp_path: Path) -> None:
    """`--n-sats 2 --repeat-per-sat 12` builds an equally-phased dual constellation
    → effective baseline 6 days (Sentinel-1 A+B).
    """
    out = tmp_path / "rutford_dual"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--n-sats", "2", "--repeat-per-sat", "12",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["constellation_n_sats"] == 2
    assert summary["constellation_B_eff_days"] == 6.0


@cats_only
def test_analyze_with_constellation_preset(tmp_path: Path) -> None:
    out = tmp_path / "rcm"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "C-BAND", "--preset", "RUTFORD",
         "--constellation", "RCM", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["constellation_n_sats"] == 3
    assert summary["constellation_B_eff_days"] == pytest.approx(4.0)


@cats_only
def test_analyze_multi_baseline_mode(tmp_path: Path) -> None:
    out = tmp_path / "rcm_multi"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--constellation", "RCM", "--mode", "multi_baseline",
         "--duration", "30",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "multi_baseline_summary.csv").exists()
    # At least one per-B CSV must exist.
    per_b = list(out.glob("sweep_B*.csv"))
    assert len(per_b) >= 2


@cats_only
def test_analyze_any_triplet_mode(tmp_path: Path) -> None:
    out = tmp_path / "dual_any"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--n-sats", "2", "--repeat-per-sat", "12",
         "--mode", "any_triplet", "--duration", "60",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "any_triplets.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"] == "any_triplet"
    assert summary["n_triplets"] > 0


@cats_only
def test_analyze_with_satellites_yaml(tmp_path: Path) -> None:
    yaml_text = """
satellites:
  - name: S1A
    phase_offset_days: 0.0
    repeat_days: 12.0
  - name: S1B
    phase_offset_days: 6.0
    repeat_days: 12.0
"""
    yaml_path = tmp_path / "constellation.yaml"
    yaml_path.write_text(yaml_text)
    out = tmp_path / "custom_const"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "C-BAND", "--preset", "RUTFORD",
         "--satellites", str(yaml_path), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "summary.json").read_text())
    assert summary["constellation_n_sats"] == 2
    assert summary["constellation_B_eff_days"] == 6.0


@cats_only
def test_analyze_lat_lon_mode(tmp_path: Path) -> None:
    out = tmp_path / "custom"
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND",
         "--lat", "-78.5", "--lon", "-83.0",
         "--name", "rutford_custom", "--ice-thickness", "2000",
         "--output", str(out), "--no-geojson", "--no-kml"],
    )
    assert result.exit_code == 0, result.output
    assert (out / "summary.json").exists()
    assert not (out / "site.geojson").exists()
    assert not (out / "site.kml").exists()


def test_analyze_rejects_missing_coords() -> None:
    """Neither --preset nor --lat/--lon → UsageError."""
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "L-BAND", "--output", "out"],
    )
    assert result.exit_code != 0
    assert "preset" in result.output.lower() or "lat" in result.output.lower()


def test_analyze_rejects_unknown_sensor() -> None:
    result = CliRunner().invoke(
        cli,
        ["analyze", "--sensor", "BOGUS", "--preset", "RUTFORD", "--output", "out"],
    )
    assert result.exit_code != 0
    assert "sensor" in result.output.lower()


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@cats_only
def test_plan_rutford_writes_csv_and_ics(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    result = CliRunner().invoke(
        cli,
        ["plan", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--start", "2026-06-01T00:00", "--end", "2026-07-15T00:00",
         "--top-n", "3", "--min-fringes", "8", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".ics").exists()
    ics_text = out.with_suffix(".ics").read_text()
    assert "BEGIN:VCALENDAR" in ics_text
    assert ics_text.count("BEGIN:VEVENT") == 3 * 3  # 3 rows x 3 vevents


def test_plan_rejects_bad_iso() -> None:
    result = CliRunner().invoke(
        cli,
        ["plan", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--start", "not-a-date", "--end", "2026-07-01T00:00",
         "--output", "p"],
    )
    assert result.exit_code != 0
    assert "ISO" in result.output or "start" in result.output.lower()


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


@cats_only
def test_synthesize_writes_geotiff(tmp_path: Path) -> None:
    out = tmp_path / "fmap"
    result = CliRunner().invoke(
        cli,
        ["synthesize", "--sensor", "L-BAND", "--preset", "RUTFORD",
         "--triplet-start-date", "2026-07-14T00:00",
         "--size", "3.0", "2.0",   # 3 km x 2 km for speed
         "--pixel-m", "50",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".tif").exists()
    # Verify GeoTIFF integrity
    import rasterio
    with rasterio.open(out.with_suffix(".tif")) as ds:
        assert ds.count == 3
        assert ds.crs.to_epsg() == 3031


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cats_only
def test_batch_with_site_presets_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "batch"
    result = CliRunner().invoke(
        cli,
        ["batch",
         "--site-preset", "THWAITES", "--site-preset", "RUTFORD",
         "--sensor", "L-BAND", "--sensor", "C-BAND",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "batch.parquet").exists()
    assert (out / "heatmap_P_usable_ge_3fr.csv").exists()


@cats_only
def test_batch_with_multiple_constellations(tmp_path: Path) -> None:
    """Grid of 1 site x 1 sensor x 2 constellations → 2 rows."""
    import pandas as pd

    out = tmp_path / "batch_constellations"
    result = CliRunner().invoke(
        cli,
        ["batch",
         "--site-preset", "RUTFORD",
         "--sensor", "L-BAND",
         "--constellation", "NISAR", "--constellation", "SENTINEL-1-DUAL",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(out / "batch.parquet")
    assert len(df) == 2
    assert set(df["n_satellites"]) == {1, 2}


@cats_only
def test_batch_with_yaml_sites(tmp_path: Path) -> None:
    yaml_text = """
sites:
  - THWAITES
  - name: custom_rutford
    lat: -78.5
    lon: -83.0
    ice_thickness_m: 2000
"""
    yaml_path = tmp_path / "sites.yaml"
    yaml_path.write_text(yaml_text)
    out = tmp_path / "batch_yaml"
    result = CliRunner().invoke(
        cli,
        ["batch", "--sites", str(yaml_path), "--sensor", "L-BAND",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "batch.parquet").exists()
    import pandas as pd
    df = pd.read_parquet(out / "batch.parquet")
    assert {"Thwaites_GL", "custom_rutford"} <= set(df["site"])


def test_batch_no_sites_errors() -> None:
    result = CliRunner().invoke(
        cli,
        ["batch", "--sensor", "L-BAND", "--output", "out"],
    )
    assert result.exit_code != 0
    assert "site" in result.output.lower()
