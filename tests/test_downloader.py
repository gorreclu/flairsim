"""Tests for flairsim.data.downloader (HubDownloader).

All tests mock the ``huggingface_hub`` dependency so no actual network
calls are made.  We verify:

- Modality resolution (valid / invalid names).
- Correct HF filenames are constructed.
- ZIPs are extracted and then deleted.
- Cleanup removes the entire temporary directory.
- Edge cases: double cleanup, download failure cleanup, missing hf_hub.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flairsim.data.downloader import HubDownloader
from flairsim.map.modality import Modality


# ---------------------------------------------------------------------------
# Modality resolution
# ---------------------------------------------------------------------------


class TestModalityResolution:
    """Tests for HubDownloader._resolve_modalities()."""

    def test_single_valid_modality(self):
        result = HubDownloader._resolve_modalities(["AERIAL_RGBI"])
        assert result == [Modality.AERIAL_RGBI]

    def test_multiple_valid_modalities(self):
        result = HubDownloader._resolve_modalities(["AERIAL_RGBI", "DEM_ELEV"])
        assert result == [Modality.AERIAL_RGBI, Modality.DEM_ELEV]

    def test_all_modalities(self):
        names = [m.name for m in Modality]
        result = HubDownloader._resolve_modalities(names)
        assert len(result) == len(Modality)

    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality 'NOT_A_MODALITY'"):
            HubDownloader._resolve_modalities(["NOT_A_MODALITY"])

    def test_mixed_valid_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown modality 'BAD'"):
            HubDownloader._resolve_modalities(["AERIAL_RGBI", "BAD"])

    def test_empty_list(self):
        result = HubDownloader._resolve_modalities([])
        assert result == []


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for HubDownloader.__init__()."""

    def test_creates_tmpdir(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        assert dl.data_dir.exists()
        assert dl.data_dir.is_dir()
        assert "flairsim_D006-2020_" in dl.data_dir.name
        dl.cleanup()

    def test_default_modality_is_aerial_rgbi(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        assert dl.modalities == [Modality.AERIAL_RGBI]
        dl.cleanup()

    def test_custom_modalities(self, tmp_path):
        dl = HubDownloader(
            "D006-2020",
            modalities=["DEM_ELEV", "SPOT_RGBI"],
            base_dir=tmp_path,
        )
        assert dl.modalities == [Modality.DEM_ELEV, Modality.SPOT_RGBI]
        dl.cleanup()

    def test_is_downloaded_initially_false(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        assert not dl.is_downloaded
        dl.cleanup()

    def test_repr(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        r = repr(dl)
        assert "D006-2020" in r
        assert "AERIAL_RGBI" in r
        dl.cleanup()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _make_fake_zip(zip_path: Path, domain: str, suffix: str) -> None:
    """Create a small ZIP that extracts to ``{domain}_{suffix}/roi/tile.tif``."""
    dir_name = f"{domain}_{suffix}"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Write a fake TIF file inside a ROI subdirectory.
        zf.writestr(f"{dir_name}/UU-S2-1/fake_tile.tif", b"fake tif data")


class TestDownload:
    """Tests for HubDownloader.download() with mocked hf_hub_download."""

    def test_download_single_modality(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)

        def fake_hf_download(*, repo_id, repo_type, filename, local_dir):
            # Create a fake ZIP at the expected location.
            zip_path = Path(local_dir) / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            _make_fake_zip(zip_path, "D006-2020", "AERIAL_RGBI")
            return str(zip_path)

        with patch(
            "flairsim.data.downloader.hf_hub_download",
            side_effect=fake_hf_download,
            create=True,
        ):
            # We need to patch the import inside the method.
            with patch.dict(
                "sys.modules",
                {"huggingface_hub": MagicMock(hf_hub_download=fake_hf_download)},
            ):
                result = dl.download()

        assert result == dl.data_dir
        assert dl.is_downloaded
        # The extracted directory should exist.
        extracted = dl.data_dir / "D006-2020_AERIAL_RGBI"
        assert extracted.is_dir()
        # The ROI subdirectory should exist.
        roi_dir = extracted / "UU-S2-1"
        assert roi_dir.is_dir()
        assert (roi_dir / "fake_tile.tif").exists()
        # The ZIP should have been deleted.
        zip_path = dl.data_dir / "data" / "D006-2020_AERIAL_RGBI.zip"
        assert not zip_path.exists()

        dl.cleanup()

    def test_download_multiple_modalities(self, tmp_path):
        dl = HubDownloader(
            "D006-2020",
            modalities=["AERIAL_RGBI", "DEM_ELEV"],
            base_dir=tmp_path,
        )

        call_log = []

        def fake_hf_download(*, repo_id, repo_type, filename, local_dir):
            call_log.append(filename)
            zip_path = Path(local_dir) / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            # Extract suffix from filename: "data/D006-2020_AERIAL_RGBI.zip" -> "AERIAL_RGBI"
            stem = Path(filename).stem  # "D006-2020_AERIAL_RGBI"
            suffix = stem.split("_", 1)[1]  # "AERIAL_RGBI"
            _make_fake_zip(zip_path, "D006-2020", suffix)
            return str(zip_path)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=fake_hf_download)},
        ):
            dl.download()

        assert len(call_log) == 2
        assert "data/D006-2020_AERIAL_RGBI.zip" in call_log
        assert "data/D006-2020_DEM_ELEV.zip" in call_log

        assert (dl.data_dir / "D006-2020_AERIAL_RGBI").is_dir()
        assert (dl.data_dir / "D006-2020_DEM_ELEV").is_dir()

        dl.cleanup()

    def test_download_constructs_correct_filenames(self, tmp_path):
        """Verify ZIP filenames match the HF naming convention for all modalities."""
        expected_filenames = {
            "AERIAL_RGBI": "data/D006-2020_AERIAL_RGBI.zip",
            "AERIAL_RLT_PAN": "data/D006-2020_AERIAL-RLT_PAN.zip",
            "DEM_ELEV": "data/D006-2020_DEM_ELEV.zip",
            "SPOT_RGBI": "data/D006-2020_SPOT_RGBI.zip",
            "SENTINEL1_ASC_TS": "data/D006-2020_SENTINEL1-ASC_TS.zip",
            "SENTINEL1_DESC_TS": "data/D006-2020_SENTINEL1-DESC_TS.zip",
            "SENTINEL2_TS": "data/D006-2020_SENTINEL2_TS.zip",
            "LABEL_COSIA": "data/D006-2020_AERIAL_LABEL-COSIA.zip",
            "LABEL_LPIS": "data/D006-2020_ALL_LABEL-LPIS.zip",
            "SENTINEL2_MSK_SC": "data/D006-2020_SENTINEL2_MSK-SC.zip",
        }

        for mod_name, expected_filename in expected_filenames.items():
            dl = HubDownloader("D006-2020", modalities=[mod_name], base_dir=tmp_path)
            call_log = []

            def fake_hf_download(*, repo_id, repo_type, filename, local_dir):
                call_log.append(filename)
                zip_path = Path(local_dir) / filename
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                stem = Path(filename).stem
                suffix = stem.split("_", 1)[1]
                _make_fake_zip(zip_path, "D006-2020", suffix)
                return str(zip_path)

            with patch.dict(
                "sys.modules",
                {"huggingface_hub": MagicMock(hf_hub_download=fake_hf_download)},
            ):
                dl.download()

            assert len(call_log) == 1, f"Expected 1 call for {mod_name}"
            assert call_log[0] == expected_filename, (
                f"Modality {mod_name}: expected {expected_filename}, got {call_log[0]}"
            )
            dl.cleanup()

    def test_download_failure_raises_runtime_error(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)

        def failing_download(*, repo_id, repo_type, filename, local_dir):
            raise ConnectionError("Network error")

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=failing_download)},
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                dl.download()

        assert not dl.is_downloaded
        dl.cleanup()

    def test_download_bad_zip_raises_runtime_error(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)

        def bad_zip_download(*, repo_id, repo_type, filename, local_dir):
            zip_path = Path(local_dir) / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            # Write invalid data (not a valid ZIP).
            zip_path.write_bytes(b"not a zip file")
            return str(zip_path)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=bad_zip_download)},
        ):
            with pytest.raises(RuntimeError, match="Failed to extract"):
                dl.download()

        dl.cleanup()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for HubDownloader.cleanup()."""

    def test_cleanup_removes_tmpdir(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        tmpdir = dl.data_dir
        assert tmpdir.exists()
        dl.cleanup()
        assert not tmpdir.exists()

    def test_cleanup_removes_contents(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        # Create some files inside.
        (dl.data_dir / "somefile.txt").write_text("hello")
        subdir = dl.data_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("world")

        dl.cleanup()
        assert not dl.data_dir.exists()

    def test_double_cleanup_is_safe(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        dl.cleanup()
        # Second call should not raise.
        dl.cleanup()

    def test_cleanup_after_download(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)

        def fake_hf_download(*, repo_id, repo_type, filename, local_dir):
            zip_path = Path(local_dir) / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            _make_fake_zip(zip_path, "D006-2020", "AERIAL_RGBI")
            return str(zip_path)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=fake_hf_download)},
        ):
            dl.download()

        assert dl.is_downloaded
        assert (dl.data_dir / "D006-2020_AERIAL_RGBI").exists()

        dl.cleanup()
        assert not dl.data_dir.exists()


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Test that missing huggingface_hub gives a clear error."""

    def test_missing_huggingface_hub_raises_import_error(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                dl.download()

        dl.cleanup()


# ---------------------------------------------------------------------------
# HF API parameters
# ---------------------------------------------------------------------------


class TestHFAPIParameters:
    """Verify correct repo_id and repo_type are passed to hf_hub_download."""

    def test_correct_repo_params(self, tmp_path):
        dl = HubDownloader("D006-2020", base_dir=tmp_path)
        call_kwargs = []

        def spy_download(*, repo_id, repo_type, filename, local_dir):
            call_kwargs.append(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "filename": filename,
                }
            )
            zip_path = Path(local_dir) / filename
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            _make_fake_zip(zip_path, "D006-2020", "AERIAL_RGBI")
            return str(zip_path)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(hf_hub_download=spy_download)},
        ):
            dl.download()

        assert len(call_kwargs) == 1
        assert call_kwargs[0]["repo_id"] == "IGNF/FLAIR-HUB"
        assert call_kwargs[0]["repo_type"] == "dataset"

        dl.cleanup()
