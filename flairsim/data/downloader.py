"""
Download FLAIR-HUB data from HuggingFace and manage temporary storage.

The :class:`HubDownloader` automates the full workflow:

1. Download ZIP files from ``IGNF/FLAIR-HUB`` on HuggingFace.
2. Unzip them into a temporary directory.
3. Delete the ZIPs to save disk space.
4. On :meth:`cleanup`, remove the entire temporary directory.

ZIP naming convention on HuggingFace::

    data/{DOMAIN}_{MODALITY_SUFFIX}.zip
    e.g. data/D006-2020_AERIAL_RGBI.zip

After extraction the layout mirrors the flat FLAIR-HUB structure::

    <tmpdir>/
      D006-2020_AERIAL_RGBI/
        UU-S2-1/
          D006_AERIAL_RGBI_UU-S2-1_000-000.tif
          ...
      D006-2020_DEM_ELEV/
        ...
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

from ..map.modality import Modality

logger = logging.getLogger(__name__)

# HuggingFace coordinates.
_HF_REPO_ID = "IGNF/FLAIR-HUB"
_HF_REPO_TYPE = "dataset"


class HubDownloader:
    """Download and manage FLAIR-HUB data from HuggingFace.

    Parameters
    ----------
    domain : str
        FLAIR-HUB domain identifier, e.g. ``"D006-2020"``.
    modalities : sequence of str
        Modality names to download (matching :class:`~flairsim.map.modality.Modality`
        member names).  Defaults to ``["AERIAL_RGBI"]``.
    base_dir : Path or None
        Parent directory for the temporary download folder.  ``None``
        uses the system default (``/tmp`` on Linux/macOS).

    Attributes
    ----------
    data_dir : Path
        Path to the temporary directory containing extracted data.
        This is the value to pass as ``--data-dir`` to the simulator.
    """

    def __init__(
        self,
        domain: str,
        modalities: Sequence[str] | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.domain = domain
        self._modality_names = list(modalities or ["AERIAL_RGBI"])
        self._resolved_modalities = self._resolve_modalities(self._modality_names)

        # Create a temporary directory for downloads.
        self._tmpdir = tempfile.mkdtemp(
            prefix=f"flairsim_{domain}_",
            dir=str(base_dir) if base_dir else None,
        )
        self.data_dir = Path(self._tmpdir)
        self._downloaded = False
        logger.info(
            "HubDownloader: domain=%s, modalities=%s, tmpdir=%s",
            domain,
            self._modality_names,
            self._tmpdir,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def download(self) -> Path:
        """Download and extract all requested modalities.

        Returns
        -------
        Path
            The temporary data directory (same as :attr:`data_dir`).

        Raises
        ------
        ImportError
            If ``huggingface_hub`` is not installed.
        RuntimeError
            If a download or extraction fails.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for automatic data download. "
                "Install it with: pip install huggingface_hub  "
                "(or: uv sync --extra server)"
            )

        total = len(self._resolved_modalities)
        for idx, modality in enumerate(self._resolved_modalities, 1):
            suffix = modality.value.dir_suffix
            zip_filename = f"data/{self.domain}_{suffix}.zip"
            logger.info("Downloading %s from HuggingFace ...", zip_filename)
            # Print directly to stdout so users see progress even without
            # DEBUG logging — the HF progress bar can appear stuck on large
            # files (especially with the Xet storage backend).
            print(
                f"[{idx}/{total}] Downloading {self.domain}_{suffix}.zip "
                f"from HuggingFace ...\n"
                f"  (Large files may take several minutes. "
                f"The progress bar may stay at 0%% while data is fetched — "
                f"this is normal.)",
                flush=True,
            )

            try:
                local_path = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    repo_type=_HF_REPO_TYPE,
                    filename=zip_filename,
                    local_dir=str(self.data_dir),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download {zip_filename} from HuggingFace: {exc}"
                ) from exc

            local_path = Path(local_path)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                "Downloaded %s (%.1f MB). Extracting ...",
                zip_filename,
                size_mb,
            )
            print(
                f"  Downloaded {size_mb:.0f} MB. Extracting ...",
                flush=True,
            )

            # Extract the ZIP into the data_dir.
            try:
                with zipfile.ZipFile(local_path, "r") as zf:
                    zf.extractall(self.data_dir)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(f"Failed to extract {local_path}: {exc}") from exc

            # Delete the ZIP to save disk space.
            local_path.unlink(missing_ok=True)
            # Also clean up the data/ subdirectory if hf_hub_download
            # created it as an artifact.
            data_subdir = self.data_dir / "data"
            if data_subdir.is_dir() and not any(data_subdir.iterdir()):
                data_subdir.rmdir()

            logger.info(
                "Extracted %s_%s into %s",
                self.domain,
                suffix,
                self.data_dir,
            )
            print(f"  Extracted {self.domain}_{suffix}.", flush=True)

        self._downloaded = True
        logger.info(
            "All %d modalities downloaded and extracted to %s",
            len(self._resolved_modalities),
            self.data_dir,
        )
        print(
            f"All {len(self._resolved_modalities)} modality(ies) ready in "
            f"{self.data_dir}",
            flush=True,
        )
        return self.data_dir

    def cleanup(self) -> None:
        """Remove the temporary download directory and all its contents.

        Safe to call multiple times.  Logs a warning if removal fails
        (e.g. due to permissions) but does not raise.
        """
        if not self.data_dir.exists():
            logger.debug("Cleanup: tmpdir already removed: %s", self.data_dir)
            return

        logger.info("Cleaning up temporary data: %s", self.data_dir)
        try:
            shutil.rmtree(self.data_dir)
            logger.info("Cleanup complete: removed %s", self.data_dir)
        except OSError as exc:
            logger.warning(
                "Failed to remove temporary directory %s: %s",
                self.data_dir,
                exc,
            )

    @property
    def is_downloaded(self) -> bool:
        """Whether :meth:`download` has completed successfully."""
        return self._downloaded

    @property
    def modalities(self) -> list[Modality]:
        """Resolved :class:`Modality` enum members to download."""
        return list(self._resolved_modalities)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_modalities(names: list[str]) -> list[Modality]:
        """Convert modality name strings to Modality enum members.

        Parameters
        ----------
        names : list of str
            Modality names, e.g. ``["AERIAL_RGBI", "DEM_ELEV"]``.

        Returns
        -------
        list[Modality]

        Raises
        ------
        ValueError
            If any name doesn't match a known Modality.
        """
        result = []
        for name in names:
            try:
                result.append(Modality[name])
            except KeyError:
                valid = [m.name for m in Modality]
                raise ValueError(
                    f"Unknown modality '{name}'. Valid modalities: {valid}"
                )
        return result

    def __repr__(self) -> str:
        return (
            f"HubDownloader(domain={self.domain!r}, "
            f"modalities={self._modality_names!r}, "
            f"data_dir={str(self.data_dir)!r})"
        )
