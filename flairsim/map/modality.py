"""
Modality definitions for the FLAIR-HUB dataset.

Each modality corresponds to a sensor/data-type combination available in
FLAIR-HUB.  The :class:`Modality` enum encapsulates the directory naming
convention, default pixel resolution, and band count so the rest of the
codebase never has to hard-code these values.

Reference
---------
FLAIR-HUB data paper -- Table V (modality overview):
    https://arxiv.org/abs/2506.07080
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModalitySpec:
    """Immutable specification of a FLAIR-HUB data modality.

    Attributes
    ----------
    dir_suffix : str
        The suffix used in the FLAIR-HUB directory naming convention.
        A full directory name is ``{DOMAIN}_{dir_suffix}``, e.g.
        ``D004-2021_AERIAL_RGBI``.
    pixel_size_m : float
        Ground sampling distance in metres per pixel.
    patch_pixels : int
        Side length of a single patch in pixels (patches are always square).
    bands : int
        Number of spectral / data bands.  For time-series modalities this
        is the number of bands *per acquisition date* -- the actual band
        count in the file is ``bands * n_dates``.
    dtype : str
        NumPy-compatible data-type string (e.g. ``"uint8"``, ``"float32"``).
    is_time_series : bool
        Whether the modality contains a temporal stack of acquisitions.
    description : str
        Human-readable description.
    """

    dir_suffix: str
    pixel_size_m: float
    patch_pixels: int
    bands: int
    dtype: str
    is_time_series: bool
    description: str


@unique
class Modality(Enum):
    """Enumeration of every data modality available in FLAIR-HUB.

    Each member wraps a :class:`ModalitySpec` that carries the physical
    and structural metadata of the corresponding data layer.

    Examples
    --------
    >>> Modality.AERIAL_RGBI.value.pixel_size_m
    0.2
    >>> Modality.AERIAL_RGBI.value.patch_pixels
    512
    >>> Modality.SENTINEL2_TS.value.is_time_series
    True
    """

    # -- Mono-temporal image modalities ---------------------------------------

    AERIAL_RGBI = ModalitySpec(
        dir_suffix="AERIAL_RGBI",
        pixel_size_m=0.2,
        patch_pixels=512,
        bands=4,
        dtype="uint8",
        is_time_series=False,
        description="BD ORTHO aerial imagery (Red, Green, Blue, Near-Infrared).",
    )

    AERIAL_RLT_PAN = ModalitySpec(
        dir_suffix="AERIAL-RLT_PAN",
        pixel_size_m=0.4,
        patch_pixels=256,
        bands=1,
        dtype="uint8",
        is_time_series=False,
        description="Historical panchromatic aerial imagery (~1950s).",
    )

    DEM_ELEV = ModalitySpec(
        dir_suffix="DEM_ELEV",
        pixel_size_m=0.2,
        patch_pixels=512,
        bands=2,
        dtype="float32",
        is_time_series=False,
        description="Digital Elevation Model (DSM + DTM channels).",
    )

    SPOT_RGBI = ModalitySpec(
        dir_suffix="SPOT_RGBI",
        pixel_size_m=1.6,
        patch_pixels=64,
        bands=4,
        dtype="uint16",
        is_time_series=False,
        description="SPOT 6-7 satellite imagery (R, G, B, NIR).",
    )

    # -- Time-series modalities -----------------------------------------------

    SENTINEL1_ASC_TS = ModalitySpec(
        dir_suffix="SENTINEL1-ASC_TS",
        pixel_size_m=10.24,
        patch_pixels=10,
        bands=2,
        dtype="float32",
        is_time_series=True,
        description="Sentinel-1 ascending SAR backscatter time series (VV, VH).",
    )

    SENTINEL1_DESC_TS = ModalitySpec(
        dir_suffix="SENTINEL1-DESC_TS",
        pixel_size_m=10.24,
        patch_pixels=10,
        bands=2,
        dtype="float32",
        is_time_series=True,
        description="Sentinel-1 descending SAR backscatter time series (VV, VH).",
    )

    SENTINEL2_TS = ModalitySpec(
        dir_suffix="SENTINEL2_TS",
        pixel_size_m=10.24,
        patch_pixels=10,
        bands=10,
        dtype="uint16",
        is_time_series=True,
        description="Sentinel-2 multispectral time series (10 bands).",
    )

    # -- Label / supervision modalities ---------------------------------------

    LABEL_COSIA = ModalitySpec(
        dir_suffix="AERIAL_LABEL-COSIA",
        pixel_size_m=0.2,
        patch_pixels=512,
        bands=1,
        dtype="uint8",
        is_time_series=False,
        description="Land-cover semantic labels (19 COSIA classes).",
    )

    LABEL_LPIS = ModalitySpec(
        dir_suffix="ALL_LABEL-LPIS",
        pixel_size_m=0.2,
        patch_pixels=512,
        bands=3,
        dtype="uint8",
        is_time_series=False,
        description="Crop-type labels from LPIS declarations (3-level hierarchy).",
    )

    # -- Mask modalities ------------------------------------------------------

    SENTINEL2_MSK_SC = ModalitySpec(
        dir_suffix="SENTINEL2_MSK-SC",
        pixel_size_m=10.24,
        patch_pixels=10,
        bands=2,
        dtype="uint16",
        is_time_series=True,
        description="Sentinel-2 snow and cloud mask time series.",
    )

    # -- Convenience properties -----------------------------------------------

    @property
    def spec(self) -> ModalitySpec:
        """Return the :class:`ModalitySpec` associated with this modality."""
        return self.value

    @property
    def patch_ground_size_m(self) -> float:
        """Side length in metres that one patch covers on the ground.

        All FLAIR-HUB modalities are designed so that every patch -- regardless
        of pixel resolution -- covers exactly the same 102.4 m x 102.4 m
        ground footprint.
        """
        return self.value.patch_pixels * self.value.pixel_size_m

    # -- Lookup helpers -------------------------------------------------------

    @classmethod
    def from_dir_suffix(cls, suffix: str) -> Optional["Modality"]:
        """Resolve a :class:`Modality` from its FLAIR-HUB directory suffix.

        Parameters
        ----------
        suffix : str
            Directory suffix such as ``"AERIAL_RGBI"`` or ``"DEM_ELEV"``.

        Returns
        -------
        Modality or None
            The matching enum member, or ``None`` if no match is found.
        """
        for member in cls:
            if member.value.dir_suffix == suffix:
                return member
        return None


# ---------------------------------------------------------------------------
# Modality discovery
# ---------------------------------------------------------------------------

# The primary modality is the one used for the ``image`` field in
# :class:`~flairsim.core.observation.Observation` when multiple
# modalities are loaded.  We prefer AERIAL_RGBI because it's the
# highest-resolution visual modality and the most useful for VLMs.
_PRIMARY_PREFERENCE = [
    "AERIAL_RGBI",
    "SPOT_RGBI",
    "AERIAL-RLT_PAN",
    "SENTINEL2_TS",
]


def _extract_domain_prefix(dirname: str) -> Optional[str]:
    """Extract the domain prefix (e.g. ``"D006-2020"``) from a FLAIR-HUB directory name.

    FLAIR-HUB directories follow the pattern ``{DOMAIN}-{YEAR}_{SUFFIX}``,
    e.g. ``D006-2020_AERIAL_RGBI``.  The domain prefix is the part before
    the first underscore: ``D006-2020``.

    The historical panchromatic modality uses a slightly different pattern:
    ``D006-195X_AERIAL-RLT_PAN`` -- the prefix is ``D006-195X``.

    Returns
    -------
    str or None
        The domain prefix, or ``None`` if the name doesn't match.
    """
    # Pattern: D{NNN}-{YEAR}_{rest}
    # The domain prefix is everything up to the first underscore.
    idx = dirname.find("_")
    if idx <= 0:
        return None
    prefix = dirname[:idx]
    # Basic validation: should contain a dash (e.g. "D006-2020").
    if "-" not in prefix:
        return None
    return prefix


def discover_modalities(
    parent_dir: str | Path,
    domain: Optional[str] = None,
) -> Dict[Modality, Path]:
    """Auto-discover FLAIR-HUB modalities under a parent directory.

    The function scans *parent_dir* for sub-directories whose names
    end with a known :class:`Modality` ``dir_suffix``.  For example,
    a directory named ``D006-2020_AERIAL_RGBI`` matches the suffix
    ``AERIAL_RGBI`` and maps to :attr:`Modality.AERIAL_RGBI`.

    When *domain* is given (e.g. ``"D006-2020"``), only directories
    whose name starts with that prefix are considered.  This is
    essential for the **flat FLAIR-HUB layout** where all domains
    coexist as sibling directories::

        FLAIR-HUB/
          D006-2020_AERIAL_RGBI/
          D006-2020_DEM_ELEV/
          D012-2019_AERIAL_RGBI/   <-- different domain, same level
          ...

    Without a domain filter, ``D012-2019_AERIAL_RGBI`` would shadow
    ``D006-2020_AERIAL_RGBI`` (or vice-versa) since both match the
    ``AERIAL_RGBI`` suffix.

    Parameters
    ----------
    parent_dir : str or Path
        Top-level data directory containing modality sub-dirs.
    domain : str or None
        Domain prefix to filter on (e.g. ``"D006-2020"``).  ``None``
        disables filtering (all directories are considered).

    Returns
    -------
    dict[Modality, Path]
        Mapping from discovered modality to its data directory path.
        Empty if no modalities are found.

    Examples
    --------
    >>> mods = discover_modalities("path/to/FLAIR-HUB", domain="D006-2020")
    >>> Modality.AERIAL_RGBI in mods
    True
    """
    parent = Path(parent_dir).resolve()
    if not parent.is_dir():
        logger.warning("Parent directory does not exist: %s", parent)
        return {}

    result: Dict[Modality, Path] = {}

    for entry in sorted(parent.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name

        # If a domain filter is set, skip directories that don't match.
        if domain is not None:
            if not name.startswith(domain + "_"):
                continue

        # Try every known suffix and check if the dir name ends with it.
        for member in Modality:
            suffix = member.value.dir_suffix
            if name.endswith(suffix):
                # Verify it's not a false partial match by checking
                # that the character before the suffix is '_' or it
                # starts the name.
                prefix_end = len(name) - len(suffix)
                if prefix_end == 0 or name[prefix_end - 1] == "_":
                    result[member] = entry
                    break

    if result:
        logger.info(
            "Discovered %d modalities in %s%s: %s",
            len(result),
            parent,
            f" (domain={domain})" if domain else "",
            [m.name for m in result],
        )
    else:
        logger.debug(
            "No modalities discovered in %s%s",
            parent,
            f" (domain={domain})" if domain else "",
        )

    return result


def infer_domain_from_dir(data_dir: str | Path) -> Optional[str]:
    """Infer the FLAIR-HUB domain prefix from a data directory name.

    Given a path like ``/data/FLAIR-HUB/D006-2020_AERIAL_RGBI``, this
    extracts the domain prefix ``"D006-2020"`` from the directory name.

    This allows backward-compatible usage: when a user passes
    ``--data-dir D006-2020_AERIAL_RGBI``, we can infer the domain and
    use it to discover sibling modalities.

    Parameters
    ----------
    data_dir : str or Path
        Path to a data directory.

    Returns
    -------
    str or None
        The domain prefix (e.g. ``"D006-2020"``), or ``None`` if
        the directory name doesn't match the FLAIR-HUB pattern.
    """
    dirname = Path(data_dir).name
    return _extract_domain_prefix(dirname)


def pick_primary_modality(modalities: Dict[Modality, Path]) -> Modality:
    """Choose the primary modality from a set of available ones.

    The primary modality is used for the main ``image`` field in
    observations.  The selection order follows :data:`_PRIMARY_PREFERENCE`.

    Parameters
    ----------
    modalities : dict[Modality, Path]
        Available modalities (must be non-empty).

    Returns
    -------
    Modality
        The chosen primary modality.

    Raises
    ------
    ValueError
        If *modalities* is empty.
    """
    if not modalities:
        raise ValueError("No modalities to choose from.")

    for suffix in _PRIMARY_PREFERENCE:
        mod = Modality.from_dir_suffix(suffix)
        if mod is not None and mod in modalities:
            return mod

    # Fallback: return the first one.
    return next(iter(modalities))


def is_single_modality_dir(data_dir: str | Path) -> bool:
    """Check whether *data_dir* is a single-modality directory.

    A single-modality directory directly contains ``.tif`` files (or
    ROI sub-directories with ``.tif`` files).  A parent/multi-modality
    directory instead contains sub-directories whose names match known
    modality suffixes.

    This is used for backward compatibility: if the user passes a
    single modality directory (the old behaviour), we skip auto-discovery.

    Parameters
    ----------
    data_dir : str or Path
        The path to check.

    Returns
    -------
    bool
        ``True`` if *data_dir* appears to be a single-modality directory.
    """
    p = Path(data_dir).resolve()
    if not p.is_dir():
        return False

    # Check for .tif files directly in the directory.
    if any(p.glob("*.tif")):
        return True

    # Check for .tif files in immediate subdirectories (ROI layout).
    for sub in p.iterdir():
        if sub.is_dir() and any(sub.glob("*.tif")):
            return True

    return False
