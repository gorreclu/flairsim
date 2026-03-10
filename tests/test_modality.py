"""Unit tests for flairsim.map.modality."""

import pytest

from flairsim.map.modality import Modality, ModalitySpec


# ---------------------------------------------------------------------------
# ModalitySpec
# ---------------------------------------------------------------------------


class TestModalitySpec:
    """Tests for the ModalitySpec dataclass."""

    def test_creation(self):
        spec = ModalitySpec(
            dir_suffix="TEST",
            pixel_size_m=0.5,
            patch_pixels=256,
            bands=3,
            dtype="uint8",
            is_time_series=False,
            description="Test modality.",
        )
        assert spec.dir_suffix == "TEST"
        assert spec.pixel_size_m == 0.5
        assert spec.patch_pixels == 256
        assert spec.bands == 3
        assert spec.dtype == "uint8"
        assert spec.is_time_series is False
        assert spec.description == "Test modality."

    def test_frozen(self):
        spec = ModalitySpec(
            dir_suffix="A",
            pixel_size_m=1.0,
            patch_pixels=10,
            bands=1,
            dtype="uint8",
            is_time_series=False,
            description="",
        )
        with pytest.raises(AttributeError):
            spec.bands = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Modality enum
# ---------------------------------------------------------------------------


class TestModality:
    """Tests for the Modality enum."""

    def test_member_count(self):
        """All 10 FLAIR-HUB modalities should be present."""
        assert len(Modality) == 10

    def test_aerial_rgbi(self):
        m = Modality.AERIAL_RGBI
        assert m.value.dir_suffix == "AERIAL_RGBI"
        assert m.value.pixel_size_m == 0.2
        assert m.value.patch_pixels == 512
        assert m.value.bands == 4
        assert m.value.dtype == "uint8"
        assert m.value.is_time_series is False

    def test_dem_elev(self):
        m = Modality.DEM_ELEV
        assert m.value.pixel_size_m == 0.2
        assert m.value.bands == 2
        assert m.value.dtype == "float32"

    def test_sentinel2_ts(self):
        m = Modality.SENTINEL2_TS
        assert m.value.is_time_series is True
        assert m.value.bands == 10
        assert m.value.patch_pixels == 10

    def test_label_cosia(self):
        m = Modality.LABEL_COSIA
        assert m.value.bands == 1
        assert m.value.dtype == "uint8"
        assert m.value.patch_pixels == 512

    def test_spec_property(self):
        """The .spec property should return the same object as .value."""
        for m in Modality:
            assert m.spec is m.value

    def test_patch_ground_size_m(self):
        """All modalities should cover 102.4m on the ground."""
        for m in Modality:
            ground = m.patch_ground_size_m
            assert abs(ground - 102.4) < 1e-6, (
                f"{m.name}: patch_ground_size_m = {ground}, expected 102.4"
            )

    def test_from_dir_suffix_found(self):
        result = Modality.from_dir_suffix("AERIAL_RGBI")
        assert result is Modality.AERIAL_RGBI

    def test_from_dir_suffix_dem(self):
        result = Modality.from_dir_suffix("DEM_ELEV")
        assert result is Modality.DEM_ELEV

    def test_from_dir_suffix_not_found(self):
        result = Modality.from_dir_suffix("NONEXISTENT")
        assert result is None

    def test_from_dir_suffix_all_round_trip(self):
        """Every modality should be retrievable by its dir_suffix."""
        for m in Modality:
            found = Modality.from_dir_suffix(m.value.dir_suffix)
            assert found is m
