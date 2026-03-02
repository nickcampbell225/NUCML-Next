"""
Unit Tests for NNDCSigmaFetcher
===============================

Tests for the NNDCSigmaFetcher class which provides evaluated nuclear
cross-section data via the IAEA NDS API with local .npz caching.

Test coverage:
- Cache path format and naming convention
- Custom cache directory handling
- Library name mapping
- Repr output
- add_endf() / add_endf_auto() delegation
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nucml_next.visualization.endf_reader import NNDCSigmaFetcher


class TestNNDCSigmaFetcher(unittest.TestCase):
    """Unit tests for NNDCSigmaFetcher cache and configuration."""

    def test_cache_path_format(self):
        """Cache path should follow expected naming convention."""
        fetcher = NNDCSigmaFetcher(library="endfb8.0")
        path = fetcher._get_cache_path(17, 35, 103)
        self.assertEqual(path.name, "Cl-35_MT103_endfb8.0.npz")

    def test_cache_path_u235(self):
        """Cache path for U-235 fission should use correct symbol."""
        fetcher = NNDCSigmaFetcher(library="endfb8.0")
        path = fetcher._get_cache_path(92, 235, 18)
        self.assertEqual(path.name, "U-235_MT18_endfb8.0.npz")

    def test_repr(self):
        """repr should include library name."""
        fetcher = NNDCSigmaFetcher(library="endfb7.1")
        r = repr(fetcher)
        self.assertIn("endfb7.1", r)

    def test_custom_cache_dir(self):
        """Custom cache_dir should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)
            self.assertEqual(fetcher.cache_dir, Path(tmpdir))

    def test_default_cache_dir(self):
        """Default cache_dir should be in home directory."""
        fetcher = NNDCSigmaFetcher()
        self.assertTrue(str(fetcher.cache_dir).endswith("nndc_cache"))

    def test_default_library(self):
        """Default library should be endfb8.0."""
        fetcher = NNDCSigmaFetcher()
        self.assertEqual(fetcher.library, "endfb8.0")

    def test_different_libraries_different_paths(self):
        """Different libraries should produce different cache paths."""
        f1 = NNDCSigmaFetcher(library="endfb8.0")
        f2 = NNDCSigmaFetcher(library="endfb7.1")
        p1 = f1._get_cache_path(92, 235, 18)
        p2 = f2._get_cache_path(92, 235, 18)
        self.assertNotEqual(p1.name, p2.name)
        self.assertIn("endfb8.0", p1.name)
        self.assertIn("endfb7.1", p2.name)

    def test_iaea_lib_name_mapping(self):
        """Known short names should map to IAEA display names."""
        cases = {
            "endfb8.0": "ENDF/B-VIII.0",
            "endfb8.1": "ENDF/B-VIII.1",
            "jeff3.3": "JEFF-3.3",
            "jendl5": "JENDL-5",
            "tendl2023": "TENDL-2023",
        }
        for short, expected in cases.items():
            fetcher = NNDCSigmaFetcher(library=short)
            self.assertEqual(fetcher._iaea_lib_name, expected)

    def test_iaea_lib_name_passthrough(self):
        """Unknown library names should pass through as-is."""
        fetcher = NNDCSigmaFetcher(library="ENDF/B-VIII.0")
        self.assertEqual(fetcher._iaea_lib_name, "ENDF/B-VIII.0")

    def test_cache_round_trip(self):
        """Data saved to cache should be loadable."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            # Simulate caching data
            energies = np.logspace(-5, 7, 500)
            xs = np.random.exponential(0.1, 500)
            cache_path = fetcher._get_cache_path(17, 35, 103)
            np.savez_compressed(cache_path, energies=energies, xs=xs)

            # Load from cache (close handle to avoid Windows file-lock)
            with np.load(cache_path) as data:
                np.testing.assert_array_equal(data['energies'], energies)
                np.testing.assert_array_equal(data['xs'], xs)

    def test_clear_cache_all(self):
        """clear_cache() with no args should remove all .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            # Create some cache files
            for z, a, mt in [(17, 35, 103), (92, 235, 18), (92, 233, 1)]:
                path = fetcher._get_cache_path(z, a, mt)
                np.savez_compressed(path,
                                    energies=np.array([1.0]),
                                    xs=np.array([1.0]))

            self.assertEqual(len(fetcher.list_cached()), 3)
            removed = fetcher.clear_cache()
            self.assertEqual(removed, 3)
            self.assertEqual(len(fetcher.list_cached()), 0)

    def test_clear_cache_selective(self):
        """clear_cache(z, a) should only remove matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            for z, a, mt in [(17, 35, 103), (92, 235, 18), (92, 233, 1)]:
                path = fetcher._get_cache_path(z, a, mt)
                np.savez_compressed(path,
                                    energies=np.array([1.0]),
                                    xs=np.array([1.0]))

            removed = fetcher.clear_cache(z=92, a=235)
            self.assertEqual(removed, 1)
            self.assertEqual(len(fetcher.list_cached()), 2)

    def test_list_cached(self):
        """list_cached() should return stems of cached .npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)
            path = fetcher._get_cache_path(17, 35, 103)
            np.savez_compressed(path,
                                energies=np.array([1.0]),
                                xs=np.array([1.0]))

            cached = fetcher.list_cached()
            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0], "Cl-35_MT103_endfb8.0")

    def test_get_cross_section_from_cache(self):
        """get_cross_section() should load from cache when available."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            # Pre-populate cache
            energies = np.logspace(-5, 7, 500)
            xs = np.random.exponential(0.1, 500)
            cache_path = fetcher._get_cache_path(17, 35, 103)
            np.savez_compressed(cache_path, energies=energies, xs=xs)

            # Should load from cache, no network call
            with patch.object(fetcher, '_fetch_from_nndc') as mock_fetch:
                result_e, result_xs = fetcher.get_cross_section(z=17, a=35, mt=103)
                mock_fetch.assert_not_called()

            np.testing.assert_array_equal(result_e, energies)
            np.testing.assert_array_equal(result_xs, xs)

    def test_get_cross_section_energy_filter(self):
        """get_cross_section() should filter by energy range."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            # Pre-populate cache with known data
            energies = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            xs = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
            cache_path = fetcher._get_cache_path(92, 235, 18)
            np.savez_compressed(cache_path, energies=energies, xs=xs)

            result_e, result_xs = fetcher.get_cross_section(
                z=92, a=235, mt=18, energy_min=1.0, energy_max=100.0
            )

            self.assertEqual(len(result_e), 3)
            np.testing.assert_array_equal(result_e, [1.0, 10.0, 100.0])
            np.testing.assert_array_equal(result_xs, [2.0, 1.0, 0.5])

    def test_get_cross_section_bypasses_cache(self):
        """get_cross_section(use_cache=False) should always fetch fresh."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            fetcher = NNDCSigmaFetcher(cache_dir=tmpdir)

            # Pre-populate cache
            cache_path = fetcher._get_cache_path(17, 35, 103)
            np.savez_compressed(cache_path,
                                energies=np.array([1.0]),
                                xs=np.array([1.0]))

            fresh_e = np.logspace(-5, 7, 500)
            fresh_xs = np.random.exponential(0.1, 500)

            with patch.object(fetcher, '_fetch_from_nndc',
                              return_value=(fresh_e, fresh_xs)) as mock_fetch:
                result_e, result_xs = fetcher.get_cross_section(
                    z=17, a=35, mt=103, use_cache=False
                )
                mock_fetch.assert_called_once()

            np.testing.assert_array_equal(result_e, fresh_e)


class TestAddEndfDelegation(unittest.TestCase):
    """Verify that CrossSectionFigure.add_endf() delegates to add_nndc_sigma()."""

    def test_add_endf_calls_nndc(self):
        """add_endf() should delegate to add_nndc_sigma()."""
        from nucml_next.visualization.cross_section_figure import CrossSectionFigure

        fig = CrossSectionFigure('U-235', mt=18)

        fake_energies = np.logspace(-5, 7, 5000)
        fake_xs = np.random.exponential(0.1, 5000)

        with patch.object(fig, 'add_nndc_sigma', return_value=fig) as mock_nndc:
            fig.add_endf()
            mock_nndc.assert_called_once()

        fig.close()

    def test_add_endf_auto_calls_add_endf(self):
        """add_endf_auto() should delegate to add_endf()."""
        from nucml_next.visualization.cross_section_figure import CrossSectionFigure

        fig = CrossSectionFigure('Cl-35', mt=103)

        with patch.object(fig, 'add_endf', return_value=fig) as mock_endf:
            fig.add_endf_auto()
            mock_endf.assert_called_once()

        fig.close()

    def test_add_endf_no_endf_dir_param(self):
        """CrossSectionFigure should not accept endf_dir parameter."""
        from nucml_next.visualization.cross_section_figure import CrossSectionFigure
        import inspect

        sig = inspect.signature(CrossSectionFigure.__init__)
        self.assertNotIn('endf_dir', sig.parameters,
                         "endf_dir should have been removed from __init__")


class TestIsotopePlotterNoEndfDir(unittest.TestCase):
    """Verify IsotopePlotter no longer takes endf_dir."""

    def test_no_endf_dir_param(self):
        """IsotopePlotter should not accept endf_dir parameter."""
        from nucml_next.visualization.isotope_plotter import IsotopePlotter
        import inspect

        sig = inspect.signature(IsotopePlotter.__init__)
        self.assertNotIn('endf_dir', sig.parameters,
                         "endf_dir should have been removed from __init__")


class TestThresholdExplorerNoEndfDir(unittest.TestCase):
    """Verify ThresholdExplorer no longer takes endf_dir."""

    def test_no_endf_dir_param(self):
        """ThresholdExplorer should not accept endf_dir parameter."""
        from nucml_next.visualization.threshold_explorer import ThresholdExplorer
        import inspect

        sig = inspect.signature(ThresholdExplorer.__init__)
        self.assertNotIn('endf_dir', sig.parameters,
                         "endf_dir should have been removed from __init__")


if __name__ == "__main__":
    unittest.main()
