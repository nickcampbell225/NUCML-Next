"""
Evaluated Cross-Section Data
=============================

Provides evaluated nuclear cross-section data via the IAEA Nuclear Data
Services (NDS) EXFOR/ENDF web API.  Data is fetched as PENDF (processed
ENDF) pointwise tables with full resonance reconstruction, and cached
locally as compressed ``.npz`` files for offline use.

The IAEA NDS API serves data from all major evaluated libraries including
ENDF/B, JEFF, JENDL, TENDL, CENDL, BROND, and more.

Classes:
    NNDCSigmaFetcher: HTTP client for IAEA NDS with local .npz caching

References:
    - IAEA NDS: https://nds.iaea.org/
    - EXFOR search API: https://nds.iaea.org/exfor/

Example:
    >>> from nucml_next.visualization import NNDCSigmaFetcher
    >>> fetcher = NNDCSigmaFetcher()
    >>> energies, xs = fetcher.get_cross_section(z=92, a=235, mt=18)
    >>> print(f"U-235 fission: {len(energies)} points")
"""

from pathlib import Path
from typing import Tuple, Optional, List, Union
import numpy as np
import warnings


class NNDCSigmaFetcher:
    """
    Fetch processed pointwise cross-section data from the IAEA NDS API.

    Uses the IAEA EXFOR/ENDF web service to retrieve PENDF (processed ENDF)
    pointwise cross-section data with full resonance reconstruction.  Covers
    all major evaluated nuclear data libraries: ENDF/B, JEFF, JENDL, TENDL,
    CENDL, BROND, and many more.

    Data is fetched via HTTP and cached locally as ``.npz`` files to avoid
    repeated downloads.

    Attributes:
        cache_dir: Directory for caching downloaded data
        library: Nuclear data library identifier

    Example:
        >>> from nucml_next.visualization import NNDCSigmaFetcher
        >>>
        >>> fetcher = NNDCSigmaFetcher()
        >>>
        >>> # Fetch Cl-35 (n,p) with full resonance detail
        >>> energies, xs = fetcher.get_cross_section(z=17, a=35, mt=103)
        >>> print(f"Got {len(energies)} points with resonance structure")
        >>>
        >>> # Clear cache if needed
        >>> fetcher.clear_cache()
    """

    # IAEA NDS EXFOR/ENDF search and data endpoints
    _SEARCH_URL = "https://nds.iaea.org/exfor/servlet/E4sSearch2"
    _DATA_URL = "https://nds.iaea.org/exfor/servlet/E4sGetTabSect"

    # Map short library names (used in cache filenames and __init__) to the
    # display names returned by the IAEA API in the 'LibName' field.
    _LIB_MAP = {
        "endfb8.1": "ENDF/B-VIII.1",
        "endfb8.0": "ENDF/B-VIII.0",
        "endfb7.1": "ENDF/B-VII.1",
        "endfb7.0": "ENDF/B-VII.0",
        "jeff4.0": "JEFF-4.0",
        "jeff3.3": "JEFF-3.3",
        "jeff3.2": "JEFF-3.2",
        "jendl5": "JENDL-5",
        "jendl4.0": "JENDL-4.0",
        "tendl2025": "TENDL-2025",
        "tendl2023": "TENDL-2023",
        "cendl3.2": "CENDL-3.2",
        "brond3.1": "BROND-3.1",
    }

    # Element symbols indexed by atomic number
    SYMBOLS = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
        93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es',
    }

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        library: str = "endfb8.0",
    ):
        """
        Initialize evaluated data fetcher.

        Args:
            cache_dir: Directory for caching downloaded data.
                       Default: ~/.nucml_next/nndc_cache
            library: Nuclear data library to use. Options:
                     - "endfb8.0" (default): ENDF/B-VIII.0
                     - "endfb8.1": ENDF/B-VIII.1
                     - "endfb7.1": ENDF/B-VII.1
                     - "jeff4.0": JEFF-4.0
                     - "jeff3.3": JEFF-3.3
                     - "jendl5": JENDL-5
                     - "tendl2023": TENDL-2023
                     Or any IAEA library name directly (e.g. "ENDF/B-VIII.0")
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".nucml_next" / "nndc_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.library = library

    @property
    def _iaea_lib_name(self) -> str:
        """Resolve the IAEA display name for the configured library."""
        return self._LIB_MAP.get(self.library, self.library)

    def _get_cache_path(self, z: int, a: int, mt: int) -> Path:
        """Get cache file path for given isotope and reaction."""
        symbol = self.SYMBOLS.get(z, f"Z{z}")
        return self.cache_dir / f"{symbol}-{a}_MT{mt}_{self.library}.npz"

    def get_cross_section(
        self,
        z: int,
        a: int,
        mt: int,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get processed pointwise cross-section data.

        Fetches PENDF data from the IAEA NDS API, which provides
        cross-sections that have been processed through NJOY/PREPRO to
        reconstruct resonance parameters into pointwise data.

        Args:
            z: Atomic number (e.g., 17 for Cl)
            a: Mass number (e.g., 35 for Cl-35)
            mt: Reaction type (MT code, e.g., 103 for (n,p))
            energy_min: Minimum energy filter in eV (optional)
            energy_max: Maximum energy filter in eV (optional)
            use_cache: Use cached data if available (default True)

        Returns:
            Tuple of (energies, cross_sections) as numpy arrays
            - energies: Energy points in eV
            - cross_sections: Cross-section values in barns

        Raises:
            RuntimeError: If data cannot be fetched
            ValueError: If the requested data is not available

        Example:
            >>> fetcher = NNDCSigmaFetcher()
            >>> energies, xs = fetcher.get_cross_section(z=17, a=35, mt=103)
            >>> print(f"Cl-35 (n,p): {len(energies)} points")
        """
        cache_path = self._get_cache_path(z, a, mt)

        # Check cache
        if use_cache and cache_path.exists():
            data = np.load(cache_path)
            energies = data['energies']
            xs = data['xs']
        else:
            # Fetch from IAEA NDS
            energies, xs = self._fetch_from_nndc(z, a, mt)

            # Save to cache
            np.savez_compressed(cache_path, energies=energies, xs=xs)

        # Apply energy filters
        if energy_min is not None or energy_max is not None:
            mask = np.ones(len(energies), dtype=bool)
            if energy_min is not None:
                mask &= energies >= energy_min
            if energy_max is not None:
                mask &= energies <= energy_max
            energies = energies[mask]
            xs = xs[mask]

        return energies, xs

    def _fetch_from_nndc(self, z: int, a: int, mt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch cross-section data from the IAEA NDS API.

        Two-step process:
        1. Search for evaluated sections matching (Z, A, MT, library)
        2. Fetch the PENDF pointwise data for the best match

        Returns data as (energies_eV, cross_sections_barns).
        """
        import json
        import urllib.request
        import urllib.parse

        symbol = self.SYMBOLS.get(z, f"Z{z}")
        target = f"{z}-{symbol}-{a}"
        lib_name = self._iaea_lib_name

        # Step 1: Search for evaluated sections
        search_params = {
            'Target': target,
            'MF': '3',
            'MT': str(mt),
            'Quantity': 'SIG',
            'json': '',
        }
        search_url = f"{self._SEARCH_URL}?{urllib.parse.urlencode(search_params)}"

        try:
            req = urllib.request.Request(
                search_url, headers={'User-Agent': 'nucml-next/1.0'},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                search_data = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(
                f"IAEA NDS search failed for {symbol}-{a} MT={mt}: {e}"
            )

        # Find best section: prefer NSUB=10 (incident neutrons) to avoid
        # picking up photonuclear (NSUB=0) or other sublibraries.
        # Priority: exact lib + NSUB=10 > exact lib > any lib + NSUB=10 > any
        sections = search_data.get('sections', [])
        pen_sect_id = None
        matched_lib = None

        def _pick(sects, lib_filter=None, nsub_filter=None):
            for s in sects:
                if not s.get('PenSectID'):
                    continue
                if lib_filter and s.get('LibName', '') != lib_filter:
                    continue
                if nsub_filter is not None and s.get('NSUB') != nsub_filter:
                    continue
                return s['PenSectID'], s.get('LibName', '?')
            return None, None

        for lib_f, nsub_f in [
            (lib_name, 10),    # exact library, neutron sublibrary
            (lib_name, None),  # exact library, any sublibrary
            (None, 10),        # any library, neutron sublibrary
            (None, None),      # any library, any sublibrary
        ]:
            pen_sect_id, matched_lib = _pick(sections, lib_f, nsub_f)
            if pen_sect_id is not None:
                if matched_lib != lib_name:
                    warnings.warn(
                        f"{lib_name} not available for {symbol}-{a} MT={mt}, "
                        f"using {matched_lib} instead.",
                        UserWarning,
                    )
                break

        if pen_sect_id is None:
            raise ValueError(
                f"No evaluated data found for {symbol}-{a} MT={mt}.\n"
                f"Searched {len(sections)} evaluations, none had PENDF data.\n"
                f"Check: https://nds.iaea.org/exfor/"
            )

        # Step 2: Fetch PENDF pointwise data
        data_url = f"{self._DATA_URL}?PenSectID={pen_sect_id}&json"

        try:
            req = urllib.request.Request(
                data_url, headers={'User-Agent': 'nucml-next/1.0'},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(
                f"IAEA NDS data fetch failed for {symbol}-{a} MT={mt} "
                f"(PenSectID={pen_sect_id}): {e}"
            )

        # Parse JSON response
        datasets = content.get('datasets', [])
        if not datasets:
            raise ValueError(
                f"Empty response from IAEA NDS for {symbol}-{a} MT={mt} "
                f"(PenSectID={pen_sect_id})."
            )

        pts = datasets[0].get('pts', [])
        if not pts:
            raise ValueError(
                f"No data points in IAEA response for {symbol}-{a} MT={mt}."
            )

        energies = np.array([p['E'] for p in pts], dtype=np.float64)
        xs_values = np.array([p['Sig'] for p in pts], dtype=np.float64)

        # Filter out non-physical values
        valid = (energies > 0) & (xs_values >= 0)
        energies = energies[valid]
        xs_values = xs_values[valid]

        if len(energies) == 0:
            raise ValueError(
                f"No valid data points for {symbol}-{a} MT={mt} from {matched_lib}."
            )

        return energies, xs_values

    def clear_cache(self, z: Optional[int] = None, a: Optional[int] = None) -> int:
        """
        Clear cached data.

        Args:
            z: Atomic number (optional, clears all if not specified)
            a: Mass number (optional, clears all if not specified)

        Returns:
            Number of cache files removed

        Example:
            >>> fetcher = NNDCSigmaFetcher()
            >>> fetcher.clear_cache()  # Clear all
            >>> fetcher.clear_cache(z=17, a=35)  # Clear only Cl-35
        """
        count = 0
        pattern = "*.npz"

        if z is not None and a is not None:
            symbol = self.SYMBOLS.get(z, f"Z{z}")
            pattern = f"{symbol}-{a}_*.npz"
        elif z is not None:
            symbol = self.SYMBOLS.get(z, f"Z{z}")
            pattern = f"{symbol}-*_*.npz"

        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            count += 1

        return count

    def list_cached(self) -> List[str]:
        """
        List all cached datasets.

        Returns:
            List of cached dataset identifiers (e.g., "Cl-35_MT103_endfb8.0")
        """
        return [f.stem for f in self.cache_dir.glob("*.npz")]

    def __repr__(self) -> str:
        """String representation."""
        return f"NNDCSigmaFetcher(library={self.library}, cache={self.cache_dir})"
