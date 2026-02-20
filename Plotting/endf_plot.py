# endf_plotter.py
from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple

# Z -> element symbol
_SYMBOL = [
    None,
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db",
    "Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]

PathLike = Union[str, Path]

def za_to_nuclide(Z: int, A: int) -> str:
    return f"{_SYMBOL[Z]}{A}"

def find_neutron_h5(h5_dir: PathLike, Z: int, A: int) -> Path:
    h5_dir = Path(h5_dir)
    nuclide = za_to_nuclide(Z, A)

    for name in (f"{nuclide}.h5", f"{_SYMBOL[Z]}-{A}.h5", f"{_SYMBOL[Z]}_{A}.h5"):
        p = h5_dir / name
        if p.exists():
            return p

    token = nuclide.lower()
    hits = [p for p in h5_dir.glob("*.h5") if token in p.name.lower()]
    if not hits:
        raise FileNotFoundError(f"Can't find {nuclide}.h5 in {h5_dir}")
    hits.sort(key=lambda p: (len(p.name), p.name))
    return hits[0]

def read_xs_schema_ugrp(
    h5_path: PathLike,
    nuclide: str,
    MT: int,
    temperature: str = "294K",
) -> Tuple["object", "object", str]:
    """
    For your files:
      <nuclide>/energy/<temp>                       -> energy grid (eV)
      <nuclide>/reactions/reaction_###/<temp>/xs    -> cross section (barns)
    Returns (E, xs, used_temperature).
    """
    h5_path = Path(h5_path)
    rx = f"{nuclide}/reactions/reaction_{MT:03d}"

    with h5py.File(h5_path, "r") as f:
        if nuclide not in f:
            raise KeyError(
                f"Missing top group '{nuclide}' in {h5_path.name}. Top keys: {list(f.keys())}"
            )

        g0 = f[nuclide]

        temps = list(g0["energy"].keys())
        if temperature not in temps:
            temperature = temps[0]

        E = g0["energy"][temperature][()]

        if rx not in f:
            mts = sorted(
                int(k.split("_")[1]) for k in g0["reactions"].keys() if k.startswith("reaction_")
            )
            raise KeyError(
                f"MT={MT} not present for {nuclide} in {h5_path.name}. "
                f"Available (first 50): {mts[:50]}"
            )

        rg = f[rx]
        if temperature not in rg:
            avail = [k for k in rg.keys() if k.endswith("K")]
            raise KeyError(
                f"{nuclide} MT {MT} has no '{temperature}'. Available temps: {avail}"
            )

        xsnode = rg[temperature]["xs"]

        if isinstance(xsnode, h5py.Dataset):
            xs = xsnode[()]
        elif isinstance(xsnode, h5py.Group) and "y" in xsnode:
            xs = xsnode["y"][()]
        else:
            if isinstance(xsnode, h5py.Group):
                ks = list(xsnode.keys())
                if len(ks) == 1 and isinstance(xsnode[ks[0]], h5py.Dataset):
                    xs = xsnode[ks[0]][()]
                else:
                    raise RuntimeError(f"Unrecognized xs layout at {rx}/{temperature}/xs")
            else:
                raise RuntimeError(f"Unrecognized xs layout at {rx}/{temperature}/xs")

    return E, xs, temperature

def endf_plot(
    Z: int,
    A: int,
    MT: int,
    h5_dir: PathLike,
    temperature: str = "294K",
    ax=None,
    label: str | None = None,
    show: bool = True,
):
    """
    Plot cross section for (Z,A,MT). Returns (E, xs, used_temperature, ax).
    If ax is provided, plot onto that axes (so you can overlay plots).
    """
    nuclide = za_to_nuclide(Z, A)
    h5_path = find_neutron_h5(h5_dir, Z, A)

    E, xs, used_T = read_xs_schema_ugrp(h5_path, nuclide, MT, temperature=temperature)

    if ax is None:
        fig, ax = plt.subplots()

    ax.loglog(E, xs, label=label or f"{nuclide} MT {MT} @ {used_T}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Cross section (barns)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()

    if show:
        plt.show()

    return E, xs, used_T, ax