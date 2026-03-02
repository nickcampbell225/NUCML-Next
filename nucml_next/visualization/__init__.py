"""
NUCML-Next Visualization Module
===============================

Comprehensive visualization tools for nuclear cross-section analysis.

Main Classes:
    CrossSectionFigure: Flexible figure class for cross-section plots
    NNDCSigmaFetcher: Fetch evaluated pointwise data from IAEA NDS
    IsotopePlotter: One-line plotting of ML predictions vs EXFOR data

Example -- low-level (full control):
    >>> from nucml_next.visualization import CrossSectionFigure
    >>>
    >>> fig = CrossSectionFigure(isotope='U-235', mt=18, title='U-235 Fission')
    >>> fig.add_exfor(exfor_df, label='EXFOR')
    >>> fig.add_endf()  # fetches from IAEA NDS (cached)
    >>> fig.add_model(energies, predictions, label='XGBoost')
    >>> fig.show()

Example -- high-level (one line per isotope):
    >>> from nucml_next.visualization import IsotopePlotter
    >>>
    >>> plotter = IsotopePlotter(
    ...     training_df=df_tier,
    ...     models={'Decision Tree': dt_model, 'XGBoost': xgb_model},
    ... )
    >>> plotter.plot(Z=92, A=233, MT=1)
    >>> plotter.plot(Z=17, A=35, MT=103, show_endf=True, save_path='cl35.png')
"""

from .cross_section_figure import CrossSectionFigure
from .endf_reader import NNDCSigmaFetcher
from .isotope_plotter import IsotopePlotter

__all__ = [
    'CrossSectionFigure',
    'NNDCSigmaFetcher',
    'IsotopePlotter',
]
