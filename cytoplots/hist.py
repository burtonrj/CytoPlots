#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module concerns plotting functions for one or two-dimensional 'flow plots' common to software such as FlowJo.
These plots support common transform methods in cytometry such as logicle (biexponential), log, hyper-log and
inverse hyperbolic arc-sine. These plotting functions have application in traditional gating and visualising
populations in low dimensional space.
Copyright 2020 Ross Burton
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import seaborn as sns
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from cytotools.transform import apply_transform, TRANSFORMERS
from .transforms import *

logger = logging.getLogger(__name__)


class PlotError(Exception):
    def __init__(self, message: Exception):
        logger.error(message)
        super().__init__(message)


def _auto_plot_kind(
    data: pd.DataFrame,
    y: Optional[str] = None,
) -> str:
    """
    Determine the best plotting method. If the number of observations is less than 1000, returns 'hist2d' otherwise
    returns 'scatter_kde'. If 'y' is None returns 'kde'.
    Parameters
    ----------
    data: Pandas.DataFrame
    y: str
    Returns
    -------
    str
    """
    if y:
        if data.shape[0] > 1000:
            return "hist2d"
        return "scatter_kde"
    return "kde"


def _hist2d_axis_limits(x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames for axis limits. DataFrames have the columns 'Min' and 'Max', and values
    are the min and max of the provided arrays
    Parameters
    ----------
    x: Numpy.Array
    y: Numpy.Array
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        X-axis limits, Y-axis limits
    """
    xlim = [np.min(x), np.max(x)]
    ylim = [np.min(y), np.max(y)]
    xlim = pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]})
    ylim = pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]})
    return xlim, ylim


def _hist2d_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: Optional[int],
    transform_x: Optional[str],
    transform_x_kwargs: Optional[Dict],
    transform_y: Optional[str],
    transform_y_kwargs: Optional[Dict],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    """
    Calculate bins and edges for 2D histogram
    Parameters
    ----------
    x: Numpy.Array
    y: Numpy.Array
    bins: int, optional
    transform_x: str, optional
    transform_x_kwargs: Dict, optional
    transform_y: str, optional
    transform_y_kwargs: Dict, optional
    Returns
    -------
    List[Numpy.Array]
    """
    nbins = bins or int(np.sqrt(x.shape[0]))
    if xlim is not None:
        if ylim is not None:
            xlim, ylim = (
                pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]}),
                pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]}),
            )
        else:
            xlim, ylim = (pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]}), _hist2d_axis_limits(x, y)[1])
    elif ylim is not None:
        xlim, ylim = (_hist2d_axis_limits(x, y)[0], pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]}))
    else:
        xlim, ylim = _hist2d_axis_limits(x, y)
    bins = []
    for lim, transform_method, transform_kwargs in zip(
        [xlim, ylim], [transform_x, transform_y], [transform_x_kwargs, transform_y_kwargs]
    ):
        if transform_method:
            transform_kwargs = transform_kwargs or {}
            lim, transformer = apply_transform(
                data=lim, features=["Min", "Max"], method=transform_method, return_transformer=True, **transform_kwargs
            )
            grid = pd.DataFrame({"x": np.linspace(lim["Min"].values[0], lim["Max"].values[0], nbins)})
            bins.append(transformer.inverse_scale(grid, features=["x"]).x.to_numpy())
        else:
            bins.append(np.linspace(lim["Min"].values[0], lim["Max"].values[0], nbins))
    return bins


def kde1d(
    data: pd.DataFrame,
    x: str,
    transform_method: Optional[str] = None,
    bw: Union[str, float] = "silverman",
    **transform_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-dimensional kernel density estimation
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    bw: Union[str, float], (default='silverman')
    transform_method: str, optional
    transform_kwargs: optional
        Additional keyword arguments passed to transform method
    Returns
    -------
    Tuple[Numpy.Array, Numpy.Array]
        The grid space and the density array
    """
    transformer = None
    if transform_method:
        data, transformer = apply_transform(
            data=data, features=[x], method=transform_method, return_transformer=True, **transform_kwargs
        )
    x_grid, y = FFTKDE(kernel="gaussian", bw=bw).fit(data[x].values).evaluate()
    data = pd.DataFrame({"x": x_grid, "y": y})
    if transform_method:
        data = transformer.inverse_scale(data=pd.DataFrame({"x": x_grid, "y": y}), features=["x"])
    return data["x"].values, data["y"].values


def hist2d(
    data: pd.DataFrame,
    x: str,
    y: str,
    transform_x: Optional[str],
    transform_y: Optional[str],
    transform_x_kwargs: Optional[Dict],
    transform_y_kwargs: Optional[Dict],
    ax: plt.Axes,
    bins: Optional[int],
    cmap: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """
    Plot two-dimensional histogram on axes
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Column to plot on x-axis
    y: str
        Column to plot on y-axis
    transform_x: str, optional
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_x_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    transform_y_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    ax: Matplotlib.Axes
        Axes to plot on
    bins: int, optional
        Number of bins to use, if not given defaults to square root of the number of observations
    cmap: str (default='jet')
    xlim: Tuple[float, float], optional
        Limit the x-axis between this range
    ylim: Tuple[float, float], optional
        Limit the y-axis between this range
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.hist2d
    Returns
    -------
    None
    """
    xbins, ybins = _hist2d_bins(
        x=data[x].values,
        y=data[y].values,
        bins=bins,
        transform_x=transform_x,
        transform_y=transform_y,
        transform_x_kwargs=transform_x_kwargs,
        transform_y_kwargs=transform_y_kwargs,
        xlim=xlim,
        ylim=ylim,
    )
    ax.hist2d(data[x].values, data[y].values, bins=[xbins, ybins], norm=LogNorm(), cmap=cmap, **kwargs)


def cyto_plot(
    data: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    kind: str = "auto",
    transform_x: Optional[str] = "asinh",
    transform_y: Optional[str] = "asinh",
    transform_x_kwargs: Optional[Dict] = None,
    transform_y_kwargs: Optional[Dict] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    bins: Optional[int] = None,
    cmap: str = "jet",
    autoscale: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Generate a generic 'flow plot', that is a one or two-dimensional plot identical to that generated by common
    cytometry software like FlowJo. These plots support common cytometry data transformations like logicle
    (biexponential), log, hyperlog, or hyperbolic arc-sine transformations, whilst translating values back
    to a linear scale on axis for improved interpretability.
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Column to plot on x-axis
    y: str, optional
        Column to plot on y-axis, will generate a one-dimensional KDE plot if not provided
    kind: str, (default='auto)
        Should be one of: 'hist2d', 'scatter', 'scatter_kde', 'kde' or 'auto'. If 'auto' then plot type is
        determined from data; If the number of observations is less than 1000, will use 'hist2d' otherwise
        kind is 'scatter_kde'. If data is one-dimensional (i.e. y is not provided), then will use 'kde'.
    transform_x: str, optional, (default='asinh')
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional, (default='asinh')
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_x_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    transform_y_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    xlim: Tuple[float, float], optional
        Limit the x-axis between this range
    ylim: Tuple[float, float], optional
        Limit the y-axis between this range
    ax: Matplotlib.Axes
        Axes to plot on
    bins: int, optional
        Number of bins to use, if not given defaults to square root of the number of observations
    cmap: str (default='jet')
        Colour palette to use for two-dimensional histogram
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    autoscale: bool (default=True)
        Allow matplotlib to autoscale the axis view to the data
    kwargs:
        Additional keyword arguments passed to plotting method:
            * seaborn.scatterplot for 'scatter' or 'scatter_kde'
            * matplotlib.Axes.hist2d for 'hist2d'
            * matplotlib.Axes.plot for 'kde'
    Returns
    -------
    Matplotlib.Axes
    """
    try:
        ax = ax or plt.subplots(figsize=figsize)[1]
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        kind = kind if kind != "auto" else _auto_plot_kind(data=data, y=y)
        kwargs = kwargs or {}
        if kind == "hist2d":
            assert y, "No y-axis variable provided"
            hist2d(
                data=data,
                x=x,
                y=y,
                transform_x=transform_x,
                transform_y=transform_y,
                transform_x_kwargs=transform_x_kwargs,
                transform_y_kwargs=transform_y_kwargs,
                ax=ax,
                bins=bins,
                cmap=cmap,
                xlim=xlim,
                ylim=ylim,
            )
        elif kind == "scatter":
            assert y, "No y-axis variable provided"
            kwargs["s"] = kwargs.get("s", 10)
            kwargs["edgecolor"] = kwargs.get("edgecolor", None)
            kwargs["linewidth"] = kwargs.get("linewidth", 0)
            sns.scatterplot(data=data, x=x, y=y, **kwargs)
        elif kind == "scatter_kde":
            assert y, "No y-axis variable provided"
            scatter_kwargs = kwargs.get("scatter_kwargs", {})
            kde_kwargs = kwargs.get("kde_kwargs", {})
            sns.kdeplot(data=data, x=x, y=y, **kde_kwargs)
            scatter_kwargs["s"] = scatter_kwargs.get("s", 10)
            scatter_kwargs["edgecolor"] = scatter_kwargs.get("edgecolor", None)
            scatter_kwargs["linewidth"] = scatter_kwargs.get("linewidth", 0)
            scatter_kwargs["color"] = scatter_kwargs.get("color", "black")
            sns.scatterplot(data=data, x=x, y=y, **scatter_kwargs)
        elif kind == "kde":
            if y:
                sns.kdeplot(data=data, x=x, y=y, **kwargs)
            else:
                bw = kwargs.pop("bw", "silverman")
                transform_x_kwargs = transform_x_kwargs or {}
                xx, pdf = kde1d(data=data, x=x, transform_method=transform_x, bw=bw, **transform_x_kwargs)
                ax.plot(xx, pdf, linewidth=kwargs.get("linewidth", 2), color=kwargs.get("color", "black"))
                ax.fill_between(xx, pdf, color=kwargs.get("fill", "#8A8A8A"), alpha=kwargs.get("alpha", 0.5))
        else:
            raise KeyError("Invalid value for 'kind', must be one of: 'auto', 'hist2d', 'scatter_kde', or 'kde'.")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if autoscale:
            ax.autoscale(enable=True)
        else:
            if xlim is None:
                xlim = (data[x].quantile(q=0.001), data[x].quantile(q=0.999))
            if ylim is None:
                ylim = (data[y].quantile(q=0.001), data[y].quantile(q=0.999))
            ax.set_xlim(*xlim)
            if y is not None:
                ax.set_ylim(*ylim)
        if transform_x:
            transform_x_kwargs = transform_x_kwargs or {}
            ax.set_xscale(transform_x, **transform_x_kwargs)
        if y and transform_y:
            transform_y_kwargs = transform_y_kwargs or {}
            ax.set_yscale(transform_y, **transform_y_kwargs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        return ax
    except AssertionError as e:
        raise PlotError(e)
    except KeyError as e:
        raise PlotError(e)


def overlay(
    x: str,
    y: str,
    background_data: pd.DataFrame,
    overlay_data: Dict[str, pd.DataFrame],
    background_colour: str = "#323232",
    transform_x: Optional[str] = "asinh",
    transform_y: Optional[str] = "asinh",
    legend_kwargs: Optional[Dict] = None,
    **plot_kwargs,
) -> plt.Axes:
    """
    Generates a two-dimensional scatterplot as background data and overlays a histogram, KDE, or scatterplot
    in the foreground. Can be useful for comparing populations and is commonly referred to as 'back-gating' in
    traditional cytometry analysis.
    Parameters
    ----------
    x: str
        Column to plot on x-axis, must be common to both 'background_data' and 'overlay_data'
    y: str
        Column to plot on y-axis, must be common to both 'background_data' and 'overlay_data'
    background_data: Pandas.DataFrame
        Data to plot in the background
    overlay_data: Pandas.DataFrame
        Data to plot in the foreground
    background_colour: str (default='#323232')
        How to colour the background data points (defaults to a grey-ish black)
    transform_x: str, optional, (default='asinh')
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional, (default='asinh')
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    legend_kwargs: optional
        Additional keyword arguments passed to legend
    plot_kwargs: optional
        Additional keyword arguments passed to cytopy.plotting.cyto.cyto_plot for foreground plot
    Returns
    -------
    Matplotlib.Axes
    """
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    if len(overlay_data) > len(colours):
        raise ValueError(f"Maximum of {len(colours)} overlaid populations.")
    ax = cyto_plot(
        data=background_data,
        x=x,
        y=y,
        transform_x=transform_x,
        transform_y=transform_y,
        color=background_colour,
        kind="scatter",
        **plot_kwargs,
    )
    legend_kwargs = legend_kwargs or {}
    for label, df in overlay_data.items():
        cyto_plot(data=df, x=x, y=y, transform_x=transform_x, transform_y=transform_y, ax=ax, **plot_kwargs)
    _default_legend(ax=ax, **legend_kwargs)
    return ax


def _default_legend(ax: plt.Axes, **legend_kwargs):
    """
    Default setting for plot legend
    Parameters
    ----------
    ax: Matplotlib.Axes
    legend_kwargs: optional
        User defined legend keyword arguments
    Returns
    -------
    None
    """
    legend_kwargs = legend_kwargs or {}
    anchor = legend_kwargs.get("bbox_to_anchor", (1.1, 0.95))
    loc = legend_kwargs.get("loc", 2)
    ncol = legend_kwargs.get("ncol", 3)
    fancy = legend_kwargs.get("fancybox", True)
    shadow = legend_kwargs.get("shadow", False)
    ax.legend(loc=loc, bbox_to_anchor=anchor, ncol=ncol, fancybox=fancy, shadow=shadow)
