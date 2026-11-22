import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as qualitative_colors
from plotly.colors import sample_colorscale


import src.utils.common as ucommon


def _stack_runs_with_nan_padding(arrays:list[np.ndarray])->np.ndarray:
    """Stack 1D arrays of possibly different lengths (eg. different number of trials across runs) into a 2D array (n_runs, max_len), padding the shorter ones with NaN."""
    max_len = max(len(a) for a in arrays)
    stacked = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        stacked[i, :len(a)] = a
    return stacked


def _hex_to_rgba(hex_color:str, alpha:float)->str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def output_figure(fig:go.Figure, file_path:str|None=None, append:bool=False)->bool:
    """
    Shows fig if file_path is None, else saves it as a standalone .html file at file_path.

    append : bool, default False
        If True and file_path already exists, fig is added at the end of that html instead of
        overwriting it, so several figures (eg. of the same study) can be grouped in one file.

    Returns True once the figure has been saved, so a function producing several figures can pass
    it as `append` for the next ones and keep them all in the same file.
    """
    if file_path is None:
        fig.show()
        return append

    if append and os.path.exists(file_path):
        with open(file_path) as f:
            html = f.read()
        div = fig.to_html(full_html=False, include_plotlyjs=False)
        i_end = html.rfind("</body>")
        html = html[:i_end] + div + "\n" + html[i_end:]
        with open(file_path, "w") as f:
            f.write(html)
    else:
        ucommon.recursive_mkdir(os.path.dirname(os.path.abspath(file_path)))
        fig.write_html(file_path, include_plotlyjs=True)
    return True


def add_sequence_group_markers(fig, sequence_groups, x=None, style="lines",
                               colors=("rgba(0,0,0,0)", "rgba(120,120,120,0.12)"),
                               show_labels=False, row=None, col=None):
    """Mark changes of sequence group along the x axis of one subplot (or the whole figure).

    sequence_groups: one group id per sample, in x order.
    x: x value of each sample (default: 0, 1, 2, ...).
    style: "lines" (dotted vertical line at each transition) or "background" (alternating bands).
    """
    groups = np.asarray(sequence_groups)
    x = np.arange(len(groups), dtype=float) if x is None else np.asarray(x, dtype=float)
    if len(x) != len(groups):
        raise ValueError("sequence_groups must have one entry per sample")
    if len(groups) == 0:
        return fig

    # Boundaries sit halfway between the last sample of a group and the first of the next
    change = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    boundaries = (x[change - 1] + x[change]) / 2

    if style == "lines":
        for b, g in zip(boundaries, groups[change]):
            # Label each line with the group that starts there (same "new text" caveat as below)
            label = dict(annotation_text=str(g), annotation_position="top right") if show_labels else {}
            fig.add_vline(x=b, line_dash="dot", line_color="gray", line_width=1,
                          row=row, col=col, **label)
        if show_labels:
            # The first group has no line before it: label it at the start of the axis
            fig.add_annotation(x=x[0], y=1, xref="x", yref="y domain", text=str(groups[0]),
                               showarrow=False, xanchor="left", yanchor="bottom",
                               row=row, col=col)
    elif style == "background":
        half_step = (x[1] - x[0]) / 2 if len(x) > 1 else 0.5
        starts = np.concatenate([[x[0] - half_step], boundaries])
        stops = np.concatenate([boundaries, [x[-1] + half_step]])
        run_ids = groups[np.concatenate([[0], change])]
        for k, (x0, x1, g) in enumerate(zip(starts, stops, run_ids)):
            # Only pass annotation args when labelling: plotly shows "new text" for an
            # annotation_position without annotation_text
            label = dict(annotation_text=str(g), annotation_position="top left") if show_labels else {}
            fig.add_vrect(x0=x0, x1=x1, fillcolor=colors[k % 2], line_width=0, layer="below",
                          row=row, col=col, **label)
    else:
        raise ValueError('style must be "lines" or "background"')
    return fig


def plot_values_across_trials(raw_data, y_names:list[str], n_bins:int=20, groups:list|None=None,
                              n_boot:int=2000, ci:float=0.95, seed:int=0,
                              sequence_groups:list|str|None=None, sequence_style:str="lines",
                              sequence_labels:bool=False,
                              file_path:str|None=None, append:bool=False):
    """
    One row of subplots per name in y_names:
    - left (or only, if n_bins==0): raw_data[y] against raw_data["trial"]
    - right: distribution of raw_data[y] across n_bins bins (how many trials fall in each bin)

    raw_data : pd.DataFrame or list[pd.DataFrame]
        One exploration trace, or a list of traces (one DataFrame per trace).

    groups : list, optional
        Only used when raw_data is a list: group label for each trace (same length/order as
        raw_data). Traces sharing a group are aggregated into one mean-per-trial curve with a
        bootstrap (percentile, no normality assumption) ci-level CI band, sharing one colour,
        and their values add up in the distribution plot. If None, every trace is its own group
        (one plain colour per trace, no aggregation/band).

    n_boot : int, default 2000
        Bootstrap resamples for the CI band. Only used for groups with more than one trace.

    ci : float, default 0.95
        Confidence level for the band, eg 0.95 for a 95% band.

    sequence_groups : list or str, optional
        Marks on the trial axis where a new sequence group starts. Either a list with one
        group id per trial (in trial order), or the name of a column of raw_data holding it
        (read from the longest trace, using its "trial" column as x).

    sequence_style : {"lines", "background"}, default "lines"
        "lines": dotted vertical line at each transition. "background": alternating bands.

    sequence_labels : bool, default False
        Print each group id at the top of the plot (first row only): at the start of its band
        with "background", next to the line where it starts with "lines".
    """
    raw_data_list = raw_data if isinstance(raw_data, list) else [raw_data]

    ungrouped = groups is None
    if ungrouped:
        groups = list(range(len(raw_data_list)))

    group_ids = list(dict.fromkeys(groups))
    group_traces = {g: [df for df,grp in zip(raw_data_list,groups) if grp == g] for g in group_ids}

    is_single = len(raw_data_list) == 1
    palette = qualitative_colors.Plotly
    group_colors = {group_ids[0]: "steelblue"} if is_single else {g: palette[i % len(palette)] for i,g in enumerate(group_ids)}
    group_names = {g: (f"trace {g}" if ungrouped else str(g)) for g in group_ids}
    show_legend_entries = len(group_ids) > 1

    show_distribution = n_bins != 0
    n_cols = 2 if show_distribution else 1
    kwargs = dict(column_widths=[0.75,0.25], horizontal_spacing=0.03) if show_distribution else {}
    fig = make_subplots(rows=len(y_names), cols=n_cols, **kwargs)

    rng = np.random.default_rng(seed)

    for i, y_name in enumerate(y_names):
        for g in group_ids:
            dfs, color, name = group_traces[g], group_colors[g], group_names[g]

            if len(dfs) == 1:
                df = dfs[0]
                fig.add_trace(
                    go.Scatter(
                        x = df["trial"], y = df[y_name], mode = "markers+lines",
                        marker_color = color, line_color = color,
                        name = name, legendgroup = str(g), showlegend = show_legend_entries and (i == 0)
                    ),
                    row=i+1, col=1
                )
            else:
                stacked = _stack_runs_with_nan_padding([df[y_name].to_numpy(dtype=float) for df in dfs])
                mean_obs = np.nanmean(stacked, axis=0)

                n_traces = stacked.shape[0]
                boot_idx = rng.integers(0, n_traces, size=(n_boot, n_traces))
                boot_means = np.nanmean(stacked[boot_idx], axis=1)

                alpha = 1 - ci
                lower = np.nanquantile(boot_means, alpha/2, axis=0)
                upper = np.nanquantile(boot_means, 1 - alpha/2, axis=0)

                x = np.arange(stacked.shape[1])
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                        fill="toself", fillcolor=_hex_to_rgba(color, 0.2),
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", legendgroup=str(g), showlegend=False
                    ),
                    row=i+1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=x, y=mean_obs, mode="lines", line=dict(color=color, width=2),
                        name=name, legendgroup=str(g), showlegend=show_legend_entries and (i == 0)
                    ),
                    row=i+1, col=1
                )
        fig.update_yaxes(title_text=y_name, row=i+1, col=1)

        if show_distribution:
            for g in group_ids:
                all_values = pd.concat([df[y_name] for df in group_traces[g]], ignore_index=True)
                fig.add_trace(
                    go.Histogram(
                        y = all_values, nbinsy = n_bins, marker_color = group_colors[g],
                        opacity = 0.6 if show_legend_entries else 1.0,
                        showlegend = False, legendgroup = str(g)
                    ),
                    row=i+1, col=2
                )
            left_axis_ref = "y" if i == 0 else f"y{2*i+1}"
            fig.update_yaxes(matches=left_axis_ref, showticklabels=False, row=i+1, col=2)
            fig.update_xaxes(title_text="count", row=i+1, col=2)

    if sequence_groups is not None:
        if isinstance(sequence_groups, str):
            ref_df = max(raw_data_list, key=len)
            seq = ref_df[sequence_groups].to_numpy()
            x_seq = ref_df["trial"].to_numpy(dtype=float)
        else:
            seq, x_seq = sequence_groups, None
        for i in range(len(y_names)):
            add_sequence_group_markers(fig, seq, x=x_seq, style=sequence_style,
                                       show_labels=sequence_labels and i == 0,
                                       row=i+1, col=1)

    fig.update_xaxes(title_text="trial", row=len(y_names), col=1)
    fig.update_layout(height=300*len(y_names), bargap=0.05, barmode="overlay", showlegend=show_legend_entries)
    append = output_figure(fig, file_path, append)



def plot_2D_actions(actions, colors, xbounds=None, ybounds=None, colorbar_title="value", file_path=None, append=False):
    """
    Scatter plot of 2D actions.

    Parameters
    ----------
    actions : array-like, shape (n, 2)
    colors : array-like, shape (n,)
        Value per action (e.g. reward). Numeric values use a Viridis colour scale;
        otherwise they are passed to Plotly as colours.
    xbounds, ybounds : [float, float], optional
        Axis ranges.
    colorbar_title : str, default "value"
    file_path, append
        See :func:`output_figure`.
    """
    actions = np.asarray(actions)
    numeric = np.issubdtype(np.asarray(colors).dtype, np.number)

    marker = dict(size=8, color=colors, opacity=0.8)
    if numeric:
        marker.update(colorscale="Viridis", showscale=True,colorbar=dict(title=colorbar_title))

    fig = go.Figure(go.Scatter(
        x=actions[:, 0],
        y=actions[:, 1],
        mode="markers",
        marker=marker,
        customdata=[[i, str(c)] for i, c in enumerate(colors)],
        hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<br>id: %{customdata[0]}<br>"+colorbar_title+": %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="action[0]", range=xbounds, constrain="domain"),
        yaxis=dict(title="action[1]", range=ybounds, constrain="domain",scaleanchor="x"),
        template="plotly_white",
    )
    append = output_figure(fig, file_path, append)


def plot_positions_to_actions_2D(positions, actions, colors, groups=None, xbounds=None, ybounds=None,
                                 colorbar_title="value", arrow_scale=None,
                                 colorscale="Viridis", file_path=None, append=False):
    """
    Plot each action as an arrow starting from the position it was taken at.

    Parameters
    ----------
    positions : array-like, shape (n, 2)
        Position when each action was chosen.
    actions : array-like, shape (n, 2)
    colors : array-like, shape (n,)
        Value per action (e.g. reward), used for markers and arrows.
    groups : array-like, shape (n,), optional
        Group id per sample (e.g. episode). Consecutive positions of the same group
        are joined by a dotted line with an arrowhead.
    xbounds, ybounds : [float, float], optional
        Axis ranges.
    colorbar_title : str, default "value"
    arrow_scale : float, optional
        Arrow length per unit of action. By default the longest arrow is 10% of the
        positions' extent.
    colorscale : str, default "Viridis"
    file_path, append
        See :func:`output_figure`.
    """
    positions = np.asarray(positions, dtype=float)
    actions = np.asarray(actions, dtype=float)
    colors_arr = np.asarray(colors)
    numeric = np.issubdtype(colors_arr.dtype, np.number)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(positions):
            raise ValueError("groups must have one entry per position")

    # Arrow length: by default, longest arrow = 10% of the positions' extent
    if arrow_scale is None:
        extent = np.ptp(positions, axis=0).max() if len(positions) > 1 else 1.0
        max_norm = np.linalg.norm(actions, axis=1).max()
        arrow_scale = 0.1 * (extent if extent > 0 else 1.0) / (max_norm if max_norm > 0 else 1.0)
    ends = positions + arrow_scale * actions

    # One color per arrow, matching the markers
    marker = dict(size=8, color=colors, opacity=0.8)
    if numeric:
        cmin, cmax = float(colors_arr.min()), float(colors_arr.max())
        marker.update(colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=True,
                      colorbar=dict(title=colorbar_title))
        norm = (colors_arr - cmin) / (cmax - cmin) if cmax > cmin else np.full(len(colors_arr), 0.5)
        arrow_colors = sample_colorscale(colorscale, norm.tolist())
    else:
        arrow_colors = list(colors_arr)

    fig = go.Figure()

    # Dotted lines between consecutive samples of each group, with a midpoint arrowhead
    if groups is not None:
        line_x, line_y = [], []
        mid_x, mid_y, mid_angle, mid_text = [], [], [], []
        for g in dict.fromkeys(groups.tolist()):          # groups in order of first appearance
            idx = np.flatnonzero(groups == g)
            for i, j in zip(idx[:-1], idx[1:]):
                p, q = positions[i], positions[j]
                line_x += [p[0], q[0], None]
                line_y += [p[1], q[1], None]
                d = q - p
                if not np.any(d):
                    continue                              # same position: no direction to show
                mid = (p + q) / 2
                mid_x.append(mid[0])
                mid_y.append(mid[1])
                # plotly marker angle: degrees clockwise from "up"
                mid_angle.append(90.0 - np.degrees(np.arctan2(d[1], d[0])))
                mid_text.append(f"group {g}: {i} → {j}")

        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                 line=dict(color="gray", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers",
                                 marker=dict(symbol="arrow", size=9, angle=mid_angle,
                                             color="gray", line=dict(width=0)),
                                 text=mid_text, hovertemplate="%{text}<extra></extra>",
                                 showlegend=False))

    # Invisible arrow tips, so autorange includes the full arrows
    fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode="markers",
                             marker=dict(opacity=0), hoverinfo="skip", showlegend=False))

    customdata = [[i, p[0], p[1], a[0], a[1], str(c)]
                  for i, (p, a, c) in enumerate(zip(positions, actions, colors))]
    hovertemplate = (
        "id: %{customdata[0]}<br>"
        "position: (%{customdata[1]:.3f}, %{customdata[2]:.3f})<br>"
        "action: (%{customdata[3]:.3f}, %{customdata[4]:.3f})<br>"
        + colorbar_title + ": %{customdata[5]}"
    )
    if groups is not None:
        for row, g in zip(customdata, groups.tolist()):
            row.append(g)
        hovertemplate += "<br>group: %{customdata[6]}"

    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode="markers",
        marker=marker,
        customdata=customdata,
        hovertemplate=hovertemplate + "<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        annotations=[
            dict(x=e[0], y=e[1], ax=p[0], ay=p[1],
                 xref="x", yref="y", axref="x", ayref="y",
                 showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                 arrowcolor=col, opacity=0.8, text="")
            for p, e, col in zip(positions, ends, arrow_colors)
        ],
        xaxis=dict(title="position[0]", range=xbounds, constrain="domain"),
        yaxis=dict(title="position[1]", range=ybounds, constrain="domain", scaleanchor="x"),
        template="plotly_white",
    )
    # return output_figure(fig, file_path, append)
    append = output_figure(fig, file_path, append)