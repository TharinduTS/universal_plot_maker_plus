# universal_plot_maker_plus

# 1) Introduction

universal_plot_maker_plus.py is a flexible and general‑purpose interactive plotting tool for exploring large tabular datasets (TSV/CSV). It produces a self‑contained HTML file with dynamic controls that allow end‑users to switch axes, filter data, search, sort, zoom, and export selected subsets — all without requiring Python or Plotly installed.
This tool is especially useful for high‑dimensional biological datasets (gene‑level, cell‑type‑level, enrichment tables, marker tables, scoring matrices, etc.) where the user needs:

Dynamic X/Y axis switching
Multiple Y metrics (raw/log/penalized/etc.)
Drop‑down filters (cell type / group / class / cluster…)
Multiple search fields
Sorting by any column
Duplicate handling
Click‑to‑inspect rows
Client‑side TSV export of selected points
Optional embedding of Plotly.js for offline sharing

You control the initial state of the figure entirely through the CLI, and the resulting HTML contains a fully reactive UI that lets end‑users interact with the dataset in real time.

# 2)Script
Following is the python script 

universal_plot_maker_plus.py
```py

#!/usr/bin/env python3
"""
Universal interactive plot maker (generalized)
- Reads a TSV/CSV into pandas
- Exports a standalone HTML with Plotly.js and a functional UI:
  * Axis selection (X/Y from CLI-provided candidates; type-aware)
  * Plot type: bar | scatter | line
  * Group coloring by a chosen column
  * Multiple dropdown filters with CLI defaults
  * Multiple search boxes with CLI defaults
  * Primary + secondary sort (asc/desc)
  * Initial zoom (#bars)
  * Click for details (columns chosen via CLI)
  * TSV export of selected points (lasso/box)
- Duplicate handling: overlay | stack | max | mean | median | first | sum

Example (with your dataframe columns):
    python universal_plot_maker_plus.py \
      --file gene_celltype_table.tsv \
      --out gene_plot.html \
      --plot-type bar \
      --title "Cell type enrichment" \
      --x-choices "Gene name|Gene" \
      --y-choices "avg_nCPM|Enrichment score|log2_enrichment|Enrichment score (tau penalized)|log2_enrichment_penalized" \
      --default-x "Gene name" \
      --default-y "log2_enrichment" \
      --color-col "Cell type group" \
      --filter-cols "Cell type|Cell type group|Cell type class" \
      --filter-defaults "Cell type group=secretory cells" \
      --search-cols "Gene name|Gene" \
      --search-defaults "" \
      --details "Gene|Gene name|Cell type|avg_nCPM|Enrichment score|log2_enrichment|specificity_tau|Cell type group|Cell type class" \
      --initial-zoom 100 \
      --sort-primary "Enrichment score" \
      --sort-primary-order desc \
      --sort-secondary "avg_nCPM" \
      --sort-secondary-order desc \
      --dup-policy mean \
      --show-legend \
      --self-contained

"""

import json
import re
import sys
import argparse
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
from plotly.graph_objects import Figure, Bar, Scatter

# ------------------------------
# Helpers: I/O & type checks
# ------------------------------

def _infer_sep(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    plu = path.lower()
    if plu.endswith(".tsv") or plu.endswith(".tab"):
        return "\t"
    if plu.endswith(".csv"):
        return ","
    return None

def load_table(path: str, sep: Optional[str]) -> pd.DataFrame:
    if sep is None:
        sep = _infer_sep(path)
    df = pd.read_csv(path, sep=sep if sep else None, engine="python")
    # Normalize column names: strip spaces/newlines
    df.columns = [str(c).strip() for c in df.columns]
    return df

def is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s)
    except Exception:
        return False

def coerce_numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s
    # Loose coercion (commas -> dots), drop "NA"/blanks
    s2 = (
        s.astype(str)
         .str.strip()
         .replace({"NA": None, "N/A": None, "null": None, "None": None, "": None})
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")

def safe_str(s) -> str:
    try:
        return str(s)
    except Exception:
        return ""

# ------------------------------
# Duplicate handling
# ------------------------------

def dedupe_rows(
    df: pd.DataFrame,
    key_cols: List[str],
    value_col: Optional[str],
    policy: str
) -> pd.DataFrame:
    """
    Pre-collapse duplicates BEFORE plotting if policy in {max, mean, median, first, sum}.
    If policy == 'overlay': keep all duplicates (single trace, overlap).
    If policy == 'stack': keep duplicates; client will render multiple traces stacked.
    """
    policy = policy.lower()
    if not key_cols or any(k not in df.columns for k in key_cols):
        # Nothing we can do; return as-is unless policy requires collapsing
        return df.copy()

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return df.copy()

    if policy in ("overlay", "stack"):
        # Plot duplicates as-is, handled in figure building
        return df.copy()

    if value_col is None or value_col not in df.columns:
        raise ValueError(f"--dup-policy '{policy}' requires --default-y (numeric) present in the data.")

    # Numeric coercion only for aggregation
    v = coerce_numeric_column(df, value_col)
    tmp = df.copy()
    tmp["__val__"] = v

    agg_map = {
        "max": "max",
        "mean": "mean",
        "median": "median",
        "first": "first",
        "sum": "sum",
    }
    if policy not in agg_map:
        raise ValueError(f"Unsupported --dup-policy '{policy}'. Use overlay|stack|max|mean|median|first|sum")

    # Aggregate by key
    gb = tmp.groupby(key_cols, dropna=False)
    agg_df = gb.agg({"__val__": agg_map[policy]}).reset_index()

    # Reattach the first row's other columns (for details)
    first_rows = gb.nth(0).reset_index()
    # Drop helper col in first_rows if present
    if "__val__" in first_rows.columns:
        first_rows = first_rows.drop(columns="__val__")

    merged = pd.merge(agg_df, first_rows, on=key_cols, how="left")
    # Replace original value_col with aggregated value for numeric
    merged[value_col] = merged["__val__"]
    merged = merged.drop(columns="__val__")
    return merged

# ------------------------------
# Figure building
# ------------------------------


def build_figure_payload(
    df: pd.DataFrame,
    plot_type: str,
    x_col: str,
    y_col: Optional[str],
    color_col: Optional[str],
    details_cols: List[str],
    top_n: Optional[int],
    sort_primary: Optional[str],
    sort_primary_order: str,
    sort_secondary: Optional[str],
    sort_secondary_order: str,
    initial_zoom: Optional[int],
    title: Optional[str],
    show_legend: bool,
    dup_policy: str,
) -> Tuple[Figure, Dict]:
    """
    Build figure + payload dictionary for client-side UI.
    """

    plot_type = plot_type.lower()
    if plot_type not in ("bar", "scatter", "line"):
        raise ValueError("Unsupported --plot-type. Use bar|scatter|line.")

    if x_col not in df.columns:
        raise ValueError(f"X column '{x_col}' not in data.")

    if plot_type == "bar":
        if y_col is None or y_col not in df.columns:
            raise ValueError("Bar plot requires a numeric Y column via --default-y.")
        y_s = coerce_numeric_column(df, y_col)
        df = df.copy()
        df[y_col] = y_s
    else:
        if y_col is None or y_col not in df.columns:
            raise ValueError(f"{plot_type} plot requires numeric --default-y.")
        x_s = coerce_numeric_column(df, x_col)
        y_s = coerce_numeric_column(df, y_col)
        df = df.copy()
        df[x_col] = x_s
        df[y_col] = y_s

    # ---- Sorting ----
    def _sort_df(dfin: pd.DataFrame) -> pd.DataFrame:
        df2 = dfin.copy()
        cols = []
        asc_flags = []
        if sort_primary and sort_primary in df2.columns:
            cols.append(sort_primary)
            asc_flags.append(sort_primary_order.lower() == "asc")
        if sort_secondary and sort_secondary in df2.columns:
            cols.append(sort_secondary)
            asc_flags.append(sort_secondary_order.lower() == "asc")
        if not cols:
            return df2
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df2[c]):
                df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.sort_values(by=cols, ascending=asc_flags, kind="mergesort")
        return df2

    df = _sort_df(df)

    # ---- Colors ----
    if color_col and color_col in df.columns:
        cats = sorted(df[color_col].astype(str).unique())      
        from plotly.express import colors
        palette = (
            colors.qualitative.Dark24 +
            colors.qualitative.Light24 +
            colors.qualitative.Set3
        )
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
        bar_colors = df[color_col].astype(str).map(lambda c: color_map.get(c, "#636EFA")).tolist()
    else:
        color_map = {}
        bar_colors = ["#636EFA"] * len(df)

    # ---- Dedupe (collapse where applicable) ----
    if dup_policy.lower() in ("max", "mean", "median", "first", "sum"):
        key_cols = [x_col] + ([color_col] if color_col else [])
        df = dedupe_rows(df, key_cols=key_cols, value_col=y_col, policy=dup_policy)

    # ---- Details (customdata) ALWAYS defined as list-of-lists ----
    detail_cols = [c for c in details_cols if c in df.columns]
    if detail_cols:
        customdata = df[detail_cols].to_numpy().tolist()
    else:
        # Ensure downstream code can index safely
        customdata = [[""] * 0 for _ in range(len(df))]

    fig = Figure()

    if plot_type == "bar":
        categories = df[x_col].astype(str).tolist()
        ticktext = categories[:]
        vals = df[y_col].astype(float).tolist()

        if dup_policy.lower() == "stack":
            from collections import defaultdict

            cat_map_vals = defaultdict(list)
            cat_map_colors = defaultdict(list)
            cat_map_cd = defaultdict(list)

            # ✅ Guard: customdata is always a list; fall back to [] per-row
            for i, cat in enumerate(categories):
                cat_map_vals[cat].append(vals[i])
                cat_map_colors[cat].append(bar_colors[i])
                cd_row = customdata[i] if (customdata and i < len(customdata)) else []
                cat_map_cd[cat].append(cd_row)

            max_dups = max((len(vs) for vs in cat_map_vals.values()), default=0)

            for j in range(max_dups):
                xL, yL, colL, cdL = [], [], [], []
                for cat in categories:
                    arr_vals = cat_map_vals.get(cat, [])
                    arr_cols = cat_map_colors.get(cat, [])
                    arr_cds  = cat_map_cd.get(cat, [])
                    if j < len(arr_vals):
                        xL.append(cat); yL.append(arr_vals[j])
                        colL.append(arr_cols[j]); cdL.append(arr_cds[j])
                    else:
                        xL.append(cat); yL.append(0.0)
                        colL.append("#00000000"); cdL.append([])
                fig.add_trace(Bar(
                    x=xL, y=yL, marker=dict(color=colL),
                    customdata=cdL,
                    hovertemplate=_make_hover_template(detail_cols, orientation="v"),
                    name=f"layer {j+1}"
                ))
            fig.update_layout(barmode="stack")

        else:
            fig.add_trace(Bar(
                x=categories,
                y=vals,
                marker=dict(color=bar_colors),
                customdata=customdata,
                hovertemplate=_make_hover_template(detail_cols, orientation="v"),
            ))
            fig.update_layout(barmode="overlay")

        fig.update_layout(
            title=title or f"{y_col} by {x_col}",
            template="plotly_white",
            hovermode="closest",
            showlegend=bool(show_legend),
            dragmode="select",
            margin=dict(l=80, r=40, t=70, b=130),
            bargap=0.2,
        )
        fig.update_xaxes(
            title=dict(text=x_col),
            tickangle=-45,
            categoryorder="array",
            categoryarray=categories,
            tickmode="array",
            tickvals=categories,
            ticktext=ticktext,
            automargin=True,
            title_standoff=10,
        )
        fig.update_yaxes(
            title=dict(text=y_col),
            automargin=True,
            title_standoff=10,
        )

        if initial_zoom is not None:
            n = max(1, min(int(initial_zoom), len(categories)))
            fig.update_xaxes(range=[-0.5, n - 0.5])

    else:
        x = df[x_col].astype(float).tolist()
        y = df[y_col].astype(float).tolist()
        fig.add_trace(Scatter(
            x=x, y=y,
            mode="markers" if plot_type == "scatter" else "lines+markers",
            marker=dict(color=bar_colors),
            customdata=customdata,
            hovertemplate=_make_hover_template(detail_cols, orientation="v"),
        ))
        fig.update_layout(
            title=title or f"{y_col} vs {x_col}",
            template="plotly_white",
            hovermode="closest",
            showlegend=bool(show_legend),
            dragmode="select",
            margin=dict(l=80, r=40, t=70, b=80),
        )
        fig.update_xaxes(title=dict(text=x_col), automargin=True, title_standoff=10)
        fig.update_yaxes(title=dict(text=y_col), automargin=True, title_standoff=10)

    payload = {
        "rows": df.to_dict(orient="records"),
        "detail_cols": detail_cols,
        "color_col": color_col,
        "color_map": color_map,
        "plot_type": plot_type,
        "x_col": x_col,
        "y_col": y_col,
        "title": title or "",
        "show_legend": bool(show_legend),
        "dup_policy": dup_policy.lower(),
        "__x__": x_col,
        "__y__": y_col,
    }
    return fig, payload



def _make_hover_template(detail_cols: List[str], orientation: str = "v") -> str:
    """
    Build hover template referencing Plotly's customdata array correctly.
    """
    lines = []
    if orientation == "v":
        lines.append("**%{x}**")
        lines.append("Value: %{y:.4g}")
    else:
        lines.append("**%{y}**")
        lines.append("Value: %{x:.4g}")

    # Proper Plotly syntax for customdata
    for i, c in enumerate(detail_cols):
        lines.append(f"{c}: %{{customdata[{i}]}}")


    return "<br>".join(lines) + "<extra></extra>"

# ------------------------------
# HTML saving + client UI
# ------------------------------

DETAILS_UI = r"""
<div id="controls" style="margin: 0 0 12px 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
  <div style="display:flex; flex-wrap:wrap; gap:10px;">
    <div>
      <label><strong>Plot type</strong></label><br>
      <select id="plotTypeSelect" aria-label="Plot type">
        <option value="bar">Bar</option>
        <option value="scatter">Scatter</option>
        <option value="line">Line</option>
      </select>
    </div>
    <div>
      <label><strong>Color by</strong></label><br>
      <select id="colorBySelect" aria-label="Color by"></select>
    </div>
    <div>
      <label><strong>X</strong></label><br>
      <select id="xSelect" aria-label="X column"></select>
    </div>
    <div>
      <label><strong>Y</strong></label><br>
      <select id="ySelect" aria-label="Y column"></select>
    </div>
    <div>
      <label><strong>Bars to show</strong></label><br>
      <input id="barsCount" type="number" value="100" min="1" step="1" style="width:80px;">
    </div>
    <div>
      <label><strong>Duplicate policy</strong></label><br>
      <select id="dupPolicySelect" aria-label="Duplicate policy">
        <option value="overlay">overlay</option>
        <option value="stack">stack</option>
        <option value="max">max</option>
        <option value="mean">mean</option>
        <option value="median">median</option>
        <option value="first">first</option>
        <option value="sum">sum</option>
      </select>
    </div>
    <div>
      <label><strong>Primary sort</strong></label><br>
      <select id="sortPrimary"></select>
      <select id="sortPrimaryOrder">
        <option value="asc">asc</option>
        <option value="desc" selected>desc</option>
      </select>
    </div>
    <div>
      <label><strong>Secondary sort</strong></label><br>
      <select id="sortSecondary"></select>
      <select id="sortSecondaryOrder">
        <option value="asc">asc</option>
        <option value="desc" selected>desc</option>
      </select>
    </div>
  </div>

  <hr style="margin:8px 0;">

  <div id="filterRow" style="display:flex; flex-wrap:wrap; gap:10px;"></div>

  <div id="searchRow" style="margin-top:8px; display:flex; flex-wrap:wrap; gap:10px;"></div>

  <div style="margin-top:8px; display:flex; gap:8px;">
    <button id="resetBtn" type="button" aria-label="Reset filters">Reset</button>
    <button id="exportBtn" type="button" aria-label="Export TSV">Export TSV</button>
  </div>
</div>

<div id="rowDetails" style="font-size: 13px; color: #333;">
  Click a point/bar to see details here. Use lasso/box-select to export only selected.
</div>

<script>
(function() {
  try {
    var NL = String.fromCharCode(10);
    var TAB = String.fromCharCode(9);
    var payloadEl = document.getElementById('__payload__');
    var P = payloadEl ? JSON.parse(payloadEl.textContent || '{}') : null;
    if (!P) return;

    var plotEl = document.querySelector('div.js-plotly-plot');
    if (!plotEl) return;

    // UI config pulled from payload "ui_cfg"
    var UI = P.ui_cfg || {};
    var xChoices = UI.x_choices || [];
    var yChoices = UI.y_choices || [];
    var filterCols = UI.filter_cols || [];
    var filterDefaults = UI.filter_defaults || {}; // col -> value
    var searchCols = UI.search_cols || [];
    var searchDefaults = UI.search_defaults || {}; // col -> term
    var sortChoices = UI.sort_choices || []; // allowed sort columns

    var dupPolicy = P.dup_policy || "overlay";
    var dupPolicySelect = document.getElementById('dupPolicySelect');
    dupPolicySelect.value = dupPolicy;

    var firstRender = true;
    var plotTypeSelect = document.getElementById('plotTypeSelect');
    var xSelect = document.getElementById('xSelect');
    var ySelect = document.getElementById('ySelect');
    var barsCount = document.getElementById('barsCount');
    var sortPrimary = document.getElementById('sortPrimary');
    var sortPrimaryOrder = document.getElementById('sortPrimaryOrder');
    var sortSecondary = document.getElementById('sortSecondary');
    var sortSecondaryOrder = document.getElementById('sortSecondaryOrder');
    var filterRow = document.getElementById('filterRow');
    var searchRow = document.getElementById('searchRow');
    var resetEl = document.getElementById('resetBtn');
    var exportEl = document.getElementById('exportBtn');
    var detailsEl = document.getElementById('rowDetails');
    var colorBySelect = document.getElementById('colorBySelect');

    // Selected points tracking (by a canonical composite key)
    var selectedKeys = [];

    // Build select options helpers
    function fillSelect(sel, items, defaultValue) {
      sel.innerHTML = '';
      items.forEach(function(it){
        var opt = document.createElement('option');
        opt.value = it;
        opt.textContent = it;
        sel.appendChild(opt);
      });
      if (defaultValue && items.indexOf(defaultValue) >= 0) {
        sel.value = defaultValue;
      } else if (items.length > 0) {
        sel.value = items[0];
      }
    }

    // Init axis selectors
    fillSelect(plotTypeSelect, ['bar','scatter','line'], P.plot_type || 'bar');
    fillSelect(xSelect, xChoices, P.x_col || (xChoices.length ? xChoices[0] : ''));
    fillSelect(ySelect, yChoices, P.y_col || (yChoices.length ? yChoices[0] : ''));

    // Init barsCount from initial zoom
    if (UI.initial_zoom && UI.initial_zoom > 0) {
      barsCount.value = UI.initial_zoom;
    }

    // Sort choices

    var prevSecondary = sortSecondary.value;
    var prevSecondaryOrder = sortSecondaryOrder.value;

    var prevPrimary = sortPrimary.value;
    var prevPrimaryOrder = sortPrimaryOrder.value;
    
    // Primary
    fillSelect(sortPrimary, sortChoices, prevPrimary || UI.sort_primary || '');

    // Secondary
    fillSelect(sortSecondary,sortChoices, prevSecondary || UI.sort_secondary || '(none)');
    
    //fillselect for color  by
    var prevColorBy = colorBySelect.value;
    fillSelect(colorBySelect, (UI.color_choices || []), prevColorBy || UI.color_default || '');
    fillSelect(colorBySelect, ['(none)'].concat(UI.color_choices || []),
           UI.color_default || '(none)');
    var colorCol = colorBySelect.value;
    if (colorCol === '(none)') colorCol = null;


    if (firstRender) {
        // Use CLI defaults on first load
        sortPrimaryOrder.value = UI.sort_primary_order || 'desc';
        sortSecondaryOrder.value = UI.sort_secondary_order || 'desc';
    } else {
        // Use user-selected values after first render
        sortPrimaryOrder.value = prevPrimaryOrder || 'desc';
        sortSecondaryOrder.value = prevSecondaryOrder || 'desc';
    }


    // Build filter dropdowns
    var filterEls = {}; // col -> select
    filterCols.forEach(function(col){
      var wrap = document.createElement('div');
      var label = document.createElement('label');
      label.innerHTML = '<strong>' + col + '</strong>';
      var sel = document.createElement('select');
      sel.setAttribute('aria-label', 'Filter: ' + col);
      // Collect unique values from rows
      var uniq = new Set();
      (P.rows || []).forEach(function(r){
        var v = (r[col] != null ? String(r[col]) : '');
        uniq.add(v);
      });
      var values = Array.from(uniq).sort();
      // Special 'All' option
      var optAll = document.createElement('option');
      optAll.value = '__ALL__';
      optAll.textContent = 'All';
      sel.appendChild(optAll);
      values.forEach(function(v){
        var opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      });
      // Default
      var def = filterDefaults[col];
      sel.value = (def && values.indexOf(def) >= 0) ? def : '__ALL__';
      wrap.appendChild(label);
      wrap.appendChild(document.createElement('br'));
      wrap.appendChild(sel);
      filterRow.appendChild(wrap);
      filterEls[col] = sel;
    });

    // Build search inputs
    var searchEls = {}; // col -> input
    searchCols.forEach(function(col){
      var wrap = document.createElement('div');
      var label = document.createElement('label');
      label.innerHTML = '<strong>Search in ' + col + '</strong>';
      var box = document.createElement('input');
      box.type = 'text';
      box.placeholder = 'Type to filter...';
      box.setAttribute('aria-label', 'Search: ' + col);
      box.value = (searchDefaults[col] || '');
      wrap.appendChild(label);
      wrap.appendChild(document.createElement('br'));
      wrap.appendChild(box);
      searchRow.appendChild(wrap);
      searchEls[col] = box;
    });

    // Selection events
    plotEl.on('plotly_selected', function(ev) {
      var pts = (ev && ev.points) ? ev.points : [];
      // Use x (or y for horizontal) as canonical key; for scatter, use JSON of (x,y)
      selectedKeys = pts.map(function(p){
        if ((P.plot_type || 'bar') === 'bar') {
          return String(p.x);
        } else {
          return JSON.stringify({x:p.x, y:p.y});
        }
      });
    });
    plotEl.on('plotly_deselect', function(){ selectedKeys = []; });

    // Click details
    plotEl.on('plotly_click', function(ev) {
      try {
        var p = (ev && ev.points && ev.points[0]) ? ev.points[0] : null;
        if (!p) return;
        var cd = p.customdata || [];
        var cols = P.detail_cols || [];
        if (!cols.length) {
          detailsEl.textContent = 'No detail columns configured.';
          return;
        }
        var html = '<table style="border-collapse:collapse;">';
        for (var i = 0; i < cols.length; i++) {
          var v = cd[i];
          var vv = (typeof v === 'number') ? (Number.isFinite(v) ? v.toPrecision(4) : String(v)) : String(v != null ? v : '');
          html += '<tr><th style="text-align:left;padding:4px 8px;">' + cols[i] + '</th>' +
                  '<td style="padding:4px 8px;">' + vv + '</td></tr>';
        }
        html += '</table>';
        detailsEl.innerHTML = html;
      } catch (e) { console.error('click -> details error:', e); }
    });

    // Filtering function
    function applyFilters(rows) {
      // Dropdown filters
      var filtered = rows.filter(function(r){
        for (var col in filterEls) {
          var selVal = filterEls[col].value;
          if (selVal === '__ALL__') continue;
          var rv = (r[col] != null ? String(r[col]) : '');
          if (rv !== selVal) return false;
        }
        // Search filters (contains, case-insensitive)
        for (var scol in searchEls) {
          var term = (searchEls[scol].value || '').toLowerCase().trim();
          if (term === '') continue;
          var rv2 = String(r[scol] != null ? r[scol] : '').toLowerCase();
          if (rv2.indexOf(term) === -1) return false;
        }
        return true;
      });
      return filtered;
    }
    // Apply client dedupe function
    
    function applyClientDedupe(rows, policy, xcol, colorCol) {
      policy = policy || "overlay";
      if (policy === "overlay" || policy === "stack")
        return rows;  // no collapsing

      // collapse duplicates into one row per category
      function key(r) {
        return colorCol ? (String(r[xcol]) + "||" + String(r[colorCol])) :
                          String(r[xcol]);
      }

      var groups = {};
      for (var r of rows) {
        var k = key(r);
        if (!(k in groups)) groups[k] = [];
        groups[k].push(r);
      }

      var valCol = ySelect.value;
      var output = [];

      for (var k in groups) {
        var g = groups[k];
        var base = g[0];  // row to copy metadata from
        var nums = g.map(r => Number(r[valCol])).filter(n => !Number.isNaN(n));

        if (nums.length === 0) nums = [0];

        var newVal;
        switch (policy) {
          case "first":  newVal = nums[0]; break;
          case "max":    newVal = Math.max(...nums); break;
          case "mean":   newVal = nums.reduce((a,b)=>a+b,0)/nums.length; break;
          case "median":
            nums.sort((a,b)=>a-b);
            newVal = nums[Math.floor(nums.length/2)];
            break;
          case "sum":    newVal = nums.reduce((a,b)=>a+b,0); break;
          default:       newVal = nums[0];
        }

        var newRow = JSON.parse(JSON.stringify(base));
        newRow[valCol] = newVal;
        output.push(newRow);
      }

      return output;
    }


    // Sort function
    function sortRows(rows, pcol, pord, scol, sord) {
    
    // Normalize sort column names
    if (pcol) pcol = pcol.trim();
    if (scol) scol = scol.trim();

      if (!Array.isArray(rows)) return [];
      var data = rows.slice();
      function asNum(v) {
        if (typeof v === 'number') return v;
        if (typeof v === 'string') {
          var s = v.replace(',', '.');
          var n = parseFloat(s);
          if (!Number.isNaN(n)) return n;
        }
        return v;
      }
      
    
    
    function cmp(a, b, col, ord) {
      let asc = (ord || 'asc').toLowerCase() === 'asc';

      let va = a[col];
      let vb = b[col];

      // Convert if possible
      let na = Number(va);
      let nb = Number(vb);

      let a_is_num = !Number.isNaN(na);
      let b_is_num = !Number.isNaN(nb);

      // Case 1: both numeric → numeric compare
      if (a_is_num && b_is_num) {
        return asc ? (na - nb) : (nb - na);
      }

      // Case 2: only one is numeric → treat both as strings
      // This avoids inconsistent ordering
      let sa = String(va);
      let sb = String(vb);

      // Case 3: pure string sort A-Z or Z-A
      return asc ? sa.localeCompare(sb) : sb.localeCompare(sa);
    }

      
      data.sort(function(a, b){
        if (
          pcol &&
          Object.prototype.hasOwnProperty.call(a, pcol) &&
          Object.prototype.hasOwnProperty.call(b, pcol)
        ) {

          var c = cmp(a,b,pcol,pord);
          if (c !== 0) return c;
        }
        if (
          scol &&
          scol !== '(none)' &&
          scol.trim() !== '' &&
          Object.prototype.hasOwnProperty.call(a, scol) &&
          Object.prototype.hasOwnProperty.call(b, scol)
        )
        {
          var d = cmp(a,b,scol,sord);
          if (d !== 0) return d;
        }
        return 0;
      });

      return data;
    }

    // Build full hover template dynamically (axis choice may change)
    function makeHover(detailCols, orientation) {
      var lines = [];
      if (orientation === 'v') {
        lines.push("**%{x}**");
        lines.push("Value: %{y:.4g}");
      } else {
        lines.push("**%{y}**");
        lines.push("Value: %{x:.4g}");
      }
      for (var i = 0; i < detailCols.length; i++) {
        lines.push(detailCols[i] + ": %{customdata[" + i + "]}");
      }
      return lines.join("<br>") + "<extra></extra>";
    }

    // Render function: rebuild single/stacked traces
    function render() {
      var rowsAll = Array.isArray(P.rows) ? P.rows.slice() : [];
      var rowsF = applyFilters(rowsAll);
      rowsF = applyClientDedupe(rowsF, currentDupPolicy, xcol, colorCol);

      var ptype = plotTypeSelect.value || 'bar';
      var xcol = xSelect.value || '';
      var ycol = ySelect.value || '';
      var currentDupPolicy = dupPolicySelect.value || "overlay";

      var pcol = sortPrimary.value.trim();
      var scol = sortSecondary.value.trim();
      rowsF = sortRows(rowsF, pcol, sortPrimaryOrder.value, scol, sortSecondaryOrder.value);

      // Sort before subsetting
      rowsF = sortRows(rowsF, sortPrimary.value, sortPrimaryOrder.value, sortSecondary.value, sortSecondaryOrder.value);

      
      // Bars count (viewport only; do NOT slice data)
      var nBars = parseInt(barsCount.value || '0', 10);
      if (!Number.isFinite(nBars) || nBars <= 0) nBars = null;


      var detailCols = P.detail_cols || [];
      var colorCol = colorBySelect.value || null;

      // custom color by

      // Build colors for current colorCol (client-side)
      var colorMap = {}; 
      var palette = [
          "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
          "#393b79", "#5254a3", "#6b6ecf", "#9c9ede",
          "#c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
          "#843c39", "#ad494a", "#d6616b", "#e7969c",
          "#7b4173", "#a55194", "#ce6dbd", "#de9ed6"
      ];
      // Collect unique categories from the rows to be plotted (rowsF)
      if (colorCol) {
          // collect all possible categories from ALL rows (not only filtered)
          var cats = Array.from(new Set(P.rows.map(r =>
              r[colorCol] != null ? String(r[colorCol]) : ""
          ))).sort();
        cats.sort();
        for (var i = 0; i < cats.length; i++) {
          colorMap[cats[i]] = palette[i % palette.length];
        }
      }

      // Build per-point colors
      var colors = [];
      for (var i = 0; i < rowsF.length; i++) {
        var r = rowsF[i];
        var gval = colorCol ? String(r[colorCol]) : null;
        var c = (colorCol && colorMap[gval]) ? colorMap[gval] : '#636EFA';
        colors.push(c);
      }
      

      // Collect data arrays
      var x = [], y = [], customdata = [];
      var categories = [], ticktext = [];

      // Determine colors
      for (var i = 0; i < rowsF.length; i++) {
        var r = rowsF[i];
        var gval = colorCol ? String(r[colorCol]) : null;
        var color = (gval && colorMap[gval]) ? colorMap[gval] : '#636EFA';
        colors.push(color);
      }

      // Build x,y/customdata based on plot type
      if (ptype === 'bar') {
        // X categorical, Y numeric
        for (var i = 0; i < rowsF.length; i++){
          var r = rowsF[i];
          var xc = String(r[xcol]);
          var yc = r[ycol];
          if (typeof yc === 'string') {
            var s = yc.replace(',', '.');
            yc = parseFloat(s);
          }
          x.push(xc);
          y.push(yc);
          categories.push(xc);
          ticktext.push(xc);
          var cd = [];
          for (var j = 0; j < detailCols.length; j++) cd.push(r[detailCols[j]]);
          customdata.push(cd);
        }

        var traces = [];
        if (currentDupPolicy === 'stack') {
          // group by category, stack layers
          var catMap = new Map();
          for (var i = 0; i < x.length; i++) {
            var cat = x[i];
            if (!catMap.has(cat)) catMap.set(cat, []);
            catMap.get(cat).push({y:y[i], color:colors[i], cd:customdata[i]});
          }
          // max depth
          var maxDepth = 0;
          catMap.forEach(function(arr){ if (arr.length > maxDepth) maxDepth = arr.length; });

          for (var layer = 0; layer < maxDepth; layer++) {
            var xL = [], yL = [], colL = [], cdL = [];
            for (var i = 0; i < categories.length; i++) {
              var cat = categories[i];
              var arr = catMap.get(cat) || [];
              if (layer < arr.length) {
                xL.push(cat); yL.push(arr[layer].y);
                colL.push(arr[layer].color);
                cdL.push(arr[layer].cd);
              } else {
                xL.push(cat); yL.push(0);
                colL.push('#00000000');
                cdL.push([]);
              }
            }
            traces.push({
              type: 'bar',
              x: xL,
              y: yL,
              customdata: cdL,
              marker: { color: colL },
              hovertemplate: makeHover(detailCols, 'v'),
              name: ('layer ' + (layer+1))
            });
          }
          
          Plotly.react(plotEl, traces, {
            title: P.title || (ycol + ' by ' + xcol),
            template: 'plotly_white',
            hovermode: 'closest',
            showlegend: !!P.show_legend,
            dragmode: 'select',
            margin: {l:80, r:40, t:70, b:130},
            xaxis: {
              title: { text: xcol },
              automargin: true,
              title_standoff: 10,
              tickangle: -45,
              categoryorder: 'array',
              categoryarray: categories,
              tickmode: 'array',
              tickvals: categories,
              ticktext: ticktext,
              // NEW:
              range: (nBars ? [-0.5, Math.min(categories.length, nBars) - 0.5] : undefined),
            },
            yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
            barmode: 'stack'
          });
        } else {
          var trace = {
            type: 'bar',
            x: x,
            y: y,
            customdata: customdata,
            marker: { color: colors },
            selected: { marker: { opacity: 1.0 } },
            unselected: { marker: { opacity: 0.5 } },
            hovertemplate: makeHover(detailCols, 'v'),
          };
          
          Plotly.react(plotEl, [trace], {
            title: P.title || (ycol + ' by ' + xcol),
            template: 'plotly_white',
            hovermode: 'closest',
            showlegend: !!P.show_legend,
            dragmode: 'select',
            margin: {l:80, r:40, t:70, b:130},
            xaxis: {
              title: { text: xcol },
              automargin: true,
              title_standoff: 10,
              tickangle: -45,
              categoryorder: 'array',
              categoryarray: categories,
              tickmode: 'array',
              tickvals: categories,
              ticktext: ticktext,
              // NEW:
              range: (nBars ? [-0.5, Math.min(categories.length, nBars) - 0.5] : undefined),
            },
            yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
            barmode: 'overlay'
          });
        }

      } else {
        // scatter/line: both numeric
        for (var i = 0; i < rowsF.length; i++){
          var r = rowsF[i];
          var xv = r[xcol];
          var yv = r[ycol];
          if (typeof xv === 'string') { var s1 = xv.replace(',', '.'); xv = parseFloat(s1); }
          if (typeof yv === 'string') { var s2 = yv.replace(',', '.'); yv = parseFloat(s2); }
          x.push(xv);
          y.push(yv);
          var cd = [];
          for (var j = 0; j < detailCols.length; j++) cd.push(r[detailCols[j]]);
          customdata.push(cd);
        }
        var trace2 = {
          type: 'scatter',
          mode: (ptype === 'scatter' ? 'markers' : 'lines+markers'),
          x: x,
          y: y,
          customdata: customdata,
          marker: { color: colors },
          hovertemplate: makeHover(detailCols, 'v'),
        };
        Plotly.react(plotEl, [trace2], {
          title: P.title || (ycol + ' vs ' + xcol),
          template: 'plotly_white',
          hovermode: 'closest',
          showlegend: !!P.show_legend,
          dragmode: 'select',
          margin: {l:80, r:40, t:70, b:80},
          xaxis: { title: { text: xcol }, automargin: true, title_standoff: 10 },
          yaxis: { title: { text: ycol }, automargin: true, title_standoff: 10 },
        });
      }
    }

    // Bind changes
    plotTypeSelect.addEventListener('change', render);
    xSelect.addEventListener('change', render);
    ySelect.addEventListener('change', render);
    barsCount.addEventListener('input', render);
    sortPrimary.addEventListener('change', render);
    sortPrimaryOrder.addEventListener('change', render);
    sortSecondary.addEventListener('change', render);
    sortSecondaryOrder.addEventListener('change', render);
    dupPolicySelect.addEventListener('change', render);
    Object.values(filterEls).forEach(function(sel){ sel.addEventListener('change', render); });
    Object.values(searchEls).forEach(function(box){ box.addEventListener('input', render); });
    colorBySelect.addEventListener('change', render);

    resetEl.addEventListener('click', function(){
      plotTypeSelect.value = (P.plot_type || 'bar');
      xSelect.value = (P.x_col || (xChoices.length ? xChoices[0] : ''));
      ySelect.value = (P.y_col || (yChoices.length ? yChoices[0] : ''));
      barsCount.value = (UI.initial_zoom || 100);
      sortPrimary.value = (UI.sort_primary || '');
      sortPrimaryOrder.value = (UI.sort_primary_order || 'desc');
      sortSecondary.value = (UI.sort_secondary || '(none)');
      sortSecondaryOrder.value = (UI.sort_secondary_order || 'desc');
      for (var col in filterEls) {
        var def = filterDefaults[col];
        var sel = filterEls[col];
        if (def && Array.from(sel.options).map(function(o){return o.value;}).indexOf(def) >= 0) {
          sel.value = def;
        } else {
          sel.value = '__ALL__';
        }
      }
      for (var scol in searchEls) {
        searchEls[scol].value = (searchDefaults[scol] || '');
      }
      selectedKeys = [];
      firstRender = false;
      render();
    });

    exportEl.addEventListener('click', function(){
      // Export TSV (filtered OR selected)
      var rowsAll = Array.isArray(P.rows) ? P.rows.slice() : [];
      var rowsF = applyFilters(rowsAll);

      // if selection exists, filter by selection set
      if (selectedKeys && selectedKeys.length > 0) {
        var sset = new Set(selectedKeys.map(String));
        rowsF = rowsF.filter(function(r){
          // bar: key is X category; scatter: JSON of (x,y)
          var ptype = plotTypeSelect.value || 'bar';
          if (ptype === 'bar') {
            var key = String(r[xSelect.value || P.x_col]);
            return sset.has(key);
          } else {
            // best-effort: exact match on (x,y) pair
            var xv = r[xSelect.value || P.x_col];
            var yv = r[ySelect.value || P.y_col];
            var key = JSON.stringify({x:xv, y:yv});
            return sset.has(key);
          }
        });
      }

      var cols = P.detail_cols || (rowsF.length ? Object.keys(rowsF[0]) : []);
      var header = cols.join(TAB);
      var lines = rowsF.map(function(r){
        return cols.map(function(c){ return (r[c] != null ? String(r[c]) : ''); }).join(TAB);
      });
      var tsv = [header].concat(lines).join(NL);
      var blob = new Blob([tsv], { type: 'text/tab-separated-values;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'export.tsv';
      a.click();
      URL.revokeObjectURL(a.href);
    });

    // Initial render
    render();

  } catch (e) {
    console.error('UI init error:', e);
  }
})();
</script>
"""

def save_html(fig: Figure, payload: Dict, ui_cfg: Dict, out_path: str, self_contained: bool, lang: str = "en"):
    # Build plotly HTML
    html = fig.to_html(full_html=True, include_plotlyjs="cdn" if not self_contained else True)
    html = html.lstrip("\ufeff\r\n\t")
    if not html.startswith("<!DOCTYPE html>"):
        html = "<!DOCTYPE html>\n" + html

    # Ensure lang attribute
    if not re.search(r"(?is)<html[^>]*>", html):
        html = f'<!DOCTYPE html>\n<html lang="{lang}">\n</html>'
    if not re.search(r"(?is)<html[^>]*\blang\s*=", html):
        html = re.sub(r"(?is)<html(\s*)>", f'<html lang="{lang}"\\1>', html, count=1)

    # Inject basic meta & title if needed
    head_inject = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{payload.get("title") or "Interactive plot"}</title>\n'
    )
    if re.search(r"(?is)<head\s*>", html):
        html = re.sub(r"(?is)<head\s*>", "<head>\n" + head_inject, html, count=1)
    else:
        html = re.sub(r"(?is)(<html[^>]*>)", r"\1\n<head>\n" + head_inject + "</head>\n", html, count=1)

    # Attach payload JSON + UI
    payload_json = json.dumps({**payload, "ui_cfg": ui_cfg}, ensure_ascii=False)
    payload_json = payload_json.replace("</script", "<\\/script")  # safe close
    payload_script = f'<script type="application/json" id="__payload__">{payload_json}</script>\n'

    # Insert our controls + client script before </body>
    if re.search(r"(?is)</body>", html):
        html = re.sub(r"(?is)</body>", payload_script + DETAILS_UI + "\n</body>", html, count=1)
    else:
        html = html + "\n" + payload_script + DETAILS_UI

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# ------------------------------
# CLI
# ------------------------------

def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # support both '|' and ',' separators
    items = re.split(r"[|,]", s)
    return [it.strip() for it in items if it.strip()]

def parse_kv_defaults(s: Optional[str]) -> Dict[str, str]:
    """
    Parse "col=value; col=value" pairs; supports ';' or '|' as pair separators.
    """
    if not s:
        return {}
    pairs = re.split(r"[;|]", s)
    out = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser(
        "Universal interactive plot maker (generalized)"
    )
    ap.add_argument("--file", "-f", required=True, help="Input table (TSV/CSV)")
    ap.add_argument("--out", "-o", default="interactive_plot.html", help="Output HTML file")
    ap.add_argument("--sep", help="Field separator (auto by extension if omitted)")
    ap.add_argument("--plot-type", default="bar", choices=["bar","scatter","line"], help="Plot type")
    ap.add_argument("--title", default="", help="Figure title")
    ap.add_argument("--color-col", help="Column for group coloring")
    ap.add_argument("--color-choices",help="Columns allowed for coloring (pipe or comma separated). If omitted, any column can be chosen.")
    ap.add_argument("--x-choices", help="Columns allowed for X axis (bar categorical; scatter/line numeric). Use '|' or ',' to separate.")
    ap.add_argument("--y-choices", help="Columns allowed for Y axis (numeric). Use '|' or ',' to separate.")
    ap.add_argument("--default-x", help="Default X column at load")
    ap.add_argument("--default-y", help="Default Y column at load")
    ap.add_argument("--filter-cols", help="Columns to add dropdown filters (pipe or comma separated)")
    ap.add_argument("--filter-defaults", help="Default selection for filters: 'col=value;col=value'")
    ap.add_argument("--search-cols", help="Columns to add search boxes")
    ap.add_argument("--search-defaults", help="Default values for search boxes: 'col=term;col=term'")
    ap.add_argument("--details", default="*", help="Columns to include in hover/details/export. '*' means all cols.")
    ap.add_argument("--initial-zoom", type=int, default=100, help="Initial number of bars/points to show")
    ap.add_argument("--sort-primary", help="Primary sort column")
    ap.add_argument("--sort-primary-order", default="desc", choices=["asc","desc"], help="Primary sort order")
    ap.add_argument("--sort-secondary", help="Secondary sort column (optional)")
    ap.add_argument("--sort-secondary-order", default="desc", choices=["asc","desc"], help="Secondary sort order")
    ap.add_argument("--dup-policy", default="overlay", choices=["overlay","stack","max","mean","median","first","sum"],
                    help="Duplicate policy: overlay/stack for visual; max/mean/median/first/sum collapse pre-plot.")
    ap.add_argument("--show-legend", action="store_true", help="Show legend (hidden otherwise)")
    ap.add_argument("--self-contained", action="store_true", help="Embed plotly.js for offline HTML")
    ap.add_argument("--lang", default="en", help="HTML lang attribute (e.g., 'en', 'en-CA')")

    args = ap.parse_args()

    # Load data
    try:
        df = load_table(args.file, sep=args.sep)
    except Exception as e:
        print(f"[error] Failed to load '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare UI configs
    x_choices = parse_list(args.x_choices) or list(df.columns)
    y_choices = parse_list(args.y_choices) or [c for c in df.columns if is_numeric_series(df[c])]
    filter_cols = parse_list(args.filter_cols)
    search_cols = parse_list(args.search_cols)
    filter_defaults = parse_kv_defaults(args.filter_defaults)
    search_defaults = parse_kv_defaults(args.search_defaults)
    color_choices = (parse_list(args.color_choices) if args.color_choices
                 else list(df.columns))  # fallback: any column
    default_color = args.color_col if (args.color_col in df.columns) else ""


    # Details columns
    if args.details.strip() == "*":
        details_cols = list(df.columns)
    else:
        details_cols = [c for c in parse_list(args.details) if c in df.columns]
        if not details_cols:
            details_cols = list(df.columns)

    # Determine defaults for axis
    default_x = args.default_x or (x_choices[0] if x_choices else None)
    default_y = args.default_y or (y_choices[0] if y_choices else None)

    # Build figure & payload on initial defaults
    try:
        fig, payload = build_figure_payload(
            df=df,
            plot_type=args.plot_type,
            x_col=default_x,
            y_col=default_y,
            color_col=args.color_col if args.color_col in df.columns else None,
            details_cols=details_cols,
            top_n=args.initial_zoom if args.initial_zoom and args.initial_zoom > 0 else None,
            sort_primary=args.sort_primary if args.sort_primary in df.columns else None,
            sort_primary_order=args.sort_primary_order,
            sort_secondary=args.sort_secondary if (args.sort_secondary and args.sort_secondary in df.columns) else None,
            sort_secondary_order=args.sort_secondary_order,
            initial_zoom=args.initial_zoom,
            title=args.title,
            show_legend=args.show_legend,
            dup_policy=args.dup_policy,
        )
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    # UI config object for client
    ui_cfg = {
        "x_choices": x_choices,
        "y_choices": y_choices,
        "filter_cols": filter_cols,
        "filter_defaults": filter_defaults,
        "search_cols": search_cols,
        "search_defaults": search_defaults,
        "initial_zoom": args.initial_zoom,
        "sort_choices": list(df.columns),  # permit any column to sort
        "sort_primary": args.sort_primary if args.sort_primary in df.columns else "",
        "sort_primary_order": args.sort_primary_order,
        "sort_secondary": args.sort_secondary if (args.sort_secondary and args.sort_secondary in df.columns) else "",
        "sort_secondary_order": args.sort_secondary_order,
        "color_choices": color_choices,
        "color_default": default_color,   # initial dropdown default
    }

    # Persist HTML
    save_html(fig, payload, ui_cfg, out_path=args.out, self_contained=bool(args.self_contained), lang=args.lang)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

```
# 2)Run
```

python universal_plot_maker_plus.py \
  --file top_10k.tsv \
  --out Celltype_Enrichment_V2_1_top_10k.html \
  --plot-type bar \
  --x-choices "Gene name | Gene" \
  --y-choices "Enrichment score|log2_enrichment| specificity_tau | Enrichment score (tau penalized)|log2_enrichment_penalized" \
  --default-x "Gene name" \
  --default-y "log2_enrichment_penalized" \
  --color-col "Cell type" \
  --color-choices "Cell type|Cell type group|Cell type class" \
  --filter-cols "Cell type class|Cell type group|Cell type" \
  --search-cols "Gene|Gene name" \
  --details "Gene|Gene name|Cell type|Cell type group|Cell type class|Enrichment score|log2_enrichment| specificity_tau |log2_enrichment_penalized|top_percent_Cell_type_count|top_percent_Cell_type_group_count|top_percent_Cell_type_class_count|overall_rank_by_Cell_type|overall_rank_by_Cell_type_group|overall_rank_by_Cell_type_class|rank_within_Cell_type|rank_within_Cell_type_group|rank_within_Cell_type_class|top_percent_Cell_types|top_percent_Cell_type_groups|top_percent_Cell_type_classes" \
  --title "Celltype Enrichmnt V 2.1" \
  --dup-policy overlay \
  --sort-primary "overall_rank_by_Cell_type" \
  --sort-primary-order asc \
  --sort-secondary "log2_enrichment_penalized" \
  --sort-secondary-order desc \
  --initial-zoom 100 \
  --self-contained \
  --lang en
```
# 3) 🧰 Command‑Line Interface (CLI) — Full Help

Below is the complete list of CLI options supported by `universal_plot_maker_plus.py`, with explanations of what each command does and how it affects the resulting interactive HTML plot.

---

## 📥 Input / Output

### `--file`, `-f`
Path to the input TSV/CSV file.

### `--out`, `-o`
Output HTML file path.  
Default: `interactive_plot.html`

### `--sep`
Manually specify a field separator.  
If omitted, auto‑detected based on file extension (`.tsv`, `.csv`, etc.).

---

## 📊 Plot Configuration

### `--plot-type {bar,scatter,line}`
Initial plot type shown in the viewer.  
End users can still change plot type later.

### `--title`
Title displayed at the top of the plot.

### `--color-col`
Column used for coloring points or bars.  
Each unique category is mapped to a unique color.

---

## 🧭 Axis Selection

### `--x-choices`
List of allowed X‑axis columns.  
Use `|` or `,` to separate multiple options.

### `--y-choices`
List of allowed Y‑axis columns (typically numeric).  
Use `|` or `,` to separate multiple options.

### `--default-x`
The X‑axis column selected at initial load.

### `--default-y`
The Y‑axis column selected at initial load.

---

## 🎚️ Sorting

### `--sort-primary`
Primary sort column used before plotting.

### `--sort-primary-order {asc,desc}`
Sorting direction for primary sort.

### `--sort-secondary`
Optional secondary sort column.

### `--sort-secondary-order {asc,desc}`
Sorting direction for secondary sort.

---

## 🔍 Filtering & Searching

### `--filter-cols`
List of columns exposed as dropdown filters in the HTML viewer.

### `--filter-defaults`
Default filter selections in the format:
``
"Column1=Value1;Column2=Value2"
Use `__ALL__` to default to "no filtering".

### `--search-cols`
Columns that get search bars in the UI.

### `--search-defaults`
Default starting values for search inputs:

"Column1=query;Column2=query"

---

## 📝 Details Panel / Hover Information

### `--details`
Columns included in:
- hover tooltip  
- click‑to‑inspect details panel  
- TSV export of selected points  

Use:
- `"*"` to include all columns  
- `"Col1|Col2|Col3"` for specific columns  

---

## 🔁 Duplicate Handling

### `--dup-policy {overlay,stack,max,mean,median,first,sum}`
Defines how duplicate X‑values (or X+color pairs) are handled:

| Policy | Description |
|--------|-------------|
| `overlay` | Plot duplicates on top of each other (default) |
| `stack` | Stack duplicates as multiple bars |
| `max` | Use only the maximum value |
| `mean` | Use the mean of duplicates |
| `median` | Use the median |
| `first` | Keep first occurrence |
| `sum` | Sum all duplicates |

---

## 🔍 Initial Zoom / Data Window

### `--initial-zoom`
Number of rows/bars initially shown.  
More rows remain accessible through sorting or increasing the zoom in the UI.

---

## 🎨 Legend & Layout

### `--show-legend`
If supplied, legend is visible by default.  
Omit to hide the legend.

### `--lang`
Set the HTML `<html lang="...">` attribute.

---

## 📦 Self‑contained HTML

### `--self-contained`
Embed Plotly.js directly in the output HTML.  
Use this when sharing the HTML offline.

---

## 📤 Example Command

```bash
python universal_plot_maker_plus.py \
    --file top_10k.tsv \
    --out top_10k.html \
    --plot-type bar \
    --x-choices "Gene name" \
    --y-choices "Enrichment score|log2_enrichment|log2_enrichment_penalized" \
    --default-x "Gene name" \
    --default-y "log2_enrichment_penalized" \
    --color-col "Cell type" \
    --filter-cols "Cell type class|Cell type group|Cell type" \
    --search-cols "Gene|Gene name" \
    --details "*" \
    --sort-primary "overall_rank_by_Cell_type" \
    --sort-primary-order asc \
    --initial-zoom 100 \
    --self-contained
```
