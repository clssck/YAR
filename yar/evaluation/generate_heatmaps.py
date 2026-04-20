#!/usr/bin/env python3
"""Render per-question metric heatmaps for concierge and RAGAS eval runs.

Given the graded-concierge JSON emitted by grade_concierge.py and a RAGAS
results JSON (e.g. yar/evaluation/results/results_YYYYMMDD_HHMMSS.json),
produce one PNG heatmap per source. Both heatmaps share the same layout
so they can be pasted side-by-side in a PPTX:

	rows: Q1..QN (test_number, truncated question text)
	cols: metric names (per-source fixed order) + composite
	cells: 0.0 - 1.0 annotated, RdYlGn colormap
	errored/missing cells: shown as NaN (greyed out)

Typical use:
	.venv/bin/python yar/evaluation/generate_heatmaps.py \
		--concierge yar/evaluation/results/concierge_graded_20260420_143013.json \
		--ragas     yar/evaluation/results/results_20260420_141924.json \
		--outdir    yar/evaluation/results/heatmaps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAGAS_METRIC_ORDER = ['faithfulness', 'answer_relevance', 'context_recall', 'context_precision']
CONCIERGE_METRIC_ORDER = ['correctness', 'relevance', 'completeness', 'faithfulness']


@dataclass
class HeatmapSpec:
    title: str
    df: pd.DataFrame
    score_col: str
    output_path: Path


def truncate(text: str, max_chars: int = 60) -> str:
    text = (text or '').replace('\n', ' ').strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + '…'


def row_label(test_number: int, question: str, max_chars: int = 60) -> str:
    return f'Q{test_number:02d}: {truncate(question, max_chars)}'


def load_concierge(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for res in data.get('results', []):
        metrics = res.get('metrics') or {}
        row = {
            'test_number': int(res.get('test_number') or 0),
            'question': res.get('question', ''),
            'status': res.get('status', 'unknown'),
        }
        for key in CONCIERGE_METRIC_ORDER:
            value = metrics.get(key)
            row[key] = float(value) if value is not None else np.nan
        row['composite'] = float(res.get('composite_score')) if res.get('composite_score') is not None else np.nan
        if row['status'] not in ('success',):
            # Keep test_number/question/label for alignment, but mark metric cells NaN.
            for key in (*CONCIERGE_METRIC_ORDER, 'composite'):
                row[key] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('test_number').reset_index(drop=True)
    df['label'] = df.apply(lambda r: row_label(r['test_number'], r['question']), axis=1)
    return df


def load_ragas(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for res in data.get('results', []):
        metrics = res.get('metrics') or {}
        row = {
            'test_number': int(res.get('test_number') or 0),
            'question': res.get('question', ''),
            'status': res.get('status', 'unknown'),
        }
        for key in RAGAS_METRIC_ORDER:
            value = metrics.get(key)
            row[key] = float(value) if value is not None else np.nan
        ragas_score = res.get('ragas_score')
        row['composite'] = float(ragas_score) if ragas_score not in (None, 0) else np.nan
        if row['status'] not in ('success',):
            for key in (*RAGAS_METRIC_ORDER, 'composite'):
                row[key] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('test_number').reset_index(drop=True)
    df['label'] = df.apply(lambda r: row_label(r['test_number'], r['question']), axis=1)
    return df


def render_heatmap(spec: HeatmapSpec) -> None:
    df = spec.df
    metric_cols = [c for c in df.columns if c not in {'test_number', 'question', 'status', 'label'}]
    matrix = df[metric_cols].to_numpy(dtype=float)
    row_labels = df['label'].tolist()

    # Figure size scales with question count so labels don't overlap.
    row_count = len(df)
    col_count = len(metric_cols)
    fig_w = max(8.0, 1.6 * col_count + 5.0)
    fig_h = max(6.0, 0.35 * row_count + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Column headers get average appended in parentheses for quick reading.
    col_means = np.nanmean(matrix, axis=0)
    col_headers = [f'{name}\n(μ={mean:.2f})' for name, mean in zip(metric_cols, col_means, strict=False)]

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor='#dddddd',
        cbar_kws={'label': 'score (0-1)'},
        xticklabels=col_headers,
        yticklabels=row_labels,
        annot_kws={'fontsize': 8},
        mask=np.isnan(matrix),
    )
    ax.set_title(spec.title, fontsize=14, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spec.output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {spec.output_path} ({row_count} rows × {col_count} cols)')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render metric heatmaps for concierge + RAGAS evaluations.')
    p.add_argument('--concierge', type=Path, default=None, help='Graded-concierge JSON (from grade_concierge.py).')
    p.add_argument('--ragas', type=Path, default=None, help='RAGAS results JSON (e.g. results_*.json).')
    p.add_argument('--outdir', type=Path, default=Path('yar/evaluation/results/heatmaps'))
    p.add_argument('--concierge-title', type=str, default='Concierge QA — judge-graded metrics')
    p.add_argument('--ragas-title', type=str, default='YAR RAGAS evaluation — per-question metrics')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.concierge and not args.ragas:
        raise SystemExit('Provide at least one of --concierge or --ragas.')

    specs: list[HeatmapSpec] = []
    if args.concierge:
        df = load_concierge(args.concierge)
        out = args.outdir / f'concierge_heatmap_{args.concierge.stem}.png'
        specs.append(HeatmapSpec(title=args.concierge_title, df=df, score_col='composite', output_path=out))
    if args.ragas:
        df = load_ragas(args.ragas)
        out = args.outdir / f'ragas_heatmap_{args.ragas.stem}.png'
        specs.append(HeatmapSpec(title=args.ragas_title, df=df, score_col='composite', output_path=out))

    for spec in specs:
        render_heatmap(spec)


if __name__ == '__main__':
    main()
