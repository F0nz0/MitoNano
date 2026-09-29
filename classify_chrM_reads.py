#!/usr/bin/env python3
"""Classify extended mitochondrial-read TSVs into four categories based on gene overlap."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)

COMPLETE_LABEL = "Complete gene"
PRECURSOR_LABEL = "Precursor | Multi-gene"
PARTIAL_LABEL = "Partial gene"
NGR_LABEL = "Non Genic Region"
CLASS_COLUMN = "chrM_primary_class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify extended chrM-read TSVs into four categories based on gene overlap."
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more extended TSV files produced by extract_chrM_info_ext_v2.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where classified TSV files will be written.",
    )
    parser.add_argument(
        "--complete-threshold",
        type=float,
        default=90.0,
        help="Minimum percentage of a single gene covered for Complete gene (default: 90).",
    )
    parser.add_argument(
        "--precursor-gene-bases",
        type=int,
        default=30,
        help="Minimum overlap in bases for precursor support (default: 30).",
    )
    parser.add_argument(
        "--precursor-gene-pct",
        type=float,
        default=30.0,
        help="Minimum percentage of a gene covered for precursor support (default: 30).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gene_columns(table: pl.DataFrame) -> list[str]:
    """Return gene-base columns that have a matching percentage column."""
    columns = set(table.columns)
    return [
        column
        for column in table.columns
        if f"{column} (%)" in columns
    ]


def classify_reads(
    table: pl.DataFrame,
    complete_threshold: float = 90.0,
    precursor_gene_bases: int = 30,
    precursor_gene_pct: float = 30.0,
) -> pl.DataFrame:
    """Add the four classification labels to an extended-read table."""
    if not isinstance(table, pl.DataFrame):
        raise TypeError("Input table must be a polars.DataFrame.")

    gene_base_columns = gene_columns(table)
    classes: list[str] = []

    for row in tqdm(table.to_dicts(), desc="Classifying reads", unit="read"):
        overlapping_genes = 0
        supported_genes = 0
        max_overlap_pct = 0.0

        for gene_id in gene_base_columns:
            overlap_bases = _to_int(row.get(gene_id, 0))
            if overlap_bases <= 0:
                continue

            overlapping_genes += 1
            overlap_pct = _to_float(row.get(f"{gene_id} (%)", 0.0))
            max_overlap_pct = max(max_overlap_pct, overlap_pct)

            if (
                overlap_bases >= precursor_gene_bases
                and overlap_pct >= precursor_gene_pct
            ):
                supported_genes += 1

        if overlapping_genes == 0:
            read_class = NGR_LABEL
        elif overlapping_genes == 1:
            read_class = (
                COMPLETE_LABEL
                if max_overlap_pct >= complete_threshold
                else PARTIAL_LABEL
            )
        elif supported_genes >= 1:
            read_class = PRECURSOR_LABEL
        else:
            # Multi-gene reads without precursor-level support are classified as Complete gene
            read_class = COMPLETE_LABEL

        classes.append(read_class)

    return table.with_columns(pl.Series(CLASS_COLUMN, classes, dtype=pl.String))


def process_table(
    input_path: Path,
    output_dir: Path,
    complete_threshold: float,
    precursor_gene_bases: int,
    precursor_gene_pct: float,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {input_path}")

    logger.info("Reading %s", input_path)
    table = pl.read_csv(input_path, separator="\t", has_header=True)
    classified_table = classify_reads(
        table,
        complete_threshold=complete_threshold,
        precursor_gene_bases=precursor_gene_bases,
        precursor_gene_pct=precursor_gene_pct,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.classified.tsv"
    classified_table.write_csv(output_path, separator="\t")
    logger.info("Wrote %d reads to %s", len(classified_table), output_path)
    return output_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    for input_path in args.input_tsv:
        process_table(
            input_path=input_path.expanduser(),
            output_dir=args.output_dir.expanduser(),
            complete_threshold=args.complete_threshold,
            precursor_gene_bases=args.precursor_gene_bases,
            precursor_gene_pct=args.precursor_gene_pct,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
