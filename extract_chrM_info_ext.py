#!/usr/bin/env python3
"""
For each strand-specific BAM, the script writes an extended TSV with read
coordinates, IsoQuant assignment if available, and strand-aware overlap against gene
annotations from a GTF file.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import pysam

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reads info from forward and reverse BAMs."
    )
    parser.add_argument("sample", help="Sample ID, e.g. GTA1")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base analysis directory containing sample folders (default: current working directory).",
    )
    parser.add_argument(
        "--chromosome",
        default="chrM",
        help="Chromosome to extract reads from (default: chrM).",
    )
    parser.add_argument(
        "--gtf",
        required=True,
        help="GTF file used to compute strand-aware gene overlaps.",
    )
    parser.add_argument(
        "--forward-bam",
        required=True,
        help="Path to the forward BAM for this sample.",
    )
    parser.add_argument(
        "--reverse-bam",
        required=True,
        help="Path to the reverse BAM for this sample.",
    )
    parser.add_argument(
        "--isoquant-assignments",
        default=None,
        help="Optional explicit path to the IsoQuant read_assignments.tsv.gz file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for progress and diagnostics (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def empty_reads_df(gene_ids: list[str]) -> pd.DataFrame:
    columns = [
        "Chr",
        "ChrStart",
        "ChrEnd",
        "read_id",
        "MAPQ",
        "strand",
        "read_length",
        "alignment_length",
        "reference_length",
        "polyA_length",
        "isoquant_assignment",
        "isoquant_strand",
        "isoquant_ENST",
        "isoquant_ENSG",
        "sample",
    ]
    for gene_id in gene_ids:
        columns.append(f"{gene_id}")
        columns.append(f"{gene_id} (%)")
    columns.extend(["overlapping_gene_ids", "overlapping_gene_names", "n_overlapping_gene_ids"])
    return pd.DataFrame(columns=columns)


def parse_gene_id(attributes: str) -> str:
    match = re.search(r'gene_id\s+"([^"]+)"', attributes)
    return match.group(1) if match else ""


def parse_gene_name(attributes: str) -> str:
    match = re.search(r'gene_name\s+"([^"]+)"', attributes)
    return match.group(1) if match else ""


def load_gtf_genes(gtf_path: Path, chromosome: str) -> pd.DataFrame:
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["Chr", "feature", "ChrStart", "ChrEnd", "strand", "attributes"],
        dtype={
            "Chr": str,
            "feature": str,
            "ChrStart": int,
            "ChrEnd": int,
            "strand": str,
            "attributes": str,
        },
    )

    genes = gtf[(gtf["Chr"] == chromosome) & (gtf["feature"] == "gene")].copy()
    genes["gene_id"] = genes["attributes"].map(parse_gene_id)
    genes["gene_name"] = genes["attributes"].map(parse_gene_name)
    genes = genes[genes["gene_id"] != ""]

    genes = (
        genes.groupby(["Chr", "strand", "gene_id"], as_index=False)
        .agg(ChrStart=("ChrStart", "min"), ChrEnd=("ChrEnd", "max"), gene_name=("gene_name", "first"))
        .sort_values(["strand", "ChrStart", "ChrEnd", "gene_id"])
        .reset_index(drop=True)
    )
    genes["start_0"] = genes["ChrStart"] - 1
    genes["end_0"] = genes["ChrEnd"]
    genes["gene_length"] = genes["end_0"] - genes["start_0"]
    return genes


def load_isoquant_assignments(assign_path: Path | None, chromosome: str) -> pd.DataFrame:
    empty_assignments = pd.DataFrame(
        columns=["read_id", "assignment_type", "isoquant_strand", "isoquant_ENST", "isoquant_ENSG"]
    )

    if assign_path is None:
        logger.info("No IsoQuant assignment file provided; continuing without IsoQuant annotations.")
        return empty_assignments

    if not assign_path.exists():
        logger.warning(
            "IsoQuant assignment file not found: %s. Continuing without IsoQuant annotations.",
            assign_path,
        )
        return empty_assignments

    try:
        assigns = pd.read_csv(
            assign_path,
            sep="\t",
            comment="#",
            compression="infer",
            header=None,
            usecols=[0, 1, 2, 3, 4, 5],
            names=["read_id", "chr", "isoquant_strand", "isoquant_ENST", "isoquant_ENSG", "assignment_type"],
            dtype=str,
        )
    except Exception as e:
        logger.warning("Error occurred while reading IsoQuant file %s: %s", assign_path, e)
        return pd.DataFrame(
            columns=["read_id", "assignment_type", "isoquant_strand", "isoquant_ENST", "isoquant_ENSG"]
        )

    assigns = assigns[assigns["read_id"] != "read_id"].copy()
    assigns = assigns[assigns["chr"] == chromosome].copy()

    if "read_id" not in assigns.columns or "assignment_type" not in assigns.columns:
        logger.warning(
            "IsoQuant file %s does not contain required columns. Available columns: %s",
            assign_path,
            list(assigns.columns),
        )
        return pd.DataFrame(
            columns=["read_id", "assignment_type", "isoquant_strand", "isoquant_ENST", "isoquant_ENSG"]
        )

    cols_to_keep = ["read_id", "assignment_type", "isoquant_strand", "isoquant_ENST", "isoquant_ENSG"]
    assigns = assigns[cols_to_keep].copy()

    agg_dict = {
        "assignment_type": lambda x: ",".join(x.astype(str)),
        "isoquant_strand": lambda x: ",".join(x.astype(str)),
        "isoquant_ENST": lambda x: ",".join(x.astype(str)),
        "isoquant_ENSG": lambda x: ",".join(x.astype(str)),
    }

    assigns = pd.DataFrame(assigns.groupby("read_id", as_index=False).agg(agg_dict))

    logger.debug("Loaded %d aggregated IsoQuant assignment rows for %s", len(assigns), chromosome)
    return assigns


def read_overlap_bases(read_start: int, read_end: int, gene_start: int, gene_end: int) -> int:
    overlap = min(read_end, gene_end) - max(read_start, gene_start)
    return max(0, overlap)


def build_read_row(read: pysam.AlignedSegment, sample: str, chromosome: str, strand: str) -> dict[str, object]:
    return {
        "Chr": str(chromosome),
        "ChrStart": int(read.reference_start),
        "ChrEnd": int(read.reference_end),
        "read_id": str(read.query_name),
        "MAPQ": int(read.mapping_quality),
        "strand": str(strand),
        "read_length": int(read.query_length),
        "alignment_length": int(read.query_alignment_length),
        "reference_length": int(read.reference_end) - int(read.reference_start),
        "polyA_length": int(read.get_tag("pt")) if read.has_tag("pt") else None,
        "sample": str(sample),
    }


def process_bam(
    bam_path: Path,
    output_path: Path,
    sample: str,
    chromosome: str,
    strand: str,
    genes: pd.DataFrame,
    gene_ids: list[str],
    isoquant_assignments: pd.DataFrame,
) -> None:
    strand_genes = genes[genes["strand"] == strand].copy()
    strand_gene_id_set = set(strand_genes["gene_id"])
    strand_gene_ids = [gene_id for gene_id in gene_ids if gene_id in strand_gene_id_set]
    gene_lookup = {
        row.gene_id: (int(row.start_0), int(row.end_0), int(row.gene_length))
        for row in strand_genes.itertuples(index=False)
    }
    gene_name_lookup = {row.gene_id: (row.gene_name or "") for row in strand_genes.itertuples(index=False)}
    isoquant_lookup = {row.read_id: row._asdict() for row in isoquant_assignments.itertuples(index=False)}

    rows: list[dict[str, object]] = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        if chromosome not in bam.references:
            logger.warning("[%s] %s not found in %s. Writing empty TSV.", sample, chromosome, bam_path.name)
            empty_reads_df(strand_gene_ids).to_csv(output_path, sep="\t", index=False)
            return

        total_reads = 0
        skipped_unmapped = 0
        skipped_intronic = 0
        skipped_strand = 0

        for read in bam.fetch(chromosome):
            total_reads += 1
            if read.is_unmapped:
                skipped_unmapped += 1
                continue

            cig = read.cigartuples
            if cig is not None:
                for op, length in cig:
                    if op == 3 and length > 0:
                        skipped_intronic += 1
                        break
                if any(op == 3 and length > 0 for op, length in cig):
                    continue

            read_strand = "-" if read.is_reverse else "+"
            if read_strand != strand:
                skipped_strand += 1
                continue

            chr_start = int(read.reference_start)
            chr_end = int(read.reference_end)
            row = build_read_row(read, sample, chromosome, strand)
            read_id = read.query_name
            if read_id in isoquant_lookup:
                iso_row = isoquant_lookup[read_id]
                row["isoquant_assignment"] = iso_row.get("assignment_type")
                row["isoquant_strand"] = iso_row.get("isoquant_strand")
                row["isoquant_ENST"] = iso_row.get("isoquant_ENST")
                row["isoquant_ENSG"] = iso_row.get("isoquant_ENSG")
            else:
                row["isoquant_assignment"] = None
                row["isoquant_strand"] = None
                row["isoquant_ENST"] = None
                row["isoquant_ENSG"] = None

            overlapping_gene_ids: list[str] = []
            for gene_id in strand_gene_ids:
                gene_start, gene_end, gene_length = gene_lookup[gene_id]
                overlap_bases = read_overlap_bases(
                    chr_start,
                    chr_end,
                    gene_start,
                    gene_end,
                )
                overlap_pct = (overlap_bases / gene_length * 100) if gene_length > 0 else 0.0
                row[f"{gene_id}"] = overlap_bases
                row[f"{gene_id} (%)"] = overlap_pct
                if overlap_bases > 0:
                    overlapping_gene_ids.append(gene_id)

            row["overlapping_gene_ids"] = ",".join(overlapping_gene_ids)
            row["overlapping_gene_names"] = ",".join(
                [gene_name_lookup.get(gene_id, "") for gene_id in overlapping_gene_ids if gene_name_lookup.get(gene_id, "")]
            )
            row["n_overlapping_gene_ids"] = len(overlapping_gene_ids)
            rows.append(row)

    if rows:
        reads_df = pd.DataFrame(rows)
    else:
        reads_df = empty_reads_df(strand_gene_ids)

    ordered_columns = [
        "Chr",
        "ChrStart",
        "ChrEnd",
        "read_id",
        "MAPQ",
        "strand",
        "read_length",
        "alignment_length",
        "reference_length",
        "polyA_length",
        "isoquant_assignment",
        "isoquant_strand",
        "isoquant_ENST",
        "isoquant_ENSG",
        "sample",
    ]
    for gene_id in strand_gene_ids:
        ordered_columns.append(f"{gene_id}")
        ordered_columns.append(f"{gene_id} (%)")
    ordered_columns.extend(["overlapping_gene_ids", "overlapping_gene_names", "n_overlapping_gene_ids"])

    reads_df = reads_df.reindex(columns=ordered_columns)
    reads_df.to_csv(output_path, sep="\t", index=False)

    logger.info("[%s] wrote %d reads to %s", sample, len(reads_df), output_path)
    logger.debug(
        "[%s] %s summary: scanned=%d unmapped=%d intronic=%d strand_mismatch=%d kept=%d",
        sample,
        bam_path.name,
        total_reads,
        skipped_unmapped,
        skipped_intronic,
        skipped_strand,
        len(reads_df),
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    sample = args.sample
    chromosome = args.chromosome
    sample_dir = Path(args.base_dir).expanduser() / sample

    forward_bam = Path(args.forward_bam).expanduser()
    reverse_bam = Path(args.reverse_bam).expanduser()
    output_forward = forward_bam.with_name(f"{forward_bam.stem}.reads.ext.tsv")
    output_reverse = reverse_bam.with_name(f"{reverse_bam.stem}.reads.ext.tsv")

    isoquant_assign = (
        Path(args.isoquant_assignments).expanduser()
        if args.isoquant_assignments
        else None
    )

    for bam_path in [forward_bam, reverse_bam]:
        if not bam_path.exists():
            raise FileNotFoundError(f"Input BAM not found: {bam_path}")

    gtf_path = Path(args.gtf).expanduser()
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    genes = load_gtf_genes(gtf_path, chromosome)
    if genes.empty:
        raise ValueError(f"No gene records found in {gtf_path} for chromosome {chromosome}.")

    isoquant_assignments = load_isoquant_assignments(isoquant_assign, chromosome)
    all_gene_ids = genes["gene_id"].tolist()

    logger.info("[%s] parsing GTF: %s", sample, gtf_path)
    logger.info("[%s] loaded %d gene records", sample, len(genes))
    logger.info("[%s] using IsoQuant assignments: %s", sample, isoquant_assign)

    process_bam(
        bam_path=forward_bam,
        output_path=output_forward,
        sample=sample,
        chromosome=chromosome,
        strand="+",
        genes=genes,
        gene_ids=all_gene_ids,
        isoquant_assignments=isoquant_assignments,
    )
    process_bam(
        bam_path=reverse_bam,
        output_path=output_reverse,
        sample=sample,
        chromosome=chromosome,
        strand="-",
        genes=genes,
        gene_ids=all_gene_ids,
        isoquant_assignments=isoquant_assignments,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
