# import required modules
import os, pysam, argparse, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sn

# util function
def get_rev_comp(base):
    conv_dict = {"A":"T", "C":"G", "G":"C", "T":"A"}
    return conv_dict[base]


def get_alignment_profile_stranded(bam_chrM_stranded_filt_fp, ref_fp, region="chrM", start1based=None, stop1based=None,
                                   Ntrim=5, strand_expected="+", max_reads=8000, threads=1, save_path=None):
    # open reference file
    ref = pysam.FastaFile(ref_fp)

    # limits for start and end of ref seq based on trimming strategy
    if start1based <= Ntrim:
        start1based = Ntrim + 1
        print("Start from:", start1based)
    if stop1based >= ref.get_reference_length(region) - Ntrim:
        stop1based = ref.get_reference_length(region) - Ntrim - 1
        print("Ending to:", stop1based)
    # define final outtable
    out_table = []
    pos1based_to_explore = range(start1based, stop1based + 1)
    print(pos1based_to_explore)
    pbar_update = ceil(len(pos1based_to_explore) / 500)
    print(f"[{datetime.now()}] Printing pbar updates every {pbar_update} iterations.")
    start_time = datetime.now()
    with tqdm(total=len(pos1based_to_explore), file=sys.stdout) as pbar:
        with pysam.AlignmentFile(bam_chrM_stranded_filt_fp, threads=threads) as bam_chrM_stranded_filt:
            # for each position to be investigated, open an interval to asses and and beginning of reads
            for _, pos1based in enumerate(pos1based_to_explore):
                out_table_row_current_site = []
                ref_base = ref.fetch(region, pos1based - 1, pos1based)
                if ref_base is "N": continue
                region_seqs = []
                region_quals = []
                pos1based_trim_start = pos1based - Ntrim + 1
                if pos1based_trim_start < 1:
                    pos1based_trim_start = 1
                pos1based_trim_stop = pos1based + Ntrim - 1
                if pos1based_trim_stop > ref.get_reference_length(region):
                    pos1based_trim_stop = ref.get_reference_length(region)
                # open interval around the pos1based to explore in order to assess if it falls close to the extremes of a read
                for pileupcolumn in bam_chrM_stranded_filt.pileup(region, pos1based_trim_start - 1, pos1based_trim_stop,
                                                                  truncate=True, max_depth=max_reads, min_base_quality=0):  # ^ and $ for start and end of reads respectively
                    seqs = pileupcolumn.get_query_sequences(mark_matches=True, add_indels=True, mark_ends=True)
                    if "^" in seqs or "$" in seqs:
                        seqs = [s for s in seqs if not s in ("^", "$")]  # remove artifacts due to mark-ends
                    qnames = pileupcolumn.get_query_names()
                    quals = pileupcolumn.get_query_qualities()
                    region_seqs.append(pd.DataFrame(seqs, index=qnames, columns=[pileupcolumn.reference_pos + 1]))
                    region_quals.append(pd.DataFrame(quals, index=qnames, columns=[pileupcolumn.reference_pos + 1]))
                    del (seqs)
                    del (qnames)
                    del (quals)
                # aggregate into two pandas dataframe
                region_seqs = pd.concat(region_seqs, axis=1)
                region_quals = pd.concat(region_quals, axis=1)
                # select only reads without start and end markers within seqs, and also <> reference skip chars (for forw and rev strand)
                region_seqs = region_seqs[~region_seqs.apply(lambda c: c.str.contains(r'[\^\$<>]', regex=True)).any(axis=1)].copy()
                # select column of interest
                try:
                    region_seqs = region_seqs[pos1based].dropna()
                except:
                    continue
                # retrieve number bases bringing insertions
                ins_n = region_seqs[region_seqs.str.contains("[+]", regex=True)].shape[0]
                # retrieve coverage
                coverage = region_seqs.shape[0]
                if coverage > 0:
                    # eliminate + - operations and make everything upper
                    region_seqs = region_seqs.apply(lambda c: c[0].upper()).copy()
                    # select qualities for the same above computed subset of reads
                    region_quals = region_quals.loc[region_seqs.index].copy()
                    region_quals = region_quals[pos1based]
                    if not region_seqs.shape == region_quals.shape: print("ERROR!!!!")  ### inner control row
                    region_seqs = pd.concat([region_seqs, region_quals], axis=1)
                    del (region_quals)  # delete to spare time
                    region_seqs.columns = ["Seq_" + str(region_seqs.columns[0]), "Q_" + str(region_seqs.columns[1])]
                    # change quality of deletions to nan
                    region_seqs.loc[region_seqs[f"Seq_{pos1based}"] == "*", f"Q_{pos1based}"] = np.nan
                    if strand_expected == "-":
                        ref_base = get_rev_comp(ref_base)
                    out_table_row_current_site += [pos1based, strand_expected, ref_base, coverage]
                    # add frequencies using coverage to normalize counts
                    counts = region_seqs[f"Seq_{pos1based}"].value_counts()
                    counts = pd.concat([pd.DataFrame([], index=["A", "C", "G", "T", "*"]), counts.to_frame()], axis=1)
                    counts.fillna(0, inplace=True)
                    counts = counts / coverage  # normalize with coverage
                    for base in "ACGT*":
                        if strand_expected == "-" and base != "*":
                            base = get_rev_comp(base)
                        out_table_row_current_site.append(counts[f"Seq_{pos1based}"].loc[base])
                    out_table_row_current_site.append(
                        ins_n / coverage)  # add number of bases inserition normalized on coverage
                    # add average quality for each base
                    avgBaseQs = region_seqs.groupby(
                        f"Seq_{pos1based}").mean().dropna()  # dropna of eventual * deletions
                    avgBaseQs = pd.concat([pd.DataFrame([], index=["A", "C", "G", "T"]), avgBaseQs], axis=1)
                    for base in "ACGT":
                        if strand_expected == "-":
                            base = get_rev_comp(base)
                        out_table_row_current_site.append(avgBaseQs[f"Q_{pos1based}"].loc[base])
                    out_table_row_current_site.append(region_seqs[f"Q_{pos1based}"].mean())
                    out_table.append(out_table_row_current_site)
                if _ >= 0 and _ % pbar_update == 0:
                    pbar.update(pbar_update)

    ref.close()
    # collect as pandas dataframe
    out_table = pd.DataFrame(out_table, columns=["pos1based", "strand", "ref", "cov", "Af", "Cf", "Gf", "Tf",
                                                 "Df", "If", "AqAvg", "CqAvg", "GqAvg", "TqAvg", "qAvg"])
    Err = []
    for i in out_table.index:
        row = out_table.loc[i]
        Err.append(1 - row[f"{row.ref}f"])
    out_table["Err"] = Err
    print(f"[{datetime.now()}] Computation Finished. Elapsed time: {datetime.now() - start_time}", flush=True)

    if save_path:
        print(f"[{datetime.now()}] Saving on disk error/alignment out-table at: {save_path}")
        out_table.to_csv(save_path, index=False, sep="\t")
    else:
        return out_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ComputeErrorsTabels.py")
    parser.add_argument("--bam_chrM_stranded_filt_fp",
                        required=True,
                        type=str,
                        help="--bam_chrM_stranded_filt_fp: \t a <str> with the fullpath for the input stranded and filtered chrM BAM file.")
    parser.add_argument("--ref_fp",
                        required=True,
                        type=str,
                        help="--ref_fp: \t a <str> with the fullpath for the used reference fasta file.")
    parser.add_argument("--region",
                        required=True,
                        type=str,
                        help="--region: \t a <str> with the region/chromosome name of interest.")
    parser.add_argument("--start1based",
                        required=True,
                        type=int,
                        help="--start1based: \t a <int> with the start position (1-based) of interest.")
    parser.add_argument("--stop1based",
                        required=True,
                        type=int,
                        help="--stop1based: \t a <int> with the stop position (1-based) of interest.")
    parser.add_argument("--Ntrim",
                        required=False,
                        type=int,
                        default=5,
                        help="--Ntrim: \t a <int> with the number of bases to be excluded from counting close N bases from the end/beginning of the read")
    parser.add_argument("--strand_expected",
                        required=True,
                        type=str,
                        help="--strand_expected: \t a <str> with the expected strand of the read [allowed values: + or -]")
    parser.add_argument("--max_reads",
                        required=False,
                        default=8000,
                        type=int,
                        help="--max_reads: \t a <int> with the maximum number of reads to randomly sample for each query position")
    parser.add_argument("--threads",
                        required=False,
                        type=int,
                        default=1,
                        help="--threads: \t a <int> with the number of threads to use to speed-up pysam queries and computation")
    parser.add_argument("--save_path",
                        required=True,
                        type=str,
                        help="--save_path: \t a <str> with the full path for the output TSV file containing the Error-Table")

    args = parser.parse_args()
    bam_chrM_stranded_filt_fp = args.bam_chrM_stranded_filt_fp
    ref_fp = args.ref_fp
    region = args.region
    start1based = args.start1based
    stop1based = args.stop1based
    Ntrim = args.Ntrim
    strand_expected = args.strand_expected
    max_reads = args.max_reads
    threads = args.threads
    save_path = args.save_path

    if not strand_expected in ["+", "-"]:
        print(f"[{datetime.now()}] ERROR! Strand has to be either + or -. Exiting", flush=True)
        sys.exit()

    get_alignment_profile_stranded(bam_chrM_stranded_filt_fp=bam_chrM_stranded_filt_fp,
                                   ref_fp=ref_fp,
                                   region=region,
                                   start1based=start1based,
                                   stop1based=stop1based,
                                   Ntrim=Ntrim,
                                   strand_expected=strand_expected,
                                   max_reads=max_reads,
                                   threads=threads,
                                   save_path=save_path)

