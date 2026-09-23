# import basic modules
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# load new classification
df = pd.read_table("/lustre/bio_running/GTEx_Long_Project/metadata/GTEx.chrM_info_extended.forward.reads.classified.v2.tsv.new_classification.tsv")
print(df)

print()
print(df.new_class.value_counts())

print()
print(df[["sample", "new_class"]].value_counts())

print()
print("STARTING COMPUTATION...")
basepath = "/lustre/bio_running/GTEx_Long_Project/analysis"
samples = df["sample"].value_counts().index.to_list()

failed = []
for _,sample in enumerate(samples):
    print()
    print("\n######################################################################################################")
    print("######################################################################################################")
    print("######################################################################################################")
    print(sample, f"Total Processed samples: {_}/{len(samples)}")
    df_sample = df[df["sample"] == sample].copy()
    unique_reads = df_sample.read_id.value_counts().index.tolist()
    number_of_unique_reads = len(unique_reads)
    print("Shape of df-reads for current sample:", df_sample.shape)
    print("Unique number of reads:", number_of_unique_reads)
    print(df_sample.head(2))
    print("CHECKING number of reads:")
    if not df_sample.shape[0] == number_of_unique_reads:
        print("!!!!!!!!ERROR!!!!!!!!!!! ")
        break
    mitoBAM_fp = os.path.join(basepath, sample, f"{sample}.genome.chrM.forward.bam")
    print("Starting mito-BAM filepath:", mitoBAM_fp)
    for new_class in ["fully_cleaved", "partially_cleaved_short", "partially_cleaved_long"]:
        print(f"\n#####\nElaborating class: {new_class}")
        df_sample_current_class = df_sample.query(f"new_class == '{new_class}'")
        if df_sample_current_class.empty:
            print("!!! Problem with current class !!! Empty reads-list for this sample/class combination.")
            failed.append(f"{sample}|{new_class}")
            continue
        print(df_sample_current_class[["sample", "new_class"]].value_counts())
        reads_list_current_class = df_sample_current_class.read_id
        print(df_sample_current_class.shape)
        print(f"Number of reads for current class: {len(reads_list_current_class)}")
        out_mitoBAM_fp = mitoBAM_fp + f".{new_class}.bam"
        out_mitoBAM_readslist_fp = out_mitoBAM_fp + ".readslist"
        out_mitoBAM_depth_fp = out_mitoBAM_fp + ".depth.txt"
        print(f"- output mitoBAM for current class: {out_mitoBAM_fp}")
        print(f"- output mitoBAM readslist for current class: {out_mitoBAM_readslist_fp}")
        print(f"- output mitoBAM depth file for current class: {out_mitoBAM_depth_fp}")
        # write readslist on disk
        with open(out_mitoBAM_readslist_fp, "w") as out_mitoBAM_readslist:
            for read in reads_list_current_class:
                out_mitoBAM_readslist.write(read + "\n")
        os.system(f"cat {out_mitoBAM_readslist_fp} | wc -l")
        # create sub-mitoBAM file, indexing it, creating corresponding depth file
        os.system(f"samtools view -F 2308 -@ 24 -b -N {out_mitoBAM_readslist_fp} {mitoBAM_fp} > {out_mitoBAM_fp}")
        os.system(f"samtools index -@ 24 {out_mitoBAM_fp}")
        os.system(f"samtools depth -a -@ 24 {out_mitoBAM_fp} > {out_mitoBAM_depth_fp}")
        os.system(f"samtools view -@ 24 {out_mitoBAM_fp} | cut -f 2,3 | sort | uniq -c | sort -nr")
        os.system(f"wc -l {out_mitoBAM_depth_fp}")

print("\n\n\nComputation Finished!")
print(f"Total Processed samples: {_+1}/{len(samples)}")
print(f"Failed sample|new_class combinations: {len(failed)}")