#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage:
  $0 \\
    --outer-jobs <N> \\
    --inner-threads <N> \\
    --reference <reference.fa> \\
    --varscan-jar <VarScan.jar> \\
    --rna-edit-bed <RNA_editing.bed> \\
    --out-dir <output_dir> \\
    --bam-files <bam1> [bam2 ...]

Notes:
  --reference requires a readable FASTA index at <reference.fa>.fai

Example:
  $0 \\
    --outer-jobs 6 \\
    --inner-threads 4 \\
    --reference genome.fa \\
    --varscan-jar VarScan.jar \\
    --rna-edit-bed RNA_edit.bed \\
    --out-dir ./out \\
    --bam-files \\
    sample1.bam sample2.bam
EOF
}

OUTER_JOBS=""      # Number of BAM files to process in parallel
INNER_THREADS=""   # Number of threads for samtools per BAM file
REF_FA=""          # Reference genome file
VARSCAN_JAR=""     # Path to VarScan.jar
RNA_EDIT_BED=""    # BED file with RNA editing sites
OUT_DIR=""         # Output directory
BAM_FILES=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --outer-jobs)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --outer-jobs" >&2
                usage
                exit 1
            fi
            OUTER_JOBS="$2"
            shift 2
            ;;
        --inner-threads)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --inner-threads" >&2
                usage
                exit 1
            fi
            INNER_THREADS="$2"
            shift 2
            ;;
        --reference)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --reference" >&2
                usage
                exit 1
            fi
            REF_FA="$2"
            shift 2
            ;;
        --varscan-jar)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --varscan-jar" >&2
                usage
                exit 1
            fi
            VARSCAN_JAR="$2"
            shift 2
            ;;
        --rna-edit-bed)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --rna-edit-bed" >&2
                usage
                exit 1
            fi
            RNA_EDIT_BED="$2"
            shift 2
            ;;
        --out-dir)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: Missing value for --out-dir" >&2
                usage
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --bam-files)
            shift
            if [ "$#" -eq 0 ] || [[ "$1" == -* ]]; then
                echo "ERROR: Missing value for --bam-files" >&2
                usage
                exit 1
            fi
            while [ "$#" -gt 0 ] && [[ "$1" != -* ]]; do
                BAM_FILES+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                BAM_FILES+=("$1")
                shift
            done
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            BAM_FILES+=("$1")
            shift
            ;;
    esac
done

if [ -z "$OUTER_JOBS" ] || [ -z "$INNER_THREADS" ] || [ -z "$REF_FA" ] || [ -z "$VARSCAN_JAR" ] || [ -z "$RNA_EDIT_BED" ] || [ -z "$OUT_DIR" ]; then
    echo "ERROR: Missing required options." >&2
    usage
    exit 1
fi


error_flag=0
is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "$OUTER_JOBS"; then
    echo "ERROR: OUTER_JOBS must be a positive integer, got: $OUTER_JOBS" >&2
    exit 1
fi

if ! is_positive_integer "$INNER_THREADS"; then
    echo "ERROR: INNER_THREADS must be a positive integer, got: $INNER_THREADS" >&2
    exit 1
fi

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $1" >&2
        error_flag=1
    fi
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: File not found: $1" >&2
        error_flag=1
        return
    fi

    if [ ! -r "$1" ]; then
        echo "ERROR: File is not readable: $1" >&2
        error_flag=1
    fi

    if [[ "$1" == *.bam ]]; then
        if [ -r "${1}.bai" ] || [ -r "${1%.bam}.bai" ]; then
            return
        fi

        if [ -f "${1}.bai" ] || [ -f "${1%.bam}.bai" ]; then
            echo "ERROR: Index file is not readable for BAM: $1" >&2
            error_flag=1
        else
            echo "ERROR: Index file not found for BAM: $1" >&2
            error_flag=1
        fi
    fi
}

check_output_dir() {
    local out_dir="$1"
    local probe

    if ! mkdir -p "$out_dir"; then
        echo "ERROR: Could not create output directory: $out_dir" >&2
        error_flag=1
        return
    fi

    probe="${out_dir}/.bam2vcf_write_test.$$"
    if ! : > "$probe"; then
        echo "ERROR: Output directory is not writable: $out_dir" >&2
        error_flag=1
        return
    fi
    rm -f "$probe"
}

get_bam_prefix() {
    local bam="$1"
    local basename

    basename=$(basename "$bam")
    printf "%s\n" "${basename%.bam}"
}

check_duplicate_bam_prefixes() {
    local i
    local j
    local bam_i
    local bam_j
    local prefix_i
    local prefix_j

    for ((i = 0; i < ${#BAM_FILES[@]}; i++)); do
        bam_i="${BAM_FILES[$i]}"
        prefix_i=$(get_bam_prefix "$bam_i")
        for ((j = i + 1; j < ${#BAM_FILES[@]}; j++)); do
            bam_j="${BAM_FILES[$j]}"
            prefix_j=$(get_bam_prefix "$bam_j")
            if [ "$prefix_i" = "$prefix_j" ]; then
                echo "ERROR: Duplicate BAM output prefix: $prefix_i" >&2
                echo "       BAM 1: $bam_i" >&2
                echo "       BAM 2: $bam_j" >&2
                echo "       Output files would collide under: ${OUT_DIR}/${prefix_i}.f1804q20.*" >&2
                error_flag=1
            fi
        done
    done
}

echo "[`date`] Running preflight checks."

check_command samtools
check_command java
check_command bedtools
check_command awk

check_file "$REF_FA"
check_file "$VARSCAN_JAR"
check_file "$RNA_EDIT_BED"

if [ ! -r "${REF_FA}.fai" ]; then
    echo "ERROR: Reference index not found or not readable: ${REF_FA}.fai" >&2
    echo "       You can generate it with: samtools faidx $REF_FA" >&2
    error_flag=1
fi

if [ "${#BAM_FILES[@]}" -eq 0 ]; then
    echo "ERROR: No BAM files provided." >&2
    exit 1
fi

for BAM in "${BAM_FILES[@]}"; do
    check_file "$BAM"
done

check_duplicate_bam_prefixes
check_output_dir "$OUT_DIR"

if [ "$error_flag" -ne 0 ]; then
    echo "ERROR: Aborting due to failed preflight checks." >&2
    exit 1
fi

echo "[`date`] Preflight checks passed."

check_chrom_naming_compat() {
    local ref_fa="$1"
    local bed_file="$2"

    local ref_contig
    ref_contig=$(awk '/^>/{gsub(/^>/, "", $1); print $1; exit}' "$ref_fa")
    local bed_contig
    bed_contig=$(awk 'NF>=1 && $1 !~ /^#/{print $1; exit}' "$bed_file")

    if [ -z "${ref_contig}" ] || [ -z "${bed_contig}" ]; then
        echo "WARN: Could not infer chromosome naming style from reference or BED." >&2
        return
    fi

    if [[ "$ref_contig" == chr* && "$bed_contig" != chr* ]]; then
        echo "WARN: Reference contigs look like 'chr*' but RNA_editing.bed contigs do not. RNA-editing filtering may miss all sites." >&2
    elif [[ "$ref_contig" != chr* && "$bed_contig" == chr* ]]; then
        echo "WARN: Reference contigs do not use 'chr*' but RNA_editing.bed does. RNA-editing filtering may miss all sites." >&2
    fi
}

check_chrom_naming_compat "$REF_FA" "$RNA_EDIT_BED"

has_nonempty_file() {
    [ -s "$1" ]
}

bam_index_exists() {
    local bam="$1"
    [ -s "${bam}.bai" ] || [ -s "${bam%.bam}.bai" ]
}

run_stdout_to_file() {
    local output="$1"
    shift

    local tmp="${output}.tmp.$$"
    rm -f "$tmp"
    if "$@" > "$tmp"; then
        mv -f "$tmp" "$output"
    else
        rm -f "$tmp"
        return 1
    fi
}


run_one_bam() {
    local BAM="$1"
    local INNER_THREADS="$2"
    local REF_FA="$3"
    local VARSCAN_JAR="$4"
    local RNA_EDIT_BED="$5"
    local OUT_DIR="$6"

    local PREFIX
    PREFIX=$(get_bam_prefix "$BAM")

    local BAM_FILT="${OUT_DIR}/${PREFIX}.f1804q20.bam"
    local MPILEUP="${OUT_DIR}/${PREFIX}.f1804q20.mpileup"
    local VCF="${OUT_DIR}/${PREFIX}.f1804q20.vcf"
    local VCF_NO_EDIT="${OUT_DIR}/${PREFIX}.f1804q20.no_rna_editing.vcf"

    echo "[`date`] START: $PREFIX"

    # Step 1: Filter BAM
    if has_nonempty_file "$BAM_FILT"; then
        echo "[`date`] SKIP existing filtered BAM: $BAM_FILT"
    else
        echo "[`date`] Filtering BAM: $BAM -> $BAM_FILT"
        run_stdout_to_file "$BAM_FILT" \
            samtools view -@ "$INNER_THREADS" -b -F 1804 -q 20 "$BAM"
    fi

    # Step 2: Index BAM
    if bam_index_exists "$BAM_FILT"; then
        echo "[`date`] SKIP existing BAM index: $BAM_FILT"
    else
        echo "[`date`] Indexing BAM: $BAM_FILT"
        local BAM_INDEX="${BAM_FILT}.bai"
        local BAM_INDEX_TMP="${BAM_FILT}.tmp.$$.bai"
        rm -f "$BAM_INDEX_TMP"
        samtools index -@ "$INNER_THREADS" "$BAM_FILT" "$BAM_INDEX_TMP"
        mv -f "$BAM_INDEX_TMP" "$BAM_INDEX"
    fi

    # Step 3: Generate mpileup
    if has_nonempty_file "$MPILEUP"; then
        echo "[`date`] SKIP existing mpileup: $MPILEUP"
    else
        echo "[`date`] Generating mpileup: $BAM_FILT -> $MPILEUP"
        run_stdout_to_file "$MPILEUP" \
            samtools mpileup -B -q 20 -Q 20 -f "$REF_FA" "$BAM_FILT"
    fi

    # Step 4: Call SNVs with VarScan
    if has_nonempty_file "$VCF"; then
        echo "[`date`] SKIP existing VarScan VCF: $VCF"
    else
        echo "[`date`] Calling SNVs with VarScan: $MPILEUP -> $VCF"
        run_stdout_to_file "$VCF" \
            java -jar "$VARSCAN_JAR" mpileup2snp "$MPILEUP" \
                --min-coverage 8 \
                --min-var-freq 0.01 \
                --output-vcf 1 \
                --min-reads2 3
    fi

    # Step 5: Remove RNA editing sites
    if has_nonempty_file "$VCF_NO_EDIT"; then
        echo "[`date`] SKIP existing RNA-editing filtered VCF: $VCF_NO_EDIT"
    else
        echo "[`date`] Removing RNA editing sites: $VCF -> $VCF_NO_EDIT"
        run_stdout_to_file "$VCF_NO_EDIT" \
            bedtools intersect -header -v -a "$VCF" -b "$RNA_EDIT_BED"
    fi

    echo "[`date`] FINISHED: $PREFIX"
    echo "Output: $VCF_NO_EDIT"
}

pids=()
for BAM in "${BAM_FILES[@]}"; do
    # Start a background task
    run_one_bam "$BAM" "$INNER_THREADS" "$REF_FA" "$VARSCAN_JAR" "$RNA_EDIT_BED" "$OUT_DIR" &
    pids+=("$!")

    # Control parallelism: if the number of background jobs >= OUTER_JOBS, wait
    while [ "$(jobs -rp | wc -l)" -ge "$OUTER_JOBS" ]; do
        sleep 1
    done
done

# Wait for all background tasks to finish
job_failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        job_failed=1
    fi
done

if [ "$job_failed" -ne 0 ]; then
    echo "[`date`] ERROR: One or more BAM jobs failed." >&2
    exit 1
fi

echo "[`date`] ALL JOBS COMPLETED."
