#!/usr/bin/env bash

set -euo pipefail


ACCESSIONS=(
    SRR3458591
    SRR3458592
    SRR3458593
    SRR10315320
    SRR10315321
    SRR10315322
    SRR14510018
    SRR14510019
)


JOBS=4
THREADS=4


REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$REPO"

SRA_DIR="$BASE_DIR/sra"
FASTQ_DIR="$BASE_DIR/fastq"
TMP_DIR="$BASE_DIR/tmp"

mkdir -p "$SRA_DIR"
mkdir -p "$FASTQ_DIR"
mkdir -p "$TMP_DIR"


is_complete() {

    local ACC="$1"
    local GZ=""

    for CAND in "$FASTQ_DIR/${ACC}.fastq.gz" "$FASTQ_DIR/${ACC}_1.fastq.gz"; do
        if [[ -s "$CAND" ]]; then
            GZ="$CAND"
            break
        fi
    done

    [[ -n "$GZ" ]] || return 1

    shopt -s nullglob
    local LEFTOVER=( "$FASTQ_DIR"/"${ACC}"*.fastq )
    shopt -u nullglob
    [[ ${#LEFTOVER[@]} -eq 0 ]]
}


TODO=()

for ACC in "${ACCESSIONS[@]}"; do
    if is_complete "$ACC"; then
        echo "present:  $ACC  -- skipping"
    else
        TODO+=( "$ACC" )
    fi
done


process_one() {

    ACC="$1"

    echo "[$(date '+%H:%M:%S')] STARTING $ACC"


    shopt -s nullglob
    FASTQ_FILES=( "$FASTQ_DIR"/"${ACC}"*.fastq )
    shopt -u nullglob

    if [[ ${#FASTQ_FILES[@]} -gt 0 ]]; then

        echo "[$(date '+%H:%M:%S')] $ACC found uncompressed FASTQ -- finishing compression"

    else

        echo "[$(date '+%H:%M:%S')] $ACC downloading..."

        SUCCESS=0

        for TRY in 1 2 3; do

            echo "[$(date '+%H:%M:%S')] $ACC download attempt $TRY/3"

            if prefetch \
                --progress \
                --max-size u \
                -O "$SRA_DIR" \
                "$ACC"
            then
                SUCCESS=1
                break
            fi

            echo "[$(date '+%H:%M:%S')] WARNING: $ACC download failed"

            sleep 10

        done


        if [[ "$SUCCESS" -ne 1 ]]; then
            echo "[$(date '+%H:%M:%S')] ERROR: $ACC failed after 3 attempts"
            return 1
        fi


        SRA_FILE="$SRA_DIR/$ACC/$ACC.sra"


        if [[ ! -f "$SRA_FILE" ]]; then

            echo "[$(date '+%H:%M:%S')] ERROR: SRA file not found:"
            echo "$SRA_FILE"

            return 1

        fi


        echo "[$(date '+%H:%M:%S')] $ACC download complete"

        ls -lh "$SRA_FILE"


        rm -f "$FASTQ_DIR/${ACC}.fastq.gz" "$FASTQ_DIR/${ACC}"_*.fastq.gz


        ACC_TMP="$TMP_DIR/$ACC"

        rm -rf "$ACC_TMP"
        mkdir -p "$ACC_TMP"


        echo "[$(date '+%H:%M:%S')] $ACC converting to FASTQ..."


        if ! fasterq-dump \
            "$SRA_FILE" \
            --split-files \
            --threads "$THREADS" \
            --outdir "$FASTQ_DIR" \
            --temp "$ACC_TMP" \
            --progress
        then

            echo "[$(date '+%H:%M:%S')] ERROR: fasterq-dump failed for $ACC"

            rm -rf "$ACC_TMP"

            return 1

        fi


        rm -rf "$ACC_TMP"


        echo "[$(date '+%H:%M:%S')] $ACC FASTQ conversion complete"


        shopt -s nullglob
        FASTQ_FILES=( "$FASTQ_DIR"/"${ACC}"*.fastq )
        shopt -u nullglob


        if [[ ${#FASTQ_FILES[@]} -eq 0 ]]; then

            echo "[$(date '+%H:%M:%S')] ERROR: No FASTQ files for $ACC"

            return 1

        fi

    fi


    echo "[$(date '+%H:%M:%S')] $ACC compressing FASTQ..."


    for FILE in "${FASTQ_FILES[@]}"; do

        echo "[$(date '+%H:%M:%S')] $ACC compressing $(basename "$FILE")"

        pigz \
            -f \
            -p "$THREADS" \
            -v \
            "$FILE"

    done


    echo "[$(date '+%H:%M:%S')] $ACC removing SRA copy..."

    rm -rf "$SRA_DIR/$ACC"


    echo
    echo "[$(date '+%H:%M:%S')] COMPLETED $ACC"

    find "$FASTQ_DIR" \
        -maxdepth 1 \
        -type f \
        -name "${ACC}*.fastq.gz" \
        -exec ls -lh {} \;

    echo

}


FASTQ_FAILED=0

if [[ ${#TODO[@]} -eq 0 ]]; then

    echo
    echo "All ${#ACCESSIONS[@]} FASTQ runs already present."

else

    for PROGRAM in prefetch fasterq-dump pigz parallel; do

        if ! command -v "$PROGRAM" >/dev/null 2>&1; then
            echo "ERROR: $PROGRAM not found (needed for ${TODO[*]})."
            exit 1
        fi

    done


    echo
    echo "============================================================"
    echo "Parallel SRA downloader"
    echo "============================================================"
    echo
    echo "To download:        ${#TODO[@]} of ${#ACCESSIONS[@]}  (${TODO[*]})"
    echo "Parallel jobs:      $JOBS"
    echo "Threads per job:    $THREADS"
    echo "Maximum CPU usage:  approximately $((JOBS * THREADS)) threads"
    echo
    echo "FASTQ output:"
    echo "  $FASTQ_DIR"
    echo
    echo "Temporary files:"
    echo "  $TMP_DIR"
    echo
    echo "============================================================"
    echo


    export -f process_one

    export SRA_DIR
    export FASTQ_DIR
    export TMP_DIR
    export THREADS


    parallel \
        --jobs "$JOBS" \
        --line-buffer \
        --tag \
        process_one {} \
        ::: "${TODO[@]}" || FASTQ_FAILED=1

fi


echo
echo "============================================================"
echo "References"
echo "============================================================"
echo

REF_FAILED=0

if ! command -v python3 >/dev/null 2>&1; then

    echo "ERROR: python3 not found -- cannot fetch references."
    echo "       Activate the utrscan environment (conda activate utrscan) and re-run."
    REF_FAILED=1

elif ! command -v bowtie2-build-s >/dev/null 2>&1 && ! command -v bowtie2-build >/dev/null 2>&1; then

    echo "ERROR: bowtie2-build not found -- needed to build the yeast index."
    echo "       Activate the utrscan environment (conda activate utrscan) and re-run."
    REF_FAILED=1

else

    ( cd "$REPO" && python3 scripts/remote/fetch_data.py --skip-fastq ) || REF_FAILED=1

fi


echo
echo
echo "============================================================"
echo "ALL JOBS FINISHED"
echo "============================================================"
echo

echo "FASTQ directory:"
echo "$FASTQ_DIR"

echo
echo "Files:"
echo

ls -lh "$FASTQ_DIR"/*.fastq.gz 2>/dev/null || true

echo
echo "Total compressed size:"
du -sh "$FASTQ_DIR"

echo

MISSING=()
for ACC in "${ACCESSIONS[@]}"; do
    is_complete "$ACC" || MISSING+=( "$ACC" )
done

if [[ ${#MISSING[@]} -gt 0 || "$FASTQ_FAILED" -ne 0 || "$REF_FAILED" -ne 0 ]]; then
    [[ ${#MISSING[@]} -gt 0 ]] && echo "INCOMPLETE FASTQs: ${MISSING[*]}"
    [[ "$REF_FAILED" -ne 0 ]] && echo "References: FAILED (see messages above)"
    echo "Re-run this script to retry; everything already present is skipped."
    echo "============================================================"
    exit 1
fi

echo "All FASTQs and references present."
echo "============================================================"
