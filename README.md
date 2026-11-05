The study asks which aspects of 5′UTR scanning can be statistically recovered from 40S footprinting
(TCP-seq and eIF3B-selective TCP-seq), at what spatial resolution, and at what sequencing depth. It then
asks whether the recoverable signal improves the prediction of reporter translation beyond sequence
alone. Failures to recover an effect are treated as measurable properties of the assay: every negative
result is paired with a power analysis that says whether more depth, more replicates or a different
observable would change it.

The repository holds the complete analysis, from raw reads to every figure and table, and the results
reported in the manuscript (`results/recoverability/v1/`). Raw reads and reference genomes are public
and are not included. One script downloads them (see [Data](#data)).


## Approach

1. **Reads to footprint positions.** Adapter and quality trimming, alignment, unique reads only, and a
   read-length-resolved P-site rule measured from start-codon metagenes.
2. **Transcript panels.** One annotation per species (leader, main AUG and 30 nt of CDS, with sequence
   and RNA structure tracks) and one count layer per dataset, so both replicates use identical
   coordinates.
3. **Where do 40S reads fall within a leader?** A conditional multinomial model over positions. A
   positional baseline is extended by one feature family at a time: local sequence, RNA structure,
   start-codon context and uORFs.
4. **Recoverability test.** Held-out-chromosome gain in log-likelihood per read (ΔLL/read) over the
   position-only model, transferred across independent replicates, with transcript-bootstrap confidence
   intervals and position-matched permutation nulls, at 1 to 20 nt resolution.
5. **Power and depth.** Synthetic data with planted effects on the real transcripts and depth
   distribution, binomial thinning of the real libraries, and an analytic Fisher-information
   approximation of the probability that an effect is recovered.
6. **Mechanism.** Fisher-information identifiability of the parameters of an absorbing Markov chain of
   40S scanning, and a comparison of a constrained mechanistic scanner with a neural rate model.
7. **Experimental design.** The depth and number of replicate libraries a new experiment needs, and which
   additional observable resolves a given mechanistic ambiguity.
8. **Reporter utility.** Whether the recovered 40S information explains reporter (MPRA) translation
   beyond sequence, in yeast and human.

Each effect and resolution ends in one status (written by `summary.py`):

| Status | Meaning |
|---|---|
| `recoverable` | held-out ΔLL/read CI excludes 0 in both cross-replicate directions, and beats the position-matched null (p ≤ 0.05) wherever a null was run at that bin width |
| `coarse_resolution_recoverable` | fails at 1 nt, recoverable at a bin width of 5 nt or more |
| `dataset_specific` | CI excludes 0 within a replicate but not across replicates |
| `depth_limited` | not recoverable, synthetic power at the observed median depth below 0.8, but at least 0.8 at a higher simulated depth |
| `not_recoverable_at_depth` | not recoverable although synthetic power at the observed depth is at least 0.8 |
| `fails_position_matched_null` | CI excludes 0 but the matched null explains it (for example structure carried by local GC) |
| `null_pending` | passes the CI criterion but no null was run at this bin width, never reported as recoverable |
| `inconclusive` | none of the above can be decided (for example no synthetic counterpart) |
| `structurally_unidentifiable` | zero Fisher information at any depth for that observation |

The power threshold (0.8) and the reference effect size are written into every row, so the statuses can
be re-derived under other choices.

## Repository layout

```
analysis/recoverability/    the study
  config.py                 datasets, accessions, grids, seeds, results path, hardware detection
  run_all.py                main pipeline, raw reads to figures (resumable)
  run_missing_evaluations.py  evaluations added after the main run (see below)
  00_ ... 12_*.py           one script per analysis step (table in "Analysis steps")
  panel.py, psite.py, splits.py, features.py, likelihood.py, gpu.py, bootstrap.py
                            panels, P-site rule, chromosome folds, feature tables, multinomial model,
                            GPU backend, transcript bootstrap
  chainfast.py              batched closed-form expected visits of the scanning chain
  design_engine.py          leave-one-library-out design engine
  plan_experiment.py        command-line planner: depth and replicates for a new experiment
  plan_observable.py        command-line planner: which observable resolves a mechanism
  summary.py                one status per effect and resolution
  make_figures.py, make_missing_figures.py, make_theory_figures.py   figures
  missing_eval.py           input audit, unselected-library table, headline report
  tests/                    unit tests (pytest)
pipeline/                   trimming, alignment, 5′-end counting, sequence extraction helpers
data/                       coverage projection, sequence and structure features, synthetic counts
model/                      absorbing Markov chain of 40S scanning (chain.py) and RateNet (rate_net.py)
scripts/                    reference download, human 5′UTR annotation, yeast leader calling
scripts/remote/             data download, server setup, detached pipeline runner
eval/                       reporter (MPRA) tables and the scripts that build them
results/recoverability/v1/  frozen results reported in the manuscript
download_srrs.sh            alternative FASTQ download through the SRA toolkit
environment.yml             conda environment (pinned to the versions the results were produced with)
requirements.txt            the same Python packages for pip
```

The evaluations in `run_missing_evaluations.py` (unselected-library nulls and power, the
selected-versus-unselected comparison, the yeast single-library summary and the final reporter
protocol) were added after the main run was frozen. They only add files and never overwrite a
main-run result. Their outputs live in `missing_eval/` and in new, non-colliding files in `nulls/`,
`synthetic/` and `downstream/`.

## Installation

The results were produced on Linux x86_64 with Python 3.12. The figure scripts, the planners and the
unit tests also run on Windows.

**Conda (recommended).** The environment includes bowtie2, cutadapt, ViennaRNA and the SRA toolkit.
PyTorch is installed separately so that you get the right build for your hardware.

```bash
conda env create -f environment.yml
conda activate utrscan
pip install "torch>=2.5"                                                   # NVIDIA GPU
pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu  # CPU only
pip install pytest                                                         # for the unit tests
```

On a fresh Linux server, `bash scripts/remote/bootstrap.sh` does all of this. It installs Miniforge if
no conda is present, creates the environment, picks the PyTorch build, checks every tool and runs the
unit tests.

**pip.** `pip install -r requirements.txt` installs the Python packages. bowtie2 is not a Python
package. Install it from bioconda (`conda install -c bioconda bowtie2`) or, on Windows, unzip the
official `bowtie2-<version>-mingw-x86_64.zip` anywhere under `bin/`, which is searched before `PATH`.

Do not run the pipeline in conda's `base` environment. `run_all` refuses to, because results produced
there were shown to differ (set `UTR_ALLOW_ANY_ENV=1` to override).

## Quick start without raw data

Everything below reads only the committed results and runs in about a minute.

```bash
python -m analysis.recoverability.make_figures            # main figures
python -m analysis.recoverability.make_missing_figures    # figures of the added evaluations
python -m analysis.recoverability.make_theory_figures     # theory validation, power curves with the analytic prediction
python -m analysis.recoverability.summary --dataset gse174329_eif3b   # recomputes summary/recoverability_summary.tsv
```

These write into `results/recoverability/v1/` and replace the committed files there, so `git diff`
shows any difference. `summary` reproduces the committed status table byte for byte. To leave the
committed files untouched, copy the results tree and point `UTR_RESULTS` at the copy.

The mechanism planner needs no data at all:

```bash
python -m analysis.recoverability.plan_observable --mechanism commitment_dropoff \
    --candidates 80S_known,80S_unknown,cds30
```

Mechanisms are `commitment_dropoff`, `leak_blur` and `reinit_direct`. The depth planner
(`plan_experiment.py`) needs the dataset panels, which the pipeline builds from the raw reads:

```bash
python -m analysis.recoverability.plan_experiment --pilot gse174329 --effect SW \
    --effect-size 0.1 --bin 10 --budget-reads 5e7 --replicates 2,3,4,5,6
```

## Data

### 40S footprinting

| Dataset key | Study | Assay | Libraries (GSM / SRX, run) | Role |
|---|---|---|---|---|
| `gse174329_eif3b` | GSE174329 | human eIF3B Sel-TCP-seq | rep1 GSM5291862, SRR14510018. rep2 GSM5291863, SRR14510019 | primary dataset with independent replicates |
| `gse139131_unsel` | GSE139131 | human unselected 40S TCP-seq | rep1 GSM4131410, SRR10315320. rep2 GSM4131411, SRR10315321 | assay noise ceiling, unselected versus selected |
| `gse139131_eif3b` | GSE139131 | human eIF3B Sel-TCP-seq | GSM4131412, SRR10315322 (single library) | deep scanning-enriched discovery dataset |
| `archer_yeast` | SRP074093 | yeast TCP-seq, small subunit | SRX1734358, SRR3458591 (single library) | cross-species validation |

The yeast study also provides the 80S fraction (SRX1734359, SRR3458592) and total-RNA input
(SRX1734360, SRR3458593). Input coverage is used to call yeast leader boundaries.

### Reporter assays (MPRA)

| Dataset key | Source | Readout | Files in `eval/` |
|---|---|---|---|
| `akirtava_yeast` | Akirtava et al., *Nucleic Acids Research*, gkaf165, supplementary workbook | TEff, MeanYFP | `akirtava_scer_wt_downstream.tsv` (final protocol) with its exclusion log, `akirtava_scer_wt.tsv` (exploratory analysis), source files in `gkaf165_supplemental_files/` |
| `sample2019_human` | Sample et al. 2019, GEO GSE114002, GSM4084997 (`varying_length_25to100`) | mean ribosome load | `sample2019.tsv` |

`eval/prepare_akirtava_downstream.py` and `eval/prepare_akirtava.py` build the yeast tables from the
workbook. `python eval/prepare_sample2019.py --in GSM4084997_varying_length_25to100.csv.gz` builds the
human table from the GEO file.

### References

| Species | Genome and annotation | Index | Transcripts |
|---|---|---|---|
| human | GENCODE release 49 primary-assembly FASTA and GTF, filtered to MANE Select | prebuilt bowtie2 GRCh38 no-alt analysis set (coordinates identical to the primary assembly on chr1 to 22, X, Y, M) | MANE Select, last 5′UTR exon |
| yeast | Ensembl R64-1-1, release 115 | bowtie2, built locally from the same FASTA | start codons, leaders from Input coverage |

rRNA sequences (human 45S and 5S from NCBI, the yeast rDNA locus cut from the genome) are used only to
measure rRNA contamination in the raw-read inspection.

### Download

```bash
python scripts/remote/fetch_data.py
```

This fetches the 8 FASTQ files from ENA (accepted only if the MD5 matches ENA's), both references, the
bowtie2 indexes and the rRNA sequences into `fastq/` and `ref/`, and writes every URL, size and
checksum to `ref/download_manifest.json`. It is resumable and skips what is already present.
`bash download_srrs.sh` is an alternative that downloads the FASTQs through the SRA toolkit in parallel
and then calls `fetch_data.py --skip-fastq` for the references.

Raw reads take about 22 GB and the references about 12 GB. Plan for 80 GB of free disk for a full run.

## Reproducing the full analysis

```bash
conda activate utrscan
python scripts/remote/fetch_data.py
python -m analysis.recoverability.run_all                   # 1. main study
python -m analysis.recoverability.run_missing_evaluations   # 2. added evaluations
python -m analysis.recoverability.11_recoverability_theory  # 3. analytic recoverability theory
python -m analysis.recoverability.make_theory_figures
python -m analysis.recoverability.12_experiment_design      # 4. experimental design
```

The order matters. The theory uses the synthetic power simulations of steps 1 and 2, and the design
step uses the theory.

By default every step writes to `results/recoverability/v1/`, the committed tree, so a rerun replaces
the frozen files and `git diff` shows what changed. To write to a separate tree instead, set
`UTR_RESULTS=results/rerun` (or any path) for all five commands.

**Resuming.** Each finished step writes a marker to `.done/<step>` (for the added evaluations,
`missing_eval/.done/<step>`), and a rerun skips it. A failed step has no marker. Both orchestrators take
`--list` (the plan and which steps are done), `--from <step>`, `--only <steps>` and `--force`.
`run_missing_evaluations --smoke` runs the whole chain on tiny counts to test the wiring. Every step logs
to `logs/<step>.log`, and `logs/run_all.log` keeps a running summary.

**On a server.** After `bash scripts/remote/bootstrap.sh`, `bash scripts/remote/run_pipeline.sh` runs
`run_all` detached from the SSH session. `run_pipeline.sh status` shows progress, `run_pipeline.sh stop`
stops it, and `run_pipeline.sh -- --from nulls_P_b1` passes options through.

**Hardware and run time.** The main run took about 3 hours on 8 CPU cores, 63 GB of RAM and one 48 GB
NVIDIA GPU. The added evaluations took 4.1 hours of step time on 20 cores, 236 GB of RAM and an 80 GB
GPU. The minimum is 16 GB of RAM. A GPU is optional: without one, model fitting falls back to an
identical SciPy implementation, and `tests/test_backends.py` checks that the two backends agree to
below 10⁻⁶ log-likelihood per read.

| Step | Device |
|---|---|
| model fits, nulls, synthetic power, depth curves (`likelihood.fit`) | GPU when present: float64 objective, float32 information matrix |
| chain ablation (constrained scanner, RateNet) | GPU when present |
| alignment (bowtie2), RNA folding (ViennaRNA), panel building | CPU, all cores |

## Analysis steps

Main run (`run_all.py`), in order:

| Step | Script | What it does | Output |
|---|---|---|---|
| align | `pipeline/02_align_count.py` | cutadapt piped into bowtie2, unique 5′ ends and spans per read | `work/` (not committed) |
| inspect | `00_inspect_reads.py` | first 100,000 reads per run: adapters, insert lengths, rRNA and unique-alignment rates | `qc/read_inspection.tsv`, `qc/insert_lengths.tsv` |
| offsets | `01_qc.py --stage offsets` | read-length-resolved P-site rule per replicate | `manifest.json` (`psite_offsets`), `qc/metagene.tsv` |
| annotation, counts | `panel.py` | annotation panel per species, count panel per dataset | `cache/recoverability/` (not committed) |
| qc_tables | `01_qc.py --stage tables` | read accounting, leader depth, sparsity, region fractions | `qc/qc_summary.tsv`, `qc/reads_per_transcript.tsv` |
| resolution | `02_replicate_resolution.py` | replicate agreement versus bin width (1 to 50 nt), by depth | `resolution/replicate_resolution.tsv` |
| effects | `03_nested_effects.py` | nested feature models, held-out chromosomes, all replicate directions | `effects/` |
| nulls | `04_shuffle_nulls.py` | position-matched permutation nulls per feature family and bin width | `nulls/` |
| identifiability | `06b_identifiability.py` | Fisher information of the chain's mechanistic parameters | `chain/identifiability*.tsv` |
| synthetic | `06_synthetic_power.py` | recovery power versus depth and effect size on planted truths | `synthetic/` |
| chain | `07_chain_ablation.py` | position-only versus constrained scanner versus RateNet with the chain | `chain/chain_ablation.tsv`, `chain/chain_params.tsv` |
| depth | `05_depth_curve.py` | binomial thinning, and matched depth selected versus unselected | `depth/depth_curve.tsv` |
| summary | `summary.py` | one status per effect and resolution | `summary/recoverability_summary.tsv` |
| figures | `make_figures.py` | main figures | `figures/` |
| downstream | `08_downstream_utility.py` | exploratory reporter analysis (superseded by the final protocol below) | `downstream/downstream_utility.tsv` |

Chromosome folds are frozen by `splits.py` on first use (`human_chromosome_folds.json`,
`yeast_chromosome_folds.json`).

Added evaluations (`run_missing_evaluations.py`):

| Step | Script | Output |
|---|---|---|
| input audit | `missing_eval.py audit` | `missing_eval/input_audit.tsv` |
| unselected-library nulls (1, 5, 10, 20 nt) and Dirichlet-multinomial power | `04_shuffle_nulls.py`, `06_synthetic_power.py` | `nulls/`, `synthetic/*_n50.tsv` |
| unselected-library statuses | `summary.py --out`, `missing_eval.py unselected_table` | `missing_eval/unselected_*.tsv` |
| selected versus unselected on the same transcripts, folds and depth | `09_assay_comparison.py --mode primary` | `missing_eval/assay_comparison.tsv` |
| same-study control (GSE139131 eIF3B versus unselected) | `09_assay_comparison.py --mode same_study` | `missing_eval/gse139131_same_study_selection.tsv` |
| yeast 20 nt nulls and multinomial power | `04_shuffle_nulls.py`, `06_synthetic_power.py` | `nulls/`, `synthetic/*_n50.tsv` |
| yeast single-library statuses | `10_yeast_single_library_summary.py` | `missing_eval/yeast_single_library_summary.tsv` |
| reporter utility, final protocol | `08_downstream_utility.py --stage choose`, `eval/prepare_akirtava_downstream.py`, `08_downstream_utility.py --stage run` | `downstream/model_choice.tsv`, `downstream/downstream_utility_final.tsv`, `downstream/reporter_exclusions.tsv`, `downstream/validation_checks.tsv`, `downstream/downstream_predictions.tsv.gz` |
| figures and headline report | `make_missing_figures.py`, `missing_eval.py report` | `figures/missing_eval_*`, `figures/fig9_reporter_utility.*`, `missing_eval/report.md` |

Theory and design:

| Script | What it does | Output |
|---|---|---|
| `11_recoverability_theory.py` | predicts recovery probability for every cell of the synthetic grid and validates it against the simulations | `theory/` |
| `make_theory_figures.py` | theory validation, power curves with the analytic prediction | `figures/theory_validation.*`, `figures/fig4_*_theory.*` |
| `12_experiment_design.py` | depth and replicate requirements, observables, retrospective pilots | `design/`, `figures/experiment_design.*` |

## Methods as implemented

**Read processing.** cutadapt with the dataset's 3′ adapter, quality trimming at 20 and a minimum
insert of 25 nt (human) or 18 nt (yeast). Yeast reads are additionally trimmed of the poly(A) stretch
that the TCP-seq protocol adds before the adapter. The GSE174329 reads are deposited already
adapter-trimmed, so only the quality and length filters act on them. Alignment with bowtie2
(`-L 18 -N 0`), reads with MAPQ ≥ 10 only. Per-read 5′ positions and spans are stored, so any P-site rule
can be applied without re-aligning.

**P-site assignment.** A single 5′-end offset does not fit selected 40S footprints: the start-codon
metagene of raw 5′ ends is a plateau of about 20 nt, while the 3′ end sits at +24 nt in every length bin.
`psite.py` therefore picks, per footprint-length bin, whichever end gives the sharper metagene peak,
and keeps the rule only if it beats the best single 5′ offset. The rule is measured on replicate 1 and
applied to both replicates. The rules are in `manifest.json` and `appendix/tables/psite_rules.tsv`.

**Leaders and structure.** Human leaders are the last 5′UTR exon of the MANE Select transcript.
Yeast leaders are called from Input coverage (`scripts/call_utr_boundaries.py`), and genes without a
call are excluded. Each stored sequence is the leader, the main AUG and 30 nt of exon-aware CDS. RNA
structure (base pairing and local ΔG in 40 nt windows) is folded with ViennaRNA on exactly that string,
so no window reaches into an intron.

**Model.** Footprint counts within a transcript follow a multinomial over positions (softmax of a
linear predictor), conditioned on the transcript's total, so abundance and loading drop out. The
positional baseline uses bins of distance to the AUG and to the 5′ end at the evaluation bin width,
capped at 300 nt, anchored so that the main-AUG bin covers the codon. Fits are penalised maximum
likelihood with a Gaussian prior (sd 1) on every coefficient, by Fisher scoring.

| Family | Features |
|---|---|
| `S1`, `S3`, `S5` | focal nucleotide, centred 3-mer, centred 5-mer |
| `SW` | additive ±10 nt sequence window |
| `ST` (`ST_paired`, `ST_dG`) | base pairing and local ΔG at the P site and 5 to 20 nt ahead of the 40S |
| `AUG` | start-codon identity (AUG and 9 near-cognates), −3 and +4 nucleotides, Kozak class, active only within ±2 nt of a start site |
| `UORF` | `AUG` plus the number of uAUGs passed, position inside a uORF frame and uORF geometry |
| combinations | `SW+ST`, `SW+AUG`, `ST+AUG`, `SW+ST+AUG` |

**Evaluation.** Chromosomes are split into depth-balanced folds (5 for human, 4 for yeast), frozen once
and reused by every script. A model is trained on the training chromosomes of one replicate (transcripts
with at least 10 leader reads) and scored on the held-out chromosomes of every replicate (transcripts
with at least 10 reads in every replicate, so all four directions use the same transcripts). Every
transcript is tested once, in its held-out fold. Confidence intervals come from bootstraps over
transcripts. The analysed bin widths are 1, 5, 10 and 20 nt.

**Nulls.** A global shuffle would be too easy to beat, because sequence composition and structure
change systematically from the cap to the AUG. Each null keeps position: sequence windows are permuted
among positions in the same distance-to-AUG and distance-from-5′-end strata, structure rows within the
same strata and the same local-GC quintile, and start-site features as whole units among start sites.
100 permutations per family and bin width.

**Synthetic power.** Everything is real except the counts: transcripts, sequences, structure,
chromosomes and the per-transcript depth distribution come from the dataset, and the positional truth is
the real position-only fit. Effects are planted on top at several sizes (truth A plants none and defines
the null threshold). Noise is multinomial or Dirichlet-multinomial, with the concentration calibrated by
simulation to reproduce the real replicate dispersion. Recovery requires the same criterion as the real
data.

**Depth.** Binomial thinning of each replicate keeps the multinomial structure exact. The matched
comparison thins selected and unselected libraries to the same number of leader reads.

**Mechanistic chain.** `model/chain.py` is an absorbing Markov chain in which a 40S loads at the cap and
at each nucleotide scans on, commits at a start codon, pauses or drops off. The expected-visits profile
is the predicted occupancy. `06b_identifiability.py` computes the Fisher information of its parameters
under binned occupancy, optionally with 80S counts: commitment versus drop-off (E1), leaky scanning
versus footprint blur (E2), reinitiation versus direct commitment (E3) and global loading (E4).
`07_chain_ablation.py` compares, on the folds and test transcripts of step 03, (A) position-only, (B) a
constrained scanner with about 14 interpretable parameters, fitted with L-BFGS to the per-read negative
log-likelihood plus an ℓ2 penalty of 10⁻⁴, and (C) RateNet with the chain, trained with Adam (learning
rate 3 × 10⁻³, gradient-norm clipping at 5) and early stopping on 15% of the training-chromosome
transcripts (checked every 10 epochs, patience 50 epochs). Both use seeds 0, 1 and 2. B is deterministic
across seeds.

**Recoverability theory.** `11_recoverability_theory.py` derives the sampling distribution of the
pooled held-out ΔLL/read from a second-order expansion around the population fits. The expansion has
the exact held-out population gain, a first-order term from fit noise, the estimation penalty and
test-count noise, with Dirichlet-multinomial inflation and Poisson depth. It predicts the recovery
probability for every cell of the synthetic grid without fitting anything to the simulations, then
compares prediction and simulation cell by cell. It also attributes each failure to effect size, bin
width, overdispersion or model dimension.

**Experimental design.** `design_engine.py` evaluates the same local model for a proposed design with R
libraries: each library is held out once and trained on the other R − 1, and the exact covariance
between these correlated statistics is kept. R = 2 is validated by the simulations and retrospective
pilots. R > 2 is an analytic extrapolation. Costs are placeholders and only illustrative. Every design
table carries these statuses as a column.

**Reporter utility.** Run only after the recoverability results were frozen. The 40S model is chosen
from the held-out 40S results alone, never from a reporter number. It is refitted with the reporter
leader's chromosome fold held out, applied to the reporter sequence and summarised as features.
Sequence-only models are compared with models that add the 40S features by concatenation, late fusion
or a residual model. Yeast uses the chromosome folds, and human uses frozen random folds
(`downstream/sample2019_folds.tsv`), with a sequence-cluster split as a sensitivity analysis.

## Results

`results/recoverability/v1/`:

| Path | Contents |
|---|---|
| `manifest.json` | provenance of every section (see [Provenance](#provenance)) |
| `*_chromosome_folds.json` | frozen chromosome folds |
| `qc/` | raw-read inspection, read accounting, metagenes, reads per transcript |
| `resolution/` | replicate agreement versus bin width |
| `effects/` | held-out ΔLL/read per model, direction and bin width (`nested_effects.tsv`), coefficients, per-transcript held-out log-likelihoods |
| `nulls/` | permutation-null summaries (`shuffle_nulls.tsv`) and every null draw |
| `depth/` | thinning curves |
| `synthetic/` | power per simulation and power curves |
| `chain/` | identifiability, identifiability surfaces, chain ablation and parameters |
| `downstream/` | reporter analyses, model choice, exclusions, folds, predictions |
| `summary/` | status per effect and resolution |
| `missing_eval/` | added evaluations and their manifest |
| `theory/` | analytic predictions, calibration, decomposition, required depth |
| `design/` | design tables (costs in `design_cost_overlay_ILLUSTRATIVE.tsv` are placeholders) |
| `figures/` | main figures (PNG and SVG) |
| `appendix/` | supplementary figures and tables |
| `logs/` | one log per pipeline step |

`theory/cache/` and `design/cache/` (570 MB) and the per-read and panel caches outside `results/` are
not committed. They are rebuilt by the steps that need them.

The figure file names follow the analysis order, not the manuscript's figure numbers:

| File | Shows |
|---|---|
| `fig1_qc` | library QC: read accounting, leader depth, start-codon metagene |
| `fig2_resolution`, `fig2b_resolution_by_depth` | replicate agreement versus bin width, overall and by depth |
| `fig3_effects_gse174329_eif3b` | held-out ΔLL/read per feature family and bin width |
| `fig4_power_*` | synthetic recovery power versus depth (`_theory`: with the analytic prediction) |
| `fig5_identifiability` | Fisher-information identifiability of the chain parameters |
| `fig6_complexity` | position-only versus constrained scanner versus RateNet |
| `fig9_reporter_utility` | reporter utility of the 40S representation |
| `missing_eval_*` | selected versus unselected, yeast single library, reporter utility in detail |
| `theory_validation` | analytic versus simulated recovery |
| `experiment_design` | depth and replicate requirements |

### Supplementary figures and tables

Files in `appendix/`. "Copy" is a byte copy of a figure in `figures/`, "new" was drawn from the listed
result tables, and S14 is the design figure without its illustrative cost panel. Every table comes as
`.tsv` (full rows, full precision, confidence intervals as extra columns) and as `.tex`.

| ID | File | Origin | Supports (manuscript section) | Source |
|---|---|---|---|---|
| S1 | `S1_footprint_lengths` | new | 3.1 P-site calibration | `qc/insert_lengths.tsv`, `manifest.json` |
| S2 | `S2_resolution_by_depth` | new | 4.1 resolution depends on depth in both assays | `resolution/replicate_resolution.tsv` |
| S3 | `S3_thinning` | new | 3.4 depth dependence by thinning, 4.2 and 4.5 selection beyond depth | `depth/depth_curve.tsv` |
| S4 | `S4_effects_unselected` | new | 4.2 unselected libraries | `effects/nested_effects.tsv`, `nulls/shuffle_nulls.tsv` |
| S5 | `S5_selected_vs_unselected` | copy | 4.2 and 4.5 assay dependence | `missing_eval/` |
| S6 | `S6_yeast_single_library` | copy | 4.2 yeast | `missing_eval/yeast_single_library_summary.tsv` |
| S7 | `S7_permutation_nulls` | new | 3.3 permutation nulls | `nulls/null_draws_*.tsv`, `nulls/shuffle_nulls.tsv` |
| S8 | `S8_coefficients` | new | 4.2 what the transferable effects are | `effects/coefficients_*.tsv` |
| S9 | `S9_recovery_unselected_n50` | copy | 4.2 analytic approximation versus simulated recovery | `synthetic/power_curve_gse139131_unsel_dm_n50.tsv`, `theory/theory_power_grid.tsv` |
| S10 | `S10_recovery_yeast_n50` | copy | 4.2 yeast power | `synthetic/power_curve_archer_yeast_multinomial_n50.tsv`, `theory/theory_power_grid.tsv` |
| S11 | `S11_theory_validation` | copy | 4.2 theory calibration, overdispersion ceiling | `theory/theory_validation.tsv`, `theory/theory_required_depth.tsv` |
| S12 | `S12_identifiability_surfaces` | new | 4.3 identifiability | `chain/identifiability_surfaces.tsv` |
| S13 | `S13_scanner_stability` | new | 3.5 and 4.3 coefficient stability (B), cross-seed rate-field correlation (C) | `chain/chain_params.tsv`, `chain/chain_ablation.tsv` |
| S14 | `S14_experiment_design` | re-rendered | 4.4 experimental design | `design/` |
| S15 | `S15_reporter_models` | copy | 4.5 and 3.7 reporter model variants | `downstream/downstream_utility_final.tsv` |
| S16 | `S16_reporter_strata` | new | 4.5 uAUG strata, sequence-cluster sensitivity | `downstream/downstream_utility_final.tsv` |
| T1 | `T1_library_qc` | | 4.1 data quality | `qc/qc_summary.tsv` |
| T2 | `T2_recoverability_status` | | 3.6 statuses | `summary/recoverability_summary.tsv`, `missing_eval/*_summary.tsv`, `nulls/shuffle_nulls.tsv` |
| T3 | `T3_theory_calibration` | | 4.2 theory calibration | `theory/theory_calibration.tsv` |
| T4 | `T4_identifiability` | | 4.3 identifiability | `chain/identifiability.tsv` |

The other tables in `appendix/tables/` are supplementary data that the manuscript does not print:
annotation, P-site rules, chromosome folds, held-out effects, permutation nulls, assay comparison,
same-study selection, required depth, information decomposition, observed effects in theory units,
observable ranking, scanner generalisation and coefficients, and the design tables.

## Configuration

| Variable | Effect |
|---|---|
| `UTR_RESULTS` | results tree to read and write (default `results/recoverability/v1`) |
| `UTR_BACKEND` | `auto` (default: PyTorch when a CUDA GPU is present, SciPy otherwise), `torch` or `scipy` |
| `UTR_WORKERS` | number of parallel workers (default from CPU count and RAM) |
| `UTR_CHUNK_MB` | GPU chunk size for the information matrix (larger is slightly faster and uses more GPU memory) |
| `UTR_ALLOW_ANY_ENV` | `1` allows running in conda's `base` environment |

Datasets, accessions, bin widths, depth strata, seeds (0, 1, 2), bootstrap and permutation counts, and
the read thresholds are defined in `analysis/recoverability/config.py` and at the top of each step.

## Tests

```bash
python -m pytest analysis/recoverability/tests
```

20 tests, a few minutes on a laptop CPU. They cover the multinomial model and its Fisher scoring
(`test_likelihood.py`), the matched permutations (`test_nulls.py`), the closed-form chain against the
exact solve in `model/chain.py` (`test_chainfast.py`), agreement of the GPU and CPU backends
(`test_backends.py`) and the design layer (`test_design.py`). The 10 design tests need `design/cache`
from a completed run and are skipped without it.

## Provenance

`manifest.json` records, for every section of the main run, the git commit, accessions and checksums,
the P-site rules, the structure method, thresholds and seeds. `missing_eval/manifest.json` records every
command of the added evaluations with its seeds, simulation counts, run time, software versions and
hardware. The recorded versions are Python 3.12.15, NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.8.0,
PyTorch 2.14.1 (CUDA 13.0), ViennaRNA 2.7.2 and matplotlib 3.10.9, with bowtie2 2.5 and cutadapt 5 from
`environment.yml`. `ref/download_manifest.json`, written by the download, records the URL, size and
checksum of every reference file.

## Citation

If you use this code or these results, please cite the *Foot2Feat* manuscript (RECOMB 2027). The full
reference will be added on publication.
