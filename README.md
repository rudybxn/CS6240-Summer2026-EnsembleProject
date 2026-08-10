# Distributed Ensemble Framework (CS 6240 Project)

A parallel framework, written from scratch in Spark Scala, that trains, applies, and tunes an
*ensemble* of in-memory base learners, benchmarked against Spark MLlib's own ensembles.

**Project task:** "Classification and prediction ensembles using existing libraries for in-memory
processing on a single machine" (counts as two problems).

**Dataset:** UCI HIGGS — 11,000,000 rows, 28 numeric features, binary label.

---

## The library / from-scratch boundary

The task permits an external single-machine ML library for fitting and predicting *individual*
models, but requires the parallel framework around them to be written from scratch. The code is
organized so that boundary is visible at the package level:

| Package | Written from scratch? | Purpose |
|---|---|---|
| `baselearner` | No — **the only package that imports Smile** | Fit/predict one model on data already in one task's memory |
| `framework` | **Yes** | Custom partitioner, sampling strategies, distributed training, broadcast prediction, aggregation |
| `hpsearch` | **Yes** | Parallel hyperparameter search + a shared-computation optimization |
| `driver` | **Yes** | Experiment entry points wiring train → predict → evaluate |
| `dataio` | n/a — data plumbing | CSV→Parquet conversion, shared loading |
| `metrics` | n/a — evaluation | Accuracy + AUC, shared by the framework and the baseline |
| `mllibbaseline` | n/a — comparison only | Independent MLlib job; imports nothing from `framework` |

`mllibbaseline` is a separate, standalone comparison point. No MLlib model, evaluator, or
ensemble is used anywhere in `framework`, `hpsearch`, or `driver`.

---

## Prerequisites

- Java 11
- Maven 3.9+
- Spark 3.3.2 and Hadoop 3.3.5 installed locally (paths configured at the top of the `Makefile`)
- For EMR runs: AWS CLI configured with valid credentials

Smile 3.0.2 is pulled in by Maven and bundled into the fat jar (it is *not* installed separately,
and is not present on the EMR cluster otherwise). It is pinned to the 3.x line because Smile 4.x
requires JDK 21 and 5.x+ requires JDK 25, while this toolchain — local and EMR — runs Java 11.

## Build

```bash
make jar          # mvn clean package, then copies the fat jar to ./ensemble.jar
```

---

## Getting the data

### What is already in this repository

| Dataset | Size | Committed? | What it enables |
|---|---|---|---|
| `data/scaling/parquet_10k` | 1.4 MB | ✅ yes | 10K scaling point |
| `data/scaling/parquet_100k` | 11 MB | ✅ yes | 100K scaling point, and the experiment, hyperparameter-search, design-lever, and MLlib-baseline runs |
| `data/scaling/parquet_1m` | 80 MB | ❌ no | 1M scaling point |
| `data/scaling/parquet_3m` | 231 MB | ❌ no | 3M scaling point |
| `data/HIGGS.csv.gz` (full) | 2.6 GB | ❌ no | full-scale and cluster runs |

**So `./scripts/run-local-experiments.sh` works immediately after `make jar`, with no download.**
It runs every experiment except the 1M scaling point, which it skips with a notice. That covers
all results in `EXPERIMENTS.md` other than the two largest scaling measurements.

Download the full dataset only if you need the larger scaling points or a full-scale run.

### Downloading the full dataset

```bash
mkdir -p data
curl -fL -o data/HIGGS.csv.gz \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

Then convert once to Parquet — every experiment reads Parquet, never the raw CSV:

```bash
# Full dataset: last 500,000 rows are the official test set
make local-csv2parquet in=data/HIGGS.csv.gz out=data/parquet_full testsize=500000
```

To build the two larger scaling subsets (the 10K and 100K ones are already committed, so they do
not need to be rebuilt):

```bash
mkdir -p data/scaling
zcat < data/HIGGS.csv.gz | head -1000000 > data/scaling/higgs_1m.csv
zcat < data/HIGGS.csv.gz | head -3000000 > data/scaling/higgs_3m.csv

make local-csv2parquet in=data/scaling/higgs_1m.csv out=data/scaling/parquet_1m testsize=100000
make local-csv2parquet in=data/scaling/higgs_3m.csv out=data/scaling/parquet_3m testsize=300000
```

Once `parquet_1m` exists, `./scripts/run-local-experiments.sh` picks it up automatically and the
full scaling table is reproduced.

Each subset is a prefix of the file. That is a fair sample here because every row is an
independently simulated collision — there is no ordering or grouping in the file for a prefix to
bias — and it keeps the subsets reproducible across machines.

---

## Running experiments

Requires the Parquet datasets from the previous section. The runner script skips any experiment
whose input is missing and says so, rather than failing outright, so a partial data setup still
produces useful output.

The fastest path — run the whole local suite and write logs to `results/`:

```bash
./scripts/run-local-experiments.sh
```

That script is the reproducible entry point behind the numbers reported in `EXPERIMENTS.md`.
It builds the jar once, runs each experiment in turn, and writes both a full log and a
summary of extracted `RESULT` / `SCALING` lines.

### Individual experiments

Each is also a Makefile target. **All of them default to the committed 100K dataset, so they run
with no arguments:**

```bash
make local-smoke            # Toolchain check: Spark local mode + Smile fit/predict
make local-experiment       # One ensemble configuration, end to end
make local-scaling label=100k   # Timed 12-configuration sweep
make local-hpsearch         # Naive parallel and nested-prefix search strategies
make local-levers           # Disjoint vs. sampled-fraction, majority vote vs. mean probability
make local-mllib-baseline   # Independent MLlib comparison (DecisionTree + RandomForest)
```

Override `train`, `test`, and `nummodels` to point at a different dataset — for example, the 1M
subset once it has been built:

```bash
make local-scaling label=1m train=data/scaling/parquet_1m/train \
                            test=data/scaling/parquet_1m/test

make local-experiment nummodels=40
```

One target needs data that is not committed: `local-fraction-check` reproduces the memory finding
documented in `EXPERIMENTS.md` and requires the 1M subset, since the failure only appears at that
scale.

```bash
make local-fraction-check train=data/scaling/parquet_1m/train nummodels=40 fraction=1.0
```

### Output format

Experiment drivers print one machine-readable line per configuration, so results can be
collected from logs without parsing prose:

```
RESULT numModels=10 maxDepth=20 maxNodes=100 nodeSize=5 testCount=10000 accuracy=0.6638 auc=0.7200944231502118
SCALING label=1m trainRows=900000 sweepSize=12 elapsedSeconds=431.89
```

---

## Running on AWS EMR

```bash
make make-bucket                                        # once
make upload-data-aws local=data/scaling/parquet_100k    # upload a train/test pair
make aws-experiment nummodels=10                        # launch cluster + step, auto-terminates
```

Monitor and, if needed, stop a run:

```bash
make list-clusters-aws
make terminate-cluster-aws clusterid=j-XXXXXXXX
```

Cluster size, instance type, and Spark sizing are variables at the top of the `Makefile`'s AWS
section (`aws.core.num.nodes`, `aws.instance.type`, `aws.executor.memory`, …).

**Note on Spark configuration for EMR:** the step submits with `--deploy-mode client` and
explicit executor sizing, with dynamic allocation disabled. This is deliberate — see
`EXPERIMENTS.md` (Phase 12) for the resource deadlock that the default configuration produced
on an undersized cluster, and why these settings avoid it.

---

## Where the results live

- **`EXPERIMENTS.md`** — the running log of measurements, design decisions, and findings,
  organized by development phase. This is the source for the report's analysis sections.
- **`results/`** — raw logs and summaries produced by `scripts/run-local-experiments.sh`.
