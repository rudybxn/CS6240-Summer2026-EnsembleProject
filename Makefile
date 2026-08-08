# Makefile for CS 6240 Project - Distributed Ensemble Framework
# Pattern copied from example_makefile (HW4): mvn builds the jar, spark-submit runs it.

spark.root=/usr/local/spark-3.3.2-bin-without-hadoop
hadoop.root=/usr/local/hadoop-3.3.5
app.name=CS6240 Ensemble
jar.name=ensemble.jar
maven.jar.name=ensemble-1.0.jar

# Job classes: one entry added per phase as each driver lands.
job.smoke=smoketest.ToolchainSmokeTest
job.baselearner-smoke=smoketest.BaseLearnerSmokeTest
job.csv2parquet=dataio.CsvToParquet
job.loader-smoke=smoketest.HiggsLoaderSmokeTest
job.trainer-smoke=smoketest.EnsembleTrainerSmokeTest
job.predictor-smoke=smoketest.EnsemblePredictorSmokeTest
job.metrics-smoke=smoketest.MetricsSmokeTest
job.experiment=driver.EnsembleExperimentDriver
job.hpsearch-smoke=smoketest.HyperparameterSearchSmokeTest
job.levers-smoke=smoketest.DesignLeverSmokeTest
job.scaling=driver.ScalingHarnessDriver
job.fraction-check=smoketest.SampledFractionMemoryCheck
job.mllib-baseline=mllibbaseline.MLlibBaselineDriver

data.dir=data
higgs.csv.gz=${data.dir}/HIGGS.csv.gz

local.master=local[4]
local.output=output

# Compiles code and builds jar (with dependencies).
jar:
	mvn clean package -q
	cp target/${maven.jar.name} ${jar.name}

# Removes local output directories.
clean-local-output:
	rm -rf ${local.output}*

# LOCAL targets

# Proves Spark (local mode) and Smile (in-memory fit/predict) both work in
# this project, built and run exactly the way every later job will be.
local-smoke: jar
	spark-submit --class ${job.smoke} --master ${local.master} \
		--name "${app.name}: Toolchain Smoke Test" ${jar.name}

# The base-learner interface: a single Smile decision tree per model - the
# ensemble's diversity comes from the framework's own partitioning, not from
# Smile's internal RandomForest (used only as a toolchain check above).
local-baselearner-smoke: jar
	spark-submit --class ${job.baselearner-smoke} --master ${local.master} \
		--name "${app.name}: Base Learner Smoke Test" ${jar.name}

# Convert a HIGGS CSV (full file or a small sample) into train/test Parquet.
# Usage: make local-csv2parquet in=data/higgs_sample.csv out=data/sample testsize=100
local-csv2parquet: jar
	spark-submit --class ${job.csv2parquet} --master ${local.master} \
		--name "${app.name}: CSV to Parquet" ${jar.name} \
		${in} ${out}/train ${out}/test ${testsize}

# Checks both loader shapes (plain arrays, and the MLlib vector frame)
# against real converted Parquet.
# Usage: make local-loader-smoke train=data/sample_parquet/train
local-loader-smoke: jar
	spark-submit --class ${job.loader-smoke} --master ${local.master} \
		--name "${app.name}: Higgs Loader Smoke Test" ${jar.name} ${train}

# Disjoint-partition training: one BaseLearner per partition, balanced row
# counts, every model produces a valid prediction.
# Usage: make local-trainer-smoke train=data/sample_parquet/train nummodels=5
local-trainer-smoke: jar
	spark-submit --class ${job.trainer-smoke} --master ${local.master} \
		--name "${app.name}: Ensemble Trainer Smoke Test" ${jar.name} ${train} ${nummodels}

# Full train -> broadcast -> predict -> majority-vote loop on real data.
# Usage: make local-predictor-smoke train=data/sample_parquet/train test=data/sample_parquet/test nummodels=5
local-predictor-smoke: jar
	spark-submit --class ${job.predictor-smoke} --master ${local.master} \
		--name "${app.name}: Ensemble Predictor Smoke Test" ${jar.name} ${train} ${test} ${nummodels}

# Full train -> predict -> evaluate loop through the real metrics module
# (BinaryClassificationMetrics for AUC, match-count ratio for accuracy).
# Usage: make local-metrics-smoke train=data/sample_parquet/train test=data/sample_parquet/test nummodels=5
local-metrics-smoke: jar
	spark-submit --class ${job.metrics-smoke} --master ${local.master} \
		--name "${app.name}: Metrics Smoke Test" ${jar.name} ${train} ${test} ${nummodels}

# The real experiment entry point: train -> predict -> evaluate one ensemble
# configuration, printed as a single RESULT line.
# Usage: make local-experiment train=data/sample_parquet/train test=data/sample_parquet/test nummodels=10
local-experiment: jar
	spark-submit --class ${job.experiment} --master ${local.master} \
		--name "${app.name}: Ensemble Experiment" ${jar.name} ${train} ${test} ${nummodels}

# Both hyperparameter-search strategies: naive parallel (varying tree
# hyperparameters, FAIR-scheduled) and nested-prefix (varying ensemble size
# off one shared training pass).
# Usage: make local-hpsearch-smoke train=data/sample_parquet/train test=data/sample_parquet/test
local-hpsearch-smoke: jar
	spark-submit --class ${job.hpsearch-smoke} --master ${local.master} \
		--name "${app.name}: Hyperparameter Search Smoke Test" ${jar.name} ${train} ${test}

# Compares both halves of each design lever: disjoint vs. sampled-fraction
# sampling, majority-vote vs. mean-probability aggregation.
# Usage: make local-levers-smoke train=data/sample_parquet/train test=data/sample_parquet/test nummodels=10
local-levers-smoke: jar
	spark-submit --class ${job.levers-smoke} --master ${local.master} \
		--name "${app.name}: Design Lever Smoke Test" ${jar.name} ${train} ${test} ${nummodels}

# Times a 12-config aggregate sweep against train/test data of whatever
# size it's pointed at - the local scaling harness.
# Usage: make local-scaling train=data/scaling/parquet_10k/train test=data/scaling/parquet_10k/test label=10k
local-scaling: jar
	spark-submit --class ${job.scaling} --master ${local.master} --driver-memory 4g \
		--name "${app.name}: Scaling Harness ${label}" ${jar.name} ${train} ${test} ${label}

# Deliberately no --driver-memory override - checks whether a given
# numModels/fraction combination survives under Spark's default memory.
# Usage: make local-fraction-check train=data/scaling/parquet_1m/train nummodels=40 fraction=1.0
local-fraction-check: jar
	spark-submit --class ${job.fraction-check} --master ${local.master} \
		--name "${app.name}: Sampled Fraction Memory Check" ${jar.name} ${train} ${nummodels} ${fraction}

# Independent MLlib baseline: a single DecisionTree and a RandomForest,
# evaluated through the same metrics module the framework uses.
# Usage: make local-mllib-baseline train=data/sample_parquet/train test=data/sample_parquet/test
local-mllib-baseline: jar
	spark-submit --class ${job.mllib-baseline} --master ${local.master} --driver-memory 4g \
		--name "${app.name}: MLlib Baseline" ${jar.name} ${train} ${test}

local: local-smoke
