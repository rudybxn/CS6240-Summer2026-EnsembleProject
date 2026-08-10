# Makefile for CS 6240 Project - Distributed Ensemble Framework

spark.root=/usr/local/spark-3.3.2-bin-without-hadoop
hadoop.root=/usr/local/hadoop-3.3.5
app.name=CS6240 Ensemble
jar.name=ensemble.jar
maven.jar.name=ensemble-1.0.jar

# Job classes.
job.smoke=smoketest.ToolchainSmokeTest
job.csv2parquet=dataio.CsvToParquet
job.experiment=driver.EnsembleExperimentDriver
job.scaling=driver.ScalingHarnessDriver
job.hpsearch=smoketest.HyperparameterSearchSmokeTest
job.levers=smoketest.DesignLeverSmokeTest
job.fraction-check=smoketest.SampledFractionMemoryCheck
job.mllib-baseline=mllibbaseline.MLlibBaselineDriver

data.dir=data
higgs.csv.gz=${data.dir}/HIGGS.csv.gz

local.master=local[4]

# Defaults point at the 100K subset, which is committed to the repository, so
# every local-* target below runs with no arguments and no data download.
# Override on the command line for a different dataset, e.g.
#   make local-experiment train=data/scaling/parquet_1m/train test=data/scaling/parquet_1m/test
train=data/scaling/parquet_100k/train
test=data/scaling/parquet_100k/test
nummodels=10

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

# Convert a HIGGS CSV (full file or a subset) into train/test Parquet.
# Usage: make local-csv2parquet in=data/HIGGS.csv.gz out=data/parquet_full testsize=500000
local-csv2parquet: jar
	spark-submit --class ${job.csv2parquet} --master ${local.master} \
		--name "${app.name}: CSV to Parquet" ${jar.name} \
		${in} ${out}/train ${out}/test ${testsize}

# One ensemble configuration end to end: train -> predict -> evaluate,
# printed as a single RESULT line.
# Usage: make local-experiment [nummodels=10] [train=... test=...]
local-experiment: jar
	spark-submit --class ${job.experiment} --master ${local.master} \
		--name "${app.name}: Ensemble Experiment" ${jar.name} ${train} ${test} ${nummodels}

# Times a 12-configuration sweep against whatever dataset it is pointed at -
# the scaling harness. Run once per input size to measure how cost grows.
# Usage: make local-scaling label=100k [train=... test=...]
local-scaling: jar
	spark-submit --class ${job.scaling} --master ${local.master} --driver-memory 4g \
		--name "${app.name}: Scaling Harness ${label}" ${jar.name} ${train} ${test} ${label}

# Both hyperparameter-search strategies: naive parallel (varying tree
# hyperparameters, FAIR-scheduled) and nested-prefix (varying ensemble size
# off one shared training pass).
# Usage: make local-hpsearch [train=... test=...]
local-hpsearch: jar
	spark-submit --class ${job.hpsearch} --master ${local.master} \
		--name "${app.name}: Hyperparameter Search" ${jar.name} ${train} ${test}

# Compares both halves of each design lever: disjoint vs. sampled-fraction
# sampling, majority-vote vs. mean-probability aggregation.
# Usage: make local-levers [nummodels=10] [train=... test=...]
local-levers: jar
	spark-submit --class ${job.levers} --master ${local.master} \
		--name "${app.name}: Design Lever Comparison" ${jar.name} ${train} ${test} ${nummodels}

# Independent MLlib baseline: a single DecisionTree and a RandomForest,
# evaluated through the same metrics module the framework uses.
# Usage: make local-mllib-baseline [train=... test=...]
local-mllib-baseline: jar
	spark-submit --class ${job.mllib-baseline} --master ${local.master} --driver-memory 4g \
		--name "${app.name}: MLlib Baseline" ${jar.name} ${train} ${test}

# Deliberately no --driver-memory override - checks whether a given
# numModels/fraction combination survives under Spark's default memory.
# Needs the 1M subset to reproduce the documented OOM; see README for how to build it.
# Usage: make local-fraction-check train=data/scaling/parquet_1m/train nummodels=40 fraction=1.0
local-fraction-check: jar
	spark-submit --class ${job.fraction-check} --master ${local.master} \
		--name "${app.name}: Sampled Fraction Memory Check" ${jar.name} ${train} ${nummodels} ${fraction}

# Runs the whole local experiment suite and writes logs to results/.
local-all:
	./scripts/run-local-experiments.sh

local: local-smoke

# AWS EMR targets


aws.emr.release=emr-6.10.0
aws.region=us-east-1
aws.bucket.name=cs6240-ensemble-rudybxn
aws.log.dir=log

# 1 primary + 2 core nodes, m4.large. 
aws.primary.num.nodes=1
aws.core.num.nodes=2
aws.instance.type=m4.large


aws.executor.memory=2g
aws.executor.cores=1
aws.executor.instances=2
aws.driver.memory=2g

aws.data.remote=s3://${aws.bucket.name}/data
aws.train.path=${aws.data.remote}/train
aws.test.path=${aws.data.remote}/test

# One-time: create the bucket the jar and data get uploaded to.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

upload-app-aws: jar
	aws s3 cp ${jar.name} s3://${aws.bucket.name}/${jar.name}

# Uploads a local train/test Parquet pair to the fixed S3 loc
upload-data-aws:
	aws s3 sync ${local}/train ${aws.train.path}
	aws s3 sync ${local}/test ${aws.test.path}


# Usage: make aws-experiment nummodels=10
aws-experiment: upload-app-aws
	aws emr create-cluster \
		--name "${app.name}: Ensemble Experiment" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.core.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":${aws.primary.num.nodes},"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps '[{"Type":"CUSTOM_JAR","Name":"EnsembleExperiment","Jar":"command-runner.jar","ActionOnFailure":"TERMINATE_CLUSTER","Args":["spark-submit","--deploy-mode","client","--conf","spark.dynamicAllocation.enabled=false","--conf","spark.executor.memory=${aws.executor.memory}","--conf","spark.executor.cores=${aws.executor.cores}","--conf","spark.executor.instances=${aws.executor.instances}","--driver-memory","${aws.driver.memory}","--class","${job.experiment}","s3://${aws.bucket.name}/${jar.name}","${aws.train.path}","${aws.test.path}","${nummodels}"]}]' \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification":"hadoop-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}},{"Classification":"spark-env","Configurations":[{"Classification":"export","Configurations":[],"Properties":{"JAVA_HOME":"/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties":{}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate

# Safety net: list active clusters, or manually terminate one by ID if
# auto-terminate doesn't fire as expected. Worth checking after every run.
list-clusters-aws:
	aws emr list-clusters --active

terminate-cluster-aws:
	aws emr terminate-clusters --cluster-ids ${clusterid}
