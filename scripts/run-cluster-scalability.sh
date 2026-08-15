#!/usr/bin/env bash
#
# Input-size scalability on a fixed cluster: runs the same 48-configuration
# sweep at 4 core nodes against 1M, 3M and 6M row subsets. Combined with the
# existing full-dataset run at 4 core nodes, that gives a four-point curve
# where cluster size is held constant and only input size varies - the
# counterpart to the speedup study, which holds input constant and varies
# cluster size.
#
#   ./scripts/run-cluster-scalability.sh

set -euo pipefail

cd "$(dirname "$0")/.."

BUCKET=cs6240-ensemble-rudybxn
NODES=4

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS credentials are not valid - refresh the Learner Lab session first." >&2
  exit 1
fi

for sz in 1m 3m 6m; do
  echo "==> uploading $sz"
  aws s3 sync "data/scaling/parquet_$sz/train" "s3://$BUCKET/data-$sz/train" --only-show-errors
  aws s3 sync "data/scaling/parquet_$sz/test"  "s3://$BUCKET/data-$sz/test"  --only-show-errors
done

for sz in 1m 3m 6m; do
  echo "==> launching sweep at $sz on $NODES core nodes"
  cluster=$(make aws-scaling \
      aws.core.num.nodes=$NODES \
      label="$sz" \
      aws.sweep.train="s3://$BUCKET/data-$sz/train" \
      aws.sweep.test="s3://$BUCKET/data-$sz/test" \
    | grep -oE 'j-[A-Z0-9]+' | head -1)
  echo "    cluster $cluster"

  until state=$(aws emr describe-cluster --cluster-id "$cluster" \
                  --query 'Cluster.Status.State' --output text 2>/dev/null) &&
        { [ "$state" = TERMINATED ] || [ "$state" = TERMINATED_WITH_ERRORS ]; }; do
    sleep 60
  done
  echo "    finished: $state"

  if [ "$state" = TERMINATED_WITH_ERRORS ]; then
    echo "    stopping - $sz failed, later sizes not launched" >&2
    exit 1
  fi

  step=$(aws emr list-steps --cluster-id "$cluster" \
           --query 'Steps[?Name==`ScalingSweep`].Id' --output text)
  mkdir -p results/emr
  aws s3 cp "s3://$BUCKET/log/$cluster/steps/$step/stdout.gz" - 2>/dev/null \
    | gunzip > "results/emr/scaling_${sz}_4node.log"
  grep -E '^SCALING' "results/emr/scaling_${sz}_4node.log" || true
done

echo
echo "=== cluster scalability at $NODES core nodes ==="
grep -hE '^SCALING' results/emr/scaling_*_4node.log results/emr/scaling_4node.log 2>/dev/null
