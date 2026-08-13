package driver

import baselearner.TreeParams
import dataio.HiggsLoader
import framework.{AggregationStrategy, SamplingStrategy}
import hpsearch.{NaiveSearch, SharedTrainingSearch, TrainingConfig}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

//times one aggregate sweep of the full design-lever
// matrix (48 configs: 4 ensemble sizes x 3 tree settings x 2 sampling
// strategies x 2 aggregation strategies) against train/test data of whatever
// size it is pointed at. Used both for the local scaling study (run once per
// input size) and for the cluster-size speedup study (same data, run once
// per cluster size), so that in each case exactly one variable changes.
//
// Aggregation is evaluated off shared training passes rather than retrained
// per configuration - see SharedTrainingSearch - so this runs 24 training
// passes for its 48 results.
object ScalingHarnessDriver {
  def main(args: Array[String]): Unit = {
    require(
      args.length >= 2,
      "usage: ScalingHarnessDriver <train-parquet-path> <test-parquet-path> [label] [shared|naive]"
    )
    val trainPath = args(0)
    val testPath = args(1)
    val label = if (args.length > 2) args(2) else trainPath
    // "naive" retrains per configuration, including for pairs that differ
    // only in aggregation. It exists so the shared-training optimization can
    // be measured against the alternative on identical work, rather than
    // asserted - the two produce the same results, only at different cost.
    val searchMode = if (args.length > 3) args(3) else "shared"
    require(searchMode == "shared" || searchMode == "naive", s"unknown search mode: $searchMode")

    val conf = new SparkConf().set("spark.scheduler.mode", "FAIR")
    val spark = SparkSession.builder().appName("ScalingHarnessDriver").config(conf).getOrCreate()

    // MEMORY_AND_DISK, not the RDD default of MEMORY_ONLY. The sweep reads
    // the training set once per configuration, and under MEMORY_ONLY any
    // partition that does not fit is dropped and *recomputed from Parquet*
    // on each of those passes. At full scale that penalty lands hardest on
    // the smallest cluster, which would inflate its time and make the
    // speedup curve reflect memory pressure rather than parallelism.
    // Spilling to local disk costs a deserialize instead of a full re-read,
    // and when the data does fit this behaves identically to MEMORY_ONLY.
    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).persist(StorageLevel.MEMORY_AND_DISK)
    val testData = HiggsLoader.loadAsArrays(spark, testPath).persist(StorageLevel.MEMORY_AND_DISK)
    val trainCount = trainData.count()
    val testCount = testData.count()
   
    trainData.foreach(_ => ())
    testData.foreach(_ => ())

    // The full design-lever matrix: 4 ensemble sizes x 3 tree settings x
    // 2 sampling strategies x 2 aggregation strategies = 48 configurations.
    val treeParamChoices = Seq(
      TreeParams(maxDepth = 5, maxNodes = 20, nodeSize = 5),
      TreeParams(maxDepth = 10, maxNodes = 50, nodeSize = 5),
      TreeParams(maxDepth = 20, maxNodes = 100, nodeSize = 5)
    )
    val numModelsChoices = Seq(10, 20, 40, 80)
    val aggregationChoices = Seq(AggregationStrategy.MajorityVote, AggregationStrategy.MeanProbability)

    // Only the levers that change what actually gets trained appear here.
    // fraction = 1/numModels keeps each model's expected sample size
    // comparable to disjoint partitioning's, rather than the classic
    // full-size bootstrap (fraction=1.0), which amplifies replicated data by
    // numModels x and exhausts per-task memory well before data volume does.
    val trainingConfigs = for {
      numModels <- numModelsChoices
      params <- treeParamChoices
      sampling <- Seq(SamplingStrategy.Disjoint, SamplingStrategy.SampledFraction(1.0 / numModels))
    } yield TrainingConfig(numModels, params, sampling)

    val sweepSize = trainingConfigs.length * aggregationChoices.length

    // Aggregation reads already-trained models rather than changing them, so
    // the sweep trains once per training config and evaluates every
    // aggregation against it: 24 training passes for 48 results, instead of
    // the 48 a naive cartesian product would run.
    val trainingPasses = if (searchMode == "shared") trainingConfigs.length else sweepSize
    println(s"[scaling] label=$label mode=$searchMode trainRows=$trainCount testRows=$testCount " +
      s"sweepSize=$sweepSize trainingPasses=$trainingPasses")

    val start = System.nanoTime()
    val results = searchMode match {
      case "shared" =>
        SharedTrainingSearch.run(spark, trainData, testData, trainingConfigs, aggregationChoices)
      case _ =>
        // Same 48 results, but every configuration trains its own ensemble -
        // including pairs that differ only in aggregation and could have
        // shared one.
        val flatConfigs = for {
          tc <- trainingConfigs
          aggregation <- aggregationChoices
        } yield ExperimentConfig(tc.numModels, tc.treeParams, tc.samplingStrategy, aggregation)
        NaiveSearch.run(spark, trainData, testData, flatConfigs)
    }
    val elapsedSeconds = (System.nanoTime() - start) / 1e9
    results.foreach { case (config, result) =>
      println(
        s"RESULT label=$label numModels=${config.numModels} maxDepth=${config.treeParams.maxDepth} " +
          s"maxNodes=${config.treeParams.maxNodes} sampling=${config.samplingStrategy} " +
          s"aggregation=${config.aggregationStrategy} " +
          s"trainRows=$trainCount testRows=$testCount accuracy=${result.accuracy} auc=${result.auc}"
      )
    }

    println(
      f"SCALING label=$label mode=$searchMode trainRows=$trainCount sweepSize=$sweepSize " +
        f"trainingPasses=$trainingPasses elapsedSeconds=$elapsedSeconds%.2f"
    )

    spark.stop()
  }
}
