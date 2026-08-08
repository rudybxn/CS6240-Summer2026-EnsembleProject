package driver

import baselearner.TreeParams
import dataio.HiggsLoader
import framework.SamplingStrategy
import hpsearch.NaiveSearch
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

// The local scaling harness: times one aggregate sweep of the same
// configuration matrix (12 configs: 3 ensemble sizes x 2 tree-depth choices
// x 2 sampling strategies) against train/test data of whatever size the
// caller points it at. Run once per data size (10K/100K/1M) to see how the
// sweep's wall-clock cost scales with data size 
object ScalingHarnessDriver {
  def main(args: Array[String]): Unit = {
    require(
      args.length >= 2,
      "usage: ScalingHarnessDriver <train-parquet-path> <test-parquet-path> [label]"
    )
    val trainPath = args(0)
    val testPath = args(1)
    val label = if (args.length > 2) args(2) else trainPath

    val conf = new SparkConf().set("spark.scheduler.mode", "FAIR")
    val spark = SparkSession.builder().appName("ScalingHarnessDriver").config(conf).getOrCreate()

    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).cache()
    val testData = HiggsLoader.loadAsArrays(spark, testPath).cache()
    val trainCount = trainData.count()
    val testCount = testData.count()
    // Force the cache to materialize before timing starts, so Parquet read
    // time isn't counted as part of the sweep's compute cost.
    trainData.foreach(_ => ())
    testData.foreach(_ => ())

    val treeParamChoices = Seq(
      TreeParams(maxDepth = 5, maxNodes = 20, nodeSize = 5),
      TreeParams(maxDepth = 20, maxNodes = 100, nodeSize = 5)
    )
    val numModelsChoices = Seq(10, 20, 40)

    // fraction = 1/numModels keeps each model's expected sample size
    // comparable to disjoint partitioning's, rather than the classic
    // full-size bootstrap (fraction=1.0) - which amplifies total replicated
    // data by numModels x and exhausts per-task memory well before real
    // data volume does 
    // Computed per numModels rather than as a fixed choice, since the fair
    // fraction depends on it.
    val configs = for {numModels <- numModelsChoices
      params <- treeParamChoices
      sampling <- Seq(SamplingStrategy.Disjoint, SamplingStrategy.SampledFraction(1.0 / numModels))
    } yield ExperimentConfig(numModels, params, sampling)

    println(s"[scaling] label=$label trainRows=$trainCount testRows=$testCount sweepSize=${configs.length}")
    val start = System.nanoTime()
    val results = NaiveSearch.run(spark, trainData, testData, configs)
    val elapsedSeconds = (System.nanoTime() - start) / 1e9
    results.foreach { case (config, result) =>
      println(
        s"RESULT label=$label numModels=${config.numModels} maxDepth=${config.treeParams.maxDepth} " +
          s"maxNodes=${config.treeParams.maxNodes} sampling=${config.samplingStrategy} " +
          s"trainRows=$trainCount testRows=$testCount accuracy=${result.accuracy} auc=${result.auc}"
      )
    }

    println(
      f"SCALING label=$label trainRows=$trainCount sweepSize=${configs.length} elapsedSeconds=$elapsedSeconds%.2f"
    )

    spark.stop()
  }
}
