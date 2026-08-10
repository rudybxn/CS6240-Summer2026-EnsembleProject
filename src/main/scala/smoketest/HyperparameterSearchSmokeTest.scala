package smoketest

import baselearner.TreeParams
import dataio.HiggsLoader
import driver.{EnsembleExperiment, ExperimentConfig}
import hpsearch.{NaiveSearch, NestedPrefixSearch}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

// Exercises both hyperparameter-search strategies against real data:
//   1. NaiveSearch - genuinely different configs (varying tree
//      hyperparameters), submitted concurrently under the FAIR scheduler.
//   2. NestedPrefixSearch - one training pass at the largest ensemble size,
//      evaluated at several smaller sizes without retraining. Its result at
//      the largest size is checked against a fresh, independent
object HyperparameterSearchSmokeTest {

  def main(args: Array[String]): Unit = {
    require(
      args.length >= 2,
      "usage: HyperparameterSearchSmokeTest <train-parquet-path> <test-parquet-path>"
    )
    val trainPath = args(0)
    val testPath = args(1)

    val conf = new SparkConf().set("spark.scheduler.mode", "FAIR")
    val spark = SparkSession.builder()
      .appName("HyperparameterSearchSmokeTest")
      .config(conf)
      .getOrCreate()

    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).cache()
    val testData = HiggsLoader.loadAsArrays(spark, testPath).cache()

    // 1. Naive parallel search over tree hyperparameters
    val candidateParams = Seq(
      TreeParams(maxDepth = 5, maxNodes = 20, nodeSize = 5),
      TreeParams(maxDepth = 10, maxNodes = 50, nodeSize = 5),
      TreeParams(maxDepth = 20, maxNodes = 100, nodeSize = 5)
    )
    val naiveConfigs = candidateParams.map(p => ExperimentConfig(numModels = 10, treeParams = p))
    val naiveResults = NaiveSearch.run(spark, trainData, testData, naiveConfigs)

    require(naiveResults.length == naiveConfigs.length, "naive search dropped a config")
    naiveResults.foreach { case (config, result) =>
      println(s"[naive] maxDepth=${config.treeParams.maxDepth} maxNodes=${config.treeParams.maxNodes} " +
        s"-> accuracy=${result.accuracy} auc=${result.auc}")
      require(result.auc >= 0.0 && result.auc <= 1.0, s"AUC out of range: ${result.auc}")
    }

    // 2. Nested-prefix search over ensemble size, sharing one training pass.
    val modelCounts = Seq(5, 10, 20)
    val fixedParams = TreeParams()
    val prefixResults = NestedPrefixSearch.run(spark, trainData, testData, modelCounts, fixedParams)
    require(prefixResults.map(_._1) == modelCounts.sorted, "prefix search returned unexpected sizes")
    prefixResults.foreach { case (numModels, result) =>
      println(s"[prefix] numModels=$numModels -> accuracy=${result.accuracy} auc=${result.auc}")
    }

    val freshAtMax = EnsembleExperiment.run(
      spark, trainData, testData, ExperimentConfig(modelCounts.max, fixedParams)
    )
    val prefixAtMax = prefixResults.find(_._1 == modelCounts.max).get._2
    require(
      freshAtMax.accuracy == prefixAtMax.accuracy && freshAtMax.auc == prefixAtMax.auc,
      s"prefix result at max size should match a fresh run exactly: fresh=$freshAtMax prefix=$prefixAtMax"
    )

    println("[hpsearch] all checks passed")
    spark.stop()
  }
}
