package smoketest

import baselearner.TreeParams
import dataio.HiggsLoader
import driver.{EnsembleExperiment, ExperimentConfig}
import framework.{AggregationStrategy, EnsembleTrainer, SamplingStrategy}
import org.apache.spark.sql.SparkSession

// Compares both halves of each design lever on the same real data: disjoint
// vs. sampled-fraction sampling, and majority-vote vs. mean-probability
// aggregation 
object DesignLeverSmokeTest {
  def main(args: Array[String]): Unit = {
    require(
      args.length >= 2,
      "usage: DesignLeverSmokeTest <train-parquet-path> <test-parquet-path> [numModels]"
    )
    val trainPath = args(0)
    val testPath = args(1)
    val numModels = if (args.length > 2) args(2).toInt else 10

    val spark = SparkSession.builder().appName("DesignLeverSmokeTest").getOrCreate()

    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).cache()
    val testData = HiggsLoader.loadAsArrays(spark, testPath).cache()
    val totalTrainRows = trainData.count()

    val params = TreeParams()

    // fraction = 1/numModels keeps each model's expected sample size
    // comparable to disjoint partitioning's, rather than the classic
    // full-size bootstrap (fraction=1.0) 
    val fairFraction = 1.0 / numModels
    val samplingChoices = Seq(
      "disjoint" -> SamplingStrategy.Disjoint,
      s"sampled-fraction($fairFraction)" -> SamplingStrategy.SampledFraction(fairFraction)
    )
    val aggregationChoices = Seq(
      "majority-vote" -> AggregationStrategy.MajorityVote,
      "mean-probability" -> AggregationStrategy.MeanProbability
    )

    for {
      (samplingName, sampling) <- samplingChoices
      (aggregationName, aggregation) <- aggregationChoices
    } {
      val config = ExperimentConfig(numModels, params, sampling, aggregation)
      val result = EnsembleExperiment.run(spark, trainData, testData, config)
      println(
        s"RESULT sampling=$samplingName aggregation=$aggregationName numModels=$numModels " +
          s"testCount=${result.count} accuracy=${result.accuracy} auc=${result.auc}"
      )
    }
    val disjointTrained = EnsembleTrainer.train(trainData, numModels, SamplingStrategy.Disjoint, params)
    val sampledTrained = EnsembleTrainer.train(trainData, numModels, SamplingStrategy.SampledFraction(1.0), params)

    val disjointTotal = disjointTrained.map(_.trainingRowCount).sum
    val sampledTotal = sampledTrained.map(_.trainingRowCount).sum
    val expectedSampledTotal = numModels * totalTrainRows

    println(s"[levers] totalTrainRows=$totalTrainRows disjointTotalTrained=$disjointTotal " +
      s"sampledTotalTrained=$sampledTotal expectedSampledTotal=$expectedSampledTotal")

    require(disjointTotal == totalTrainRows,
      s"disjoint partitioning must use every row exactly once: $disjointTotal != $totalTrainRows")
    require(
      math.abs(sampledTotal - expectedSampledTotal) < 0.1 * expectedSampledTotal,
      s"sampled-fraction total ($sampledTotal) too far from expected ($expectedSampledTotal)"
    )

    println("[levers] all checks passed")
    spark.stop()
  }
}
