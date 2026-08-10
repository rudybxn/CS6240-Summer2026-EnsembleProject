package hpsearch

import baselearner.TreeParams
import framework.{EnsemblePredictor, EnsembleTrainer}
import metrics.{EvaluationResult, Metrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object NestedPrefixSearch {

  // The shared-computation optimization: trains one ensemble at the largest
  // requested size, then evaluates every smaller requested size as a prefix
  // of those same trained models - no retraining per size, since only
  // prediction+aggregation+evaluation (much cheaper than a disjoint-
  // partition training pass) repeats per size.
  def run(spark: SparkSession,trainData: RDD[(Array[Double], Int)],
      testData: RDD[(Array[Double], Int)],
      modelCounts: Seq[Int],
      treeParams: TreeParams
  ): Seq[(Int, EvaluationResult)] = {
    require(modelCounts.nonEmpty, "modelCounts must not be empty")
    val maxModels = modelCounts.max
    val trained = EnsembleTrainer.trainDisjoint(trainData, maxModels, treeParams)
    val allModels = trained.map(_.learner)

    modelCounts.distinct.sorted.map { numModels =>
      val prefix = allModels.take(numModels)
      val predictions = EnsemblePredictor.predictMajorityVote(spark, prefix, testData)
      val scoresAndLabels = predictions.map { case (prediction, trueLabel) =>
        (prediction.score, trueLabel)
      }
      numModels -> Metrics.evaluate(scoresAndLabels)
    }
  }
}
