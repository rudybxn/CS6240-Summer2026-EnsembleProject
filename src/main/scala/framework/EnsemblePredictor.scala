package framework

import baselearner.BaseLearner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// One test row's ensemble-level result: a continuous score in [0, 1]. the
// fraction of models voting for label 1 under majority-vote, or the mean
// probability under mean-probability aggregation . Whichever aggregation produced it, the score alone is
// enough for both accuracy (threshold at 0.5) and AUC (rank by score).
case class Prediction(score: Double, predictedLabel: Int)

object EnsemblePredictor {

  // Dispatches to whichever aggregation strategy a config asks for
  def predict(spark: SparkSession,models: Array[BaseLearner],testData: RDD[(Array[Double], Int)],strategy: AggregationStrategy): RDD[(Prediction, Int)] = strategy match {
    case AggregationStrategy.MajorityVote =>
      predictMajorityVote(spark, models, testData)
    case AggregationStrategy.MeanProbability =>
      predictMeanProbability(spark, models, testData)
  }

  // Broadcasts the trained ensemble once, then applies every model to every
  // test row in parallel. The models (small: B single trees) travel to
  // wherever the test data already lives, rather than shuffling the (much
  // larger) test set to the models.
  def predictMajorityVote(spark: SparkSession,models: Array[BaseLearner],testData: RDD[(Array[Double], Int)]): RDD[(Prediction, Int)] = {
    val broadcastModels = spark.sparkContext.broadcast(models)

    testData.map { case (features, trueLabel) =>
      val ensemble = broadcastModels.value
      val votesFor1 = ensemble.count(model => model.predictProbability(features) >= 0.5)
      val fraction = votesFor1.toDouble / ensemble.length
      // Ties go to 0 
      val predictedLabel = if (votesFor1 * 2 > ensemble.length) 1 else 0
      (Prediction(fraction, predictedLabel), trueLabel)
    }
  }

  // Same broadcast, but averages each model's raw probability directly
  // instead of thresholding each model's vote first
  def predictMeanProbability(spark: SparkSession,models: Array[BaseLearner],testData: RDD[(Array[Double], Int)]): RDD[(Prediction, Int)] = {
    val broadcastModels = spark.sparkContext.broadcast(models)

    testData.map { case (features, trueLabel) =>
      val ensemble = broadcastModels.value
      val meanProbability = ensemble.map(_.predictProbability(features)).sum / ensemble.length
      val predictedLabel = if (meanProbability >= 0.5) 1 else 0
      (Prediction(meanProbability, predictedLabel), trueLabel)
    }
  }
}
