package driver
import baselearner.TreeParams
import framework.{AggregationStrategy, EnsemblePredictor, EnsembleTrainer, SamplingStrategy}
import metrics.{EvaluationResult, Metrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// One ensemble configuration: how many base models, what hyperparameters
// each trains with, and which half of each design lever to use. 
case class ExperimentConfig(
    numModels: Int,
    treeParams: TreeParams,
    samplingStrategy: SamplingStrategy = SamplingStrategy.Disjoint,
    aggregationStrategy: AggregationStrategy = AggregationStrategy.MajorityVote
)

object EnsembleExperiment {

  // Trains one ensemble and evaluates it against a held-out test set. This
  // is the one place train -> predict -> evaluate is wired together. the
  // hyperparameter-search module calls this repeatedly across many configs,
  // so it takes already-loaded RDDs rather than paths: the caller loads and
  // caches train/test once, then reuses them across every call here.
  def run(spark: SparkSession,trainData: RDD[(Array[Double], Int)],testData: RDD[(Array[Double], Int)],config: ExperimentConfig): EvaluationResult = {
    val trained = EnsembleTrainer.train(trainData, config.numModels, config.samplingStrategy, config.treeParams)
    val models = trained.map(_.learner)
    val predictions = EnsemblePredictor.predict(spark, models, testData, config.aggregationStrategy)
    val scoresAndLabels = predictions.map { case (prediction, trueLabel) =>
      (prediction.score, trueLabel)
    }

    Metrics.evaluate(scoresAndLabels)
  }
}
