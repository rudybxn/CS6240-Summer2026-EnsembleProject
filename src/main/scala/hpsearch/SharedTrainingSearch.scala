package hpsearch

import baselearner.TreeParams
import driver.ExperimentConfig
import framework.{AggregationStrategy, EnsemblePredictor, EnsembleTrainer, SamplingStrategy}
import metrics.{EvaluationResult, Metrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import java.util.concurrent.Executors
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

// Everything that determines what an ensemble actually is. Aggregation is
// deliberately absent as  it changes how already-trained models are read, not
// what gets trained.
case class TrainingConfig(
    numModels: Int,
    treeParams: TreeParams,
    samplingStrategy: SamplingStrategy
)

object SharedTrainingSearch {

  // The second shared-computation optimization
  //
  // A full sweep over all four design levers is the cartesian product of
  // ensemble size x tree settings x sampling x aggregation. Handing that
  // whole product to NaiveSearch would train one ensemble per combination -
  // but aggregation only affects prediction, so every pair of configs that
  // differs in aggregation would train the same models twice.
 
  def run(spark: SparkSession,trainData: RDD[(Array[Double], Int)],testData: RDD[(Array[Double], Int)],trainingConfigs: Seq[TrainingConfig],aggregations: Seq[AggregationStrategy]): Seq[(ExperimentConfig, EvaluationResult)] = {
    require(trainingConfigs.nonEmpty, "trainingConfigs must not be empty")
    require(aggregations.nonEmpty, "aggregations must not be empty")

    val pool = Executors.newFixedThreadPool(trainingConfigs.size max 1)
    implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(pool)

    val futures = trainingConfigs.map { tc =>
      Future {
        // One training pass, shared by every aggregation below.
        val trained = EnsembleTrainer.train(trainData, tc.numModels, tc.samplingStrategy, tc.treeParams)
        val models = trained.map(_.learner)

        aggregations.map { aggregation =>
          val predictions = EnsemblePredictor.predict(spark, models, testData, aggregation)
          val scoresAndLabels = predictions.map { case (prediction, trueLabel) =>
            (prediction.score, trueLabel)
          }
          val config = ExperimentConfig(tc.numModels, tc.treeParams, tc.samplingStrategy, aggregation)
          config -> Metrics.evaluate(scoresAndLabels)
        }
      }
    }

    try {
      Await.result(Future.sequence(futures), Duration.Inf).flatten
    } finally {
      pool.shutdown()
    }
  }
}
