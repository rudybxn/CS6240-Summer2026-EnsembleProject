package hpsearch
import driver.{EnsembleExperiment, ExperimentConfig}
import metrics.EvaluationResult
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import java.util.concurrent.Executors
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

object NaiveSearch {

  // Runs every config as its own Spark job, submitted concurrently from a
  // thread pool rather than one after another There is no way to
  // avoid retraining here: each config's tree hyperparameters 
  // change what gets fit, unlike the ensemble-size sweep below.
  def run(spark: SparkSession,trainData: RDD[(Array[Double], Int)],testData: RDD[(Array[Double], Int)],configs: Seq[ExperimentConfig]): Seq[(ExperimentConfig, EvaluationResult)] = {
    val pool = Executors.newFixedThreadPool(configs.size max 1)
    implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(pool)
    val futures = configs.map { config =>
      Future(config -> EnsembleExperiment.run(spark, trainData, testData, config))
    }

    try {
      Await.result(Future.sequence(futures), Duration.Inf)
    } finally {
      pool.shutdown()
    }
  }
}
