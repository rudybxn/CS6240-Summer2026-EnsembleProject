package smoketest

import baselearner.TreeParams
import dataio.HiggsLoader
import framework.EnsembleTrainer
import org.apache.spark.sql.SparkSession

// does reducing "fraction" (rather than throwing more driver memory at it) avoid the OOM that
// SampledFraction(1.0) + high numModels hit at 1M rows? Times a single
// trainSampledFraction call and reports whether it survived under whatever
// memory the caller gave spark-submit
object SampledFractionMemoryCheck {

  def main(args: Array[String]): Unit = {
    require(
      args.length >= 3,
      "usage: SampledFractionMemoryCheck <train-parquet-path> <numModels> <fraction>"
    )
    val trainPath = args(0)
    val numModels = args(1).toInt
    val fraction = args(2).toDouble

    val spark = SparkSession.builder().appName("SampledFractionMemoryCheck").getOrCreate()

    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).cache()
    val totalRows = trainData.count()

    val start = System.nanoTime()
    val trained = EnsembleTrainer.trainSampledFraction(trainData, numModels, fraction, TreeParams())
    val elapsed = (System.nanoTime() - start) / 1e9

    val totalTrained = trained.map(_.trainingRowCount).sum
    val expected = numModels * totalRows * fraction
    println(
      f"CHECK numModels=$numModels fraction=$fraction totalRows=$totalRows " +
        f"totalTrained=$totalTrained expected=$expected%.0f elapsedSeconds=$elapsed%.2f"
    )

    spark.stop()
  }
}
