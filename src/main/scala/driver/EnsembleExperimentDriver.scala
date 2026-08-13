package driver
import baselearner.TreeParams
import dataio.HiggsLoader
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

// The experiment entry point: load train/test once, cache them, run one
// ensemble configuration, print the result as a single line so
// many runs can later be collecte into a table without re-parsing output.
object EnsembleExperimentDriver {

  def main(args: Array[String]): Unit = {
    require(
      args.length >= 2,
      "usage: EnsembleExperimentDriver <train-parquet-path> <test-parquet-path> [numModels]"
    )
    val trainPath = args(0)
    val testPath = args(1)
    val numModels = if (args.length > 2) args(2).toInt else 10
    val spark = SparkSession.builder().appName("EnsembleExperimentDriver").getOrCreate()

    val trainData = HiggsLoader.loadAsArrays(spark, trainPath).persist(StorageLevel.MEMORY_AND_DISK)
    val testData = HiggsLoader.loadAsArrays(spark, testPath).persist(StorageLevel.MEMORY_AND_DISK)
    val config = ExperimentConfig(numModels, TreeParams())
    val result = EnsembleExperiment.run(spark, trainData, testData, config)

    println(
      s"RESULT numModels=${config.numModels} maxDepth=${config.treeParams.maxDepth} " +
        s"maxNodes=${config.treeParams.maxNodes} nodeSize=${config.treeParams.nodeSize} " +
        s"testCount=${result.count} accuracy=${result.accuracy} auc=${result.auc}"
    )
    spark.stop()
  }
}
