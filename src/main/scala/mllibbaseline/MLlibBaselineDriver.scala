package mllibbaseline
import dataio.HiggsLoader
import metrics.Metrics
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

// Completely independent of the ensemble framework. imports nothing from
// baselearner/framework/hpsearch. Exists only as a comparison point for the
// hand-written framework's results
object MLlibBaselineDriver {
  def main(args: Array[String]): Unit = {
    require(args.length >= 2, "usage: MLlibBaselineDriver <train-parquet-path> <test-parquet-path>")
    val trainPath = args(0)
    val testPath = args(1)

    val spark = SparkSession.builder().appName("MLlibBaselineDriver").getOrCreate()

    val trainFrame = HiggsLoader.loadAsMllibFrame(spark, trainPath).cache()
    val testFrame = HiggsLoader.loadAsMllibFrame(spark, testPath).cache()
    trainFrame.count()
    testFrame.count()

    // maxDepth alone, matched to our framework's maxDepth=20, is not a fair
    // comparison: our TreeParams also caps maxNodes=100, and MLlib's tree
    // params have no direct node-count cap - at maxDepth=20 with nothing
    // bounding node count, a tree can grow far larger than ours ever could.
    // A full binary tree of depth 6 tops out at 127 nodes, close to our
    // 100-node cap, so depth is bounded here instead to keep the comparison
    // meaningful and the model tractable
    val comparableMaxDepth = 6

    // (1) Single model - a decision tree, the same model type our own base
    // learner wraps, so the two "single model" numbers are comparable.
    val tree = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(comparableMaxDepth)

    val treeStart = System.nanoTime()
    val treeModel = tree.fit(trainFrame)
    report("DecisionTree", treeModel, (System.nanoTime() - treeStart) / 1e9, testFrame)

    // (2) Ensemble - Random Forest, MLlib's own built-in equivalent of what
    // the framework instead builds by hand across the cluster.
    val forest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(40)
      .setMaxDepth(comparableMaxDepth)

    val forestStart = System.nanoTime()
    val forestModel = forest.fit(trainFrame)
    report("RandomForest", forestModel, (System.nanoTime() - forestStart) / 1e9, testFrame)

    spark.stop()
  }

  private def report(name: String, model: Model[_], trainSeconds: Double, testFrame: DataFrame): Unit = {
    val predictions = model.transform(testFrame)
    val result = Metrics.evaluate(scoresAndLabels(predictions))
    println(
      f"RESULT model=$name trainSeconds=$trainSeconds%.2f " +
        s"accuracy=${result.accuracy} auc=${result.auc}"
    )
  }

  // (P(label=1), trueLabel). the same shape metrics.Metrics.evaluate
  // expects from the framework's own predictions, so both are graded by
  // identical evaluation code.
  private def scoresAndLabels(predictions: DataFrame): RDD[(Double, Int)] = {
    predictions.select("probability", "label").rdd.map { row =>
      val probability = row.getAs[Vector]("probability")
      val label = row.getAs[Double]("label").toInt
      (probability(1), label)
    }
  }
}
