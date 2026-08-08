package smoketest
import org.apache.spark.sql.SparkSession
import smile.classification.RandomForest
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.data.vector.{DoubleVector, IntVector}


//   1. Spark runs locally (SparkSession, an RDD transformation, a collect).
//   2. Smile fits and predicts a classifier from data built in memory - not
//      read from a file - the trickiest part of integrating an external
//      single-machine ML library into a Spark task.
object ToolchainSmokeTest {

  def main(args: Array[String]): Unit = {
    checkSpark()
    checkSmile()
  }

  private def checkSpark(): Unit = {
    val spark = SparkSession.builder()
      .appName("ToolchainSmokeTest")
      .master("local[*]")
      .getOrCreate()

    val total = spark.sparkContext.parallelize(1 to 1000).map(_.toLong).sum().toLong
    println(s"[spark] sum of 1..1000 via RDD = $total (expected 500500)")

    spark.stop()
  }

  private def checkSmile(): Unit = {
    // Two well-separated clusters, 20 points each.
    val n = 20
    val x1 = Array.tabulate(2 * n)(i => if (i < n) (i % 5) * 0.2 else 10.0 + ((i - n) % 5) * 0.2)
    val x2 = Array.tabulate(2 * n)(i => if (i < n) (i / 5) * 0.2 else 10.0 + ((i - n) / 5) * 0.2)
    val y  = Array.tabulate(2 * n)(i => if (i < n) 0 else 1)

    val df = DataFrame.of(
      DoubleVector.of("x1", x1),
      DoubleVector.of("x2", x2),
      IntVector.of("label", y)
    )

    val model = RandomForest.fit(Formula.lhs("label"), df)

    val posteriori = new Array[Double](2)
    val predicted = model.predict(df.get(0), posteriori)

    println(s"[smile] predicted class for row 0 = $predicted, " +
      s"posteriori = ${posteriori.mkString("[", ", ", "]")}")
  }
}
