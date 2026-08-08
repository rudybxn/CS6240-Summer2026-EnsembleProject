package dataio

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

// Shared data loading
// Both the hand-written framework and the independent MLlib baseline read
// through here.
object HiggsLoader {

  // What the framework needs: plain arrays, so nothing downstream of this
  // ever has to import an MLlib or Spark ML type.
  def loadAsArrays(spark: SparkSession, parquetPath: String): RDD[(Array[Double], Int)] = {
    val df: DataFrame = spark.read.parquet(parquetPath)
    df.rdd.map { row =>
      val features = HiggsSchema.featureColumnNames.map(name => row.getAs[Double](name))
      val label = row.getAs[Double]("label").toInt
      (features, label)
    }
  }

  // What the MLlib baseline wants: a feature Vector + label column. 
  def loadAsMllibFrame(spark: SparkSession, parquetPath: String): DataFrame = {
    import org.apache.spark.ml.feature.VectorAssembler
    val raw = spark.read.parquet(parquetPath)
    new VectorAssembler()
      .setInputCols(HiggsSchema.featureColumnNames)
      .setOutputCol("features")
      .transform(raw)
      .select("features", "label")
  }
}
