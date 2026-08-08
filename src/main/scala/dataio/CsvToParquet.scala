package dataio

import org.apache.spark.sql.SparkSession

// One-time conversion: raw HIGGS CSV -> train/test Parquet, split by the
// official convention (last `testSize` rows = test set). Uses
// RDD.zipWithIndex, which preserves the
// original file's row order across partitions 
object CsvToParquet {

  def main(args: Array[String]): Unit = {
    require(
      args.length >= 3,
      "usage: CsvToParquet <input-csv[.gz]> <train-out-parquet> <test-out-parquet> [testSize]"
    )
    val inputPath = args(0)
    val trainOutPath = args(1)
    val testOutPath = args(2)
    val testSize = if (args.length > 3) args(3).toLong else 500000L

    val spark = SparkSession.builder().appName("CsvToParquet").getOrCreate()

    val raw = spark.read.schema(HiggsSchema.raw).csv(inputPath)

    val indexed = raw.rdd.zipWithIndex()
    val total = indexed.count()
    val trainCutoff = total - testSize
    println(s"[csvToParquet] total rows = $total, train = $trainCutoff, test = $testSize")

    val trainRdd = indexed.filter { case (_, idx) => idx < trainCutoff }.map(_._1)
    val testRdd = indexed.filter { case (_, idx) => idx >= trainCutoff }.map(_._1)

    spark.createDataFrame(trainRdd, HiggsSchema.raw).write.mode("overwrite").parquet(trainOutPath)
    spark.createDataFrame(testRdd, HiggsSchema.raw).write.mode("overwrite").parquet(testOutPath)

    println(s"[csvToParquet] wrote train -> $trainOutPath, test -> $testOutPath")
    spark.stop()
  }
}
