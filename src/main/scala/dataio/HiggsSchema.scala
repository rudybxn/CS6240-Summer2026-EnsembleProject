package dataio

import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

// HIGGS.csv has no header: column 0 is the label, columns 1-28 are features,
// all real-valued. 
object HiggsSchema {

  val numFeatures = 28

  val featureColumnNames: Array[String] = Array.tabulate(numFeatures)(i => s"f${i + 1}")

  val raw: StructType = StructType(
    StructField("label", DoubleType, nullable = false) +:
      featureColumnNames.map(name => StructField(name, DoubleType, nullable = false))
  )
}
