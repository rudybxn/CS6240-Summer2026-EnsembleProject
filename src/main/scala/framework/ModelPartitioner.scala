package framework

import org.apache.spark.Partitioner

// Custome partititoner. Routes each row to one of numModels model-partitions. Callers key
// rows by (rowIndex % numModels) before using this with partitionBy, so
// getPartition only needs to trust that the key is already a valid id.
class ModelPartitioner(val numModels: Int) extends Partitioner {
  require(numModels > 0, "numModels must be positive")

  override def numPartitions: Int = numModels

  override def getPartition(key: Any): Int = key.asInstanceOf[Int]

  override def equals(other: Any): Boolean = other match {
    case p: ModelPartitioner => p.numModels == numModels
    case _ => false
  }

  override def hashCode(): Int = numModels
}
