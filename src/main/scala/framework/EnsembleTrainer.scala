package framework
import baselearner.{BaseLearner, TreeParams}
import org.apache.spark.rdd.RDD
import scala.util.Random

// A trained base learner plus how many rows it was trained on. The row count
// is what "rows-per-model must fit in a task's
// memory" (the partitioning-granularity constraint) actually gets measured
// against.
case class TrainedModel(learner: BaseLearner, trainingRowCount: Int)

object EnsembleTrainer {

  // Dispatches to whichever sampling strategy a config asks for, so callers
  // (the experiment driver, hyperparameter search) don't need their own
  // pattern match.
  def train(data: RDD[(Array[Double], Int)],numModels: Int,strategy: SamplingStrategy,params: TreeParams): Array[TrainedModel] = strategy match {
    case SamplingStrategy.Disjoint =>
      trainDisjoint(data, numModels, params)
    case SamplingStrategy.SampledFraction(fraction) =>
      trainSampledFraction(data, numModels, fraction, params)
  }

  // Disjoint-partition: every row is assigned to exactly one of
  // numModels partitions, by row order rather than row content, so partition
  // sizes are as balanced as the input allows (each gets either
  // floor(n/numModels) or one more). One BaseLearner is trained per
  // partition, entirely from data local to that partition's task. no
  // replication, no shuffle beyond the single repartition below.
  def trainDisjoint(data: RDD[(Array[Double], Int)],numModels: Int,params: TreeParams): Array[TrainedModel] = {
    val keyed = data.zipWithIndex().map { case (row, idx) =>
      val partitionId = (idx % numModels).toInt
      (partitionId, row)
    }
    val partitioned = keyed.partitionBy(new ModelPartitioner(numModels))

    val models = partitioned.mapPartitions { rows =>
      val rowsArray = rows.toArray
      val features = rowsArray.map(_._2._1)
      val labels = rowsArray.map(_._2._2)
      val learner = BaseLearner.fit(features, labels, params)
      Iterator(TrainedModel(learner, rowsArray.length))
    }

    models.collect()
  }

  // Sampled-fraction ("bagging"): each of numModels models independently
  // samples `fraction` of the data, with replacement.a row can land in
  // zero, one, or many models' training sets, and can repeat within one
  // model's set. Implemented in a single pass with a Poisson approximation
  // to sampling-with-replacement (the standard technique for this - drawing
  // k rows with replacement from n rows means any given row's count in the
  // sample is ~Poisson(k/n)), rather than re-scanning the data
  // numModels times.
  def trainSampledFraction(data: RDD[(Array[Double], Int)],numModels: Int,fraction: Double,params: TreeParams,seed: Long = 42L): Array[TrainedModel] = {
    require(numModels > 0, "numModels must be positive")
    require(fraction > 0.0, "fraction must be positive")

    val replicated = data.mapPartitionsWithIndex { (partitionIndex, rows) =>
      val rnd = new Random(seed + partitionIndex)
      rows.flatMap { row =>
        (0 until numModels).iterator.flatMap { modelId =>
          val copies = poissonDraw(fraction, rnd)
          Iterator.fill(copies)((modelId, row))
        }
      }
    }

    val partitioned = replicated.partitionBy(new ModelPartitioner(numModels))

    val models = partitioned.mapPartitions { rows =>
      val rowsArray = rows.toArray
      val features = rowsArray.map(_._2._1)
      val labels = rowsArray.map(_._2._2)
      val learner = BaseLearner.fit(features, labels, params)
      Iterator(TrainedModel(learner, rowsArray.length))
    }

    models.collect()
  }

  // Knuth's algorithm for a Poisson-distributed random draw.
  private def poissonDraw(mean: Double, rnd: Random): Int = {
    val threshold = math.exp(-mean)
    var k = 0
    var p = 1.0
    do {
      k += 1
      p *= rnd.nextDouble()
    } while (p > threshold)
    k - 1
  }
}
