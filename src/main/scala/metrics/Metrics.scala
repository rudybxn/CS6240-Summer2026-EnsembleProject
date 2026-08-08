package metrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

case class EvaluationResult(accuracy: Double, auc: Double, count: Long)

object Metrics {
  // Shared evaluation for both the ensemble framework and the MLlib
  // baseline: (score, trueLabel) pairs in, accuracy + AUC out. thresholded at 0.5 for accuracy the same way.
  def evaluate(scoresAndLabels: RDD[(Double, Int)]): EvaluationResult = {
    val cached = scoresAndLabels.cache()
    val total = cached.count()

    val correct = cached.filter { case (score, label) =>
      val predicted = if (score >= 0.5) 1 else 0
      predicted == label
    }.count()

    val auc = new BinaryClassificationMetrics(
      cached.map { case (score, label) => (score, label.toDouble) }
    ).areaUnderROC()

    EvaluationResult(correct.toDouble / total, auc, total)
  }
}
