package framework

sealed trait AggregationStrategy

object AggregationStrategy {
  // Each model's probability thresholded at 0.5 first, then votes counted.
  case object MajorityVote extends AggregationStrategy

  // Raw probabilities averaged directly, thresholded once at the end.
  case object MeanProbability extends AggregationStrategy
}
