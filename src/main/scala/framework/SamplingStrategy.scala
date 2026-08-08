package framework

sealed trait SamplingStrategy

object SamplingStrategy {
  // Disjoint-partition ("pasting"): every row assigned to exactly one model.
  case object Disjoint extends SamplingStrategy

  // Each model independently samples `fraction` of the data, with
  // replacement ("bagging"). fraction = 1.0 is the classic bootstrap.
  final case class SampledFraction(fraction: Double) extends SamplingStrategy
}
