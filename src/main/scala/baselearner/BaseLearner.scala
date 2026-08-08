package baselearner

import smile.base.cart.SplitRule
import smile.classification.DecisionTree
import smile.data.{DataFrame, Tuple}
import smile.data.formula.Formula
import smile.data.`type`.StructType
import smile.data.vector.{BaseVector, DoubleVector, IntVector}

// Hyperparameters for one base-learner tree. Kept separate from Smile's own
// types so the hyperparameter-search module never has to import Smile -
// only this module does.
case class TreeParams(maxDepth: Int = 20, maxNodes: Int = 100, nodeSize: Int = 5)

// A trained base learner. Serializable so the framework can broadcast a
// whole ensemble of these to every executor for parallel prediction.
trait BaseLearner extends Serializable {
  // P(label = 1) for one feature vector. Both aggregation strategies -
  // majority vote (threshold at 0.5) or mean probability - need only this.
  def predictProbability(features: Array[Double]): Double
}

// The only place smile-core is imported anywhere in this project.
object BaseLearner {

  def fit(features: Array[Array[Double]], labels: Array[Int], params: TreeParams): BaseLearner = {
    val numFeatures = if (features.nonEmpty) features(0).length else 0

    val featureColumns = Array.tabulate(numFeatures) { j =>
      DoubleVector.of(s"x$j", Array.tabulate(features.length)(i => features(i)(j)))
    }
    val labelColumn = IntVector.of("label", labels)

    // Built as an explicitly-typed array (not via :+) because Scala can't
    // cleanly infer a common BaseVector type when mixing DoubleVector and
    // IntVector through array concatenation.
    val allColumns = new Array[BaseVector[_, _, _]](numFeatures + 1)
    for (j <- 0 until numFeatures) allColumns(j) = featureColumns(j)
    allColumns(numFeatures) = labelColumn

    val trainDf = DataFrame.of(allColumns: _*)

    val tree = DecisionTree.fit(
      Formula.lhs("label"), trainDf, SplitRule.GINI, params.maxDepth, params.maxNodes, params.nodeSize
    )

    // The model's formula binds against the full schema (it looks up "label"
    // by name even though prediction never reads its value) - so single-row
    // prediction Tuples need that same full schema, label slot included.
    new SmileDecisionTreeLearner(tree, trainDf.schema())
  }
}

private class SmileDecisionTreeLearner(tree: DecisionTree, fullSchema: StructType)
    extends BaseLearner {

  override def predictProbability(features: Array[Double]): Double = {
    val values: Array[Object] =
      Array.tabulate[Object](features.length + 1) { i =>
        if (i < features.length) java.lang.Double.valueOf(features(i))
        else Integer.valueOf(0) // dummy label - declared in the schema, never read by predict
      }
    val row = Tuple.of(values, fullSchema)

    val posteriori = new Array[Double](2)
    tree.predict(row, posteriori)
    posteriori(1)
  }
}
