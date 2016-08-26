/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{UserDefinedFunction, DataFrame, SQLContext}


/**
  * :: Experimental ::
  * Parameter trait for `WilsonScoreInterval`
  */
private[feature]
trait WilsonScoreIntervalParams extends Params with HasOutputCol {

  /**
    * Set the column name for the number of positive reviews
    * @group param
    */
  final val positiveCol = new Param[String](this, "positiveCol", "positive column name")

  /** @group getParam */
  def getPositiveCol: String = $(positiveCol)

  /**
    * Set the column name for the number of negative reviews
    * @group param
    */
  final val negativeCol = new Param[String](this, "negativeCol", "negative column name")

  /** @group getParam */
  def getNegativeCol: String = $(negativeCol)
}


/**
  * :: Experimental ::
  * This transformer is used for calculating scores which is lower bound of
  * Wilson score confidence interval for a Bernoulli parameter
  *
  * https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
  */
@Experimental
class WilsonScoreInterval(override val uid: String)
  extends Transformer with DefaultParamsWritable with WilsonScoreIntervalParams {

  // Sets the default values
  setDefault(
    outputCol -> "score"
  )

  def this() = this(Identifiable.randomUID("wilsonscore"))

  /** @group setParam */
  def setPositiveCol(value: String): this.type = set(positiveCol, value)

  /** @group setParam */
  def setNegativeCol(value: String): this.type = set(negativeCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def copy(extra: ParamMap): WilsonScoreInterval = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val positiveColType = schema($(positiveCol)).dataType
    validatePositiveColType(positiveColType)
    val negativeColType = schema($(negativeCol)).dataType
    validateNegativeColumnType(negativeColType)

    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+ StructField($(outputCol), DoubleType, nullable = false)
    StructType(outputFields)
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val scoreUdf = udf(WilsonScoreInterval.createTransformFunc)
    dataset.withColumn($(outputCol), scoreUdf(col(getPositiveCol), col(getNegativeCol)))
  }

  protected def outputDataType: DataType = new ArrayType(StringType, true)

  protected def validatePositiveColType(positiveColType: DataType): Unit = {
    require(positiveColType == LongType,
      s"Positive column type must be long type but got $positiveColType.")
  }

  protected def validateNegativeColumnType(negativeColtype: DataType): Unit = {
    require(negativeColtype == LongType,
      s"Negative column type must be long type but got $negativeColtype.")
  }
}


/**
  * :: Experimental ::
  * A companion object for `WilsonScoreInterval`
  */
@Experimental
object WilsonScoreInterval extends DefaultParamsReadable[WilsonScoreInterval] {

  override def load(path: String): WilsonScoreInterval = super.load(path)

  protected def createTransformFunc: (Long, Long) => Double = {
    (pos: Long, neg: Long) => WilsonScoreInterval.confidence(pos, neg)
  }

  def defineUDF(sqlContext: SQLContext): UserDefinedFunction = {
    sqlContext.udf.register("wilson_score_interval", WilsonScoreInterval.createTransformFunc)
  }

  /**
    * Calculates lower bound of Wilson score confidence interval for a Bernoulli parameter
    *
    * SEE ALSO:
    * https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    *
    * @param positives the number of positive reviews
    * @param negatives the number of negative reviews
    * @return wilson score interval
    */
  def confidence(positives: Long, negatives: Long): Double = {
    if (positives < 0) {
      throw new IllegalArgumentException(s"the number of positives should be >= 0, but $positives")
    }
    if (negatives < 0) {
      throw new IllegalArgumentException(s"the number of negatives should be >= 0, but $negatives")
    }

    val n = positives + negatives
    if (n == 0) {
      return 0.0
    }
    else {
      val z = 1.281551565545
      val p = 1.0 * positives / n

      val left = p + 1 / (2 * n) * z * z
      val right = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
      val under = 1 + 1 / n * z * z

      return (left - right) / under
    }
  }
}
