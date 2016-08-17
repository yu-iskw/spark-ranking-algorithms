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

import breeze.linalg.{Vector => BVector}
import org.apache.hadoop.fs.Path

import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * :: Experimental ::
  * Parameter trait for `OkapiBM25`
  */
private[feature]
trait OkapiBM25Params extends Params with HasFeaturesCol with HasOutputCol {

  /**
    * Controls non-linear term frequency normalization (saturation).
    * @group expertParam
    */
  final val k1 = new DoubleParam(this, "k1",
    "Controls non-linear term frequency normalization (saturation)",
    (v: Double) => v >= 1.0 && v <= 2.0)

  /** @group expertGetParam */
  def getK1: Double = $(k1)

  /**
    * Controls to what degree document length normalizes tf values.
    * @group expertParam
    */
  final val b = new DoubleParam(this, "b",
    "Controls to what degree document length normalizes tf values",
    (v: Double) => v >= 0.0 && v <= 1.0)

  /** @group expertGetParam */
  def getB: Double = $(b)

  /**
    * Validates and transforms the input schema.
    *
    * @param schema input schema
    * @return output schema
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
  * :: Experimental ::
  * Okapi BM25 (BM stands for Best Matching) is a ranking function used
  * by search engines to rank matching documents according to their
  * relevance to a given search query.
  */
@Experimental
class OkapiBM25(override val uid: String)
  extends Estimator[OkapiBM25Model] with DefaultParamsWritable with OkapiBM25Params {

  // Sets the default values
  setDefault(
    k1 -> 1.2,
    b -> 0.75
  )

  def this() = this(Identifiable.randomUID("wilsonscore"))

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group expertSetParam */
  def setK1(value: Double): this.type = set(k1, value)

  /** @group expertSetParam */
  def setB(value: Double): this.type = set(b, value)

  override def copy(extra: ParamMap): Estimator[OkapiBM25Model] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def fit(dataset: DataFrame): OkapiBM25Model = {
    // Computes IDF
    val idf = new IDF().setInputCol($(featuresCol)).setOutputCol("idf_output")
    val idfModel = idf.fit(dataset)
    // Computes mean document length
    val documentLengthUDf = udf((vector: Vector) => Vectors.norm(vector, 1.0))
    val meanDocumentLength =
      dataset.select(avg(documentLengthUDf(col($(featuresCol))))).head().getAs[Double](0)

    val model = new OkapiBM25Model(uid, idfModel.idf, meanDocumentLength)
    copyValues(model)
  }

}

/**
  * :: Experimental ::
  * A companion object for `OkapiBM25`
  */
@Experimental
object OkapiBM25 extends DefaultParamsReadable[OkapiBM25] {
  override def load(path: String): OkapiBM25 = super.load(path)
}

/**
  * :: Experimental ::
  * Model fitted by Okapi BM25
  */
@Experimental
class OkapiBM25Model private[ml](override val uid: String,
    idf: Vector,
    meanDocumentLength: Double)
  extends Model[OkapiBM25Model] with OkapiBM25Params with MLWritable {

  def getIdf: Vector = this.idf

  def getMeanDocumentLength: Double = this.meanDocumentLength

  /** @group expertSetParam */
  def setK1(value: Double): this.type = set(k1, value)

  /** @group expertSetParam */
  def setB(value: Double): this.type = set(b, value)

  override def copy(extra: ParamMap): OkapiBM25Model = {
    val copied = new OkapiBM25Model(uid, this.idf, this.meanDocumentLength)
    copyValues(copied, extra)
  }

  override def write: MLWriter = new OkapiBM25Model.OkapiBM25Writer(this)

  override def transform(dataset: DataFrame): DataFrame = {
    // Creates an UDF to calculate Okapi BM25
    val bm25 = udf((termFreq: Vector) =>
      OkapiBM25Model.calculateBM25(termFreq, this.idf, this.meanDocumentLength, getK1, getB))
    dataset.withColumn($(outputCol), bm25(col($(featuresCol))))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}


/**
  * :: Experimental ::
  * XXX
  */
@Experimental
object OkapiBM25Model extends MLReadable[OkapiBM25Model] {

  override def read: MLReader[OkapiBM25Model] = new OkapiBM25Reader

  override def load(path: String): OkapiBM25Model = super.load(path)

  /** [[MLWriter]] instance for [[OkapiBM25]] */
  private[OkapiBM25Model] class OkapiBM25Writer(instance: OkapiBM25Model) extends MLWriter {

    private case class Data(idf: Vector, meanDocumentLength: Double)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: cluster centers
      val data = Data(instance.getIdf, instance.getMeanDocumentLength)
      val dataPath = new Path(path, "data").toString
      sqlContext.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class OkapiBM25Reader extends MLReader[OkapiBM25Model] {

    /** Checked against metadata when loading model */
    private val className = classOf[OkapiBM25Model].getName

    override def load(path: String): OkapiBM25Model = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sqlContext.read.parquet(dataPath).select("idf", "meanDocumentLength").head()
      val idf = data.getAs[Vector](0)
      val meanDocumentLength = data.getDouble(1)
      val model = new OkapiBM25Model(metadata.uid, idf, meanDocumentLength)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  /**
    * Computes Okapi BM25
    *
    * @param termFreq term frequency
    * @param idf inversed document frequency
    * @param meanDocumentLength mean document length
    * @param k1 controls non-linear term frequency normalization (saturation).
    * @param b controls to what degree document length normalizes tf values.
    * @return Okapi BM25 as Vector
    */
  private[feature]
  def calculateBM25(termFreq: Vector,
      idf: Vector,
      meanDocumentLength: Double,
      k1: Double = 1.2,
      b: Double = 0.75): Vector = {
    if (k1 < 1.0 && k1 > 2.0) {
      throw new IllegalArgumentException(s"k1 should be >= 1.0 and <= 2.0, but $k1")
    }
    if (b < 0.0 && b > 1.0) {
      throw new IllegalArgumentException(s"b should be >= 0.0 and <= 1.0, but $b")
    }
    // Calculates normalized document length
    val documentLength = Vectors.norm(termFreq, 1.0)
    val ndl = documentLength / meanDocumentLength
    // CW(i,j) = [ TF(i,j) * IDF(i) * (K1 + 1) ] / [ K1 * (1 - b + (b * NDL(j)) + TF(i,j) ]
    val numerator: BVector[Double] = termFreq.toBreeze :* idf.toBreeze :* (k1 + 1)
    val denominator: BVector[Double] = termFreq.toBreeze :+ (k1 * (1 - b + b * ndl))
    Vectors.fromBreeze(numerator :/ denominator).toSparse
  }
}
