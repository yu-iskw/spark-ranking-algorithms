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

import java.io.File

import scala.beans.BeanInfo

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.util.MLlibTestSparkContext

@BeanInfo
case class TestWilsonScoreIntervalData(docId: Long, positives: Long, negatives: Long)

class WilsonScoreIntervalSuite extends SparkFunSuite with MLlibTestSparkContext {

  private val data = Seq(
    TestWilsonScoreIntervalData(1L, 2L, 1L),
    TestWilsonScoreIntervalData(2L, 20L, 10L),
    TestWilsonScoreIntervalData(3L, 200L, 100L),
    TestWilsonScoreIntervalData(4L, 2000L, 1000L)
  )

  test("transform") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val wilsonScore = new WilsonScoreInterval()
      .setPositiveCol("positives")
      .setNegativeCol("negatives")
      .setOutputCol("score")

    val transformed = wilsonScore.transform(df)
    val scores = transformed.select("score").collect()
    assert(scores.length === 4)
    assert(scores(0).getDouble(0) === 0.22328763310073402)
    assert(scores(1).getDouble(0) === 0.553022430377575)
    assert(scores(2).getDouble(0) === 0.6316800063346981)
    assert(scores(3).getDouble(0) === 0.6556334308906774)
  }

  test("pipeline") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val wilsonScore = new WilsonScoreInterval()
      .setPositiveCol("positives")
      .setNegativeCol("negatives")
      .setOutputCol("score")
    val pipeline = new Pipeline()
        .setStages(Array(wilsonScore))
    val model = pipeline.fit(df)
    val transformed = model.transform(df)
    val scores = transformed.select("score").collect()
    assert(scores.length === 4)
    assert(scores(0).getDouble(0) === 0.22328763310073402)
    assert(scores(1).getDouble(0) === 0.553022430377575)
    assert(scores(2).getDouble(0) === 0.6316800063346981)
    assert(scores(3).getDouble(0) === 0.6556334308906774)
  }

  test("save/load") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val wilsonScore = new WilsonScoreInterval()
      .setPositiveCol("positives")
      .setNegativeCol("negatives")
      .setOutputCol("score")

    // save
    val path = File.createTempFile("spark-wilson-score", "").getAbsolutePath
    wilsonScore.write.overwrite().save(path)

    // load
    val loadedModel = WilsonScoreInterval.load(path)
    val transformed = loadedModel.transform(df)
    val scores = transformed.select("score").collect()
    assert(scores.length === 4)
    assert(scores(0).getDouble(0) === 0.22328763310073402)
    assert(scores(1).getDouble(0) === 0.553022430377575)
    assert(scores(2).getDouble(0) === 0.6316800063346981)
    assert(scores(3).getDouble(0) === 0.6556334308906774)
  }

  test("positives and negatives should be >= 0") {
    intercept[IllegalArgumentException] {
      WilsonScoreInterval.confidence(-1, 10)
    }
    intercept[IllegalArgumentException] {
      WilsonScoreInterval.confidence(10, -1)
    }
  }

  test("create an UDF for Wilson-score interval") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)
    val wilsonScoreInterval = WilsonScoreInterval.defineUDF(sqlContext)

    val scores = df.select(wilsonScoreInterval(df("positives"), df("negatives"))).collect()
    assert(scores(0).getDouble(0) === 0.22328763310073402)

    df.registerTempTable("test_data")
    val scores2 =
      sqlContext.sql("SELECT wilson_score_interval(positives, negatives) FROM test_data").collect()
    assert(scores2(0).getDouble(0) === 0.22328763310073402)
  }
}
