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
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLlibTestSparkContext

@BeanInfo
case class TestData(text: String)

class OkapiBM25Suite extends SparkFunSuite with MLlibTestSparkContext {

  private val data = Seq(
    TestData("this is a pen"),
    TestData("this is an apple"),
    TestData("this is a pen this is a pen"),
    TestData("this is an apple this is an apple"),
    TestData("those are pens"),
    TestData("those are apples"),
    TestData("those are pens those are pens"),
    TestData("those are apples those are apples")
  )

  test("transform") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val tokenedDF = tokenizer.transform(df)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("token_count")
    val vectorizerModel = vectorizer.fit(tokenedDF)
    val vectorizedDF = vectorizerModel.transform(tokenedDF)

    val bm25 = new OkapiBM25()
      .setFeaturesCol("token_count")
      .setOutputCol("bm25")

    val model = bm25.fit(vectorizedDF)
    val transformed = model.transform(vectorizedDF)
    val bm25Vectors = transformed.select("bm25").collect().map(_.getAs[Vector](0).toSparse)
    assert(bm25Vectors.length === 8)
    assert(bm25Vectors(0).size === 10)
    assert(bm25Vectors(0).numActives === 4)
    assert(bm25Vectors(1).numActives === 4)
    assert(bm25Vectors(2).numActives === 4)
    assert(bm25Vectors(3).numActives === 4)

    // print debug
    //vectorizedDF.select("token_count").map{row => row.getAs[Vector](0).toDense}.foreach(println)
    //vectorizerModel.vocabulary.zipWithIndex.foreach{case (v, i) => println(s"$i: $v")}
    //bm25Vectors.foreach(println)
  }

  test("pipeline") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("token_count")
    val bm25 = new OkapiBM25()
      .setFeaturesCol("token_count")
      .setOutputCol("bm25")

    val pipeline = new Pipeline()
        .setStages(Array(tokenizer, vectorizer, bm25))
    val model = pipeline.fit(df)
    val transformed = model.transform(df)
    val bm25Vectors = transformed.select("bm25").collect().map(_.getAs[Vector](0).toSparse)
    assert(bm25Vectors.length === 8)
    assert(bm25Vectors(0).size === 10)
    assert(bm25Vectors(0).numActives === 4)
    assert(bm25Vectors(1).numActives === 4)
    assert(bm25Vectors(2).numActives === 4)
    assert(bm25Vectors(3).numActives === 4)
  }

  test("save/load") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val tokenedDF = tokenizer.transform(df)
    val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("token_count")
    val vectorizerModel = vectorizer.fit(tokenedDF)
    val vectorizedDF = vectorizerModel.transform(tokenedDF)

    val bm25 = new OkapiBM25()
      .setFeaturesCol("token_count")
      .setOutputCol("bm25")

    // save
    val model = bm25.fit(vectorizedDF)
    val path = File.createTempFile("spark-okapi-bm25", "").getAbsolutePath
    model.write.overwrite().save(path)

    // load
    val loadedModel = OkapiBM25Model.load(path)
    val transformed = loadedModel.transform(vectorizedDF)
    val bm25Vectors = transformed.select("bm25").collect().map(_.getAs[Vector](0).toSparse)
    assert(bm25Vectors.length === 8)
    assert(bm25Vectors(0).size === 10)
    assert(bm25Vectors(0).numActives === 4)
    assert(bm25Vectors(1).numActives === 4)
    assert(bm25Vectors(2).numActives === 4)
    assert(bm25Vectors(3).numActives === 4)
  }

  test("check paramters") {
    val bm25 = new OkapiBM25().setK1(1.9).setB(0.9)
    assert(bm25.getK1 === 1.9)
    assert(bm25.getB === 0.9)

    intercept[IllegalArgumentException] {
      new OkapiBM25().setK1(-0.1)
    }
    intercept[IllegalArgumentException] {
      new OkapiBM25().setK1(2.1)
    }
    intercept[IllegalArgumentException] {
      new OkapiBM25().setB(-0.1)
    }
    intercept[IllegalArgumentException] {
      new OkapiBM25().setB(1.1)
    }
  }
}
