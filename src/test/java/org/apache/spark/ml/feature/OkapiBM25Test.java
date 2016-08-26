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

package org.apache.spark.ml.feature;


import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class OkapiBM25Test {
  private transient JavaSparkContext jsc;
  private transient SQLContext jsql;

  private List<Row> data = Arrays.asList(
      RowFactory.create("this is a pen"),
      RowFactory.create("this is an apple"),
      RowFactory.create("this is a pen this is a pen"),
      RowFactory.create("this is an apple this is an apple"),
      RowFactory.create("those are pens"),
      RowFactory.create("those are apples"),
      RowFactory.create("those are pens those are pens"),
      RowFactory.create("those are apples those are apples")
  );

  @Before
  public void setUp() {
    jsc = new JavaSparkContext("local", "JavaOkapiBM25Suite");
    jsql = new SQLContext(jsc);
  }

  @After
  public void tearDown() {
    jsc.stop();
    jsc = null;
  }

  @Test
  public void testRun() {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("text", DataTypes.StringType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens");
    DataFrame tokenizedDF = tokenizer.transform(df);
    CountVectorizer vectorizer = new CountVectorizer()
        .setInputCol("tokens").setOutputCol("token_count");
    CountVectorizerModel vectorizerModel = vectorizer.fit(tokenizedDF);
    DataFrame vectorizedDF = vectorizerModel.transform(tokenizedDF);

    OkapiBM25 bm25 = new OkapiBM25()
        .setFeaturesCol("token_count")
        .setOutputCol("bm25");

    OkapiBM25Model model = bm25.fit(vectorizedDF);
    DataFrame transformed = model.transform(vectorizedDF);
    long count = transformed.select("bm25").count();
    assertEquals(8, count);
  }

  @Test
  public void testPipeline() {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("text", DataTypes.StringType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens");
    CountVectorizer vectorizer = new CountVectorizer()
        .setInputCol("tokens").setOutputCol("token_count");
    OkapiBM25 bm25 = new OkapiBM25().setFeaturesCol("token_count").setOutputCol("bm25");

    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] {tokenizer, vectorizer, bm25});

    PipelineModel model = pipeline.fit(df);
    DataFrame transformed = model.transform(df);
    long count = transformed.select("bm25").count();
    assertEquals(8, count);
  }

  @Test
  public void testSaveAndLoad() throws IOException {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("text", DataTypes.StringType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens");
    DataFrame tokenizedDF = tokenizer.transform(df);
    CountVectorizer vectorizer = new CountVectorizer()
        .setInputCol("tokens").setOutputCol("token_count");
    CountVectorizerModel vectorizerModel = vectorizer.fit(tokenizedDF);
    DataFrame vectorizedDF = vectorizerModel.transform(tokenizedDF);

    OkapiBM25 bm25 = new OkapiBM25().setFeaturesCol("token_count").setOutputCol("bm25");
    OkapiBM25Model model = bm25.fit(vectorizedDF);

    String path = File.createTempFile("spark-wilson-score", "java").getAbsolutePath();
    model.write().overwrite().save(path);
    OkapiBM25Model loaded = OkapiBM25Model.load(path);

    DataFrame transformed = loaded.transform(vectorizedDF);
    long count = transformed.select("bm25").count();
    assertEquals(8, count);
  }
}
