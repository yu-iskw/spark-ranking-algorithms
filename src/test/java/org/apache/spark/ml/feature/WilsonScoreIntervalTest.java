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

public class WilsonScoreIntervalTest {
  private transient JavaSparkContext jsc;
  private transient SQLContext jsql;

  private List<Row> data = Arrays.asList(
      RowFactory.create(1L, 2L, 1L),
      RowFactory.create(2L, 20L, 10L),
      RowFactory.create(2L, 200L, 100L),
      RowFactory.create(2L, 2000L, 1000L)
  );

  @Before
  public void setUp() {
    jsc = new JavaSparkContext("local", "JavaKuromojiTokenizerSuite");
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
        DataTypes.createStructField("docId", DataTypes.LongType, false),
        DataTypes.createStructField("positives", DataTypes.LongType, false),
        DataTypes.createStructField("negatives", DataTypes.LongType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    WilsonScoreInterval wilsonScore = new WilsonScoreInterval()
        .setPositiveCol("positives")
        .setNegativeCol("negatives")
        .setOutputCol("score");
    DataFrame transformed = wilsonScore.transform(df);
    List<Row> scoreList = transformed.select("score").collectAsList();
    assertEquals(0.22328763310073402, scoreList.get(0).getDouble(0), 1e-5);
    assertEquals(0.553022430377575, scoreList.get(1).getDouble(0), 1e-5);
    assertEquals(0.6316800063346981, scoreList.get(2).getDouble(0), 1e-5);
    assertEquals(0.6556334308906774, scoreList.get(3).getDouble(0), 1e-5);
  }

  @Test
  public void testPipeline() {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("docId", DataTypes.LongType, false),
        DataTypes.createStructField("positives", DataTypes.LongType, false),
        DataTypes.createStructField("negatives", DataTypes.LongType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    WilsonScoreInterval wilsonScore = new WilsonScoreInterval()
        .setPositiveCol("positives")
        .setNegativeCol("negatives")
        .setOutputCol("score");
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] {wilsonScore});

    PipelineModel model = pipeline.fit(df);
    DataFrame transformed = model.transform(df);
    List<Row> scoreList = transformed.select("score").collectAsList();
    assertEquals(0.22328763310073402, scoreList.get(0).getDouble(0), 1e-5);
    assertEquals(0.553022430377575, scoreList.get(1).getDouble(0), 1e-5);
    assertEquals(0.6316800063346981, scoreList.get(2).getDouble(0), 1e-5);
    assertEquals(0.6556334308906774, scoreList.get(3).getDouble(0), 1e-5);
  }

  @Test
  public void testSaveAndLoad() throws IOException {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("docId", DataTypes.LongType, false),
        DataTypes.createStructField("positives", DataTypes.LongType, false),
        DataTypes.createStructField("negatives", DataTypes.LongType, false)
    });
    DataFrame df = jsql.createDataFrame(rdd, schema);

    WilsonScoreInterval wilsonScore = new WilsonScoreInterval()
        .setPositiveCol("positives")
        .setNegativeCol("negatives")
        .setOutputCol("score");
    String path = File.createTempFile("spark-wilson-score", "java").getAbsolutePath();
    wilsonScore.write().overwrite().save(path);
    WilsonScoreInterval loaded = WilsonScoreInterval.load(path);

    DataFrame transformed = loaded.transform(df);
    List<Row> scoreList = transformed.select("score").collectAsList();
    assertEquals(0.22328763310073402, scoreList.get(0).getDouble(0), 1e-5);
    assertEquals(0.553022430377575, scoreList.get(1).getDouble(0), 1e-5);
    assertEquals(0.6316800063346981, scoreList.get(2).getDouble(0), 1e-5);
    assertEquals(0.6556334308906774, scoreList.get(3).getDouble(0), 1e-5);
  }
}
