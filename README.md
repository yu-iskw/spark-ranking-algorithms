# Ranking Algorithms for Spark Machine Learning Pipeline

[![License](http://img.shields.io/:license-Apache%202-red.svg)](http://www.apache.org/licenses/LICENSE-2.0.txt)
[![Build Status](https://travis-ci.org/yu-iskw/spark-ranking-algorithms.svg?branch=master)](https://travis-ci.org/yu-iskw/spark-ranking-algorithms)
[![codecov](https://codecov.io/gh/yu-iskw/spark-ranking-algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/yu-iskw/spark-ranking-algorithms)

This package offers some ranking algorithms on Apache Spark machine learning pipeline.

## Wilson score interval

> The Wilson interval is an improvement (the actual coverage probability is closer to the nominal value) over the normal approximation interval and was first developed by Edwin Bidwell Wilson (1927).
> This interval has good properties even for a small number of trials and/or an extreme probability.

### Example in Scala

```{scala}
// Creates test DataFrame
case class TestData(docId: Long, positives: Long, negatives: Long)
val data = Seq(
  TestData(1L, 2L, 1L), TestData(2L, 20L, 10L),
  TestData(3L, 200L, 100L), TestData(4L, 2000L, 1000L)
)
val rdd = sc.parallelize(data)
val df = sqlContext.createDataFrame(rdd)

// Create a class for Wilson-score interval
val wilsonScore = new WilsonScoreIntervalInterval()
  .setPositiveCol("positives")  // Sets the column name for the number of positives
  .setNegativeCol("negatives")  // Sets the column name for the number of negatives
  .setOutputCol("score")        // Sets the column name for the output

wilsonScore.transform(df)
```

### Input
The input values are the number of positive reviews and the number of negative reviews for each documents.

| docId | positives | negatives | 
|-------|-----------|-----------| 
| 1     | 2         | 1         | 
| 2     | 20        | 10        | 
| 3     | 200       | 100       | 
| 4     | 2000      | 1000      | 

### Output
Each output is Wilson score interval.

| score               | 
|---------------------| 
| 0.22328763310073402 | 
| 0.553022430377575   | 
| 0.6316800063346981  | 
| 0.6556334308906774  | 

### Links

- [How Not To Sort By Average Rating](http://www.evanmiller.org/how-not-to-sort-by-average-rating.html)
- [How Reddit ranking algorithms work — Hacking and Gonzo — Medium](https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9#.v0k0nqnkv)
- [Binomial proportion confidence interval \- Wikipedia, the free encyclopedia](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
