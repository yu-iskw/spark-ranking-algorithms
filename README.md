# Ranking Algorithms for Spark Machine Learning Pipeline

[![License](http://img.shields.io/:license-Apache%202-red.svg)](http://www.apache.org/licenses/LICENSE-2.0.txt)
[![Build Status](https://travis-ci.org/yu-iskw/spark-ranking-algorithms.svg?branch=master)](https://travis-ci.org/yu-iskw/spark-ranking-algorithms)
[![codecov](https://codecov.io/gh/yu-iskw/spark-ranking-algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/yu-iskw/spark-ranking-algorithms)

This package offers some ranking algorithms on Apache Spark machine learning pipeline.

- [Wilson score interval](./docs/wilson-score-interval.md)
- [Okapi BM25](./docs/okapi-bm25.md)

## Note

```{shell}
# make a packaged jar
sbt package

# make a assembled JAR
sbt assembly

# run unit tests
sbt test

# check coding style
sbt scalastyle
```