# Okapi BM25
> Okapi BM25 (BM stands for Best Matching) is a ranking function used by search engines to rank matching documents according to their relevance to a given search query.
> It is based on the probabilistic retrieval framework developed in the 1970s and 1980s by Stephen E. Robertson, Karen Spärck Jones, and others.

## Example in Scala

```{scala}
// Prepares test data
case class TestData(text: String)
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
val rdd = sc.parallelize(data)
val df = sqlContext.createDataFrame(rdd)

// Converts text to bag-of-words
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
val tokenedDF = tokenizer.transform(df)
val vectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("token_count")
val vectorizerModel = vectorizer.fit(tokenedDF)
val vectorizedDF = vectorizerModel.transform(tokenedDF)

// Transforms bag-of-words to Okapi BM25
val bm25 = new OkapiBM25()
  .setFeaturesCol("token_count")
  .setOutputCol("bm25")

val model = bm25.fit(vectorizedDF)
model.transform(vectorizedDF)
```

## Input

|text                             |tokens                                    |token_count                     |
|---------------------------------|------------------------------------------|--------------------------------|
|this is a pen                    |[this, is, a, pen]                        |(10,[1,2,6,7],[1.0,1.0,1.0,1.0])|
|this is an apple                 |[this, is, an, apple]                     |(10,[1,2,4,9],[1.0,1.0,1.0,1.0])|
|this is a pen this is a pen      |[this, is, a, pen, this, is, a, pen]      |(10,[1,2,6,7],[2.0,2.0,2.0,2.0])|
|this is an apple this is an apple|[this, is, an, apple, this, is, an, apple]|(10,[1,2,4,9],[2.0,2.0,2.0,2.0])|
|those are pens                   |[those, are, pens]                        |(10,[0,3,5],[1.0,1.0,1.0])      |
|those are apples                 |[those, are, apples]                      |(10,[0,3,8],[1.0,1.0,1.0])      |
|those are pens those are pens    |[those, are, pens, those, are, pens]      |(10,[0,3,5],[2.0,2.0,2.0])      |
|those are apples those are apples|[those, are, apples, those, are, apples]  |(10,[0,3,8],[2.0,2.0,2.0])      |


## Output

|bm25                                                                                        |
|--------------------------------------------------------------------------------------------|
|(10,[1,2,6,7],[0.6512168805390384,0.6512168805390384,1.2171675716179058,1.2171675716179058])|
|(10,[1,2,4,9],[0.6512168805390384,0.6512168805390384,1.2171675716179058,1.2171675716179058])|
|(10,[1,2,6,7],[0.7044291548243294,0.7044291548243294,1.3166248440069177,1.3166248440069177])|
|(10,[1,2,4,9],[0.7044291548243294,0.7044291548243294,1.3166248440069177,1.3166248440069177])|
|(10,[0,3,5],[0.7127491842120184,0.7127491842120184,1.3321755311408576])                     |
|(10,[0,3,8],[0.7127491842120184,0.7127491842120184,1.3321755311408576])                     |
|(10,[0,3,5],[0.7769883810723291,0.7769883810723291,1.4522428536900338])                     |
|(10,[0,3,8],[0.7769883810723291,0.7769883810723291,1.4522428536900338])                     |

## Links

- [Okapi BM25: a non\-binary model](http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)
- [Okapi BM25 \- Wikipedia, the free encyclopedia](https://en.wikipedia.org/wiki/Okapi_BM25)
