from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    spark = SparkSession.builder.appName('HW2').getOrCreate()
    filename = 'goodreads_reviews_spoiler.json.gz'
    df = spark.read.json(filename)
    df.show(10)
    df.printSchema()
    customer = '8842281e1d1347389f2ab93d60773d4d'

    # Adding StringIndexer to asin and reviewerID columns
    indexer = StringIndexer(inputCols=['book_id','user_id'],outputCols=['bookNum','userNum'])
    indexed = indexer.fit(df).transform(df)
    indexed.show(5,truncate=False)
    train_set, test_set = indexed.randomSplit([0.7,0.3])

    num_user = train_set.select('userNum').distinct().count()
    num_book = train_set.select('bookNum').distinct().count()
    print(f"Number of Users in Training data = {num_user}")
    print(f"Number of Books in Training data = {num_book}")

    # Building the ALS model
    als = ALS(maxIter=5,regParam=0.01,
              userCol='userNum',
              itemCol='bookNum',
              ratingCol='rating')
    model = als.fit(train_set)
    predictions = model.transform(test_set)

    # Evaluating the model
    evaluator = RegressionEvaluator(metricName="rmse", labelCol='rating',
                                    predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error: {rmse}")

    # Defining a specific user
    user = indexed.select('userNum').distinct().where(indexed["user_id"] == customer)
    userRecs = model.recommendForUserSubset(user, 10)

    # Showing the Recommendations for the user
    recs = userRecs.select("recommendations").collect()[0][0]
    print(recs)
    recomm = spark.createDataFrame(recs)
    final = recomm.join(indexed,recomm.bookNum == indexed.bookNum).select(recomm['bookNum'],recomm['rating'],'book_id')
    final.distinct().orderBy('rating',ascending=False).show()

    spark.stop()

if __name__=="__main__":
    main()