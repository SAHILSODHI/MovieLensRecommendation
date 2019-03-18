import os
import sys
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


'''
INTRODUCTION

MovieLens dataset sample provided with Spark and
available in directory `data`.

'''

'''
HELPER FUNCTIONS

'''

#Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark

#Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row]))\
           .reduce(lambda x,y: os.linesep.join([x,y]))
    return a + os.linesep

def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None

def basic_als_recommender(filename, seed):
    '''
    The following parameters are used in the ALS
    optimizer:
    - maxIter: 5
    - rank: 70
    - regParam: 0.01
    - coldStartStrategy: 'drop'
    Test file: tests/test_basic_als.py
    '''

    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", rank=70)
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return rmse

def global_average(filename, seed):
    '''
    This function prints the global average rating for all users and
    all movies in the training set.
    Test file: tests/test_global_average.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)
    averageRatings = training.groupBy().avg('rating')
    res = averageRatings.first()[0]
    return res

def global_average_recommender(filename, seed):
    '''
    This function prints the RMSE of recommendations obtained
    through global average, that is, the predicted rating for each
    user-movie pair must be the global average computed in the previous
    task.
    Test file: tests/test_global_average_recommender.py
    '''

    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed = seed)
    averageRatings = training.groupBy().avg('rating')
    trainRes = averageRatings.first()[0]
    avgTestRating = test.groupBy().avg('rating')
    testRes = avgTestRating.first()[0]
    training = training.withColumn('training_avg', lit(trainRes))
    test = test.withColumn('testing_avg', lit(testRes))

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", rank=70)
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="testing_avg")

    rmse = evaluator.evaluate(predictions)
    return rmse

def means_and_interaction(filename, seed, n):
    '''
    This function returns the n first elements of a DataFrame
    containing, for each (userId, movieId, rating) triple, the
    corresponding user mean (computed on the training set), item mean
    (computed on the training set) and user-item interaction i defined
    as i=rating-(user_mean+item_mean-global_mean).
    The DataFrame contains the following columns:
    - userId # as in the input file
    - movieId #  as in the input file
    - rating # as in the input file
    - user_mean # computed on the training set
    - item_mean # computed on the training set
    - user_item_interaction # i = rating - (user_mean+item_mean-global_mean)
    Test file: tests/test_means_and_interaction.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)
    avgUserId = training.groupBy('userId').mean('rating').sort(['userId'])
    res=training.join(avgUserId, on='userId')
    res = res.withColumnRenamed("avg(rating)", "user_mean")
    item_df = training.groupBy('movieId').mean('rating')
    res2 = res.join(item_df, on='movieId')
    res2=res2.sort(['userId','movieId']).withColumnRenamed("avg(rating)", "item_mean")
    averageRatings = training.groupBy().avg('rating')
    global_mean = averageRatings.first()[0]
    des = res2.withColumn('user_item_interaction', lit(None))
    des = res2.withColumn('user_item_interaction', lit(des['rating']-(des['user_mean']+des['item_mean']-global_mean)))
    des=des.sort(['userId','movieId'])
    return des.head(n)

def als_with_bias_recommender(filename, seed):
    '''
    This function returns the RMSE of recommendations obtained
    using ALS + biases.
    Test file: tests/test_als_with_bias_recommender.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)
    res = training.groupby('userId').mean('rating').withColumnRenamed("avg(rating)", "user_mean").sort(['userId'])
    res2 = training.groupby('movieId').mean('rating').withColumnRenamed("avg(rating)", "item_mean").sort(['movieId'])
    res3 = test.groupby('userId').mean('rating').withColumnRenamed("avg(rating)", "user_mean").sort(['userId'])
    res4 = test.groupby('movieId').mean('rating').withColumnRenamed("avg(rating)", "item_mean").sort(['movieId'])
    global_mean = training.groupBy().avg('rating').first()[0]
    training = training.join(res, 'userId').join(res2, 'movieId')
    des = training.withColumn('user_item_interaction', lit(None))
    des = training.withColumn('user_item_interaction',
                              lit(des['rating'] - (des['user_mean'] + des['item_mean'] - global_mean)))
    des = des.sort(['userId', 'movieId'])
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user_item_interaction",
              coldStartStrategy="drop", rank=70, seed=seed)
    model = als.fit(des)
    predictions = model.transform(test)
    predictions = predictions.join(res, 'userId').join(res2, 'movieId')
    predictions = predictions.withColumn('ratings_pred',
                                         predictions['prediction'] + predictions['user_mean'] + predictions[
                                             'item_mean'] - global_mean)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="ratings_pred")
    rmse = evaluator.evaluate(predictions)
    return rmse
