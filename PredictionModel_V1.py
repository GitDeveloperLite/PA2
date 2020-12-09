import sys
import os
os.environ['HADOOP_HOME'] = "C:/Users/web97/Downloads/spark-3.0.1-bin-hadoop2.7"
sys.path.append("C:/Users/web97/Downloads/spark-3.0.1-bin-hadoop2.7/bin")
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
#from pyspark import implicits
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
#OneHotEncoderEstimator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import SQLContext
from pyspark.sql import DataFrameNaFunctions
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import avg
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
#from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTEENN
#from sklearn.model_selection import train_test_split
from collections import Counter


# fn = sys.argv[1]
# if os.path.exists(fn):
    # print os.path.basename(fn)
    # file exists

def create_df(path):
    spark_read=spark.read.csv(path,header='true',inferSchema='true',sep=';')
    spark_read_pd=spark_read.toPandas()
    spark_read_pd.columns=spark_read_pd.columns.str.strip('""')
    return spark.createDataFrame(spark_read_pd)

#Below prints contents
with open(sys.argv[1], 'r') as f:
   contents = f.read()
#print (contents)
path=sys.argv[1] if len(sys.argv) > 1 else "somevalue"
spark = SparkSession.builder.appName("Predict Model").getOrCreate()
#print(return_parsed_df(path))
file_df=create_df(path)
print(file_df)
file_df.printSchema()
#val rows: RDD[Row] = results.rdd
#rows: RDD[Row] = file_df.rdd
file_df.describe().toPandas().transpose()
print("Reading DATA")
#assembler=VectorAssembler(inputCols=['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'],outputCol='features')
assembler=VectorAssembler().setInputCols(['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']).setOutputCol('features')
#(inputCols=['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'],outputCol='features')

#assembler=VectorAssembler(inputCols=['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'])
output_data=assembler.transform(file_df)
file_df.show()
print("Showing Output data")
output_data.show()
print("Output data head")
output_data.head(1)
#data = spark.read.csv(contents, header=True, inferSchema=True)
#spark = SparkSession.builder.master('local[*]')\.appName('wine-rf_model')\.getOrCreate()
print("After data")
#score_data=output_data.select('quality')
score_data=output_data.select('quality').show(truncate=False)
print("score_data")
print(score_data)
#train,test=score_data.randomSplit([0.7,0.3])
train,test=output_data.randomSplit([0.7,0.3])
model=LogisticRegression(labelCol='quality')
model=model.fit(train)
summary=model.summary
summary.predictions.describe().show()
predictions=model.evaluate(test)
print("Showing predictions")
predictions.predictions.show()
predictions.predictions.select('rawPrediction', 'prediction', 'probability').show(10)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
#lr = LogisticRegression(maxIter=10, regParam=0.01)
#accuracy = evaluator.evaluate(predictions)
#accuracy = evaluator.evaluate(predictions.predictions)
#accuracy = evaluator.evaluate(prediction) Prediction is not defined
#print("Test Error = %g" %(1.0 - accuracy))
#print("Test Area under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})))
stages=[]
rf = RandomForestClassifier(numTrees=100, maxDepth=5, maxBins=5, labelCol="features",featuresCol="features",seed=42)
#rf=assembler
stages += [rf]
#trainingData=temp_df.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
trainingData=file_df.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
trainingData.show()
#params =
pipeline = Pipeline(stages = stages)
lr = LogisticRegression().setFeaturesCol("features")
params=ParamGridBuilder().build()
#params=ParamGridBuilder().addGrid(lr.maxIter, [500]).addGrid(lr.regParam, [0]).addGrid(lr.elasticNetParam, [1]).build()
#cvModel
cv = CrossValidator(estimator=pipeline,
            estimatorParamMaps=params,
            evaluator=evaluator,
            numFolds=5)

#cvModel = cv.fit(file_df) #IllegalArgumentException features does not exist
#cvModel=cv.fit(output_data) #Illegal type
cvModel=cv.fit(output_data.select('features')) #Illegal type
#cvModel = cv.fit(trainingData)
#predictions=cvModel.transform(file_df)
#cvModel=cv.fit(train) #IllegalArgumentException label does not exist
#Above IllegalArgumentException: requirement failed: Column features must be of type numeric but was actually of type struct<type:tinyint,size:int,indices:array<int>,values:array<double>>.
#prediction=cvModel.transform(test)
predictions=cvModel.transform(test)
predictions_pandas=predictions.toPandas()
print("Test Area under Pr: ", evaluator.evaluate(predictions))
f1 = f1_score(predictions_pandas.lable, predictions_pandas.prediction, average='weighted')
print(f1)

#lr = LogisticRegression(maxIter=10^5)
    
# Train the 10-fold Cross Validator
#cvModel = CrossValidator(estimator=Pipeline(stages = [lr]),
 #           estimatorParamMaps=ParamGridBuilder() \
  #                              .addGrid(lr.regParam, [0.1, 0.01]) \
   #                             .build(),
    #        evaluator=BinaryClassificationEvaluator(metricName='areaUnderPR'),
     #       numFolds=10).fit(file_df)

#numFolds=10).fit(df)

# Save the best model for later usage
#cvModel.bestModel.save("{}/model".format(opt.output))
#algo = LinearRegression(featuresCol="features", labelCol="medv")
#feature_columns = data.columns[:-1] # here we omit the final column
#print("After feature_columns")
#assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
#data_2 = assembler.transform(data)
#data_2.show()
#train, test = contents.randomSplit([0.7, 0.3])

#model = algo.fit(train)
