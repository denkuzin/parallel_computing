from math import sqrt
import numpy as np
from os.path import join
from time import time
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

N_PARTITIONS = 4000
N_WORKERS = 40

input_data = "gs://my_backet/embeding/part-*"
output = "gs://my_backet/results"
max_iteration = 100

conf = SparkConf().setAppName("ImpAnomaly").setMaster("yarn")
conf.set('spark.scheduler.listenerbus.eventqueue.size', 1000000)
conf.set('yarn.nodemanager.vmem.check.enabled', False)
conf.set('spark.yarn.executor.memoryOverhead', '20000m')
sc = SparkContext(conf=conf)


data = sc.textFile(input_data)
parsedData = data.map(lambda line: np.array([float(x) for x in line.split('\t')[2:]])).cache()


print("\n number of keys is {} \n".format(parsedData.count()))


for K in [2,5,10,50,100,150,200,250,300,350,400,450,500,600,700,800,900] + \
                                             list(range(1000,10001,500)):

    ts = time()
    print ("\n start to train model, K = {} \n".format(K))
    model = KMeans.train(parsedData, K, maxIterations=max_iteration, 
               initializationMode="k-means||", initializationSteps=2, epsilon=1e-6)

    def error(point):
        center = model.centers[model.predict(point)]
        return sum([x**2 for x in (point - center)])

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("\nK = {};  WSSSE = {}; elapsed time = {} minutes \n".format(K, WSSSE, (time()-ts)/60))
    model.save(sc, join(output, str(K), 'model'))
    #sameModel = KMeansModel.load(sc, join(output, 'model'))
