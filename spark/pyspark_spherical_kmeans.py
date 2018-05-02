import numpy as np
from pyspark import SparkContext, SparkConf
from numpy import array
from time import time
from os.path import join


def distance(p1, p2):
    return 1 - np.dot(p1, p2)


def find_closest_centroid(datapoint, centers):
    # find the index of the closest centroid of the given data point.
    return min(enumerate(centers), key=lambda x: distance(datapoint, x[1]))[0]


def calc_centers(parsed, centers_prev_brodcast):
    centers = (parsed.
               map(lambda x: (find_closest_centroid(x, centers_prev_brodcast.value), (x, 1))).
               reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).map(lambda x: (x[0], x[1][0] / x[1][1])).
               collect())
    centers = [b for a, b in sorted(centers, key=lambda x: x[0])]
    centers = [x / np.linalg.norm(x) for x in centers]  # normalization
    return centers


def calc_total_error(parsed, centers_brodcast):
    total_error = (parsed.
                   map(lambda x: (x, find_closest_centroid(x, centers_brodcast.value))).
                   map(lambda x: (x[0], centers_brodcast.value[x[1]])).
                   map(lambda x: distance(*x)).
                   reduce(lambda x, y: x + y))
    return total_error


class K_mean_model:
    def __init__(self):
        self.K = K
        self.dim = parsed.take(1)[0].shape[0]
        self.parsed = parsed
        self.centers_prev = self.init_centers()
        self.centers = None
        self.iter = 0

    def init_centers(self):
        """randomize initial centroids"""
        centers = [x for x in np.random.randn(self.K, self.dim)]
        centers = [x / np.linalg.norm(x) for x in centers]
        return centers

    def check_converge(self):
        distances = np.array([distance(c, o) for c, o in zip(self.centers, self.centers_prev)])
        if (distances <= EPSILON).all():
            return True
        return False

    def train(self, parsed):
        while True:
            centers_prev_brodcast = sc.broadcast(self.centers_prev)
            self.centers = calc_centers(parsed, centers_prev_brodcast)
            is_converged = self.check_converge()
            self.centers_prev = self.centers

            self.iter += 1
            print("iter = {}".format(self.iter))
            if self.iter > MAX_ITERATIONS:
                print("max iterations limit is reached, iter = {}".format(self.iter))
                break
            if is_converged:
                print("convergency is reached, iter = {}".format(self.iter))
                break

    def predict(self, datapoint):
        return find_closest_centroid(datapoint, self.centers)

    def error(self, parsed):
        centers_brodcast = sc.broadcast(self.centers)
        return calc_total_error(parsed, centers_brodcast)

    def save(self, path):
        string = str(model.centers)
        sc.parallelize([string]).saveAsTextFile(path)

    def load(self, path):
        string_list = sc.textFile(path).collect()
        string = string_list[0]
        centers = eval(string)
        self.centers = centers


conf = SparkConf().setAppName("ImpAnomaly").setMaster("yarn")
conf.set('spark.scheduler.listenerbus.eventqueue.size', 1000000)
conf.set('yarn.nodemanager.vmem.check.enabled', False)
conf.set('spark.yarn.executor.memoryOverhead', '20000m')
sc = SparkContext(conf=conf)


INPUT = "gs://my_backet/embeding/part-*"
MODEL_PATH = "gs://my_backet/results"

MAX_ITERATIONS = 100
EPSILON = 1e-7

data = sc.textFile(INPUT)
parsed = (data.
              map(lambda line: np.array([float(x) for x in line.split('\t')[2:]])).
              map(lambda x: x / np.linalg.norm(x)).
              cache())


for K in [2,5,10,50,100,150,200,250,300,350,400,450,500,600,700,800,900] + \
                                             list(range(1000,10001,500)):
    ts = time()
    print ("\n start to train model, K = {} \n".format(K))

    model = K_mean_model()
    model.train(parsed)
    err = model.error(parsed)
    print("\nK = {};  Error = {}; elapsed time = {} minutes \n".format(K, err, (time() - ts) / 60))
    model.save(join(MODEL_PATH, str(K)))
