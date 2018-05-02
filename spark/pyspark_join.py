#!/usr/bin/python2.7
import numpy as np
from operator import add
from pyspark import SparkConf, SparkContext
from time import time
import sys
import json
import re
from os.path import join
import random


N_PARTITIONS = 4000
N_WORKERS = 40


class Join:
    """
        optimized inner join to handle skewed distributed keys
        rdd1 - have frequent keys
        rdd2 = doesn't have frequent keys
        Usage: Join().innerJoin(rdd1,rdd2,max_num=1000, freq_threshold=100000)
        output
    """
    def __init__(self):
        pass

    @staticmethod
    def get_freq_keys(rdd, threshold=1000):
        """ input in format of pairs (a, b) """
        """ TODO idea how to optimase this func: apply prelimenary filtration of low freq keys)"""
        return set(rdd.map(lambda (a, b): (a, 1)).
                   reduceByKey(lambda a, b: a + b).filter(lambda (key, val): val > threshold).
                   map(lambda (key, val): key).collect())

    @staticmethod
    def split_key(item, freq_keys, max_num=1000):
        """ (key,val) --> ((key, int), val)  """
        key, val = item
        if key in freq_keys:
            return ((key, random.randint(0, max_num - 1)), val)
        return ((key, 0), val)

    @staticmethod
    def replicate_key(item, freq_keys, max_num=1000):
        """ (key,val) --> ((key, int), val); called by flatMap """
        key, val = item
        if key in freq_keys:
            return [((key, i), val) for i in range(max_num)]
        return [((key, 0), val)]

    def innerJoin(self, rdd1, rdd2, max_num=1000, freq_threshold=100000):
        freq_keys = self.get_freq_keys(rdd1, threshold=freq_threshold)
        print ("we have found {} frequent keys".format(len(freq_keys)))
        results = rdd1.map(lambda x: self.split_key(x, freq_keys, max_num=max_num)). \
            join(rdd2.flatMap(lambda x: self.replicate_key(x, freq_keys, max_num=max_num))). \
            map(lambda ((key, _), (val1, val2)): (key, (val1, val2)))
        return results


conf = SparkConf().setAppName("dkuzin").setMaster("yarn")
conf.set('spark.scheduler.listenerbus.eventqueue.size', 1000000)
conf.set('yarn.nodemanager.vmem.check.enabled', False)
conf.set('spark.yarn.executor.memoryOverhead', '20000m')
sc = SparkContext(conf=conf)

path1 = "gs://dkuzin/experiments/ips"
path2 = "gs://dkuzin/experiments/sites"

rdd1 = sc.textFile(path1, minPartitions=None).coalesce(N_PARTITIONS, shuffle=False)
rdd2 = sc.textFile(path2, minPartitions=None).coalesce(N_PARTITIONS, shuffle=False)

results = Join().innerJoin(rdd1, rdd2, max_num=1000, freq_threshold=100000).collect()
