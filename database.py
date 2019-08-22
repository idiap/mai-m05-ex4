#!/usr/bin/env python

import numpy


PROTOCOLS = {
        'proto1': {'train': range(0, 30), 'test': range(30, 50)},
        'proto2': {'train': range(20, 50), 'test': range(0, 20)},
        }

SUBSETS = [
        'train',
        'test',
        ]

CLASSES = [
        'setosa',
        'versicolor',
        'virginica',
        ]

VARIABLES = [
        'sepal length',
        'sepal width',
        'petal length',
        'petal width',
        ]


def load():
  import csv
  data = dict([(k,[]) for k in CLASSES])
  with open('data.csv', 'rt') as f:
    reader = csv.reader(f)
    for k, row in enumerate(reader):
      if not k: continue
      data[row[4]].append(numpy.array([float(z) for z in row[:4]]))
  for k in CLASSES:
    data[k] = numpy.vstack(data[k])
  return data


def split_data(data, subset, splits):
  return dict([(k, data[k][splits[subset]]) for k in data])


def get(protocol, subset, classes=CLASSES, variables=VARIABLES):
  retval = split_data(load(), subset, PROTOCOLS[protocol])
  varindex = [VARIABLES.index(k) for k in variables]
  retval = dict([(k, retval[k][:,varindex]) for k in classes])
  return numpy.array([retval[k] for k in classes])
