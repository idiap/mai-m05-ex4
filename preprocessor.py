#!/usr/bin/env python

import numpy


def estimate_norm(X):
  return X.mean(axis=0), X.std(axis=0, ddof=1)


def normalize(X, norm):
  return numpy.array([(k - norm[0]) / norm[1] for k in X])
