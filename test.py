#!/usr/bin/env python

import analysis
import numpy


def doit(predictions, true_labels, expected):
  predictions = numpy.array(predictions)
  true_labels = numpy.array(true_labels)
  cer = analysis.CER(predictions, true_labels)
  assert numpy.isclose(cer, expected), 'Expected %r, but got %r' % (expected, cer)


def test_50_50():
  doit([1, 1], [0, 1], 0.5)

def test_20_80():
  doit([1, 1, 1, 1, 1], [0, 1, 1, 1, 1], 0.2)
