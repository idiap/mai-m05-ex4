#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 19 Jun 2015 18:51:39 CEST

'''Test unit for analysis code'''

import analysis
import numpy


def doit(predictions, true_labels, expected):
  '''Runs a single test case


  Parameters:

    predictions (list): A list of integer predictions to input
    true_labels (list): Ground truth values to compare to
    expected (float): The expected classification-error rate


  Raises:

    AssertionError: in case something goes wrong

  '''

  predictions = numpy.array(predictions)
  true_labels = numpy.array(true_labels)

  cer = analysis.CER(predictions, true_labels)

  assert numpy.isclose(cer, expected), 'Expected %r, but got %r' % (expected, cer)


def test_50_50():
  doit([1, 1], [0, 1], 0.5)

def test_20_80():
  doit([1, 1, 1, 1, 1], [0, 1, 1, 1, 1], 0.2)
