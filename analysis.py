#!/usr/bin/env python

def CER(prediction, true_labels):
  errors = (prediction != true_labels).sum()
  return errors/len(prediction)
