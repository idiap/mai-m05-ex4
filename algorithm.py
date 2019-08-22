#!/usr/bin/env python

import logging
logger = logging.getLogger()

import numpy
import scipy.optimize


def make_labels(X):
  return numpy.hstack([k*numpy.ones(len(X[k]), dtype=int) for k in range(len(X))])


class Machine:
  def __init__(self, theta):
    self.theta = numpy.array(theta).copy()


  def __call__(self, X):
    Xp = numpy.hstack((numpy.ones((len(X),1)), X)) #add bias term
    return 1. / (1. + numpy.exp(-numpy.dot(Xp, self.theta)))


  def predict(self, X):
    retval = self(X)
    retval[retval<0.5] = 0.
    retval[retval>=0.5] = 1.
    return retval.astype(int)


  def J(self, X, regularizer=0.0):
    h = numpy.hstack([self(X[k]) for k in (0,1)])
    y = make_labels(X)

    logh = numpy.nan_to_num(numpy.log(h))
    log1h = numpy.nan_to_num(numpy.log(1-h))
    regularization_term = regularizer*(self.theta[1:]**2).sum()
    main_term = -(y*logh + ((1-y)*log1h)).mean()
    return main_term + regularization_term


  def dJ(self, X, regularizer=0.0):
    Xflat = numpy.vstack([k for k in X])
    Xp = numpy.hstack((numpy.ones((len(Xflat),1)), Xflat)) #add bias term
    y = make_labels(X)

    retval = ((self(Xflat) - y) * Xp.T).T.mean(axis=0)
    retval[1:] += (regularizer*self.theta[1:])/len(X)
    return retval


class Trainer:
  def __init__(self, regularizer=0.0):
    self.regularizer = regularizer


  def J(self, theta, machine, X):
    machine.theta = theta
    return machine.J(X, self.regularizer)


  def dJ(self, theta, machine, X):
    machine.theta = theta
    return machine.dJ(X, self.regularizer)


  def train(self, X):
    # check data dimensionality if not organized in a matrix
    if not isinstance(X, numpy.ndarray):
      baseline = X[0].shape[1]
      for k in X:
        if k.shape[1] != baseline:
          raise RuntimeError("Mismatch on the dimensionality of input `X`")

    # prepare the machine
    theta0 = numpy.zeros(X[0].shape[1]+1) #include bias terms
    machine = Machine(theta0)

    logger.debug('Settings:')
    logger.debug('  * initial guess = %s', [k for k in theta0])
    logger.debug('  * cost (J) = %g', machine.J(X, self.regularizer))
    logger.debug('Training using scipy.optimize.fmin_l_bfgs_b()...')

    # Fill in the right parameters so that the minimization can take place
    theta, cost, d = scipy.optimize.fmin_l_bfgs_b(
        self.J,
        theta0,
        self.dJ,
        (machine, X),
        )

    if d['warnflag'] == 0:

      logger.info("** LBFGS converged successfuly **")
      machine.theta = theta
      logger.debug('Final settings:')
      logger.debug('  * theta = %s', [k for k in theta])
      logger.debug('  * cost (J) = %g', cost)
      return machine

    else:
      message = "LBFGS did **not** converged:"
      if d['warnflag'] == 1:
        message += " Too many function evaluations"
      elif d['warnflag'] == 2:
        message += "  %s" % d['task']
      raise RuntimeError(message)


class MultiClassMachine:

  def __init__(self, machines):
    self.machines = machines


  def __call__(self, X):
    return numpy.vstack([m(X) for m in self.machines]).T


  def predict(self, X):
    return self(X).argmax(axis=1)


class MultiClassTrainer:

  def __init__(self, regularizer=0.0):
    self.regularizer = regularizer


  def train(self, X):
    _trainer = Trainer(self.regularizer)

    if len(X) == 2: #trains and returns a single logistic regression classifer

      return _trainer.train(X)

    else: #trains and returns a multi-class logistic regression classifier

      # use one-versus-all strategy
      machines = []
      for k in range(len(X)):
        NC_range = list(range(0,k)) + list(range(k+1,len(X)))
        Xp = numpy.array([numpy.vstack(X[NC_range]), X[k]])
        machines.append(_trainer.train(Xp))

      return MultiClassMachine(machines)
