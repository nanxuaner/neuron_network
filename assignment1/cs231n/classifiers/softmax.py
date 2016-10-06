import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
        scores = X[i].dot(W)
        
        #normalization scores
        log_c = np.max(scores)
        scores -= log_c
        correct_class_score = scores[y[i]]
        sum_exp = 0.0
        for j in xrange(num_classes):
            sum_exp += np.exp(scores[j])
        loss += -correct_class_score + np.log(sum_exp)
        for j in xrange(num_classes):
            #gradient dW_j = x_i*np.exp(f_j)/sum_exp
            #gradient dW_y_i += -x_i if j != y[i]: x_i*p else: x_i*(p-1)
            p = np.exp(scores[j]) / sum_exp
            dW[:, j] += X[i, :] * (p - (j == y[i]))
        #dW[:, y[i]] -= X[i, :]
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)
  dW = dW/num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
   Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)  #shape (N, C)
  #normalization scores
  scores -= np.max(scores, axis=1).reshape(num_train, 1)  #shape (N,C)
  correct_class_score = scores[np.arange(num_train), y] #shape (N,)
  loss = np.sum(-correct_class_score + np.log(np.sum(np.exp(scores), axis=1)))
  loss = loss/num_train + 0.5 * reg * np.sum(W * W)

  #gradient dW_j = x_i*np.exp(f_j)/sum_exp = x_i * p
  #gradient dW_y_i += -x_i if j != y[i]: x_i*p else: x_i*(p-1)
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1) #shape (N,C)
  ind = np.zeros(p.shape)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot((p-ind))
  dW = dW/num_train + reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

