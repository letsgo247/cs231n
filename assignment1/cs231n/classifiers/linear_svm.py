from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
                
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -= X[i]  #편미분 방정식 대입해보면, loss 0 이상인 j 항목마다 xi를 더해줘야 함
                dW[:,j] += X[i] 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W   #얘는 scalar가 아니라 matrix 이므로, sum 할 필요 X!

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    # print(scores.shape)
    x = list(range(num_train))
    correct_class_score = scores[x,y]   #정답의 score 골라내서
    # print(correct_class_score.shape)
    losses = scores - correct_class_score.reshape(num_train,1) + 1    #margin(=각 항목의 loss) 계산하고
    losses[x,y] = 0   #정답의 loss는 0으로 세팅 
    loss = (np.sum(losses[losses>0]) + reg * np.sum(W * W))/num_train   #loss 계산
    
    #margin 생긴거 count 해주기 위한 수동 matrix 작업 ㄷㄷ
    bool_idx = ( losses>0 )  
    count = np.sum(bool_idx, axis = 1)
    bool_idx = bool_idx * 1
    bool_idx[x,y] = -count
    



    # for i in range(num_train):
    #     scores = X[i].dot(W)
    #     correct_class_score = scores[y[i]]
    #     for j in range(num_classes):
    #         if j == y[i]:
    #             continue
                
    #         margin = scores[j] - correct_class_score + 1 # note delta = 1
    #         if margin > 0:
    #             loss += margin
    #             dW[:,y[i]] -= X[i]
    #             dW[:,j] += X[i]

    # # Right now the loss is a sum over all training examples, but we want it
    # # to be an average instead so we divide by num_train.
    # loss /= num_train
    # dW /= num_train

    # # Add regularization to the loss.
    # loss += reg * np.sum(W * W)
    # dW += reg * 2 * np.sum(W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = (X.T).dot(bool_idx) / num_train    #역함수 개념으로 접근... dW 매트릭스 모양 만들기 위해 X transpos 한거 활용!!


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
