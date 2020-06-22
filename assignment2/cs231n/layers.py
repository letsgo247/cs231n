from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    D, M = w.shape

    x_copy = x.reshape(N,-1)
    # print(x.shape)
    
    out = np.dot(x_copy,w) + b
    # print(out.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    input_shape = x.shape[1:]
    D, M = w.shape

    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T).reshape(N, *input_shape)
    dw = np.dot(x.reshape(N,-1).T, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x * (x>0)
    # print(out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)

        out = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = {}
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['x_hat'] = x_hat
        cache['x'] = x
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['eps'] = eps 


        # mask = np.random.choice(N, int(0.1*N))
        # x_sample = x[mask]

        # sample_mean = np.mean(x_sample, axis=0)
        # sample_var = np.var(x_sample, axis=0)

        # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var

        # x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        # out = gamma * x_hat + beta

        # cache = {}
        # cache['sample_mean'] = sample_mean
        # cache['sample_var'] = sample_var
        # cache['x_hat'] = x_hat
        # cache['x'] = x
        # cache['gamma'] = gamma
        # cache['beta'] = beta
        # cache['eps'] = eps 

        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var)
        out = x_hat * gamma + beta

        # x_norm = (x - running_mean) / np.sqrt(running_var + eps)

        # out = gamma * x_norm + beta



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    m = dout.shape[0]

    dx_hat = dout * cache['gamma'] 
    dsample_var = np.sum(dx_hat * (cache['x']-cache['sample_mean']) * (-0.5) * (cache['sample_var'] + cache['eps'])**(-1.5), axis=0)
    dsample_mean = np.sum(dx_hat * (-1/np.sqrt(cache['sample_var'] + cache['eps'])) , axis=0) + dsample_var * ((np.sum(-2*(cache['x']-cache['sample_mean']))) / m)

    dx = dx_hat * (1/np.sqrt(cache['sample_var'] + cache['eps'])) + \
        dsample_var * (2*(cache['x']-cache['sample_mean'])/m) + \
        dsample_mean/m

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * cache['x_hat'], axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # sample_mean, sample_var, x_hat, x, gamma, beta, eps = cache

    sample_mean = cache['sample_mean']
    sample_var = cache['sample_var']
    x_hat = cache['x_hat']
    x = cache['x']
    gamma = cache['gamma']
    beta = cache['beta']
    eps = cache['eps']

    # print(x)

    N = x.shape[0]
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x - sample_mean) + 1.0 / N * dmean
    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = x.T
    sample_mean = np.mean(x, axis = 0)
    sample_var = np.var(x, axis = 0)
    x_after = (x - sample_mean) / np.sqrt(sample_var + eps)
    x_after = x_after.T
    out = gamma * x_after + beta
    inv_var = 1.0 / np.sqrt(sample_var + eps)
    cache = (x, x_after, gamma, beta, inv_var, sample_mean)
    

    # sample_mean = np.mean(x, axis=0)
    # sample_var = np.var(x, axis=0)
    # x_hat = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)

    # out = x_hat * gamma + beta

    # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    # running_var = momentum * running_var + (1 - momentum) * sample_var

    # cache = {}
    # cache['sample_mean'] = sample_mean
    # cache['sample_var'] = sample_var
    # cache['x_hat'] = x_hat
    # cache['x'] = x
    # cache['gamma'] = gamma
    # cache['beta'] = beta
    # cache['eps'] = eps 


    # mean = np.mean(x,axis=0)
    # var = np.var(x,axis=0)
    # x_hat = (x - mean) / np.sqrt(var + eps)
    # out = gamma*x_hat + beta

    # cache = {}
    # cache['x'] = x
    # cache['gamma'] = gamma
    # cache['beta'] = beta
    # cache['mean'] = mean
    # cache['var'] = var
    # cache['eps'] = eps
    # cache['x_hat'] = x_hat



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_after, gamma, beta, inv_var, sample_mean = cache
    N, D = dout.shape[0], dout.shape[1]
    dgamma = np.ones(N).dot(dout * x_after)
    dbeta = np.ones(N).dot(dout)
    dx_after = dout * gamma
    dx_after = dx_after.T
    dvar = np.ones(D).dot(dx_after * (-0.5) * (x - sample_mean) * np.power(inv_var, 3))
    dmean = np.ones(D).dot(dx_after * (-inv_var)) + np.ones(D).dot(dvar * (-2.0 / D) * (x - sample_mean))
    dx = dx_after * inv_var + dvar * 2.0 * (x - sample_mean) / D + dmean / D 
    dx = dx.T

    # m = dout.shape[0]

    # dx_hat = dout * cache['gamma'] 
    # dvar = np.sum(dx_hat * (cache['x']-cache['mean']) * (-0.5) * (cache['var'] + cache['eps'])**(-1.5), axis=0)
    # dmean = np.sum(dx_hat * (-1/np.sqrt(cache['var'] + cache['eps'])) , axis=0) + dvar * ((np.sum(-2*(cache['x']-cache['mean']))) / m)

    # dx = dx_hat * (1/np.sqrt(cache['var'] + cache['eps'])) + \
    #     dvar * (2*(cache['x']-cache['mean'])/m) + \
    #     dmean/m

    # dbeta = np.sum(dout, axis=1)
    # dgamma = np.sum(dout * cache['x_hat'], axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None
    # cache = (dropout_param, mask)

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p   #/p를 마스크에서 해줘야되는 거였다! out 할때 쓰면 backpropagation 시 적용 X!
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    N, C, H, W = x.shape
    F, CC, HH, WW = w.shape
    assert C == CC

    H_out = int(1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride'])
    W_out = int(1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride'])

    out = np.zeros((N, F, H_out, W_out))

    # padding
    pad = conv_param['pad']
    x_with_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    _, _, H, W = x_with_pad.shape

    # convolving
    stride = conv_param['stride']

    for i in range(0, N):
      x_data = x_with_pad[i]

      xx, yy = -1, -1
      for j in range(0, H-HH+1, stride):
        yy += 1
        for k in range(0, W-WW+1, stride):
          xx += 1
          x_rf = x_data[:, j:j+HH, k:k+WW]

          for l in range(0, F):
            conv_value = np.sum(x_rf * w[l]) + b[l]
            out[i, l, yy, xx] = conv_value

        xx = -1

    cache = (x, w, b, conv_param)
    return out, cache


    # N, C, H, W = x.shape
    # F, C, HH, WW = w.shape

    # stride = conv_param['stride']
    # pad = conv_param['pad']

    # HO = int(1 + (H + 2 * pad - HH) / stride)
    # WO = int(1 + (W + 2 * pad - WW) / stride)
    # # print(HO, WO)

    # x_pad = np.zeros((N,C,H + 2*pad, W + 2*pad))
    # x_pad[:,:,pad:-pad,pad:-pad] = x    #padding
    
    # param_num = C*HH*WW

    # out = np.zeros((N,F,HO,WO))
    
    # for n in range(N):
    #   x_n = x_pad[n]
    #   # print(x_n.shape)

    #   X_col = np.zeros((param_num,HO*WO))

    #   D =list(range(HO*WO))

    #   for i in range(HO):
    #     for j in range(WO):
    #       print(i,j)
    #       field = x_n[:, i*stride : i*stride + HH, j*stride : j*stride + WW]
    #       # print(field.shape)
    #       d = D.pop()
    #       X_col[:,d] = field.flatten().T

    #   W_row = np.zeros((F,param_num))
      
    #   for f in range(F):
    #     wf = w[f]        
    #     W_row[f,:] = wf.flatten()

    #   temp = np.dot(W_row, X_col)

    #   print(temp.reshape(F,HO,WO))
      
    #   out[n] = temp.reshape(F,HO,WO) + b




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    print(out.shape)
    print(out)
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']
    x_with_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)

    N, F, Hdout, Wdout = dout.shape

    H_out = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
    W_out = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

    db = np.zeros((b.shape))
    for i in range(0, F):
      db[i] = np.sum(dout[:, i, :, :])

    dw = np.zeros((F, C, HH, WW))
    for i in range(0, F):
      for j in range(0, C):
        for k in range(0, HH):
          for l in range(0, WW):
            dw[i, j, k, l] = np.sum(dout[:, i, :, :] * x_with_pad[:, j, k:k + Hdout * stride:stride, l:l + Wdout * stride:stride])

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
      for i in range(H):
        for j in range(W):
          for f in range(F):
            for k in range(Hdout):
              for l in range(Wdout):
                mask1 = np.zeros_like(w[f, :, :, :])
                mask2 = np.zeros_like(w[f, :, :, :])
                if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                  mask1[:, i + pad - k * stride, :] = 1.0
                if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                  mask2[:, :, j + pad - l * stride] = 1.0

                w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    HO = int(1 + (H - pool_height) / stride)
    WO = int(1 + (W - pool_width) / stride)

    out = np.zeros((N,C,HO,WO))

    for n in range(N):
      x_n = x[n]
      for c in range(C):
        x_n_c = x_n[c]
        for i in range(HO):          
          for j in range(WO):
            RF = x_n_c[i*stride : i*stride + pool_height, j*stride : j*stride + pool_width]
            m = np.max(RF)
            
            out[n,c,i,j] = m


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros((N, C, H, W))
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride

    for i in range(0, N):
      x_data = x[i]

      xx, yy = -1, -1
      for j in range(0, H-pool_height+1, stride):
        yy += 1
        for k in range(0, W-pool_width+1, stride):
          xx += 1
          x_rf = x_data[:, j:j+pool_height, k:k+pool_width]
          for l in range(0, C):
            x_pool = x_rf[l]
            mask = x_pool == np.max(x_pool)
            dx[i, l, j:j+pool_height, k:k+pool_width] += dout[i, l, yy, xx] * mask

        xx = -1



    # N,C,HO,WO = dout.shape
    # x, pool_param = cache
    # pool_height = pool_param['pool_height']
    # pool_width = pool_param['pool_width']
    # stride = pool_param['stride']

    # # HO = int(1 + (H - pool_height) / stride)
    # # WO = int(1 + (W - pool_width) / stride)

    # dx = np.zeros_like(x)

    # for n in range(N):
    #   dout_n = dout[n]
    #   x_n = x[n]

    #   for c in range(C):
    #     dout_n_c = dout_n[c]
    #     x_n_c = x_n[c]
        
    #     for i in range(HO):
    #       for j in range(WO):
    #         delta = dout_n_c[i,j]
            
    #         RF = x_n_c[i*stride : i*stride + pool_height, j*stride : j*stride + pool_width]
    #         max_index = np.unravel_index(np.argmax(RF, axis=None), RF.shape)
    #         print(max_index)

    #         dx[n,c,i*stride+max_index[0], j+stride+max_index[1]] = delta

    # print(x)
    # print(dx)
      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # mode = bn_param["mode"]
    # eps = bn_param.get("eps", 1e-5)
    # momentum = bn_param.get("momentum", 0.9)

    # N, C, H, W = x.shape
    # running_mean = bn_param.get("running_mean", np.zeros(C, dtype=x.dtype))
    # running_var = bn_param.get("running_var", np.zeros(C, dtype=x.dtype))

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_reshaped = x.transpose(0,2,3,1).reshape(N*H*W, C)
    out_tmp, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2) 



    # sample_mean = np.mean(x, axis=(0,2,3))
    # sample_var = np.var(x, axis=(0,2,3))
    # x_hat = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)
    # out = x_hat * gamma + beta

    # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    # running_var = momentum * running_var + (1 - momentum) * sample_var

    # cache = {}
    # cache.update({'sample_mean':sample_mean, 'sample_var':sample_var, 'x_hat':x_hat, 'x':x, 'gamma':gamma, 'beta':beta, 'eps':eps})



    # sample_mean = np.mean(x, axis=0)
    # sample_var = np.var(x, axis=0)
    # x_hat = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)

    # out = x_hat * gamma + beta

    # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    # running_var = momentum * running_var + (1 - momentum) * sample_var

    # cache = {}
    # cache['sample_mean'] = sample_mean
    # cache['sample_var'] = sample_var
    # cache['x_hat'] = x_hat
    # cache['x'] = x
    # cache['gamma'] = gamma
    # cache['beta'] = beta
    # cache['eps'] = eps 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx_tmp, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = x.reshape(N, G, C // G, H, W)
    sample_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    sample_var = np.var(x, axis=(2, 3, 4), keepdims=True)

    vareps = sample_var + eps
    x_normalized = (x - sample_mean) / np.sqrt(vareps)
    out = x_normalized.reshape(N, C, H, W)
    out = out * gamma + beta

    x = x.reshape(N, C, H, W)
    cache = (x, gamma, sample_mean, vareps, x_normalized, G)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx, dgamma, dbeta = None, None, None

    x = cache[0]
    gamma = cache[1]
    sample_mean = cache[2]
    vareps = cache[3]
    x_normalized = cache[4]
    G = cache[5]

    N, C, H, W = x.shape
    D = (C//G) * H * W
    x = x.reshape(N, G, C // G, H, W)

    dbeta = np.sum(dout, axis=(0, 2, 3))
    dbeta = dbeta.reshape(1, C, 1, 1)

    x_normalized = x_normalized.reshape(N, C, H, W)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))
    dgamma = dgamma.reshape(1, C, 1, 1)

    x_mu = x - sample_mean
    dx_norm = dout * gamma
    dx_norm = dx_norm.reshape(N, G, C // G, H, W)

    std_inv = 1/np.sqrt(vareps)

    summ = np.sum(dx_norm * x_mu, axis=(2, 3, 4), keepdims=True)
    dvar = summ * -.5 * np.power(vareps, -3/2)

    dmu_term1 = np.sum((dx_norm * -std_inv), axis=(2, 3, 4), keepdims=True)

    dmu = dmu_term1 + \
        dvar * np.mean(-2. * x_mu, axis=(2, 3, 4), keepdims=True)

    dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / D) + (dmu / D)

    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
