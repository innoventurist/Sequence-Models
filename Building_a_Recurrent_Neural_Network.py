
# coding: utf-8

# # Building Recurrent Neural Network
# 
# Implement key components of a Recurrent Neural Network in numpy.
# 

# Import all packages needed

import numpy as np
from rnn_utils import *


# ## 1 - Forward propagation for the basic Recurrent Neural Network
# 

# 

# Function: rnn_cell_forward

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba) # compute next activation state using the formula given above
    yt_pred = softmax(np.dot(Wya, a_next) + by)  # compute output of the current cell using the formula given above
    
    
    cache = (a_next, a_prev, xt, parameters) # store values you need for backward propagation in cache
    
    return a_next, yt_pred, cache # return values


# 

np.random.seed(1) # Makes random numbers predictable for the parameters
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_pred_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp) # calculated the prediction of what is defined
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = \n", a_next_tmp.shape)
print("yt_pred[1] =\n", yt_pred_tmp[1])
print("yt_pred.shape = \n", yt_pred_tmp.shape)


# **Expected Output**: 
# ```Python
# a_next[4] = 
#  [ 0.59584544  0.18141802  0.61311866  0.99808218  0.85016201  0.99980978
#  -0.18887155  0.99815551  0.6531151   0.82872037]
# a_next.shape = 
#  (5, 10)
# yt_pred[1] =
#  [ 0.9888161   0.01682021  0.21140899  0.36817467  0.98988387  0.88945212
#   0.36920224  0.9966312   0.9982559   0.17746526]
# yt_pred.shape = 
#  (2, 10)
# 
# ```

# ## 1.2 - RNN forward pass 
# 

# #### Additional Notes
# - [np.zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html)
# - If a 3 dimensional numpy array and are indexing by its third dimension, can use array slicing like this: `var_name[:,:,i]`.

#

# Function: rnn_forward

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []
    
    # Retrieve dimensions from shapes of x and parameters["Wya"]
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # initialize "a" and "y_pred" with zeros 
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # Initialize a_next (≈1 line)
    a_next = a0
    
    # loop over all time-steps of the input 'x' 
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        xt = a_next
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:,:,t] = a_next
        # Save the value of the prediction in y 
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" 
        caches.append(cache)

    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches


# 

np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][1] = \n", a_tmp[4][1])
print("a.shape = \n", a_tmp.shape)
print("y_pred[1][3] =\n", y_pred_tmp[1][3])
print("y_pred.shape = \n", y_pred_tmp.shape)
print("caches[1][1][3] =\n", caches_tmp[1][1][3])
print("len(caches) = \n", len(caches_tmp))


# **Expected Output**:
# 
# ```Python
# a[4][1] = 
#  [-0.99999375  0.77911235 -0.99861469 -0.99833267]
# a.shape = 
#  (5, 10, 4)
# y_pred[1][3] =
#  [ 0.79560373  0.86224861  0.11118257  0.81515947]
# y_pred.shape = 
#  (2, 10, 4)
# caches[1][1][3] =
#  [-1.1425182  -0.34934272 -0.20889423  0.58662319]
# len(caches) = 
#  2
# ```

# Have now built the forward propagation of a recurrent neural network from scratch. 
# 
# #### Situations when this RNN will perform better:
# - This will work well enough for some applications, but it suffers from the vanishing gradient problems. 
# - The RNN works best when each output can be estimated using "local" context.  
# - "Local" context refers to information that is close to the prediction's time step $t$.
# - More formally, local context refers to inputs and predictions.
# 
# ## 2 - Long Short-Term Memory (LSTM) network
# LSTM is better at addressing vanishing gradients. It will be able to remember a piece of information and save it for many timesteps. 
# 
# 

# #### Additional Information
# * You can use [numpy.concatenate](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html).  Check which value to use for the `axis` parameter.
# * The functions `sigmoid()` and `softmax` are imported from `rnn_utils.py`.
# * [numpy.tanh](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tanh.html)
# * Use [np.dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) for matrix multiplication.
# * Notice that the variable names `Wi`, `bi` refer to the weights and biases of the **update** gate.  There are no variables named "Wu" or "bu" in this function.

# 

# Function: lstm_cell_forward

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt 
    concat = np.concatenate((a_prev, xt))

    # Compute values for ft, it, cct, c_next, ot, a_next 
    ft = sigmoid(np.dot(Wf, concat) + bf)        # forget gate equation (ft) what information should be thrown away or kept
    it = sigmoid(np.dot(Wi, concat) + bi)        # update gate equation (it) to decide aspect of candidate
    cct = np.tanh(np.dot(Wc, concat) + bc)       # candidate value equation (cct) containing values between 0 and 1
    c_next = ft * c_prev + it * cct              # cell state equation (c_next) transfers relevant information down the sequence chain
    ot = sigmoid(np.dot(Wo, concat)+ bo)         # output gate equation (ot) decides what the next hidden state should be
    a_next = ot * np.tanh(c_next)                # hidden state equation (a_next) has information of previous inputs and used for prediction
    
    
    yt_pred = softmax(np.dot(Wy, a_next) + by) # Compute prediction of the LSTM cell 

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


# In[7]:

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
c_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
print("a_next[4] = \n", a_next_tmp[4])
print("a_next.shape = ", c_next_tmp.shape)
print("c_next[2] = \n", c_next_tmp[2])
print("c_next.shape = ", c_next_tmp.shape)
print("yt[1] =", yt_tmp[1])
print("yt.shape = ", yt_tmp.shape)
print("cache[1][3] =\n", cache_tmp[1][3])
print("len(cache) = ", len(cache_tmp))


# **Expected Output**:
# 
# ```Python
# a_next[4] = 
#  [-0.66408471  0.0036921   0.02088357  0.22834167 -0.85575339  0.00138482
#   0.76566531  0.34631421 -0.00215674  0.43827275]
# a_next.shape =  (5, 10)
# c_next[2] = 
#  [ 0.63267805  1.00570849  0.35504474  0.20690913 -1.64566718  0.11832942
#   0.76449811 -0.0981561  -0.74348425 -0.26810932]
# c_next.shape =  (5, 10)
# yt[1] = [ 0.79913913  0.15986619  0.22412122  0.15606108  0.97057211  0.31146381
#   0.00943007  0.12666353  0.39380172  0.07828381]
# yt.shape =  (2, 10)
# cache[1][3] =
#  [-0.16263996  1.03729328  0.72938082 -0.54101719  0.02752074 -0.30821874
#   0.07651101 -1.03752894  1.41219977 -0.37647422]
# len(cache) =  10
# ```

# ### 2.2 - Forward pass for LSTM
# 
# Now have implemented one step LSTM, can now iterate this over this using a for-loop to process a sequence of inputs. 
# 

# Function: lstm_forward

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []
    
    Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']

    # Retrieve dimensions from shapes of x and parameters['Wy'] 
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next 
    a_next = a0
    c_next = np.zeros(a_next.shape)
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:, :, t]
        # Update next hidden state, next memory state, compute the prediction, get the cache
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a 
        a[:,:,t] = a_next
        # Save the value of the next cell state 
        c[:,:,t]  = c_next
        # Save the value of the prediction in y
        y[:,:,t] = yt
        # Append the cache into caches 
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


#

np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi']= np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
print("a[4][3][6] = ", a_tmp[4][3][6])
print("a.shape = ", a_tmp.shape)
print("y[1][4][3] =", y_tmp[1][4][3])
print("y.shape = ", y_tmp.shape)
print("caches[1][1][1] =\n", caches_tmp[1][1][1])
print("c[1][2][1]", c_tmp[1][2][1])
print("len(caches) = ", len(caches_tmp))


# **Expected Output**:
# 
# ```Python
# a[4][3][6] =  0.172117767533
# a.shape =  (5, 10, 7)
# y[1][4][3] = 0.95087346185
# y.shape =  (2, 10, 7)
# caches[1][1][1] =
#  [ 0.82797464  0.23009474  0.76201118 -0.22232814 -0.20075807  0.18656139
#   0.41005165]
# c[1][2][1] -0.855544916718
# len(caches) =  2
# ```

# Have now implemented the forward passes for the basic RNN and LSTM.

# ## 3 - Backpropagation in recurrent neural networks 

# Note: this does not implement the backward path from the Loss 'J' backwards to 'a'.
# This would have included the dense layer and softmax which are a part of the forward path.
# This is assumed to be calculated elsewhere and the result passed to rnn_backward in 'da'.
# It is further assumed that loss has been adjusted for batch size (m) and division by the number of examples is not required here.

# ### 3.1 - Basic RNN backward pass
#
def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    
    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache
    
    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute the gradient of the loss with respect to z 
    dz = (1 - a_next**2) * da_next

    # compute the gradient of the loss with respect to Wax 
    dxt = np.dot(Wax.T, dz)
    dWax = np.dot(dz, xt.T)

    # compute the gradient with respect to Waa 
    da_prev = np.dot(Waa.T, dz)
    dWaa = np.dot(dz, a_prev.T)

    # compute the gradient with respect to b 
    dba = np.sum(dz, 1, keepdims=True)
    
    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients


# 

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, yt_tmp, cache_tmp = rnn_cell_forward(xt_tmp, a_prev_tmp, parameters_tmp)

da_next_tmp = np.random.randn(5,10)
gradients_tmp = rnn_cell_backward(da_next_tmp, cache_tmp)
print("gradients[\"dxt\"][1][2] =", gradients_tmp["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients_tmp["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients_tmp["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients_tmp["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients_tmp["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients_tmp["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients_tmp["dba"][4])
print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **gradients["dxt"][1][2]** =
#         </td>
#         <td>
#            -1.3872130506
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dxt"].shape** =
#         </td>
#         <td>
#            (3, 10)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da_prev"][2][3]** =
#         </td>
#         <td>
#            -0.152399493774
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da_prev"].shape** =
#         </td>
#         <td>
#            (5, 10)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWax"][3][1]** =
#         </td>
#         <td>
#            0.410772824935
#         </td>
#     </tr>
#             <tr>
#         <td>
#             **gradients["dWax"].shape** =
#         </td>
#         <td>
#            (5, 3)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWaa"][1][2]** = 
#         </td>
#         <td>
#            1.15034506685
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWaa"].shape** =
#         </td>
#         <td>
#            (5, 5)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dba"][4]** = 
#         </td>
#         <td>
#            [ 0.20023491]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dba"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
# </table>

# #### Backward pass through the RNN
# 
#

def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
        
    
    # Retrieve values from the first cache (t=1) of caches
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    # Retrieve dimensions from da's and x1's shapes 
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients (return values) of zeroes with the right sizes
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    
    # Loop through all the time steps
    for t in reversed(range(T_x)):
        
        # Compute gradients at time step t with the output path (da) and the previous timesteps (da_prevt) 
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t]) 
        # Retrieve derivatives from gradients
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    # Set da0 to the gradient of a which has been backpropagated through all time-steps 
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients


# 

np.random.seed(1)
x_tmp = np.random.randn(3,10,4)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wax'] = np.random.randn(5,3)
parameters_tmp['Waa'] = np.random.randn(5,5)
parameters_tmp['Wya'] = np.random.randn(2,5)
parameters_tmp['ba'] = np.random.randn(5,1)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_tmp, caches_tmp = rnn_forward(x_tmp, a0_tmp, parameters_tmp)
da_tmp = np.random.randn(5, 10, 4)
gradients_tmp = rnn_backward(da_tmp, caches_tmp)

print("gradients[\"dx\"][1][2] =", gradients_tmp["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients_tmp["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients_tmp["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients_tmp["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients_tmp["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients_tmp["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients_tmp["dba"][4])
print("gradients[\"dba\"].shape =", gradients_tmp["dba"].shape)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **gradients["dx"][1][2]** =
#         </td>
#         <td>
#            [-2.07101689 -0.59255627  0.02466855  0.01483317]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dx"].shape** =
#         </td>
#         <td>
#            (3, 10, 4)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da0"][2][3]** =
#         </td>
#         <td>
#            -0.314942375127
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da0"].shape** =
#         </td>
#         <td>
#            (5, 10)
#         </td>
#     </tr>
#          <tr>
#         <td>
#             **gradients["dWax"][3][1]** =
#         </td>
#         <td>
#            11.2641044965
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWax"].shape** =
#         </td>
#         <td>
#            (5, 3)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWaa"][1][2]** = 
#         </td>
#         <td>
#            2.30333312658
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWaa"].shape** =
#         </td>
#         <td>
#            (5, 5)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dba"][4]** = 
#         </td>
#         <td>
#            [-0.74747722]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dba"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
# </table>

# ## 3.2 - LSTM backward pass

# ### 3.2.2 gate derivatives
# This is convenient for computing parameter derivatives in the next step. 
# 
# $d\gamma_o^{\langle t \rangle} = da_{next}*\tanh(c_{next}) * \Gamma_o^{\langle t \rangle}*\left(1-\Gamma_o^{\langle t \rangle}\right)\tag{7}$
# 
# $dp\widetilde{c}^{\langle t \rangle} = \left(dc_{next}*\Gamma_u^{\langle t \rangle}+ \Gamma_o^{\langle t \rangle}* (1-\tanh^2(c_{next})) * \Gamma_u^{\langle t \rangle} * da_{next} \right) * \left(1-\left(\widetilde c^{\langle t \rangle}\right)^2\right) \tag{8}$
# 
# $d\gamma_u^{\langle t \rangle} = \left(dc_{next}*\widetilde{c}^{\langle t \rangle} + \Gamma_o^{\langle t \rangle}* (1-\tanh^2(c_{next})) * \widetilde{c}^{\langle t \rangle} * da_{next}\right)*\Gamma_u^{\langle t \rangle}*\left(1-\Gamma_u^{\langle t \rangle}\right)\tag{9}$
# 
# $d\gamma_f^{\langle t \rangle} = \left(dc_{next}* c_{prev} + \Gamma_o^{\langle t \rangle} * (1-\tanh^2(c_{next})) * c_{prev} * da_{next}\right)*\Gamma_f^{\langle t \rangle}*\left(1-\Gamma_f^{\langle t \rangle}\right)\tag{10}$
# 
# ### 3.2.3 parameter derivatives 
# 
# $ dW_f = d\gamma_f^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{11} $
# $ dW_u = d\gamma_u^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{12} $
# $ dW_c = dp\widetilde c^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{13} $
# $ dW_o = d\gamma_o^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{14}$
# 
# To calculate $db_f, db_u, db_c, db_o$ you just need to sum across the horizontal (axis= 1) axis on $d\gamma_f^{\langle t \rangle}, d\gamma_u^{\langle t \rangle}, dp\widetilde c^{\langle t \rangle}, d\gamma_o^{\langle t \rangle}$ respectively. Note that you should have the `keepdims = True` option.
# 
# $\displaystyle db_f = \sum_{batch}d\gamma_f^{\langle t \rangle}\tag{15}$
# $\displaystyle db_u = \sum_{batch}d\gamma_u^{\langle t \rangle}\tag{16}$
# $\displaystyle db_c = \sum_{batch}d\gamma_c^{\langle t \rangle}\tag{17}$
# $\displaystyle db_o = \sum_{batch}d\gamma_o^{\langle t \rangle}\tag{18}$
# 
# Finally, compute the derivative with respect to the previous hidden state, previous memory state, and input.
# 
# $ da_{prev} = W_f^T d\gamma_f^{\langle t \rangle} + W_u^T   d\gamma_u^{\langle t \rangle}+ W_c^T dp\widetilde c^{\langle t \rangle} + W_o^T d\gamma_o^{\langle t \rangle} \tag{19}$
# 
# Here, to account for concatenation, the weights for equations 19 are the first n_a, (i.e. $W_f = W_f[:,:n_a]$ etc...)
# 
# $ dc_{prev} = dc_{next}*\Gamma_f^{\langle t \rangle} + \Gamma_o^{\langle t \rangle} * (1- \tanh^2(c_{next}))*\Gamma_f^{\langle t \rangle}*da_{next} \tag{20}$
# 
# $ dx^{\langle t \rangle} = W_f^T d\gamma_f^{\langle t \rangle} + W_u^T  d\gamma_u^{\langle t \rangle}+ W_c^T dp\widetilde c^{\langle t \rangle} + W_o^T d\gamma_o^{\langle t \rangle}\tag{21} $
# 
# where the weights for equation 21 are from n_a to the end, (i.e. $W_f = W_f[:,n_a:]$ etc...)
# 
#
def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # Compute gates related derivatives
    # $d\gamma_o^{\langle t \rangle}$ is represented by (dot)    
    # $dp\widetilde{c}^{\langle t \rangle}$ is represented by (dcct  
    # $d\gamma_u^{\langle t \rangle}$ is represented by (dit)  
    # $d\gamma_f^{\langle t \rangle}$ is represented by (dft)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot) 
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct)) 
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it) 
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next)) * c_prev * da_next) * ft * (1 -ft) 
    
    # Compute parameters related derivatives equations. 
    dWf = np.dot(dft, np.hstack([a_prev.T, xt.T]))
    dWi = np.dot(dit, np.hstack([a_prev.T, xt.T]))
    dWc = np.dot(dcct, np.hstack([a_prev.T, xt.T]))
    dWo = np.dot(dot, np.hstack([a_prev.T, xt.T]))
    dbf = np.sum(dft, axis=1, keepdims = True)
    dbi = np.sum(dit, axis=1, keepdims = True)
    dbc = np.sum(dcct, axis=1, keepdims = True)
    dbo = np.sum(dot, axis=1, keepdims = True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input.
    da_prev = np.dot(Wf[:, :n_a:].T, dft) + np.dot(Wu[:, :n_a:].T, dit) + np.dot(Wc[:, :n_a:].T, dcct) + np.dot(Wo[:, :n_a:].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(Wf[:, :n_a:].T, dft) + np.dot(Wu[:, :n_a:].T, dit) + np.dot(Wc[:, :n_a:].T, dcct) + np.dot(Wo[:, :n_a:].T, dot)
    
    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients


# In[46]:

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
c_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)

da_next_tmp = np.random.randn(5,10)
dc_next_tmp = np.random.randn(5,10)
gradients_tmp = lstm_cell_backward(da_next_tmp, dc_next_tmp, cache_tmp)
print("gradients[\"dxt\"][1][2] =", gradients_tmp["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients_tmp["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients_tmp["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients_tmp["da_prev"].shape)
print("gradients[\"dc_prev\"][2][3] =", gradients_tmp["dc_prev"][2][3])
print("gradients[\"dc_prev\"].shape =", gradients_tmp["dc_prev"].shape)
print("gradients[\"dWf\"][3][1] =", gradients_tmp["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients_tmp["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients_tmp["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients_tmp["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients_tmp["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients_tmp["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients_tmp["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients_tmp["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients_tmp["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients_tmp["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients_tmp["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients_tmp["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients_tmp["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients_tmp["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients_tmp["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients_tmp["dbo"].shape)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **gradients["dxt"][1][2]** =
#         </td>
#         <td>
#            3.23055911511
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dxt"].shape** =
#         </td>
#         <td>
#            (3, 10)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da_prev"][2][3]** =
#         </td>
#         <td>
#            -0.0639621419711
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da_prev"].shape** =
#         </td>
#         <td>
#            (5, 10)
#         </td>
#     </tr>
#          <tr>
#         <td>
#             **gradients["dc_prev"][2][3]** =
#         </td>
#         <td>
#            0.797522038797
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dc_prev"].shape** =
#         </td>
#         <td>
#            (5, 10)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWf"][3][1]** = 
#         </td>
#         <td>
#            -0.147954838164
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWf"].shape** =
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWi"][1][2]** = 
#         </td>
#         <td>
#            1.05749805523
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWi"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dWc"][3][1]** = 
#         </td>
#         <td>
#            2.30456216369
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWc"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dWo"][1][2]** = 
#         </td>
#         <td>
#            0.331311595289
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWo"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dbf"][4]** = 
#         </td>
#         <td>
#            [ 0.18864637]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbf"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dbi"][4]** = 
#         </td>
#         <td>
#            [-0.40142491]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbi"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbc"][4]** = 
#         </td>
#         <td>
#            [ 0.25587763]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbc"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbo"][4]** = 
#         </td>
#         <td>
#            [ 0.13893342]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbo"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
# </table>

# ### 3.3 Backward pass through the LSTM RNN
# 

def lstm_backward(da, caches):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Create variables of the same dimension as return variables.
    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t] + da_prev_t, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        da_prevt = gradients["dat"]
        dc_prevt = gradients["dct"]
        dx[:,:,t] = gradients["dxt"] # dxt, specifically, is stored
        dWf = gradients["dWf"]
        dWi = gradients["dWi"]
        dWc = gradients["dWc"]
        dWo = gradients["dWo"]
        dbf = gradients["dbf"]
        dbi = gradients["dbi"]
        dbc = gradients["dbc"]
        dbo = gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients


# 

np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)

parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.zeros((2,5))       # unused, but needed for lstm_forward
parameters_tmp['by'] = np.zeros((2,1))       # unused, but needed for lstm_forward

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)

da_tmp = np.random.randn(5, 10, 4)
gradients_tmp = lstm_backward(da_tmp, caches_tmp)

print("gradients[\"dx\"][1][2] =", gradients_tmp["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients_tmp["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients_tmp["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients_tmp["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients_tmp["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients_tmp["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients_tmp["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients_tmp["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients_tmp["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients_tmp["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients_tmp["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients_tmp["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients_tmp["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients_tmp["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients_tmp["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients_tmp["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients_tmp["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients_tmp["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients_tmp["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients_tmp["dbo"].shape)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **gradients["dx"][1][2]** =
#         </td>
#         <td>
#            [0.00218254  0.28205375 -0.48292508 -0.43281115]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dx"].shape** =
#         </td>
#         <td>
#            (3, 10, 4)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da0"][2][3]** =
#         </td>
#         <td>
#            0.312770310257
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["da0"].shape** =
#         </td>
#         <td>
#            (5, 10)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWf"][3][1]** = 
#         </td>
#         <td>
#            -0.0809802310938
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWf"].shape** =
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWi"][1][2]** = 
#         </td>
#         <td>
#            0.40512433093
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWi"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dWc"][3][1]** = 
#         </td>
#         <td>
#            -0.0793746735512
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWc"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dWo"][1][2]** = 
#         </td>
#         <td>
#            0.038948775763
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dWo"].shape** = 
#         </td>
#         <td>
#            (5, 8)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dbf"][4]** = 
#         </td>
#         <td>
#            [-0.15745657]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbf"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **gradients["dbi"][4]** = 
#         </td>
#         <td>
#            [-0.50848333]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbi"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbc"][4]** = 
#         </td>
#         <td>
#            [-0.42510818]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbc"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbo"][4]** = 
#         </td>
#         <td>
#            [ -0.17958196]
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **gradients["dbo"].shape** = 
#         </td>
#         <td>
#            (5, 1)
#         </td>
#     </tr>
# </table>
#
# 

# End of Code



