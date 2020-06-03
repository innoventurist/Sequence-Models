
# coding: utf-8

# # Character level language model 
# 
# 
# Goal:
# 
# - How to store text data for processing using an RNN 
# - How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
# - How to build a character-level text generation recurrent neural network
# - Why clipping the gradients is important


# Import all packages needed

import numpy as np
from utils import *
import random
import pprint


# ## 1 - Problem Statement
# 
# ### 1.1 - Dataset and Preprocessing
#  

data = open('dinos.txt', 'r').read() # read dataset of dinosoar names
data= data.lower() # let data be lowercase
chars = list(set(data)) # creare list on unique  characters
data_size, vocab_size = len(data), len(chars) # compute dataset and vocabulary size
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))


# 

# In[3]:

chars = sorted(chars) # sorting the characters from a - z
print(chars)


# In[4]:

char_to_ix = { ch:i for i,ch in enumerate(chars) } # create python dictionary to map each character to an index from 0-26
ix_to_char = { i:ch for i,ch in enumerate(chars) } # second python dictionary, mapping each index to corresponding character
pp = pprint.PrettyPrinter(indent=4) # this returns the formatted representation of an object ; helps load objects not fundamental to Python
pp.pprint(ix_to_char)


# ### 1.2 - Overview of the model
# 
# What the model is doing: 
# 
# - Initialize parameters 
# - Run the optimization loop
#     - Forward propagation to compute the loss function
#     - Backward propagation to compute the gradients with respect to the loss function
#     - Clip the gradients to avoid exploding gradients
#     - Using the gradients, update your parameters with the gradient descent update rule.
# - Return the learned parameters 
#     

# ## 2 - Building blocks of the model
# 
# In this part, build two important blocks of the overall model:
# - Gradient clipping: to avoid exploding gradients
# - Sampling: a technique used to generate characters
# 
# Then apply these two functions to build the model.

# ### 2.1 - Clipping the gradients in the optimization loop
# 
# In this section you will implement the `clip` function that you will call inside of your optimization loop. 
# 
# #### Exploding gradients
# * When gradients are very large, they're called "exploding gradients."  
# * Exploding gradients make the training process more difficult, because the updates may be so large that they "overshoot" the optimal values during back propagation.
# 
# Recall the overall loop structure usually consists of:
# * forward pass, 
# * cost computation, 
# * backward pass, 
# * parameter update. 
# 
# 

# Function: clip

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWax, dWaa, dWya, db, dby]: # loop over [dWax, dWaa, dWya, db, dby].
        np.clip(gradient, -maxValue, maxValue, out=gradient) # clip [dWax, dWaa, dWya, db, dby] to mitigate exploding gradients
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby} 
    
    return gradients

#

# Test with a maxvalue of 10 with gradients
maxValue = 10
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, maxValue)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])


# 

# ** Expected output:**
# 
# ```Python
# gradients["dWaa"][1][2] = 10.0
# gradients["dWax"][3][1] = -10.0
# gradients["dWya"][1][2] = 0.29713815361
# gradients["db"][4] = [ 10.]
# gradients["dby"][1] = [ 8.45833407]
# ```

#

# Test now with a maxValue of 5
maxValue = 5
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, maxValue)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])


# ** Expected Output: **
# ```Python
# gradients["dWaa"][1][2] = 5.0
# gradients["dWax"][3][1] = -5.0
# gradients["dWya"][1][2] = 0.29713815361
# gradients["db"][4] = [ 5.]
# gradients["dby"][1] = [ 5.]
# ```

# ### 2.2 - Sampling
# 

# In[8]:

import numpy as np


# In[9]:

matrix1 = np.array([[1,1],[2,2],[3,3]]) # (3,2) 
matrix2 = np.array([[0],[0],[0]]) # (3,1) ; input "dummy" vector of zeros ; default input before before generating any characters
vector1D = np.array([1,1]) # (2,) 
vector2D = np.array([[1],[1]]) # (2,1) # use 2D arrays instead of 1D
print("matrix1 \n", matrix1,"\n")
print("matrix2 \n", matrix2,"\n")
print("vector1D \n", vector1D,"\n")
print("vector2D \n", vector2D)


# In[10]:

print("Multiply 2D and 1D arrays: result is a 1D array\n", 
      np.dot(matrix1,vector1D))
print("Multiply 2D and 2D arrays: result is a 2D array\n", 
      np.dot(matrix1,vector2D))


# In[11]:

print("Adding (3 x 1) vector to a (3 x 1) vector is a (3 x 1) vector\n",
      "This is what we want here!\n", 
      np.dot(matrix1,vector2D) + matrix2) 


# In[12]:

print("Adding a (3,) vector to a (3 x 1) vector\n",
      "broadcasts the 1D array across the second dimension\n",
      "Not what we want here!\n",
      np.dot(matrix1,vector1D) + matrix2
     )


# - **Step 3**: Sampling: 
#     - Now that we have $y^{\langle t+1 \rangle}$, we want to select the next letter in the dinosaur name. If we select the most probable, the model will always generate the same result given a starting letter. 
#         - To make the results more interesting, we will use np.random.choice to select a next letter that is likely, but not always the same.
#     - Sampling is the selection of a value from a group of values, where each value has a probability of being picked.  
#     - Sampling allows us to generate random sequences of values.
#     - Pick the next character's index according to the probability distribution specified by $\hat{y}^{\langle t+1 \rangle }$. 
#     - This means that if $\hat{y}^{\langle t+1 \rangle }_i = 0.16$, you will pick the index "i" with 16% probability. 
#     - You can use [np.random.choice](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html).
# 
#     Example of how to use `np.random.choice()`:
#     ```python
#     np.random.seed(0)
#     probs = np.array([0.1, 0.0, 0.7, 0.2])
#     idx = np.random.choice([0, 1, 2, 3] p = probs)
#     ```
#     - This means that you will pick the index (`idx`) according to the distribution: 
# 
#     $P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$.
# 
#     - Note that the value that's set to `p` should be set to a 1D vector.
#     - Also notice that $\hat{y}^{\langle t+1 \rangle}$, which is `y` in the code, is a 2D array.

# ##### Additional Hints
# - [range](https://docs.python.org/3/library/functions.html#func-range)
# - [numpy.ravel](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html) takes a multi-dimensional array and returns its contents inside of a 1D vector.
# ```Python
# arr = np.array([[1,2],[3,4]])
# print("arr")
# print(arr)
# print("arr.ravel()")
# print(arr.ravel())
# ```
# Output:
# ```Python
# arr
# [[1 2]
#  [3 4]]
# arr.ravel()
# [1 2 3 4]
# ```
# 
# - Note that `append` is an "in-place" operation.  In other words, don't do this:
# ```Python
# fun_hobbies = fun_hobbies.append('learning')  ## Doesn't give you what you want
# ```

# - **Step 4**: Update to $x^{\langle t \rangle }$ 
#     - The last step to implement in `sample()` is to update the variable `x`, which currently stores $x^{\langle t \rangle }$, with the value of $x^{\langle t + 1 \rangle }$. 
#     - You will represent $x^{\langle t + 1 \rangle }$ by creating a one-hot vector corresponding to the character that you have chosen as your prediction. 
#     - You will then forward propagate $x^{\langle t + 1 \rangle }$ in Step 1 and keep repeating the process until you get a "\n" character, indicating that you have reached the end of the dinosaur name. 


# - In order to reset `x` before setting it to the new one-hot vector, want to set all the values to zero.
#     - Can either create a new numpy array: [numpy.zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html)
#     - Or fill all values with a single number: [numpy.ndarray.fill](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.fill.html)

# In[13]:

# GRADED FUNCTION: sample

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b'] # retreive parameters from parameters dictionary
    vocab_size = by.shape[0] # retrieve relavent shapes
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size, 1)) # Create the a zero vector x that can be used as the one-hot vector ; representing the first character (initializing the sequence generation) 
    a_prev = np.zeros((n_a, 1)) # Initialize a_prev as zeros 
    
    indices = [] # create empty list of indices that will contain the list of indices of the characters to generate
    
    # idx is the index of the one-hot vector x that is set to 1
    # All other positions in x are zero.
    idx = -1 # initialize idx to -1
    
    # Loop over time-steps t. At each time-step:
 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)           #hidden state
        z = np.dot(Wya, a) + by                               #activation
        y = softmax(z)      #prediction
        
        # for grading purposes
        np.random.seed(counter+seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(list(range(vocab_size)), p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input x with one that corresponds to the sampled index `idx`.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        # for grading purposes
        seed += 1
        counter +=1
        

    if (counter == 50):
        indices.append(char_to_ix['\n']) # append indices and stop if reaching 50
    
    return indices


# In[14]:

np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:\n", indices)
print("list of sampled characters:\n", [ix_to_char[i] for i in indices])


# ** Expected output:**
# 
# ```Python
# Sampling:
# list of sampled indices:
#  [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 17, 24, 12, 13, 24, 0]
# list of sampled characters:
#  ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'q', 'x', 'l', 'm', 'x', '\n']
# ```

# ## 3 - Building the language model 
# 
# Build the character-level language model for text generation. 
# 
# 
# ### 3.1 - Gradient descent 
#

# Function: optimize

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
  
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, maxValue = 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    
    return loss, gradients, a[len(X)-1]


# 

np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])


# ** Expected output:**
# 
# ```Python
# Loss = 126.503975722
# gradients["dWaa"][1][2] = 0.194709315347
# np.argmax(gradients["dWax"]) = 93
# gradients["dWya"][1][2] = -0.007773876032
# gradients["db"][4] = [-0.06809825]
# gradients["dby"][1] = [ 0.01538192]
# a_last[4] = [-1.]
# ```

# ### 3.2 - Training the model 

#

# Function: model

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(num_iterations):
        
        
        # Set the index `idx` (see instructions above)
        idx = j % len(examples)
        
        # Set the input X (see instructions above)
        single_example = None
        single_example_chars = None
        single_example_ix = None
        X = [None] + [char_to_ix[ch] for ch in examples[idx]]
        
        # Set the labels Y (see instructions above)
        ix_newline = None
        Y = X[1:] + [char_to_ix['\n']]       
        
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # To get the same result (for grading purposes), increment the seed by one. 
      
            print('\n')
        
    return parameters

# 

parameters = model(data, ix_to_char, char_to_ix) # run cell and observe the model outputting random-looking characters at the first iteration
# After a few thousand iterations, the model should learn to generate reasonable-looking names. 


# ** Expected Output**
# 
# The output of your model may look different, but it will look something like this:
# 
# ```Python
# Iteration: 34000, Loss: 22.447230
# 
# Onyxipaledisons
# Kiabaeropa
# Lussiamang
# Pacaeptabalsaurus
# Xosalong
# Eiacoteg
# Troia
# ```

### Conclusion
# 
# Algorithm has started to generate plausible dinosaur names towards the end of the training.
#   -At first, it was generating random characters, but towards the end you could see dinosaur names with cool endings. 
# 
# If it generates some non-cool names, don't blame the model entirely--not all actual dinosaur names sound cool. 
# 

# 
# Learn more about Keras Team's text generation implementation on GitHub: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py.

# **References**:
# - This exercise took inspiration from Andrej Karpathy's implementation: https://gist.github.com/karpathy/d4dee566867f8291f086. To learn more about text generation, also check out Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
# - For the Shakespearian poem generator, our implementation was based on the implementation of an LSTM text generator by the Keras team: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 

#



