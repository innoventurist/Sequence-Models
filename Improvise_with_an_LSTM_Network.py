
# coding: utf-8

# # LSTM Network
#
# **Goal:**
# - Apply an LSTM to music generation.
# - Generate own jazz music with deep learning.
# 

# Run and load all the packages required

#

from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


# ## 1 - Problem statement
# 
# Train a network to generate novel jazz solos in a style representative of a body of performed work.
# 

# 
# ### 1.1 - Dataset
# 
# Run the cell below to listen to a snippet of the audio from the training set:

IPython.display.Audio('./data/30s_seq.mp3') 


# Have taken care of the preprocessing of the musical data to render it in terms of musical "values." 
#

# Run the following code to load the raw music data and preprocess it into values.

X, Y, n_values, indices_values = load_music_utils()
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('shape of X:', X.shape)
print('Shape of Y:', Y.shape)


# Have just loaded the following:
# 
# - `X`: This is an (m, $T_x$, 78) dimensional array ; is one of 78 different possible values, represented as a one-hot vector. 
#
# - `Y`: a $(T_y, m, 78)$ dimensional array; essentially the same as `X`, but shifted one step to the left (to the past).
# 
# - `n_values`: The number of unique values in this dataset. This should be 78. 
# 
# - `indices_values`: python dictionary mapping integers 0 through 77 to musical values.
# 

# ## 2 - Building the model
# 
# * In this part you will build and train a model that will learn musical patterns. 
# * The model takes input X of shape $(m, T_x, 78)$ and labels Y of shape $(T_y, m, 78)$. 

#

# number of dimensions for the hidden state of each LSTM cell.
n_a = 64 

# 
# * Please read the Keras documentation and understand these layers: 
#     - [Reshape()](https://keras.io/layers/core/#reshape): Reshapes an output to a certain shape.
#     - [LSTM()](https://keras.io/layers/recurrent/#lstm): Long Short-Term Memory layer
#     - [Dense()](https://keras.io/layers/core/#dense): A regular fully-connected neural network layer.
# 

# Defined the layers of objects needed as global values - run the cell below and create them
# * `reshapor`, `LSTM_cell` and `densor` are globally defined layer objects
n_values = 78                                      # number of unique music values
reshapor = Reshape((1, n_values))                  # Will be used in djmodel(), below - takes previous layer as its input argument
LSTM_cell = LSTM(n_a, return_state = True)         # Use LSTM with number of dimesions for each hidden state
densor = Dense(n_values, activation='softmax')     # Propagate LSTM's hidden state through dense+softmax layer

#
# * In order to propagate a Keras tensor object X through one of these layers, use `layer_object()`.
#     - For one input, use `layer_object(X)`
#     - For more than one input, put the inputs in a list: `layer_object([X1,X2])`:
# 
# * Choose the appropriate variables for the input tensor, hidden state, cell state, and output.
# * See the documentation for [Model](https://keras.io/models/model/)

# 

# Function: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
     
    # Create empty list to append the outputs while you iterate 
    outputs = []
    
    # Loop through time step
    for t in range(Tx):
        
        # Select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:, t, :])(X)
        # Use reshapor to reshape x to be (1, n_values) 
        x = reshapor(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Add the output to "outputs"
        outputs.append(out)
        
    # Create model instance
    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model


# Creat the model object to define the model (n_values=78)

model = djmodel(Tx = 30 , n_a = 64, n_values = 78) # n_a and Tx are dimension of LSTM Activations

# 

# Check the model
model.summary()


# **Expected Output**  
# Scroll to the bottom of the output, and you'll see the following:
# 
# ```Python
# Total params: 41,678
# Trainable params: 41,678
# Non-trainable params: 0
# ```
#

# Compile the model for training, using Adam optimizer and categorical cross-entropy loss

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#

### Initialize hidden state and cell state for the LSTM's initial state to be zero. 

m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

#

# Train and fit the model

model.fit([X, a0, c0], list(Y), epochs=100) # turn Y into a list and train on 1-- epochs


# #### Expected Output
# 
# The model loss will start high, (100 or so), and after 100 epochs, it should be in the single digits.  These won't be the exact number that you'll see, due to random initialization of weights.  
# For example:
# ```
# Epoch 1/100
# 60/60 [==============================] - 3s - loss: 125.7673
# ...
# ```
# Scroll to the bottom to check Epoch 100
# ```
# ...
# Epoch 100/100
# 60/60 [==============================] - 0s - loss: 6.1861
# ```
# 
# Now that the model has been train, final section implements an inference algorithm, and generate some music! 

### 3 - Generating music
# 
# Now use this model to synthesize new music. 
#
# * Read the documentation for [keras.argmax](https://www.tensorflow.org/api_docs/python/tf/keras/backend/argmax).
# * Apply the custom one_hot encoding using the [Lambda](https://keras.io/layers/core/#lambda) layer.  
#

# Function: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Step 1: Create an empty list of "outputs" to later store your predicted values 
    outputs = []
    
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state = [a, c])
        
        # Apply Dense layer to the hidden state output of the LSTM_cell 
        out = densor(a)

        # Append the prediction "out" to "outputs". out.shape = (None, 78) 
        outputs.append(out)
        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = Lambda(one_hot)(out)
        
    # Create model instance with the correct "inputs" and "outputs" 
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    return inference_model


# Run the cell below to define your inference model. This model is hard coded to generate 50 values.

inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)

#

# Check the inference model
inference_model.summary()


# ** Expected Output**
# ```
# Total params: 41,678
# Trainable params: 41,678
# Non-trainable params: 0
# ```

# Initialize inference model that creates the zero-valued vectors used to initialize `x` and the LSTM state variables `a` and `c`. 
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

#

# 

# Function: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    # Use inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis = -1)
    # Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices, num_classes = 78)
    
    return results, indices


# Use a dimension from the given parameters of `predict_and_sample()'

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))


# **Expected (Approximate) Output**: 
# 
# * Rsults **may likely differ** because Keras' results are not completely predictable. 
# * However, if trained LSTM_cell with model.fit() for exactly 100 epochs as described above: 
#     * Should very likely observe a sequence of indices that are not all identical. 
#     * Moreover, should observe that: 
#         * np.argmax(results[12]) is the first element of list(indices[12:18]) 
#         * and np.argmax(results[17]) is the last element of list(indices[12:18]). 
# 
# <table>
#     <tr>
#         <td>
#             **np.argmax(results[12])** =
#         </td>
#         <td>
#         1
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **np.argmax(results[17])** =
#         </td>
#         <td>
#         42
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **list(indices[12:18])** =
#         </td>
#         <td>
#             [array([1]), array([42]), array([54]), array([17]), array([1]), array([42])]
#         </td>
#     </tr>
# </table>

# #### 3.3 - Generate music 
# 
# RNN generates a sequence of values. The following code generates music by first calling `predict_and_sample()` function. These values are then post-processed into musical chords (meaning that multiple values or notes can be played at the same time). 
# 
# Most computational music algorithms use some post-processing because it is difficult to generate music that sounds good without such post-processing. The post-processing does things such as clean up the generated audio by making sure the same sound is not repeated too many times, that two successive notes are not too far from each other in pitch, and so on. One could argue that a lot of these post-processing steps are hacks; also, a lot of the music generation literature has also focused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the RNN. But this post-processing does make a huge difference, so let's use it in our implementation as well. 
# 
# Let's make some music! 

# Run the following cell to generate music and record it into the `out_stream`.
out_stream = generate_music(inference_model)


# To listen to music, click File->Open... Then go to "output/" and download "my_music.midi".
# Either play it the your computer with an application that can read midi files if needed,
# or use one of the free online "MIDI to mp3" conversion tools to convert this to mp3.  
# 
# As a reference, here is a 30 second audio clip we generated using this algorithm. 
IPython.display.Audio('./data/30s_trained_model.mp3')


# 
# 
# ##Takeaway
# - A sequence model can be used to generate musical values, which are then post-processed into midi music. 
# - Fairly similar models can be used to generate dinosaur names or to generate music, with the major difference being the input fed to the model.  
# - In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps $1, \ldots, T_x$. 

# **References**
# 
# The ideas presented in this notebook came primarily from three computational music papers cited below and used many components from Ji-Sung Kim's GitHub repository.
# 
# - Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
# - Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
# - Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
# - François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)

# End of Code



