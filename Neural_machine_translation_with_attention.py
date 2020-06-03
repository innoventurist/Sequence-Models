
# coding: utf-8

# # Neural Machine Translation
#
# Goal:
# * Build a Neural Machine Translation (NMT) model to translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25"). 
# * Do this using an attention model, one of the most sophisticated sequence-to-sequence models. 
# 
# This repository was produced together with NVIDIA's Deep Learning Institute. 
#

# Load all the packages needed
#

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## 1 - Translating human readable dates into machine readable dates
# 
# * The model you will build here could be used to translate from one language to another, such as translating from English to Hindi. 
# * However, language translation requires massive datasets and usually takes days of training on GPUs. 
# * To give you a place to experiment with these models without using massive datasets, we will perform a simpler "date translation" task. 
# * The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) 
# * The network will translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). 
# * We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 
# 
# <!-- 
# Take a look at [nmt_utils.py](./nmt_utils.py) to see all the formatting. Count and figure out how the formats work, you will need this knowledge later. !--> 

# ### 1.1 - Dataset
# 

m = 10000 # Model will be trained on a 10,000 human readable dates
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m) # Load the dataset of human equivalent, standardize, machine readable dates
# 

dataset[:10]
#
# Loaded:
# - `dataset`: a list of tuples of (human readable date, machine readable date).
# - `human_vocab`: a python dictionary mapping all characters used in the human readable dates to an integer-valued index.
# - `machine_vocab`: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. 
#     - **Note**: These indices are not necessarily consistent with `human_vocab`. 
# - `inv_machine_vocab`: the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 
#

# Preprocess the data
Tx = 30 # Maximum length of human readable date
Ty = 10 # represents number of characters long
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty) # map the raw data into the index values

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

# Now have:
# - `X`: a processed version of the human readable dates in the training set.
# - `Y`: a processed version of the machine readable dates in the training set.
# - `Xoh`: one-hot version of `X`
# - `Yoh`: one-hot version of `Y`
#

# Index cell below to navigate the dataset and see how source/target dates are preprocessed. 
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])


# ## 2 - Neural machine translation with attention
#
# * Would read/re-read and focus on the parts of the translation paragraph corresponding to the parts of the other translation being written down. 
#
# ### 2.1 - Implementing Attention Mechanism
# This tells a Neural Machine Translation model where it should pay attention to at any step.
#

# Defined shared layers as global variables
repeator = RepeatVector(Tx) # repeats the input at whatever amount of times
concatenator = Concatenate(axis=-1) # Combining two separate strings, merging them until they become one
densor1 = Dense(10, activation = "tanh") # implements output operation where activation is an element-wise activation
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

#

# Function: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" 
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis 
    # For grading purposes, please list 'a' first and 's_prev' second, in this order.
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. 
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell 
    context = dotor([alphas, a])
    
    return context


# Able to check the expected output of `one_step_attention()` after coding the `model()` function.

n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-attention LSTM 
output_layer = Dense(len(machine_vocab), activation=softmax) # SoftMax generates a prediction


# Nowcan use these layers $T_y$ times in a `for` loop to generate the outputs, and their parameters will not be reinitialized. Carry out the following steps: 
# 
# 1. Propagate the input `X` into a bi-directional LSTM.
#     * [Bidirectional](https://keras.io/layers/wrappers/#bidirectional) 
#     * [LSTM](https://keras.io/layers/recurrent/#lstm)
#     * Remember that we want the LSTM to return a full sequence instead of just the last hidden state.  
#     
# Sample code:
# 
# ```Python
# sequence_of_hidden_states = Bidirectional(LSTM(units=..., return_sequences=...))(the_input_X)
# ```
#     
# 2. Iterate for $t = 0, \cdots, T_y-1$: 
#     1. Call `one_step_attention()`, passing in the sequence of hidden states $[a^{\langle 1 \rangle},a^{\langle 2 \rangle}, ..., a^{ \langle T_x \rangle}]$ from the pre-attention bi-directional LSTM, and the previous hidden state $s^{<t-1>}$ from the post-attention LSTM to calculate the context vector $context^{<t>}$.
#     2. Give $context^{<t>}$ to the post-attention LSTM cell. 
#         - Remember to pass in the previous hidden-state $s^{\langle t-1\rangle}$ and cell-states $c^{\langle t-1\rangle}$ of this LSTM 
#         * This outputs the new hidden state $s^{<t>}$ and the new cell state $c^{<t>}$.  
# 
#         Sample code:
#         ```Python
#         next_hidden_state, _ , next_cell_state = 
#             post_activation_LSTM_cell(inputs=..., initial_state=[prev_hidden_state, prev_cell_state])
#         ```   
#         Please note that the layer is actually the "post attention LSTM cell".  For the purposes of passing the automatic grader, please do not modify the naming of this global variable.  This will be fixed when we deploy updates to the automatic grader.
#     3. Apply a dense, softmax layer to $s^{<t>}$, get the output.  
#         Sample code:
#         ```Python
#         output = output_layer(inputs=...)
#         ```
#     4. Save the output by adding it to the list of outputs.
# 
# 3. Create your Keras model instance.
#     * It should have three inputs:
#         * `X`, the one-hot encoded inputs to the model, of shape ($T_{x}, humanVocabSize)$
#         * $s^{\langle 0 \rangle}$, the initial hidden state of the post-attention LSTM
#         * $c^{\langle 0 \rangle}$), the initial cell state of the post-attention LSTM
#     * The output is the list of outputs.  
#     Sample code
#     ```Python
#     model = Model(inputs=[...,...,...], outputs=...)
#     ```

# Function: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    X = Input(shape=(Tx, human_vocab_size))
    # Define s0 (initial hidden state) and c0 (initial cell state)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    # for the decoder LSTM with shape (n_s,)
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    # Define your pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(n_a, return_sequences = True))(X)
    
    # Iterate for Ty steps
    for t in range(Ty):
    
        # Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a, s)
        
        # Apply the post-attention LSTM cell to the "context" vector and pass: initial_state = [hidden state, cell state] 
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        
        # Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)
        
        # Append "out" to the "outputs" list
        outputs.append(out)
    
    # Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    
    return model


# Run the following cell to create the model.
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

#
# Obtain a summary of the model to check if it matches the expected output.
model.summary()


# **Expected Output**:
# 
# Here is the summary you should see
# <table>
#     <tr>
#         <td>
#             **Total params:**
#         </td>
#         <td>
#          52,960
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **Trainable params:**
#         </td>
#         <td>
#          52,960
#         </td>
#     </tr>
#             <tr>
#         <td>
#             **Non-trainable params:**
#         </td>
#         <td>
#          0
#         </td>
#     </tr>
#                     <tr>
#         <td>
#             **bidirectional_1's output shape **
#         </td>
#         <td>
#          (None, 30, 64)  
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **repeat_vector_1's output shape **
#         </td>
#         <td>
#          (None, 30, 64) 
#         </td>
#     </tr>
#                 <tr>
#         <td>
#             **concatenate_1's output shape **
#         </td>
#         <td>
#          (None, 30, 128) 
#         </td>
#     </tr>
#             <tr>
#         <td>
#             **attention_weights's output shape **
#         </td>
#         <td>
#          (None, 30, 1)  
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **dot_1's output shape **
#         </td>
#         <td>
#          (None, 1, 64)
#         </td>
#     </tr>
#            <tr>
#         <td>
#             **dense_3's output shape **
#         </td>
#         <td>
#          (None, 11) 
#         </td>
#     </tr>
# </table>
# 

#
opt = Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01) # Define optimizer
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy']) # Compile the model defining loss function, optimizer and metrics

#

# Define inputs and outputs
# Create `s0` and `c0` to initialize `post_attention_LSTM_cell` with zeros.
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

# 

# Now fit the model and run it for one epoch.

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

#

model.load_weights('models/model.h5') # Load the weights to obtain a model of similar accuracy, saving time


# Now see the results on new examples.

#

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output),"\n")



### 3 - Visualizing Attention (Optional / Ungraded)
# Give a better sense of what the attention mechanism is doing such as:
#   - what part of the input the network is paying attention to when generating a particular output character. 

#
# ### 3.1 - Getting the attention weights from the network
# 
# Now, visualize the attention values in the network. 
# 

# First, print a summary of the model 
model.summary()

#

# Use function `attention_map()` to pull out the attention values from the model and plots them.
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);


# On the generated plot, can observe the values of the attention weights for each character of the predicted output.
# Examine this plot and check that the places where the network is paying attention makes sense.
# 
# In the date translation application, will observe that most of the time attention helps predict the year, and doesn't have much impact on predicting the day or month.

# 
### Takeway:
# 
# - Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages,but also for tasks like date format translation. 
# - An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
# - A network using an attention mechanism can translate from inputs of length $T_x$ to outputs of length $T_y$, where $T_x$ and $T_y$ can be different. 
# - Cabn visualize attention weights $\alpha^{\langle t,t' \rangle}$ to see what the network is paying attention to while generating each output.
