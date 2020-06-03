
# coding: utf-8

# ## Trigger Word Detection
# 
# 
# Learned about applying deep learning to speech recognition by constructing a speech dataset
# and implement an algorithm for trigger word detection (sometimes also called keyword detection, or wake word detection). 
# 
# * Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.  

# 
# Goal: 
# - Structure a speech recognition project
# - Synthesize and process audio recordings to create train/dev datasets
# - Train a trigger word detection model and make predictions


# Import all packages to use

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
get_ipython().magic('matplotlib inline')


# # 1 - Data synthesis: Creating a speech dataset 
# 
# ## 1.1 - Listening to the data   
# 

# Running and displaying all 10 second clip to listen to the 3 examples given

IPython.display.Audio("./raw_data/activates/1.wav")

#

IPython.display.Audio("./raw_data/negatives/4.wav")

# 

IPython.display.Audio("./raw_data/backgrounds/1.wav")


# Use these three types of recordings (positives/negatives/backgrounds) to create a labeled dataset.

# ## 1.2 - From audio recordings to spectrograms
# 
# What really is an audio recording? 
# * A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear also perceives as sound. 
# * You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. 
# * We will use audio sampled at 44100 Hz (or 44100 Hertz). 
#     * This means the microphone gives us 44,100 numbers per second. 
#     * Thus, a 10 second audio clip is represented by 441,000 numbers (= $10 \times 44,100$). 
# 
# #### Spectrogram
# * It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said. 
# * In  order to help the sequence model more easily learn to detect trigger words,  compute a *spectrogram* of the audio. 
# * The spectrogram tells how much different frequencies are present in an audio clip at any moment in time. 
# * What signal processing or on Fourier transforms does:
#     * A spectrogram is computed by sliding a window over the raw audio signal, and calculating the most active frequencies in each window using a Fourier transform. 
#     * If you don't understand the previous sentence, don't worry about it.
# 

# Below, an example of taking the the audio recording and graphing it into a spectrogram

IPython.display.Audio("audio_examples/example_train.wav")


# In[6]:

x = graph_spectrogram("audio_examples/example_train.wav")


# The graph above represents how active each frequency is (y axis) over a number of time-steps (x axis). 
# 
# 
# * The color in the spectrogram shows the degree to which different frequencies are present (loud) in the audio at different points in time. 
# * Green means a certain frequency is more active or more present in the audio clip (louder).
# * Blue squares denote less active frequencies.
# * The dimension of the output spectrogram depends upon the hyperparameters of the spectrogram software and the length of the input. 
#

# 

_, data = wavfile.read("audio_examples/example_train.wav") # return the sample rate and data rate from a WAV file
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)


# Now, can define:

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram


# #### Dividing into time-intervals
# Note that may divide a 10 second interval of time with different units (steps).
# * Raw audio divides 10 seconds into 441,000 units.
# * A spectrogram divides 10 seconds into 5,511 units.
#     * $T_x = 5511$
# * Use a Python module `pydub` to synthesize audio, and it divides 10 seconds into 10,000 units.
# * The output of the model will divide 10 seconds into 1,375 units.
#     * $T_y = 1375$
#     * For each of the 1375 time steps, the model predicts whether someone recently finished saying the trigger word "activate." 
# * All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone). 
# * Chosen values that are within the standard range used for speech systems.

# In[9]:

Ty = 1375 # The number of time steps in the output of our model


# ## 1.3 - Generating a single training example
# 
# #### Benefits of synthesizing data
# * It is quite slow to record lots of 10 second audio clips with random "activates" in it. 
# * Instead, it is easier to record lots of positives and negative words, and record background noise separately (or download background noise from free online sources). 
# 
# #### Process for Synthesizing an audio clip
# * To synthesize a single training example:
#     - Pick a random 10 second background audio clip
#     - Randomly insert 0-4 audio clips of "activate" into this 10sec clip
#     - Randomly insert 0-2 audio clips of negative words into this 10sec clip
# 
# #### Pydub
# * Use pydub package to manipulate audio. 
# * Pydub converts raw audio files into lists of Pydub data structures.
# * Pydub uses 1ms as the discretization interval (1ms is 1 millisecond = 1/1000 seconds).
#     * This is why a 10 second clip is always represented using 10,000 steps. 

# 

# Load audio segments using pydub 
activates, negatives, backgrounds = load_raw_audio()

print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")


# ### Overlaying positive/negative 'word' audio clips on top of the background audio
# 
# * Given a 10 second background clip and a short audio clip containing a positive or negative word, need to be able to "add" the word audio clip on top of the background audio.
# * Will be inserting multiple clips of positive/negative words into the background, and don't want to insert an "activate" or a random word somewhere that overlaps with another clip previously added. 
#     * To ensure that the 'word' audio segments do not overlap when inserted, you will keep track of the times of previously inserted audio clips. 
# * To be clear,  inserting a 1 second "activate" onto a 10 second clip of cafe noise, **do not end up with an 11 sec clip** 
#     * The resulting audio clip is still 10 seconds long.
#     * See later how pydub allows you to do this. 

# #### Label the positive/negative words
# 
# #### Helper functions
# 
# To implement the training set synthesis process, you will use the following helper functions. 
# * All of these functions will use a 1ms discretization interval
# * The 10 seconds of audio is always discretized into 10,000 steps. 
# 
# 
# 1. `get_random_time_segment(segment_ms)`
#     * Retrieves a random time segment from the background audio.
# 2. `is_overlapping(segment_time, existing_segments)`
#     * Checks if a time segment overlaps with existing segments
# 3. `insert_audio_clip(background, audio_clip, existing_times)`
#     * Inserts an audio segment at a random time in the background audio
#     * Uses the functions `get_random_time_segment` and `is_overlapping`
# 4. `insert_ones(y, segment_end_ms)`
#     * Inserts additional 1's into the label vector y after the word "activate"

# 

def get_random_time_segment(segment_ms): # this returns a random time segment onto which we can insert an audio clip of duration `segment_ms`.
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


# #### Check if audio clips are overlapping
# 
#

# Function: is_overlapping

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
  
    overlap = False  # Initialize overlap as a "False" flag.

    for previous_start, previous_end in previous_segments: # loop over the previous_segments start and end times.
        if segment_start <= previous_end and segment_end >= previous_start: #compare these times to the start and end times
            overlap = True # Set the flag to True if there is an overlap
 
    return overlap


# In[13]:

overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
print("Overlap 1 = ", overlap1)
print("Overlap 2 = ", overlap2)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Overlap 1**
#         </td>
#         <td>
#            False
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **Overlap 2**
#         </td>
#         <td>
#            True
#         </td>
#     </tr>
# </table>

# #### Insert audio clip
# 

# Function: insert_audio_clip

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration (length) of the audio clip in ms
    segment_ms = len(audio_clip)

    # Use one of the helper functions to pick a random time segment onto which to insert the new audio clip. 
    segment_time = get_random_time_segment(segment_ms)
    
    # Check if the new segment_time overlaps with one of the previous_segments.
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms) # If so, keep picking new segment_time at random until it doesn't overlap

    # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    
    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time


# In[15]:

np.random.seed(5)
audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
audio_clip.export("insert_test.wav", format="wav")
print("Segment Time: ", segment_time)
IPython.display.Audio("insert_test.wav")


# **Expected Output**
# 
# <table>
#     <tr>
#         <td>
#             **Segment Time**
#         </td>
#         <td>
#            (2254, 3169)
#         </td>
#     </tr>
# </table>

# In[16]:

# Expected audio
IPython.display.Audio("audio_examples/insert_reference.wav")


# #### Insert ones for the labels of the positive target
# 
# * Implement code to update the labels $y^{\langle t \rangle}$, assuming you just inserted an "activate" audio clip.
# * In the code below, `y` is a `(1,1375)` dimensional vector, since $T_y = 1375$. 
# * If the "activate" audio clip ends at time step $t$, then set $y^{\langle t+1 \rangle} = 1$ and also set the next 49 additional consecutive values to 1.
#     * Notice that if the target word appears near the end of the entire audio clip, there may not be 50 additional time steps to set to 1.
#     * Make sure you don't run off the end of the array and try to update `y[0][1375]`, since the valid indices are `y[0][0]` through `y[0][1374]` because $T_y = 1375$. 
#     * So if "activate" ends at step 1370, you would get only set `y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1`
# 
# **Exercise**: 
# Implement `insert_ones()`. 
# * You can use a for loop. 
# * If you want to use Python's array slicing operations, you can do so as well.
# * If a segment ends at `segment_end_ms` (using a 10000 step discretization),
#     * To convert it to the indexing for the outputs $y$ (using a $1375$ step discretization), we will use this formula:  
# ```
#     segment_end_y = int(segment_end_ms * Ty / 10000.0)
# ```

# In[21]:

# Function: insert_ones

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    for i in range(segment_end_y + 1, segment_end_y + 51): # use for loop 
        if i < Ty:
            y[0, i] = 1 

    
    return y


# 

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0,:]) # making a change to the figure
print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])


# **Expected Output**
# <table>
#     <tr>
#         <td>
#             **sanity checks**:
#         </td>
#         <td>
#            0.0 1.0 0.0
#         </td>
#     </tr>
# </table>
# <img src="images/ones_reference.png" style="width:320;height:240px;">

# #### Creating a training example

# Function: create_training_example

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    
    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    
    # Set the random seed
    np.random.seed(18)
    
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros 
    y = np.zeros((1, Ty))

    # Initialize segment times as an empty list 
    previous_segments = []
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end_ms = segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y

# 

x, y = create_training_example(backgrounds[0], activates, negatives)


# **Expected Output**
# <img src="images/train_reference.png" style="width:320;height:240px;">

# 

IPython.display.Audio("train.wav") # can listen to training example created and compare to the spectrogram generated above.


# **Expected Output**

IPython.display.Audio("audio_examples/train_reference.wav")

# 

plt.plot(y[0]) # finally, plot the associated labels for the generated training example.


# **Expected Output**
# <img src="images/train_label.png" style="width:320;height:240px;">

# ## 1.4 - Full training set
#

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")


# ## 1.5 - Development set
# 

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


# # 2 - Model
# 
# Write and train a trigger word detection model! 
# The model will use 1-D convolutional layers, GRU layers, and dense layers. 
#  
# Load the packages that allows to use these layers in Keras.

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


# ## 2.1 - Build the model
# 
# ##### 1D convolutional layer
# * It inputs the 5511 step spectrogram.  Each step is a vector of 101 units.
# * It outputs a 1375 step output
# * This output is further processed by multiple layers to get the final $T_y = 1375$ step output. 
# * This 1D convolutional layer plays a role similar to the 2D convolutions of extracting low-level features and then possibly generating an output of a smaller dimension. 
# * Computationally, the 1-D conv layer also helps speed up the model because now the GRU  can process only 1375 timesteps rather than 5511 timesteps. 
# 
# ##### GRU, dense and sigmoid
# * The two GRU layers read the sequence of inputs from left to right.
# * A dense plus sigmoid layer makes a prediction for $y^{\langle t \rangle}$. 
# * Use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said "activate."
# 
# #### Unidirectional RNN
# * This is really important for trigger word detection, since we want to be able to detect the trigger word almost immediately after it is said. 
# * If using bidirectional RNN, have to wait for the whole 10sec of audio to be recorded before telling if "activate" was said in the first second of the audio clip.  

# #### Implement the model
#
# Function: model

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size = 15, strides = 4)(X_input)   # CONV1D
    X = BatchNormalization()(X)                               # Batch normalization
    X = Activation("relu")(X)                                 # ReLu activation
    X = Dropout(rate = 0.8)(X)                                # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)          # GRU (use 128 units and return the sequences)
    X = Dropout(rate = 0.8)(X)                                # dropout (use 0.8)
    X = BatchNormalization()(X)                               # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units = 128, return_sequences = True)(X)   # GRU (use 128 units and return the sequences)
    X = Dropout(rate = 0.8)(X)                         # dropout (use 0.8)
    X = BatchNormalization()(X)                        # Batch normalization
    X = Dropout(rate = 0.8)(X)                         # dropout (use 0.8)
    
    # Time-distributed dense layer (sigmoid)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) 

    model = Model(inputs = X_input, outputs = X)
    
    return model  


# 

model = model(input_shape = (Tx, n_freq))

#

model.summary() # Print the model summary to keep track of the shapes.


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **Total params**
#         </td>
#         <td>
#            522,561
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **Trainable params**
#         </td>
#         <td>
#            521,657
#         </td>
#     </tr>
#     <tr>
#         <td>
#             **Non-trainable params**
#         </td>
#         <td>
#            904
#         </td>
#     </tr>
# </table>

# The output of the network is of shape (None, 1375, 1) while the input is (None, 5511, 101). The Conv1D has reduced the number of steps from 5511 to 1375. 

# ## 2.2 - Fit the model

# * Trigger word detection takes a long time to train. 
# * To save time, a model has been trained for about 3 hours on a GPU using the architecture built above, and a large training set of about 4000 examples. 

#

model = load_model('./models/tr_model.h5') # Load the model

# 

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01) # Train the model further using Adam
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"]) # and using binary cross entropy

# Training just for one epoch and with a small training set of 26 examples. 


# 

model.fit(X, Y, batch_size = 5, epochs=1)


# ## 2.3 - Test the model

# 

loss, acc = model.evaluate(X_dev, Y_dev) # let the model perform on the dev set
print("Dev set accuracy = ", acc)


# This looks pretty good! 
# * However, accuracy isn't a great metric for this task
#     * Since the labels are heavily skewed to 0's, a neural network that just outputs 0's would get slightly over 90% accuracy. 
# * Could define more useful metrics such as F1 score or Precision/Recall. 
#     * Just empirically see how the model does with some predictions.

# # 3 - Making Predictions
# 
# Now that working model is built for trigger word detection, now make a predictions. This code snippet runs audio (saved in a wav file) through the network. 


#

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename) # compute spectrogram for the audio file
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1) # used to reshape input size
    x = np.expand_dims(x, axis=0) # reshapes input size
    predictions = model.predict(x)

    # Use forward propagation on model to comput the prediction at each output step
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


# #### Insert a chime to acknowledge the "activate" trigger

# 

chime_file = "audio_examples/chime.wav" # Can trigger a chime sound to plat then probability is above a certain threshold
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Loop over the output steps in the y
    for i in range(Ty):
        # Increment consecutive output steps
        consecutive_timesteps += 1
        # If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')


# ## 3.3 - Test on dev examples

# Let's explore how the model performs on two unseen audio clips from the development set.  

# Lets first listen to the two dev set clips.

IPython.display.Audio("./raw_data/dev/1.wav")

#

IPython.display.Audio("./raw_data/dev/2.wav")


# Now lets run the model on these audio clips and see if it adds a chime after "activate"!

# 

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")

# 

filename  = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")

# 
# ## Takeaway:
# - Data synthesis is an effective way to create a large training set for speech problems, specifically trigger word detection. 
# - Using a spectrogram and optionally a 1D conv layer is a common pre-processing step prior to passing audio data to an RNN, GRU or LSTM.
# - An end-to-end deep learning approach can be used to build a very effective trigger word detection system. 
# 


# 
