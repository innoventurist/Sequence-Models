
# coding: utf-8

# # Operations on word vectors
# 
# **Goal:**
# 
# - Load pre-trained word vectors, and measure similarity using cosine similarity
# - Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______. 
# - Modify word embeddings to reduce their gender bias 
# 
# 

# Run the following cell to load the packages needed.

import numpy as np
from w2v_utils import *


### Load the word vectors of words, word_to_vec_map. 
words, word_to_vec_map = read_glove_vecs('../../readonly/glove.6B.50d.txt') # will use 50-dimensional GloVe vectors to represent words

# What's loaded:
# - `words`: set of words in the vocabulary.
# - `word_to_vec_map`: dictionary mapping words to their GloVe vector representation.
# 
# # 1 - Cosine similarity
# 

#
# Function: cosine_similarity

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    # Compute the dot product between u and v 
    dot = np.dot(u, v)
    # Compute the L2 norm of u 
    norm_u = np.linalg.norm(u)
    
    # Compute the L2 norm of v 
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity defined by the norm formulas 
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity


# Mapping words of their representation

father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **cosine_similarity(father, mother)** =
#         </td>
#         <td>
#          0.890903844289
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **cosine_similarity(ball, crocodile)** =
#         </td>
#         <td>
#          0.274392462614
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **cosine_similarity(france - paris, rome - italy)** =
#         </td>
#         <td>
#          -0.675147930817
#         </td>
#     </tr>
# </table>

# #### Try different words!

# Function: complete_analogy

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
    e_a = word_to_vec_map.get(word_a)
    e_b = word_to_vec_map.get(word_b)
    e_c = word_to_vec_map.get(word_c)
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # to avoid best_word being one of the input words, skip the input words
    # place the input words in a set for faster searching than a list
    input_words_set = set([word_a, word_b, word_c]) # Will re-use this set of input words inside the for-loop
    
    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, skip the input words
        if w in input_words_set:
            continue
       
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  
        cosine_sim = cosine_similarity(np.subtract(e_b, e_a), np.subtract(word_to_vec_map.get(w), e_c))
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word 
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


# Run the cell below to test the code.
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **italy -> italian** ::
#         </td>
#         <td>
#          spain -> spanish
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **india -> delhi** ::
#         </td>
#         <td>
#          japan -> tokyo
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **man -> woman ** ::
#         </td>
#         <td>
#          boy -> girl
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **small -> smaller ** ::
#         </td>
#         <td>
#          large -> larger
#         </td>
#     </tr>
# </table>

# * Once get the correct expected output, please feel free to modify the input cells above to test differnt analogies. 
# * Try to find some other analogy pairs that do work, but also find some where the algorithm doesn't give the right answer:

# Takeaway for this portion:
# 
# - Cosine similarity is a good way to compare the similarity between pairs of word vectors.
#     - Note that L2 (Euclidean) distance also works.
# - For NLP applications, using a pre-trained set of word vectors is often a good way to get started.
# - Even though you have finished the graded portions, we recommend you take a look at the rest of this notebook to learn about debiasing word vectors.

### 3 - Debiasing word vectors

# Examine gender biases that can be reflected in a word embedding, and explore algorithms for reducing the bias.
# Tto learning about the topic of debiasing, will also help understand what word vectors are doing.
# This section involves a bit of linear algebra.
# 

# Compute a vector that represents the word vector corresponds to the other word
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)            # This encodes the concept of "gender"


# Now, consider the cosine similarity of different words with "g". Consider what a positive value of similarity means vs a negative cosine similarity. 
print ('List of names and their similarities with constructed vector:')

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))


# Female first names have a positive cosine similarity with vector "g", while male first names have a negative cosine similarity. The result seems acceptable. 
# 
# But let's try with some other words.
print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))


# DIt is astonishing how these results reflect certain unhealthy gender stereotypes. 
# Below how to reduce the bias of these vectors, using an algorithm due to [Boliukbasi et al., 2016](https://arxiv.org/abs/1607.06520).

# ### 3.1 - Neutralize bias for non-gender specific words 
# 

#

def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    # Select word vector representation of "word". Use word_to_vec_map. 
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent. 
    e_biascomponent = (np.dot(e,g) / np.linalg.norm(g)**2) * g
 
    # Neutralize e by subtracting e_biascomponent from it 
    # e_debiased should be equal to its orthogonal projection. 
    e_debiased = e - e_biascomponent
    
    return e_debiased

# Remove the biases of words such as "receptionist" or "scientist". 
e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))
# Can compute with debiased formula
e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))


# **Expected Output**: The second result is essentially 0, up to numerical rounding (on the order of $10^{-17}$).
# 
# 
# <table>
#     <tr>
#         <td>
#             **cosine similarity between receptionist and g, before neutralizing:** :
#         </td>
#         <td>
#          0.330779417506
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **cosine similarity between receptionist and g, after neutralizing:** :
#         </td>
#         <td>
#          -3.26732746085e-17
#     </tr>
# </table>

# ### 3.2 - Equalization algorithm for gender-specific words
# 
# The key idea behind equalization is to make sure that a particular pair of words are equi-distant from the 49-dimensional $g_\perp$.
# The equalization step also ensures that the two equalized steps are now the same distance from $e_{receptionist}^{debiased}$, or from any other work that has been neutralized. 
# 
# The derivation of the linear algebra to do this is a bit more complex. (See Bolukbasi et al., 2016 for details.) But the key equations are: 
# 
# $$ \mu = \frac{e_{w1} + e_{w2}}{2}\tag{4}$$ 
# 
# $$ \mu_{B} = \frac {\mu \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
# \tag{5}$$ 
# 
# $$\mu_{\perp} = \mu - \mu_{B} \tag{6}$$
# 
# $$ e_{w1B} = \frac {e_{w1} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
# \tag{7}$$ 
# $$ e_{w2B} = \frac {e_{w2} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
# \tag{8}$$
# 
# 
# $$e_{w1B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w1B}} - \mu_B} {||(e_{w1} - \mu_{\perp}) - \mu_B||} \tag{9}$$
# 
# 
# $$e_{w2B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w2B}} - \mu_B} {||(e_{w2} - \mu_{\perp}) - \mu_B||} \tag{10}$$
# 
# $$e_1 = e_{w1B}^{corrected} + \mu_{\perp} \tag{11}$$
# $$e_2 = e_{w2B}^{corrected} + \mu_{\perp} \tag{12}$$
# 
#
# 

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Select word vector representation of "word". Use word_to_vec_map. 
    w1, w2 = pair[0], pair[1]
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # Compute the mean of e_w1 and e_w2 
    mu = (e_w1 + e_w2) / 2

    # Compute the projections of mu over the bias axis and the orthogonal axis 
    mu_B = (np.dot(mu, bias_axis)) / (np.linalg.norm(bias_axis)**2) * bias_axis
    mu_orth = mu - mu_B

    # Compute e_w1B and e_w2B 
    e_w1B = (np.dot(e_w1, bias_axis)) / (np.linalg.norm(bias_axis)**2) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis)) / (np.linalg.norm(bias_axis)**2) * bias_axis
        
    # Step 5: Adjust the Bias part of e_w1B and e_w2B  
    corrected_e_w1B = (np.linalg.norm(1 - (mu_orth))**2) * ((e_w1B - mu_B) /((e_w1 - mu_orth) - mu_B))
    corrected_e_w2B = (np.linalg.norm(1 - (mu_orth))**2) * ((e_w2B - mu_B) /((e_w2 - mu_orth) - mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections 
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w1B + mu_orth
    
    return e1, e2


# 

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))


# **Expected Output**:
# 
# cosine similarities before equalizing:
# <table>
#     <tr>
#         <td>
#             **cosine_similarity(word_to_vec_map["man"], gender)** =
#         </td>
#         <td>
#          -0.117110957653
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **cosine_similarity(word_to_vec_map["woman"], gender)** =
#         </td>
#         <td>
#          0.356666188463
#         </td>
#     </tr>
# </table>
# 
# cosine similarities after equalizing:
# <table>
#     <tr>
#         <td>
#             **cosine_similarity(u1, gender)** =
#         </td>
#         <td>
#          -0.700436428931
#         </td>
#     </tr>
#         <tr>
#         <td>
#             **cosine_similarity(u2, gender)** =
#         </td>
#         <td>
#          0.700436428931
#         </td>
#     </tr>
# </table>

# Feel free to play with the input words in the cell above, to apply equalization to other pairs of words. 
# 
# These debiasing algorithms are very helpful for reducing bias, but are not perfect and do not eliminate all traces of bias. 

# **References**:
# - The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to
# Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
# - The GloVe word embeddings were due to Jeffrey Pennington, Richard Socher, and Christopher D. Manning. (https://nlp.stanford.edu/projects/glove/)
# 
