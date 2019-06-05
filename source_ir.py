# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:54:58 2018

@author: rkrit
"""
from bs4 import BeautifulSoup
import re
import os 
import urllib3
import zipfile
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import tensorflow as tf
import collections
import string
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import keras.backend as K
from keras.objectives import cosine_proximity
import numpy as np
from collections import defaultdict
import urllib
import collections
import os
import zipfile
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
def main():
    collect_data()
    #total_data = collect_data()
    #print(len(total_data))
    #print(total_data[1])
    #file = open("C:\\Users\\rkrit\\Documents\\CS 584_Data Mining\\documentation_java\\docs\\api\\java\\lang\\ArithmeticException.html")
def collect_data():
    
    filename = "C:\\Users\\rkrit\\Documents\\CS 584_Data Mining\\source_code.zip"
    vocabulary,code = read_data(filename)
    code = np.squeeze(np.asarray(code))
    print(len(vocabulary))
    #print(vocabulary[:7])
    vocabulary_size = 83123
    sum = 0
    code3=code
    #data, count, dictionary, reverse_dictionary =
    vocab_size =83123
    window_size = 100
    data, count, dictionary, reverse_dictionary,code_data = build_dataset(vocabulary,vocabulary_size,code3)
    print(code_data)
    couples, labels = skipgrams(data, vocab_size, window_size=window_size,shuffle=False)
    vocab_size = 83123
    print(data) 
    vocab_size =83123
    window_size = 100
    
    print(data[:7])
    import numpy as np
    window_size = 5
    vector_dim = 300
    epochs = 2000
    
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    
    sampling_table = sequence.make_sampling_table(16)
    couples, labels = skipgrams(data, vocab_size, window_size=window_size)
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")
    print(couples[:10], labels[:10])
    
    # create some input variables
    input_target = Input((1,))
    input_context = Input((1,))
    
    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
    
    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)
    print(len(embedding.get_weights()[0]))
    sum1=0
    final =[]
    
    param =embedding.get_weights()[0]
    indices=[]
    
        
    prev=0
    for i in vocabulary:
       
       indices.append([prev,len(i)+prev])
       prev=len(i)
    sum=0
    for ind in indices:
        final.append(param[ind[0]:ind[1]])
        sum=sum+len(param[ind[0]:ind[1]])
        
        
    print(len(indices))
    print(sum)
    print(sum1)
    print(len(final))
    print(len(param))
    print(len(vocabulary[124]))
    print(len(final[124]))
    
    for code1,code2 in code_data:
        
        print("The cosine similarity for  ",code1,"  is  ", K.eval(cos_distance(np.array(param[code2]),np.array(param[code2+100:code2+200]))))   
    # setup a cosine similarity operation which will be output in a secondary model
   
    model = Sequential()
    target = keras.layers.Input(shape=(300,))
    model.add(target)
    context = keras.layers.Input(shape=(300,))
    model.add(Dense(units=50, activation='sigmoid')(context))
    print(couples[:10], labels[:10])
    print(K.eval(cos_distance(x1, x2)))
    model = Sequential()
    model.add(Dense(units=50,activation="relu",input_shape=(300,)))
    model.add(Dense(units=50,activation="relu",input_shape=(300,)))
    model.add(Dense(units=10,activation="softmax"))
    model.compile(optimizer=SGD(0.001),loss="binary_crossentropy",metrics=["accuracy"],optimizer='rmsprop')
   
    #
    
    # setup a cosine similarity operation which will be output in a secondary model
    similarity = merge([target, context], mode='cos', dot_axes=0)
    
    # now perform the dot product operation to get a similarity measure
    dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)
    # create the primary training model
    model = Model(input=[input_target, input_context], output=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    # create a secondary validation model to run our similarity checks during training
    validation_model = Model(input=[input_target, input_context], output=similarity)
    #del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

track = defaultdict(int)
def build_dataset(words, n_words,code):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
   
    c=0
    
    #print(words[0])
    
    s = defaultdict(int)
    
    for word in words:
        
        for w in word:
            s[w] = s[w]+1
            track[c] =track[c]+ 1
        c= c+1    
    #print(track[1])
    #print(s)
    #print(len(s))
    #print(type(s))
    print("S Item" ,s.__getitem__)
    #sorted(s.items(),key = s.__getitem__,reverse=True)
    #count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for k,v in s.items():
        dictionary[k] = len(dictionary)
    data = list()
    code_data =list()
    unk_count = 0
    for word in s.keys():
        if word in dictionary:
            index = dictionary[word]
            if word in code :
                code_data.append((word,index))
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
        
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    return data, count, dictionary, reversed_dictionary,code_data

c_i=[]
code_data1=[]
def read_data(filename):
   
    """Extract the first file enclosed in a zip file as a list of words."""
    f = zipfile.ZipFile(filename)
    total_data1=[]
   
    contents4=""
    count = 0
    c_i=[]
    code_data1=[]
    for (i,item) in enumerate(f.namelist()):
        
        data = f.namelist()[i]
        soup = BeautifulSoup(f.read(data))
        
        contents1 = soup.findAll("div",{"block"})
        
        contents3=""
        for con in contents1:
            #print(con.text)
            
            contents2 = con.find("code")
            re.sub(' +', ' ',con.text)
            re.sub('\n', ' ',con.text)
            #print(con.text)
            if contents2 != None:
                #print(contents2.text)
                cont = con.text.strip("\n\t")+contents2.text.strip("\n\t")
                contents3 = contents3+cont.strip("\n\t") #contains the name of the file
                contents4 =contents2.text.strip("\n\t")
            else:
                cont=con.text.strip("\n\t")
                contents3 = contents3+cont.strip("\n\t")              
        
        
        word_tokens =word_tokenize(contents3)
        stop_words = set(stopwords.words('english')) 
        final = [w for w in word_tokens if not w in stop_words]
        exclude = set(string.punctuation)
        final = [w for w in final if not w in exclude]   
       
        
            
        #ind=  final.index(contents4)
        #c_i = c_i + [ind]
        
        total_data1.append(final)
        if contents4 != "x":
            code_data1 = code_data1 +  [contents4]
        print(code_data1)
        
        
            #final= [ e.translate(str.maketrans(" ",str(string.punctuation)))for e in filtered_sentence]	
    
           
    

  
    return total_data1,code_data1#,c_i


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=0)
    y_pred = K.l2_normalize(y_pred, axis=0)
    return K.mean(K.sum((y_true * y_pred), axis=0))

#    print(k,track[k])
#    print(len(param))
#   # print(param[:track[k]])
#    print(type(param[:v]))
#    
#    final1.append(param[:track[k]])
#    del param[:track[k]]
#    print(len(param))
#    break


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()
if __name__ == '__main__':
    main()