import os
import tensorflow as tf
import collections
import datetime as dt

data_path='C:\\Users\\padarsh\\PycharmProjects\\LSTM-PTB-Data\\simple-examples\\data'
vocab_size=50000


def read_words(file):

    with tf.gfile.GFile(file,'r') as f:
        return f.read().decode('utf-8').replace('\n','<eos>').split()

def build_vocab(file):
    data=read_words(file)

    dictionary=dict()
    count=[['UNK',-1]]
    count.append(collections.Counter(data).most_common(vocab_size-1))

    for word,_ in count:
        dictionary[word]=len(dictionary)

    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary,reverse_dictionary

def  file_to_word_id(file,vocab_dict):
    data =read_words(file)
    return [ vocab_dict[word] for word in data if word in vocab_dict]


def load_data():

    train_path=os.path.join(data_path,'ptb.train.txt')
    test_path=os.path.join(data_path,'ptb.test.txt')
    valid_path=os.path.join(data_path,'ptb.valid.txt')

    word_to_id,reverse_dictionary=build_vocab(train_path)
    train_data=file_to_word_id(train_path,word_to_id)
    test_data=file_to_word_id(test_path,word_to_id)
    valid_data=file_to_word_id(valid_path,word_to_id)

    vocabulary=len(word_to_id)

    return train_data,test_data,valid_data,vocabulary,reverse_dictionary



