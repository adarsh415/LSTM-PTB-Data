import os
import tensorflow as tf
from tensorflow import contrib
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


def batch_producer(raw_data,batch_size,num_steps):

    raw_data=tf.convert_to_tensor(raw_data,dtype=tf.int32,name='raw_data')

    data_len=tf.size(raw_data)
    batch_len=data_len//batch_size
    data= tf.reshape(raw_data[0:batch_size*batch_len],[batch_size,batch_len])

    epoch_size=(batch_len-1)//num_steps

    i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
    x=data[:,i*num_steps:(i+1)*num_steps]
    x.set_shape([batch_size,num_steps])
    y=data[:,i*num_steps+1:(i+1)*num_steps+1]
    y.set_shape([batch_size,num_steps])

    return x,y



class Input(object):
    def __init__(self,data,num_steps,batch_size):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps
        self.input_data,self.targets=batch_producer(data,batch_size,num_steps)


class Model(object):
    def __init__(self,input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5,init_scale=0.05):
        self.is_training=is_training
        self.input_obj=input
        self.hidden_size=hidden_size
        self.batch_size=input.batch_size
        self.num_steps=input.num_steps

        #create word-embedding
        with tf.device('/cpu:0'):
            embedding=tf.Variable(tf.random_uniform([vocab_size,self.hidden_size],-init_scale,init_scale))
            inputs=tf.nn.embedding_lookup(embedding,self.input_obj.input_data)

        if self.is_training and dropout < 1:
            inputs=tf.nn.dropout(inputs,dropout)

        # setup the state storage/extraction
        self.init_state=tf.placeholder(tf.float32,[num_layers,2,self.batch_size,self.hidden_size])

        state_per_layer_list=tf.unstack(self.init_state,axis=0)
        rnn_state_tuple=tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0,state_per_layer_list[idx][1]]) for idx in range(num_layers)])


        #create LSTM cell to unroll over number time input
        cell=tf.contrib.rnn.LSTMCell(hidden_size,forget_bias=1.0)

        #Add dropout wrapper during training
        if is_training and dropout > 1:
            cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_pron=dropout)
        if num_layers > 1:
            cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)],state_is_tuple=True)

        output, self.state=tf.nn.dynamic_rnn(cell, inputs,dtype=tf.float32,initial_state=rnn_state_tuple)

        # reshape to (batch_size * num_steps, hidden_size)
        output=tf.reshape(output,[-1,hidden_size])

        softmax_w=tf.Variable(tf.random_uniform([hidden_size,vocab_size],-init_scale,init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits=tf.nn.xw_plus_b(output,softmax_w,softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits=tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss=tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size,self.num_steps]),
            average_across_timesteps=False,
            average_across_batch=True
        )

        #Update the cost
        self.cost=tf.reduce_sum(loss)

        #get the prediction accuracy
        self.softmax_out=tf.nn.softmax(tf.reshape(logits,[-1,vocab_size]))
        self.predict=tf.cast(tf.argmax(self.softmax_out,axis=1),tf.int32)
        correct_prediction=tf.equal(self.predict,tf.reshape(self.input_obj.targes,[-1]))
        self.accoracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


        if not is_training:
            return











