import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()

MAX_DOC_LENGTH = 500 # kích thước tối đa cho 1 document
NUM_CLASSES = 20 # số lượng các label 
class Rnn:
    def __init__(self, vocab_size ,embedding_size, lstm_size, batch_size):
        self.vocab_size= vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.data = tf.placeholder(tf.int32, shape = [self.batch_size,MAX_DOC_LENGTH])
        self.label = tf.placeholder(tf.int32, shape = [self.batch_size,])
        self.sentence_lengths = tf.placeholder(tf.int32, shape = [self.batch_size,])
      #  self.final_tokens = tf.placeholder(tf.int32, shape = [self.batch_size,])

    def build_graph(self): # graph bao gồm các khối: input, embedding word( embedding vector được học trực tiếp từ word, không sử dụng w2v pretrained), LSTM, lstm output = average output lstm tại các thời điểm
                           # và được đưa vào fully connected layer cuối, tính softmax và predict label
        label_one_hot = tf.one_hot(indices = self.label,depth = NUM_CLASSES,dtype =  tf.float32 ) # tạo one-hot label dựa trên output nhập

        embedding = self.embedding_layer(self.data)
        lstm_output = self.lstm_layer(embedding) 
        weights = tf.get_variable(name = 'final_layer_weight', shape = (self.lstm_size,NUM_CLASSES), initializer= tf.random_normal_initializer(seed = 2020))
        biases = tf.get_variable(name = 'final_layer_biases' ,shape = (NUM_CLASSES), initializer = tf.random_normal_initializer(seed = 2020))
        logits = tf.matmul(lstm_output, weights) + biases
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = label_one_hot, logits = logits) # cross entropy sau khi đã tự softmax-> không cần layer softmax
        loss=  tf.reduce_mean(loss)
        probs = tf.nn.softmax(logits)     #lớp softmax tính xác suất thuộc các lớp
        predicted_label = tf.argmax(probs,axis = 1) # axis = 1 : lấy xác xuất lớn nhất theo hàng ngang(hòng dọc: batch size)
        predicted_label = tf.squeeze(predicted_label)
        return loss, predicted_label


    def embedding_layer(self, indice) : # embedding layer của graph
        embedding_vector  = []
        embedding_vector.append(np.zeros(self.embedding_size)) # khởi tao embedding vetor cho padding word or unknow word
        for i in range(self.vocab_size +1): # khởi tạo embedding vetor cho các word còn lại
            embedding_vector.append(np.random.normal(loc = 0., scale = 1., size = self.embedding_size))
        embedding_vector = np.array(embedding_vector)
        self.embedding_matrix = tf.get_variable(name = 'embedding', shape = (self.vocab_size+2,self.embedding_size), initializer = tf.constant_initializer(embedding_vector))
        return tf.nn.embedding_lookup(params = self.embedding_matrix,ids= indice)



    def lstm_layer(self,embedding): # tạo lstm cell từ embedding vector, đóng gói vào trong 1 RNN, unroll theo size của doc, tính giá trị trung bình lstm output tại các thời điểm mỗi doc
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        zeros_state = tf.zeros(shape = (self.batch_size,self.lstm_size))
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(zeros_state, zeros_state)

        lstm_input = tf.unstack(tf.transpose(embedding, perm = [1,0,2]))
        lstm_output,last_state = tf.nn.static_rnn(cell = lstm_cell, inputs = lstm_input, initial_state = initial_state, sequence_length  = self.sentence_lengths )
        lstm_output = tf.unstack(tf.transpose(lstm_output, perm  = [1,0,2]))
        lstm_output = tf.concat(lstm_output, axis = 0)

        mask = tf.sequence_mask(lengths = self.sentence_lengths,maxlen = MAX_DOC_LENGTH, dtype = tf.float32  )
        mask = tf.concat(tf.unstack(mask, axis = 0 ), axis = 0)
        mask = tf.expand_dims(mask, -1)
        lstm_output = mask*lstm_output
        lstm_output_split = tf.split(lstm_output, num_or_size_splits = self.batch_size)
        lstm_output_sum = tf.reduce_mean(lstm_output_split,axis = 1)
        lstm_output_average = lstm_output_sum/tf.expand_dims(tf.cast(self.sentence_lengths, tf.float32), -1)
        return lstm_output_average
    def trainer(self, loss,lr): # train theo loss của graph
        return tf.train.AdamOptimizer(lr).minimize(loss)


