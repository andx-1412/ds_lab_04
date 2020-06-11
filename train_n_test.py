import data_reader
import model
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
MAX_STEP = 500 # max train epoch
NUM_CLASSES = 20 # số lượng các label 
def train_and_evaluate_rnn(): # train model và đánh giá accurate trên tập test mỗi epoch
    with open('vocab.txt','r') as f:
        VOCAB_SIZE = len(f.read().splitlines() )
    rnn = model.Rnn(VOCAB_SIZE,embedding_size = 60, lstm_size = 50, batch_size= 50) # khởi tạo và load LSTM
    loss, pred = rnn.build_graph()
    train_op = rnn.trainer(loss, lr = 0.01)
    with tf.Session() as sess:
        train_data = data_reader.Data_reader(vocab_size = VOCAB_SIZE, file_path= 'train_data.txt', batch_size= 50)
        test_data = data_reader.Data_reader(VOCAB_SIZE,batch_size = 50,file_path = 'test_data.txt')
       
        step = 0
        sess.run(tf.global_variables_initializer())
       
        while step < MAX_STEP:
            batch_data, batch_label, batch_length = train_data.next_batch()
            predict_label, batch_loss,_ = sess.run([pred, loss,train_op], feed_dict= {rnn.data : batch_data, rnn.label : batch_label,rnn.sentence_lengths : batch_length }) # train LSTM
            step +=1
            if(train_data.start ==0): # kết thúc mỗi epoch print ra loss và đánh giá trên tập test
                print('loss', batch_loss)
                true_pred= 0 # biến đếm số lần predict đúng của LSTM trên tập test
                while True:
                    test_batch_data, test_batch_label, test_batch_length = test_data.next_batch()
                    predicted_label  = sess.run([pred], feed_dict= {rnn.data : test_batch_data, rnn.label : test_batch_label,rnn.sentence_lengths : test_batch_length })
                    match = np.equal(predicted_label,test_batch_label)
                    for i in match:
                        count = 0
                        for j in i:
                            if(j== True):
                                count +=1
                        if(count == NUM_CLASSES ):
                            true_pred+=1
                    if(test_data.start == 0):
                        break
                print('accurate on test data:', true_pred/(test_data.data.shape[0]))
train_and_evaluate_rnn()