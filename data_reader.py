import os
import re
import numpy as np
parent_path = '20news-bydate.tar'
vocab_path = 'vocab.txt'

MAX_DOC_LENGTH = 500 # kích thước tối đa cho 1 document
NUM_CLASSES = 20 # số lượng các label 
def build_vocab(parent_path):
    train_path = 0
    test_path = 0
    for path in os.listdir(parent_path):
        if(not os.path.isfile(parent_path+'/' + path) ):
            
            if('train' in path):
                train_path = path
            else:
                test_path = path
    train_sub_folders = []
    word_count = dict()
    
    for train_sub_folder in os.listdir(parent_path+'/'+train_path):
        if not os.path.isfile(parent_path + '/' + train_path+'/' + train_sub_folder):
            
            train_sub_folder =  train_sub_folder
            train_sub_folders.append(train_sub_folder)
    parent_path = parent_path + '/' + train_path+'/'
    count = 0
    for id_folder, train_sub_folder in enumerate(train_sub_folders):
        train_sub_folder = parent_path + train_sub_folder
        data_reader(train_sub_folder, word_count)
        if(count > 2):
            break
    words = []
    for word in word_count:
        if(word_count[word] > 10):
            words.append(word)
    data = '\n'.join(words)
    with open('vocab.txt','w') as f:
        f.write(data)       # xây bô từ điển từ tập train
def data_reader(folder, word_count):
    list_file = os.listdir(folder)
    list_file.sort()
    total_data = []
    for file_path in list_file:
        with open(folder +'/' + file_path) as file:
            text = file.read().lower()
            words = re.split('\W+',text)
            for word in words:
                if(word_count.get(word,None) == None):
                    word_count[word] = 1
                else:
                    word_count[word] +=1    #đọc các word trong các document, đếm số lần xuất hiện. loại bỏ nếu số lần xuất hiện < 10
def encoder(parent_path,vocab_path):
    global MAX_DOC_LENGTH
    train_path = 0
    test_path = 0
    for path in os.listdir(parent_path):
        if(not os.path.isfile(parent_path+'/' + path) ):
            
            if('train' in path):
                train_path = path
            else:
                test_path = path
    train_sub_folders = []
    words = dict()
    with open(vocab_path,'r') as f:
        for id,word in enumerate( f.read().splitlines() ):
            words[word] = id
    
    for train_sub_folder in os.listdir(parent_path+'/'+train_path):
        if not os.path.isfile(parent_path + '/' + train_path+'/' + train_sub_folder):
            
            train_sub_folder =  train_sub_folder
            train_sub_folders.append(train_sub_folder)
    parent_path = parent_path + '/' + train_path+'/'
    count = 0
    total_data= []
    for id_folder, train_sub_folder in enumerate(train_sub_folders):
        train_sub_folder = parent_path + train_sub_folder
        for file in os.listdir(train_sub_folder):
            sentence_length = 0
            words_in_doc= []
            with open(train_sub_folder + '/'+ file,'r') as f:
                text = f.read().split()[:MAX_DOC_LENGTH]
                
                for word in text:
                    if(words.get(word,None) != None):
                        words_in_doc.append(str(words[word] +2 ))
                sentence_length = len(words_in_doc)
            
            if(sentence_length < MAX_DOC_LENGTH):
                for _ in range(MAX_DOC_LENGTH - sentence_length):
                    words_in_doc.append(str(2))
            data = ' '.join(words_in_doc)
          
            data = str(id_folder) + '<fff>' + file +'<fff>' + str(sentence_length)  + '<fff>'+ data 
            total_data.append(data)
        break
    total_data = '\n'.join(total_data)
    with open('train_data.txt','w') as f:
        f.write(total_data)    # mã hóa document dựa vào bộ từ điển đã xây dựng 


class Data_reader:
    def __init__(self, vocab_size, batch_size, file_path):
        self.start = 0
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data = np.array([[0 for i in range(MAX_DOC_LENGTH)]])
        self.label = np.array([0])
        self.setence_length = np.array([0])
        with open(file = file_path) as f:
            lines = f.read().splitlines()
            
            for line in lines:
                data = []
                feature = line.split('<fff>')
                length = int(feature[2])
                label = int(feature[0])
                for token in feature[3].split():
                       data.append(int(token))
                data = np.array([data])
                self.data = np.concatenate((self.data, data))
                self.setence_length = np.concatenate((self.setence_length, np.array([length])))
                self.label = np.concatenate((self.label, np.array([label])))
        self.data= self.data[1:]
        self.label = self.label[1:]
        self.setence_length = self.setence_length[1:]
      #  print(self.setence_length.shape)
    def next_batch(self):
        start = self.start
        
        if(start + self.batch_size > self.data.shape[0]):
            self.start = 0
            start = 0
            indice = np.array(range(self.data.shape[0]))
            self.data= self.data[indice]
            self.label = self.label[indice]
            self.setence_length = self.setence_length[indice]
            batch_data = self.data[start:start+ self.batch_size]
            batch_label = self.label[start:start+self.batch_size]
            batch_length = self.setence_length[start:start+self.batch_size]
        else:
            batch_data = self.data[start:start+ self.batch_size]
            batch_label = self.label[start:start+self.batch_size]
            batch_length = self.setence_length[start:start+self.batch_size]
            self.start = start+ self.batch_size
       
        return batch_data,batch_label, batch_length

encoder(parent_path, vocab_path)