import json
import numpy as np
import random
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel




# some constant parameters
seq_len=100
batch_size=100
split=0.9


# label class
classes=[]

# data tokenization
def tokenization(data_set):
    xids = np.zeros((num_samples,seq_len))
    xmask = np.zeros((num_samples,seq_len))
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    for i,data in enumerate(data_set):
        tokens=tokenizer.encode_plus(data[0],max_length=seq_len,truncation=True,padding='max_length',add_special_tokens=True,return_tensors='tf')
        xids[i,:]=tokens['input_ids']
        xmask[i,:]=tokens['attention_mask']
    return xids, xmask

def findLabels(data_set):
    for i,j in data_set:
        if j not in classes:
            classes.append(j)
            
    labels = np.zeros((num_samples,len(classes)))
    for i,data in enumerate(data_set):
        labels[i,classes.index(data[1])]=1
    return labels

def mapFunction(xid,xmask,label):
    return {'input_ids':xid,'attention_mask':xmask},label

def preprocessing(xids,xmask,labels):
    dataset= tf.data.Dataset.from_tensor_slices((xids,xmask,labels))
    dataset=dataset.map(mapFunction)
    dataset =dataset.shuffle(10000).batch(batch_size,drop_remainder=True)
    size = int((num_samples/batch_size)*split)
    train=dataset.take(size)
    val_data=dataset.skip(size)
    return train,val_data

def createModel():
    bert = TFBertModel.from_pretrained('bert-base-cased')
    input_ids= tf.keras.layers.Input(shape=(seq_len,),name='input_ids',dtype='int32')
    mask=tf.keras.layers.Input(shape=(seq_len,),name='attention_mask',dtype='int32')
    embedding= bert.bert(input_ids,attention_mask=mask)[1]
    x=tf.keras.layers.Dense(1024,activation='relu')(embedding)
    y=tf.keras.layers.Dense(len(classes),activation='softmax',name='output')(x)
    model=tf.keras.Model(inputs=[input_ids,mask],outputs=y)

    optimizer = tf.keras.optimizers.Adam(lr=1e-5, decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    return model


# data loading part
data_file = open('is_train.json').read()
data_set = json.loads(data_file)
random.shuffle(data_set)
num_samples=len(data_set)

xids,xmask = tokenization(data_set)
labels=findLabels(data_set)
train,validation=preprocessing(xids,xmask,labels)
model = createModel()

history = model.fit(train,epochs=30,validation_data=validation)
model.save('bert_model.h5')