import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from keras.models import Model
from keras.layers import Dense,TimeDistributed,RepeatVector,Input, LSTM, GRU, Lambda, Masking,Reshape, Activation,Conv2D
from keras.layers.merge import Concatenate, Dot, Add
from keras import initializers,regularizers,constraints
from keras.engine.topology import Layer
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf

class MaskedSoftmax(Layer):
    # This layers ensures zero probability for zero padded option vectors
    def build(self, input_shape):
        assert len(input_shape[0]) == 2
        assert len(input_shape[1]) == 4
    def call(self,inputs):
        arr1 = inputs[0]
        arr2 = inputs[1]
        arr3 = tf.norm(arr2, ord=2,axis=3)
        arr4 = K.sum(arr3, axis=2)
        x_mask = tf.where(tf.equal(arr4,tf.zeros_like(arr4)),tf.fill(tf.shape(arr1),np.NINF),arr1)
        sm = tf.nn.softmax(x_mask)
        return sm
    def compute_output_shape(self,input_shape):
        return input_shape[0]

def get_cnn_model():
    word_vec_size = 300
    max_q_length = 68
    max_option_length = 12       
    max_opt_count = 4
    max_sent_para = 10
    max_words_sent = 25

    get_diag = Lambda(lambda xin: K.sum(xin*tf.eye(max_opt_count),axis=2),output_shape=(max_opt_count,))
    transp_out = Lambda(lambda xin: K.permute_dimensions(xin,(0,2,1)),output_shape=(max_opt_count,word_vec_size))
    apply_weights = Lambda(lambda xin: K.sum(K.expand_dims(xin[0],axis=-1)*K.expand_dims(xin[1],axis=2),axis=1), output_shape=(word_vec_size,max_opt_count))
    tile_q = Lambda(lambda xin: K.tile(xin,(1,max_opt_count,1,1)),output_shape=(max_opt_count,max_q_length,word_vec_size))
    exp_dims = Lambda(lambda xin: K.expand_dims(xin,1), output_shape=(1,max_q_length,word_vec_size))
    exp_dims2 = Lambda(lambda xin: K.expand_dims(xin,3), output_shape=(None,word_vec_size,1))
    exp_layer = Lambda(lambda xin: K.exp(xin), output_shape=(max_sent_para,max_opt_count))
    final_weights = Lambda(lambda xin: xin/K.cast(K.sum(xin, axis=1, keepdims=True), K.floatx()),output_shape=(max_sent_para,max_opt_count))
    mask_weights = Lambda(lambda xin: tf.where(tf.equal(xin,tf.zeros_like(xin)),tf.fill(tf.shape(xin),np.NINF),xin), output_shape=(max_sent_para,max_opt_count))
    glob_pool = Lambda(lambda xin: K.mean(xin, axis=[1, 2]),output_shape=(100,))

    filter_sizes = [2,3,4]
    num_filters = 100
    q_input = Input(shape=(max_q_length, word_vec_size), name='question_input')
    q_exp = exp_dims(q_input)
    q_rep = tile_q(q_exp)
    option_input = Input(shape=(max_opt_count, max_option_length,word_vec_size), name='option_input')
    opt_q = Concatenate(axis=2)([q_rep,option_input])

    cnn_input = Input(shape=(None, word_vec_size), name='cnn_input')
    cnn_reshape = exp_dims2(cnn_input)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)

    meanpool_0 = glob_pool(conv_0)
    meanpool_1 = glob_pool(conv_1)
    meanpool_2 = glob_pool(conv_2)
    concatenated_tensor = Concatenate(axis=1)([meanpool_0, meanpool_1, meanpool_2])

    cnn_model = Model(inputs=cnn_input,outputs=concatenated_tensor)
    cnn_td_opt = TimeDistributed(cnn_model)(opt_q)
    
    doc_input = Input(shape=(max_sent_para, max_words_sent, word_vec_size), name='doc_input')
    cnn_doc = TimeDistributed(cnn_model)(doc_input)
    att_wts = Dot(axes=2,normalize=True)([cnn_doc,cnn_td_opt])
    att_wts = mask_weights(att_wts)
    att_wts = exp_layer(att_wts)
    att_wts = final_weights(att_wts)
    out = apply_weights([cnn_doc,att_wts])

    out = transp_out(out)
    dp = Dot(axes=2,normalize=True)([out,cnn_td_opt])
    out = get_diag(dp)
    probs = MaskedSoftmax()([out,option_input])
    main_model = Model(inputs=[q_input,doc_input,option_input],outputs=probs)
    sgd = SGD(lr=0.1, decay=0., momentum=0., nesterov=False)
    main_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return main_model