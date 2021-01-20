import os
import time
import random
from load_sequence import Data_Factory
from tensorflow.python import debug as tf_debug
import tensorflow as tf
from collections import defaultdict
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability ")
tf.flags.DEFINE_string("object_path", "new/sjtu_data", "Dataset path")
tf.flags.DEFINE_string("train_path", "train.txt", "Train data path")
tf.flags.DEFINE_string(
    "validation_path",
    "validation.txt",
    "Validation data path")
tf.flags.DEFINE_string("test_path", "test.txt", "Test data path")
tf.flags.DEFINE_boolean(
    "allow_soft_placement",
    True,
    "Allow device soft device placement")
tf.flags.DEFINE_boolean(
    "log_device_placement",
    False,
    "Log placement of ops on devices")

# Training parameters
tf.flags.DEFINE_string(
    'loss_type',
    'square_loss',
    'Specify a loss type (square_loss or log_loss).')
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size ")
tf.flags.DEFINE_list('net_channel',[32,32,32,32,32,32],'net_channel, should be 6 layers here')
tf.flags.DEFINE_integer("num_epochs", 800, "Number of training epochs ")
tf.flags.DEFINE_integer(
    'pretrain', -1, 'flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
tf.flags.DEFINE_string(
    'optimizer',
    'AdagradOptimizer',
    'Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
tf.flags.DEFINE_float('lamda', 0, 'Regularizer for bilinear part.')
tf.flags.DEFINE_float('lr', 0.1, 'Learning rate.')
tf.flags.DEFINE_integer(
    'verbose',
    1,
    'Show the results per X epochs (0, 1 ... any positive integer)')
tf.flags.DEFINE_integer('hidden_factor', 64, 'Number of hidden factors.')
tf.flags.DEFINE_integer(
    'batch_norm',
    0,
    'Whether to perform batch normaization (0 or 1)')


class DeepModel:
    def __init__(
            self,
            user_field_M,
            item_field_M,
            pretrain_flag,
            save_file,
            hidden_factor,
            epoch,
            batch_size,
            learning_rate,
            lamda_bilinear,
            keep,
            optimizer_type,
            batch_norm,
            verbose,
            random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.nc = FLAGS.net_channel
        # performance of each epoch
        self.rec = 0

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        tf.set_random_seed(self.random_seed)
        # input data
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.user_features = tf.placeholder(tf.int32, shape=[None, None])
        self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
        self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
        self.clc_num = tf.placeholder(tf.float32, shape=[None, self.item_field_M-df.item_bind_M])
        self.positive_word = tf.placeholder(tf.int32, shape=[None, None])
        self.negative_word = tf.placeholder(tf.int32, shape=[None, None])
        self.clc_index = tf.placeholder(tf.int32, shape=[None,self.item_field_M-df.item_bind_M])
        self.train_phase = tf.placeholder(tf.bool)

        iszs = [3] + self.nc[:-1]
        oszs = self.nc
        self.P = []
        for i in range(5):
            self.P.append(self._conv_weight(iszs[i], oszs[i]))  # first 5 layers
        self.P.append(self._conv_weight(iszs[5], oszs[5]))
        self.W = self.weight_variable([self.nc[-1], 1])  # last layer
        self.b = self.weight_variable([1])

        # Variables.
        self.weights = self._initialize_weights()
        # _________ sum_square part for positive (u,i)_____________
        self.user_feature_embeddings = tf.nn.embedding_lookup(
            self.weights['user_feature_embeddings'], self.user_features)
        self.positive_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_feature_embeddings'], self.positive_features)
        self.negative_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_feature_embeddings'], self.negative_features)

        self.clc_list = tf.convert_to_tensor(
            df.school_list, dtype=tf.int32)  # [1*570]

        self.positive_item_school = tf.gather(
            self.clc_list, tf.reshape(tf.squeeze(self.user_features[:, 1]), [-1]))
        self.positive_item_school_emb = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.positive_item_school)  # [None,256]

        self.words = tf.Variable(
            tf.random_normal([df.wordn + 1, self.hidden_factor], 0.0, 0.1),
            name='word_embeddings')
        self.p_word2vec = tf.reduce_max(tf.nn.embedding_lookup(self.words, self.positive_word), axis=1, keep_dims=True)
        self.n_word2vec = tf.reduce_max(tf.nn.embedding_lookup(self.words, self.negative_word), axis=1, keep_dims=True)

        self.clc_index_emb = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.clc_index)
        self.item_per = self.attention(self.positive_item_school_emb,self.clc_index_emb,self.clc_num)
        self.positive_item = tf.concat(
            [self.user_feature_embeddings,self.positive_item_embeddings, self.p_word2vec, self.item_per], axis=1)

        self.negative_item = tf.concat(
            [self.user_feature_embeddings,self.negative_item_embeddings, self.n_word2vec, self.item_per], axis=1)
        self.user_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['user_feature_bias'],
                self.user_features),
            1)  # None * 1

        #positive conver
        self.item_concat_p = tf.concat([self.user_feature_embeddings[:,1:,:], self.p_word2vec],1)
        split0,split1,split2 = tf.split(self.item_concat_p,num_or_size_splits=3,axis=1)
        for i in range(3):
            for j in range(i+1,3):
                cube_i = eval('split'+str(i))
                cube_j = eval('split'+str(j))
                relation = tf.matmul(tf.transpose(cube_i,perm=[0,2,1]),cube_j)
                net_input = tf.expand_dims(relation,-1)
                if i==0 and j==1:
                    self.cube = net_input
                else:
                    self.cube = tf.concat([self.cube,net_input],-1)
        self.positive_cube = self.cube

        self.layer = []
        positive_input = self.positive_cube
        if self.batch_norm:
            positive_input = self.batch_norm_layer(positive_input, train_phase=self.train_phase, scope_bn='bn_cn_po')
        for p in self.P:
            self.layer.append(self._conv_layer(positive_input, p))
            positive_input = self.layer[-1]
        #positive prediction
        self.dropout = tf.nn.dropout(self.layer[-1], self.keep)
        self.pos_out = tf.matmul(tf.reshape(self.dropout, [-1, self.nc[-1]]), self.W) + self.b
        #negative conver
        self.item_concat_n = tf.concat([self.user_feature_embeddings[:,1:,:], self.n_word2vec],1)
        split0, split1, split2 = tf.split(self.item_concat_n, num_or_size_splits=3, axis=1)
        for i in range(3):
            for j in range(i + 1, 3):
                cube_i = eval('split' + str(i))
                cube_j = eval('split' + str(j))
                relation = tf.matmul(tf.transpose(cube_i, perm=[0, 2, 1]), cube_j)
                net_input = tf.expand_dims(relation, -1)
                if i == 0 and j == 1:
                    self.cube = net_input
                else:
                    self.cube = tf.concat([self.cube, net_input], -1)
        self.negative_cube = self.cube
        self.layer = []
        negative_input = self.negative_cube
        if self.batch_norm:
            negative_input = self.batch_norm_layer(negative_input, train_phase=self.train_phase, scope_bn='bn_cn_ne')
        for p in self.P:
            self.layer.append(self._conv_layer(negative_input, p))
            negative_input = self.layer[-1]
        #negative prediction
        self.dropout = tf.nn.dropout(self.layer[-1], self.keep)
        self.neg_out = tf.matmul(tf.reshape(self.dropout, [-1, self.nc[-1]]), self.W) + self.b

        pos_element_wise_product_list = []
        pos_count = 0
        for i in range(0, 7):
            for j in range(i + 1, 7):
                pos_element_wise_product_list.append(
                    tf.multiply(self.positive_item[:, i, :], self.positive_item[:, j, :]))
                pos_count += 1
        self.pos_element_wise_product = tf.stack(pos_element_wise_product_list)  # (M'*(M'-1)) * None * K
        self.pos_element_wise_product = tf.transpose(self.pos_element_wise_product, perm=[1, 0, 2],
                                                 name="pos_element_wise_product")  # None * (M'*(M'-1)) * K

        self.FM = tf.reduce_sum(self.pos_element_wise_product, 1, name="afm")  # None * K
        if self.batch_norm:
            self.FM = self.batch_norm_layer(
                self.FM, train_phase=self.train_phase, scope_bn='bn_fm_po')
        # dropout at the FM layer
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)
        Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
        self.positive_item_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['item_feature_bias'],
                self.positive_features),
            1)  # None * 1
        self.positive = 0.5*self.pos_out + 0.5*tf.add_n(
            [Bilinear, self.user_bias, self.positive_item_bias])  # None * 1

        neg_element_wise_product_list = []
        neg_count = 0
        for i in range(0, 7):
            for j in range(i + 1, 7):
                neg_element_wise_product_list.append(
                    tf.multiply(self.negative_item[:, i, :], self.negative_item[:, j, :]))
                neg_count += 1
        self.neg_element_wise_product = tf.stack(neg_element_wise_product_list)  # (M'*(M'-1)) * None * K
        self.neg_element_wise_product = tf.transpose(self.neg_element_wise_product, perm=[1, 0, 2],
                                                 name="neg_element_wise_product")  # None * (M'*(M'-1)) * K

        self.FM_negative = tf.reduce_sum(self.neg_element_wise_product, 1, name="nfm")  # None * K
        if self.batch_norm:
            self.FM_negative = self.batch_norm_layer(
                self.FM_negative, train_phase=self.train_phase, scope_bn='bn_fm_ne')
        self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep_prob)  # dropout at the FM layer
        # _________out _________
        Bilinear_negative = tf.reduce_sum(
            self.FM_negative, 1, keep_dims=True)  # None * 1
        self.negative_item_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['item_feature_bias'],
                self.negative_features),
            1)  # None * 1

        self.negative = 0.5*self.neg_out + 0.5*tf.add_n(
            [Bilinear_negative, self.user_bias, self.negative_item_bias])  # None * 1

        # Compute the loss.
        self.loss = -tf.log(tf.sigmoid(self.positive - self.negative))
        # self.loss = tf.log(1 + tf.exp(-tf.sigmoid(self.positive - self.negative)))
        self.loss = tf.reduce_sum(self.loss)
        # self._loss = tf.add(self.loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
        #                     name='objective')
        # Optimizer.
        if self.optimizer_type == 'AdamOptimizer':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'AdagradOptimizer':
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
                initial_accumulator_value=0.1).minimize(self.loss)
            # trainable_params = tf.trainable_variables()
            # gradients = tf.gradients(self.loss, trainable_params)
            # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            # self.train_op = self.optimizer.apply_gradients(
            #     zip(clip_gradients, trainable_params))
        elif self.optimizer_type == 'GradientDescentOptimizer':
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == 'MomentumOptimizer':
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=0.95).minimize(
                self.loss)
        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init)
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def _conv_layer(self, input, P):
        conv = tf.nn.conv2d(input, P[0], strides=[1, 2, 2, 1], padding='SAME')  # strides = [batch= 1, height, width, channels=1] ,same 补0
        return tf.nn.relu(conv + P[1])

    def _conv_weight(self, isz, osz):
        return (self.weight_variable([2,2,isz,osz]), self.bias_variable([osz]))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name(
                'user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name(
                'item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name(
                'user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name(
                'item_feature_bias:0')
            # bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(
                ue, trainable=False, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(
                ie, trainable=False, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(
                ub, trainable=False, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(
                ib, trainable=False, dtype=tf.float32)
            print("load!")
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(tf.random_uniform(
                [self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(tf.random_uniform(
                [self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(
            x,
            decay=0.9,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=True,
            reuse=None,
            trainable=True,
            scope=scope_bn)
        bn_inference = batch_norm(
            x,
            decay=0.9,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=False,
            reuse=True,
            trainable=True,
            scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def attention(self,queries,keys,values):
        '''
          queries:     [B, H]
          keys:        [B, T, H]
          values: [B,T]
        '''
        queries_hidden_units = queries.get_shape().as_list()[-1]  #128
        queries = tf.expand_dims(queries, 1)
        queries = tf.tile(queries, [1, tf.shape(keys)[1],1])  #[N, 211*128]
        # values = tf.expand_dims(values,-1)
        # values = tf.tile(values,[1,1,tf.shape(keys)[2]])
        # queries = tf.reshape(
        #     queries, [-1, tf.shape(keys)[1], queries_hidden_units])
        # self.values = tf.reshape(
        #     values, [-1, tf.shape(keys)[1], queries_hidden_units])  #values: [B, T, H]
        din_all = tf.concat(
            [queries, keys], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all,128,activation=tf.nn.tanh,name='f1_att',reuse=tf.AUTO_REUSE)
        #d_layer_2_all = tf.layers.dense(d_layer_1_all, 64, activation=tf.nn.tanh, name='f2_att', reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.layers.dense(d_layer_1_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # [B, 1, T]
        outputs = d_layer_3_all
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        self.outputs = tf.multiply(tf.nn.softmax(outputs),tf.expand_dims(values,1))  # [B, 1, H]
        self.results = tf.matmul(self.outputs, keys)
        return self.results

    def partial_fit(self, data):  # fit a batch
        positive_feature = []
        positive_word = []
        negative_feature = []
        negative_word = []
        for X_po in data['X_positive']:
            feature = []
            feature.append(X_po[0])
            feature.append(X_po[1])
            words = X_po[2].strip().split(" ")
            positive_feature.append(feature)
            word = [0]*df.max_l
            for i in range(len(words)):
                word[i] = int(words[i])+1
            positive_word.append(word)

        for X_ne in data['X_negative']:
            feature = []
            feature.append(X_ne[0])
            feature.append(X_ne[1])
            words = X_ne[2].strip().split()
            negative_feature.append(feature)
            word = [0]*df.max_l
            for i in range(len(words)):
                word[i] = int(words[i])+1
            negative_word.append(word)

        feed_dict = {
            self.user_features: data['X_user'],
            self.positive_features: positive_feature,
            self.positive_word: positive_word,
            self.clc_num: data['X_history'],
            self.clc_index: data['X_clc_index'],
            self.negative_features: negative_feature,
            self.negative_word: negative_word,
            self.dropout_keep_prob: FLAGS.dropout_keep_prob,
            self.train_phase: True}
        loss, opt, outputs, results= self.sess.run(
            (self.loss, self.optimizer,self.outputs, self.results), feed_dict=feed_dict)
        return loss

    def train(self, Train_data, Test_data):  # fit a dataset
        # Check Init performance
        lastLoss = 100000000000
        #model.evaluate()
        for epoch in range(self.epoch):
            total_loss = 0
            t1 = time.time()
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(
                    Train_data, self.batch_size)
                # Fit training
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            t2 = time.time()
            print(
                "the total loss in %d th iteration is: %f cost [%.1f s]" %
                (epoch, total_loss, t2 - t1))
            if abs(lastLoss - total_loss) < 1:
                print("converge!")
                # break;
            lastLoss = total_loss
            if (epoch + 1) % 800 == 0:
                start_time = time.time()
                print("epoch:%f" % epoch)
                model.evaluate()
                end_time = time.time()
                print("Evaluation cost [%.1f s]" % (end_time - start_time))
        print("end train begin save")
        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    # generate a random block of training data
    def get_random_block_from_data(self, train_data, batch_size):
        X_user, X_positive, X_negative, X_history = [], [], [], []
        X_clc_index = []
        all_items = df.binded_items.values()
        # get sample
        while len(X_user) < batch_size*1:
            index = np.random.randint(0, len(train_data['X_user']))
            # uniform sampler
            user_features = ";".join(
                [str(item) for item in train_data['X_user'][index][0:]])
            user_id = df.binded_users[user_features]  # get userID
            # get positive list for the userID
            pos = df.user_positive_list[user_id]
            # uniform sample a negative itemID from negative set
            neg = np.random.randint(len(all_items))
            neg_list = []
            history = [0]*(self.item_field_M-df.item_bind_M)
            his_str = df.clc_number[user_features].strip().split()
            for i in range(len(history)):
                history[i]=float(his_str[i].split(':')[1])
            for i in range(1):
                while neg in pos or neg in neg_list:
                    neg = np.random.randint(len(all_items))
                neg_list.append(neg)
                negative_feature = df.item_map[neg].strip().split(
                    ';')  # get negative item feature
                nf = []
                nf.append(int(negative_feature[0]))
                nf.append(int(negative_feature[1]))
                nf.append(negative_feature[2])
                X_clc_index.append(df.clc_index)
                X_history.append(history)
                X_negative.append(nf)
                X_user.append(train_data['X_user'][index])
                X_positive.append(train_data['X_item'][index])
        return {
            'X_user': X_user,
            'X_positive': X_positive,
            'X_negative': X_negative,
            'X_history': X_history,
            'X_clc_index': X_clc_index
        }

    def evaluate(self):  # evaluate the results for an input set
        users_list = df.binded_users.keys()
        test_rating_map = df.user_positive_list_test
        test_predict_5 = defaultdict(set)
        test_predict_10 = defaultdict(set)
        test_predict_20 = defaultdict(set)
        count = 0
        ndcg_5 = 0.0
        ndcg_10 = 0.0
        ndcg_20 = 0.0
        One_features, One_words = [], []

        for X_it in df.All_data['X_item']:
            feature = []
            feature.append(X_it[0])
            feature.append(X_it[1])
            words = X_it[2].strip().split()
            One_features.append(feature)
            word = [0]*df.max_l
            for i in range(len(words)):
                word[i] = int(words[i])+1
            One_words.append(word)
        for user_key in users_list:
            if user_key not in df.clc_number:
                continue
            us = user_key.split(';')
            user_feature = [int(u) for u in us]
            One_users = [user_feature for i in range(df.item_bind_M)]
            One_history = []
            One_index = []
            pred_re = []
            history = [0]*(self.item_field_M-df.item_bind_M)
            his_str = df.clc_number[user_key].strip().split()
            for i in range(len(history)):
                history[i]=float(his_str[i].split(':')[1])
            for i in range(df.item_bind_M):
                One_history.append(history)
                One_index.append(df.clc_index)
            batch = FLAGS.batch_size #200
            batches = df.item_bind_M//batch #56020//200
            for i in range(batches+1):
                t = batch * (i + 1) if batch * \
                    (i + 1) < df.item_bind_M else df.item_bind_M
                feed_dict = {self.user_features: One_users[i * batch:t],
                             self.positive_features: One_features[i * batch:t],
                             self.positive_word: One_words[i * batch:t],
                             self.clc_num: One_history[i * batch:t],
                             self.clc_index: One_index[i * batch:t],
                             self.dropout_keep_prob: 1.0,
                             self.train_phase: False}
                pred_fm = self.sess.run(self.positive, feed_dict=feed_dict)
                pred_fm = np.reshape(pred_fm, -1)
                pred_re.extend(pred_fm)
            pred_re = np.reshape(pred_re, -1)
            pred_index = np.argsort(-pred_re)  # 排序，得分最高的在最前面
            rank_list_5 = []
            rank_list_10 = []
            rank_list_20 = []
            for i in range(len(pred_index)):
                if count == 5:
                    count = 0
                    break
                if df.binded_users[user_key] in df.user_positive_list:
                    if not df.user_positive_list[df.binded_users[user_key]].__contains__(
                            pred_index[i]):
                        count += 1
                        test_predict_5[df.binded_users[user_key]].add(pred_index[i])
                        rank_list_5.append(pred_index[i])
            for i in range(len(pred_index)):
                if count == 10:
                    count = 0
                    break
                if df.binded_users[user_key] in df.user_positive_list:
                    if not df.user_positive_list[df.binded_users[user_key]].__contains__(
                            pred_index[i]):
                        count += 1
                        test_predict_10[df.binded_users[user_key]].add(pred_index[i])
                        rank_list_10.append(pred_index[i])
            for i in range(len(pred_index)):
                if count == 20:
                    count = 0
                    break
                if df.binded_users[user_key] in df.user_positive_list:
                    if not df.user_positive_list[df.binded_users[user_key]].__contains__(
                            pred_index[i]):
                        count += 1
                        test_predict_20[df.binded_users[user_key]].add(pred_index[i])
                        rank_list_20.append(pred_index[i])
            ndcg_5 += getNDCG(rank_list_5, test_rating_map[df.binded_users[user_key]])
            ndcg_10 += getNDCG(rank_list_10, test_rating_map[df.binded_users[user_key]])
            ndcg_20 += getNDCG(rank_list_20, test_rating_map[df.binded_users[user_key]])
        self.rec_prec(test_predict_5, test_rating_map, 5)
        self.rec_prec(test_predict_10, test_rating_map, 10)
        self.rec_prec(test_predict_20, test_rating_map, 20)
        print('NDCG@5: ' + str(ndcg_5 * 1.0 / len(df.user_positive_list.keys())))
        print('NDCG@10: ' + str(ndcg_10 * 1.0 / len(df.user_positive_list.keys())))
        print('NDCG@20: ' + str(ndcg_20 * 1.0 / len(df.user_positive_list.keys())))

    def rec_prec(self, predict, test, K):
        test_sum, user_K = 0, 0
        hitsum = 0
        for key in df.user_positive_list:
            if key in test:
                test_sum += len(test[key])
                user_K += K
                values = sorted(list(predict[key]), reverse=True)
                values = values[:K]
                for value in test[key]:
                    if values.__contains__(value):
                        hitsum += 1
        rec = hitsum * 1.0 / test_sum
        precision = hitsum * 1.0 / user_K
        print('Rec@%d: ' % K + str(rec) + ' Pre@%d: ' % K + str(precision))
        return rec

def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if (item_id not in target_items):
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg
if __name__ == '__main__':
    # data loading
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    df = Data_Factory(0.8, FLAGS.object_path + '/')
    if FLAGS.verbose > 0:
        print(
            "AttenConvFM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d" %
            (FLAGS.object_path,
             FLAGS.hidden_factor,
             FLAGS.loss_type,
             FLAGS.num_epochs,
             FLAGS.batch_size,
             FLAGS.lr,
             FLAGS.lamda,
             FLAGS.dropout_keep_prob,
             FLAGS.optimizer,
             FLAGS.batch_norm))
    save_file = 'pretrain/%s_%d' % (FLAGS.object_path, FLAGS.hidden_factor)
    # Training
    t1 = time.time()
    model = DeepModel(
        df.user_field_M,
        df.item_field_M,
        FLAGS.pretrain,
        save_file,
        FLAGS.hidden_factor,
        FLAGS.num_epochs,
        FLAGS.batch_size,
        FLAGS.lr,
        FLAGS.lamda,
        FLAGS.dropout_keep_prob,
        FLAGS.optimizer,
        FLAGS.batch_norm,
        FLAGS.verbose)
    print("begin train")
    model.train(df.Train_data, df.Test_data)
    print("end train")
    print("finish")
