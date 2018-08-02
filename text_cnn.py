#coding=utf-8
import tensorflow as tf
import numpy as np
#注意如果是单向的卷积核filter_shape = [filter_size, embedding_size, 1, num_filters]
#注意如果是双向的卷积核filter_shape = [filter_size, 2*embedding_size, 1, num_filters]

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            # # GRU
            # self.cell_sentence = tf.nn.rnn_cell.GRUCell(num_units=embedding_size)
            # input_batch_size = tf.shape(self.input_x)[0]
            # self.initial_state_sentence = self.cell_sentence.zero_state(input_batch_size, dtype=tf.float32)
            # self.inpout1_gru, _states = tf.nn.dynamic_rnn(self.cell_sentence, self.embedded_chars, dtype=tf.float32,
            #                                              initial_state=self.initial_state_sentence)
            # self.embedded_chars_expanded = tf.expand_dims(self.inpout1_gru, axis=-1,
            #                                                        name="embedded_chars_inp1_expanded")  # conv2d needs 4d tensor of shape [batch, width(inp1_seq_len),

            # # 双向lstm
            # # lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(embedding_size)
            # # lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(embedding_size)
            # # lstm模型　正方向传播的RNN
            # lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            # # 反方向传播的RNN
            # lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            # (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, inputs=self.embedded_chars, dtype=tf.float32)
            # relation_seg_embedding = tf.concat([output_fw, output_bw], axis=2)
            # self.embedded_chars_expanded = tf.expand_dims(relation_seg_embedding, -1)


            # 单向lstm
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            input_batch_size = tf.shape(self.input_x)[0]
            self.initial_state_sentence = lstm_cell_fw.zero_state(input_batch_size, dtype=tf.float32)
            self.inpout1_lstm, _states = tf.nn.dynamic_rnn(lstm_cell_fw, self.embedded_chars, dtype=tf.float32,
                                                         initial_state=self.initial_state_sentence)
            self.embedded_chars_expanded = tf.expand_dims(self.inpout1_lstm, -1)

        # Create a convolution + maxpool layer for each filter size
        #filter_sizes表示不同的卷积窗口
        #tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
        # num_filters表示针对上面每个大小的卷积窗口有多少个
        #tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #2D卷积
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # filter_shape = [filter_size, 2*embedding_size, 1, num_filters]

                # [filter_height, filter_width, in_channels, out_channels]
                # 在使用中,因为一般不对Input的第一维和第四维进行卷积操作,所以strides 一般为[1,X,X,1]
                #说明https://blog.csdn.net/fontthrone/article/details/76652753

                # filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # https://blog.csdn.net/fontthrone/article/details/76652753

                # w = tf.get_variable('w', shape=filter_size,initializer=tf.contrib.layers.xavier_initializer())
                # conv = tf.nn.conv1d(self.embedded_chars_expanded, w, stride=1, padding='SAME')
                # h = tf.nn.relu(conv, name="relu")


                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_flat = tf.reshape(self.h_pool, [tf.shape(self.h_pool)[0]+0, -1])

        # Add dropout
        print("self.h_pool_flat")
        print(pooled_outputs)
        print(self.h_pool)
        print(self.h_pool_flat)
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                # shape=[self.h_pool_flat.shape.as_list()[1], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # Computes matmul(x, weights) + biases.
            self.scores_temp = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.softmax(self.scores_temp)
            # self.scores= tf.layers.dense(inputs=self.h_drop, units=2, activation=tf.nn.relu)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # https://blog.csdn.net/qq_32791307/article/details/80982897
        # Calculate mean cross-entropy loss
        #
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")