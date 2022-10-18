import tensorflow as tf
from layer import Dense,Dense_sd,Dense_sv

class NPLAM(tf.keras.models.Model):
    def __init__(self, x_d, nn_node, lambda_1, lambda_2, lambda_3):
        '''
        :param x_d: the number of input features
        :param nn_node: the number of hidden layer' nodes
        :param lambda_1: the regularity coefficient of g1
        :param lambda_2: the regularity coefficient of g2
        :param lambda_3: the regularity coefficient of a
        '''
        self.x_d = x_d
        self.nn_node = nn_node
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        super(NPLAM, self).__init__()

    def build(self, input_shape):
        # the layer of feature selection
        self.layer_vs = Dense_sv(self.x_d, activation=None, use_bias=False, kernel_initializer="one",w_train=True,b_train=False)

        # the layer of structure discovery, no activate function, don't have b
        self.layer_sd = [None] * self.x_d
        # the result of nonlinear model and linear model
        self.up_output_train = [None] * self.x_d
        self.down_output_train = [None] * self.x_d
        # nonlinear model
        self.up_layer_1 = [None] * self.x_d
        self.up_layer_2 = [None] * self.x_d
        self.up_output_1 = [None] * self.x_d
        self.up_output_2 = [None] * self.x_d
        # linear model
        self.down_layer_1 = [None] * self.x_d
        self.down_output_1 = [None] * self.x_d
        # add the nonlinear and linear
        self.concat_train = [None] * self.x_d

        self.final_single_layer = [None] * self.x_d
        self.final_single_output = [None] * self.x_d

        for i in range(self.x_d):
            # linear
            self.down_layer_1[i] = Dense(1, activation=None, kernel_initializer='one',
                                         w_train=True, use_bias=False, b_train=False)

            self.final_single_layer[i] = Dense(1, activation=None, kernel_initializer='ones',
                                               w_train=False, use_bias=False, b_train=False)

            # feature selection gates
            self.layer_sd[i] = Dense_sd(1, activation=None, use_bias=False,
                                                         kernel_initializer=tf.random_uniform_initializer(maxval=0.71,minval=0.69),
                                                         w_train=True, b_train=False)
            # nonlinear
            self.up_layer_1[i] = Dense(self.nn_node,
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=tf.random_normal_initializer(0, 1.0),
                                       bias_initializer=tf.random_normal_initializer(0.0, 1.0),
                                       # kernel_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                       # bias_initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                       w_train=False,
                                       use_bias=True,
                                       b_train=False)
            self.up_layer_2[i] = Dense(1,
                                       activation=None,
                                       kernel_initializer="zero",
                                       w_train=True,
                                       use_bias=False,
                                       b_train=False)
        # the last layer
        self.final_multi_layer = Dense(1, activation=None, kernel_initializer="ones", w_train=False, use_bias=True, b_train=True)
        super(NPLAM, self).build(input_shape)

    def call(self, inputs):
        self.up_output_train_vs = self.layer_vs(inputs)

        for i in range(self.x_d):
            #up
            self.up_output_train[i], self.down_output_train[i] = self.layer_sd[i](tf.transpose(self.up_output_train_vs[i]))
            self.up_output_1[i] = self.up_layer_1[i](self.up_output_train[i])
            self.up_output_2[i] = self.up_layer_2[i](self.up_output_1[i])
            #down
            self.down_output_1[i] = self.down_layer_1[i](self.down_output_train[i])
            #add
            self.concat_train[i] = tf.concat([self.up_output_2[i], self.down_output_1[i]], axis=-1)
            self.final_single_output[i] = self.final_single_layer[i](self.concat_train[i])
        self.concat_all = tf.concat([self.final_single_output[i] for i in range(self.x_d)], axis=-1)
        self.final_multi_output = self.final_multi_layer(self.concat_all)
        return self.final_multi_output

    def model_loss_mse(self, y_true, y_pred):
        self.regression_loss_MSE = tf.keras.losses.mean_squared_error(y_true, y_pred)

        self.regression_loss_g_a_L1 = 0
        for i in range(self.x_d):
            self.regression_loss_g_a_L1 +=  tf.reduce_mean(tf.abs(self.up_layer_2[i].kernel))
        self.regression_loss_g_a_L1 /= self.x_d

        self.regression_loss_g1 = 0
        self.regression_loss_g2 = 0
        for i in range(self.x_d):
            self.regression_loss_g1 += tf.reduce_mean(tf.abs(self.layer_vs.kernel[0][i]))
            self.regression_loss_g2 += tf.reduce_mean(tf.abs(self.layer_sd[i].kernel_up))
        self.regression_loss_g1 /= self.x_d
        self.regression_loss_g2 /= self.x_d
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        self.regression_loss = self.regression_loss_MSE + lambda_1 * self.regression_loss_g1 + lambda_2 * self.regression_loss_g2+ lambda_3 * self.regression_loss_g_a_L1

        return self.regression_loss

    def get_g1(self):
        # feature selection gates
        return self.layer_vs.kernel.numpy()[0].tolist()
    def get_g2(self):
        # structure discovery gates
        temp_t = []
        for show_i in range(self.x_d):
            temp_t.append(self.layer_sd[show_i].kernel_up.numpy()[0][0])
        return temp_t
    def get_g(self):
        g1 = self.layer_vs.kernel.numpy()[0].tolist()
        g2 = []
        w = []
        a = []
        for show_i in range(self.x_d):
            g2.append(self.layer_sd[show_i].kernel_up.numpy()[0][0])
            w.append(tf.reduce_mean(tf.abs(tf.multiply(tf.multiply(self.down_layer_1[show_i].kernel, (1 - self.layer_sd[show_i].kernel_up)), self.layer_vs.kernel[0][show_i]))).numpy())
            a.append(tf.reduce_mean(tf.abs(self.up_layer_2[show_i].kernel)).numpy())
        return g1, g2, w, a
