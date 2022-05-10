import tensorflow as tf
from layer import Dense_l1_sv, Dense, Dense_l1_sd_3, Dense_split

class NPLAM(tf.keras.models.Model):
    def __init__(self, x_d, lambda_1, lambda_2,lambda_3):
        super(NPLAM, self).__init__()
        self.x_d = x_d
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        # Fratue Selection layer
        self.l1_obj_vs = Dense_l1_sv(self.x_d, activation=None, use_bias=False, kernel_initializer="one")
        # Split layer
        self.layer_up_1 = Dense_split(self.x_d,activation=None, use_bias=False,kernel_initializer='ones')

        self.l1_gate_before_model = [None] * self.x_d
        self.up_output_train = [None] * self.x_d
        self.down_output_train = [None] * self.x_d
        self.up_layer_1 = [None] * self.x_d
        self.up_layer_2 = [None] * self.x_d
        self.up_output_1 = [None] * self.x_d
        self.up_output_2 = [None] * self.x_d
        self.down_layer_1 = [None] * self.x_d
        self.down_output_1 = [None] * self.x_d
        self.concat_train = [None] * self.x_d
        self.final_single_layer = [None] * self.x_d
        self.final_single_output = [None] * self.x_d

        for i in range(self.x_d):
            # This linear layer and only weight can be trained.
            self.down_layer_1[i] = Dense(1, activation=None, kernel_initializer='one', w_train=True, use_bias=False, b_train=False)
            # This is the final layer (combining linear and nonlinear)
            self.final_single_layer[i] = Dense(1, activation=None, kernel_initializer='ones', w_train=False, use_bias=False, b_train=False)
            # Structure discovery layer
            self.l1_gate_before_model[i] = Dense_l1_sd_3(1, activation=None, use_bias=False, kernel_initializer=tf.random_uniform_initializer(maxval=0.71,minval=0.69), w_train=True, b_train=False)
            # neural network layer
            self.up_layer_1[i] = Dense(50,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(0, 1.0),
                                       bias_initializer=tf.random_normal_initializer(0.0, 1.0),
                                       w_train=False,
                                       use_bias=True,
                                       b_train=False)
            self.up_layer_2[i] = Dense(1,
                                       activation=None,
                                       kernel_initializer="zero",
                                       w_train=True,
                                       use_bias=False,
                                       b_train=False)
        self.final_multi_layer = Dense(1, activation=None, kernel_initializer="ones", w_train=False, use_bias=True, b_train=True)

    def call(self, inputs):
        self.up_output_train_vs = self.l1_obj_vs(inputs)

        for i in range(self.x_d):
            # neural network
            self.up_output_train[i], self.down_output_train[i] = self.l1_gate_before_model[i](tf.transpose(self.up_output_train_vs[i]))
            self.up_output_1[i] = self.up_layer_1[i](self.up_output_train[i])
            self.up_output_2[i] = self.up_layer_2[i](self.up_output_1[i])
            # linear
            self.down_output_1[i] = self.down_layer_1[i](self.down_output_train[i])
            # Combining the linea and neural network
            self.concat_train[i] = tf.concat([self.up_output_2[i], self.down_output_1[i]], axis=-1)
            self.final_single_output[i] = self.final_single_layer[i](self.concat_train[i])
        self.concat_all = tf.concat([self.final_single_output[i] for i in range(self.x_d)], axis=-1)
        self.final_multi_output = self.final_multi_layer(self.concat_all)
        return self.final_multi_output

    def model_loss6(self, y_true, y_pred):
        self.regression_loss_MSE = tf.keras.losses.mean_squared_error(y_true, y_pred)
        self.regression_loss_g_a_L1 = 0
        for i in range(self.x_d):
            self.regression_loss_g_a_L1 +=  tf.reduce_mean(tf.abs(self.up_layer_2[i].kernel))
        self.regression_loss_g_a_L1 /= self.x_d
        self.regression_loss_g1 = 0
        self.regression_loss_g2 = 0
        for i in range(self.x_d):
            self.regression_loss_g1 += tf.reduce_mean(tf.abs(self.l1_obj_vs.kernel[0][i]))
            self.regression_loss_g2 += tf.reduce_mean(tf.abs(self.l1_gate_before_model[i].kernel_up))
        self.regression_loss_g1 /= self.x_d
        self.regression_loss_g2 /= self.x_d
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        self.regression_loss = self.regression_loss_MSE + lambda_1 *  self.regression_loss_g1 + lambda_2 * self.regression_loss_g2 + lambda_3 * self.regression_loss_g_a_L1
        return self.regression_loss

    def getkernel(self):
        # show the feature selection gate
        return self.l1_obj_vs.kernel.numpy()[0].tolist()
    def showkernel(self):
        # show the structure discovery gate
        temp_t = []
        for show_i in range(self.x_d):
            temp_t.append(self.l1_gate_before_model[show_i].kernel_up.numpy()[0][0])
        return temp_t
    def show_gatekernel(self):
        # get the double gates.
        temp_t = []
        temp_t_w = []
        temp_t_a = []
        for show_i in range(self.x_d):
            temp_t.append(self.l1_gate_before_model[show_i].kernel_up.numpy()[0][0])
            temp_t_w.append(tf.reduce_mean(tf.abs(tf.multiply(tf.multiply(self.down_layer_1[show_i].kernel, (1 - self.l1_gate_before_model[show_i].kernel_up)), self.l1_obj_vs.kernel[0][show_i]))).numpy())
            temp_t_a.append(tf.reduce_mean(tf.abs(self.up_layer_2[show_i].kernel)).numpy())
        return  self.l1_obj_vs.kernel.numpy()[0].tolist(),temp_t,temp_t_w, temp_t_a
