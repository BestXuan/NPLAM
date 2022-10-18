import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import openpyxl
from model import NPLAM


def simulation_result(x_train, y_train, x_val, y_val, x_test, y_test, number=0, lr=0.01, nn_node=50,  lambda_1=0.01, lambda_2=0.01,lambda_3=0.0,file_name='30'):
    file_name = file_name
    NPLAM_model = NPLAM(x_train.shape[-1], nn_node, lambda_1, lambda_2, lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss_mse, metrics=['mse'])

    # create the address to save model
    checkpoint_path = "model/NPLAM/cp.ckpt"
    # checkpoint
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_mse', save_weights_only=True, save_best_only=True, verbose=0)
    # save gate
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.get_g()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback,weight_callback], verbose=1)

    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    print(mse_NPLAM_1)
    t1_1 = NPLAM_model.get_g1()
    t2_1 = NPLAM_model.get_g2()
    ft_1 = [str(i) for i in t1_1]
    ft_2 = [str(i) for i in t2_1]
    print(' '.join(ft_1))
    print(' '.join(ft_2))

    # plot the figure of gates
    show_gate_1 = []
    show_gate_2 = []
    show_gate_3 = []
    show_gate_4 = []
    for i in range(len(weights_dict)):
        show_gate_1.append(weights_dict[i][0])
        show_gate_2.append(weights_dict[i][1])
        show_gate_3.append(weights_dict[i][2])
        show_gate_4.append(weights_dict[i][3])
    show_gate_1 = np.array(show_gate_1)
    show_gate_2 = np.array(show_gate_2)
    show_gate_3 = np.array(show_gate_3)
    show_gate_4 = np.array(show_gate_4)
    np.savetxt('./gate_data/gate_g1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/gate_g2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/gate_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/gate_a_'+str(number)+'.txt',show_gate_4)
    #下面是生成图片
    gate_1_data = np.loadtxt('./gate_data/gate_g1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/gate_g2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(x_train.shape[-1]):
        plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
    plt.subplot(212)
    for i in range(x_train.shape[-1]):
        plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
    # plt.show()
    plt.savefig('./gate_data/'+file_name+'_'+str(lambda_1)+'_'+str(lambda_2)+'_'+str(lambda_3)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], nn_node, lambda_1, lambda_2, lambda_3)
    NPLAM_model2.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model2.load_weights(checkpoint_path)
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    print(mse_NPLAM_2)
    t1_2 = NPLAM_model2.get_g1()
    t2_2 = NPLAM_model2.get_g2()
    ft_1 = [str(i) for i in t1_2]
    ft_2 = [str(i) for i in t2_2]
    print(' '.join(ft_1))
    print(' '.join(ft_2))
