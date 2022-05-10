import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import openpyxl
from model import NPLAM


def simulation_result_30(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2, lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_30.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    # save the g1 g2 A w.
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
    np.savetxt('./gate_data/30/30_gate_1_z1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/30/30_gate_1_z2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/30/30_gate_1_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/30/30_gate_1_a_'+str(number)+'.txt',show_gate_4)

    gate_1_data = np.loadtxt('./gate_data/30/30_gate_1_z1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/30/30_gate_1_z2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(30):
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
    plt.subplot(212)
    for i in range(30):
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
    # plt.show()
    plt.savefig('./fig/30/30_'+str(number)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_30.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    # save the result
    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[0]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(30):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(30):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = ''
    workbook.save(path)

def simulation_result_300(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_300.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()
    # save the g1 g2 A w.
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
    np.savetxt('./gate_data/300/300_gate_1_z1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/300/300_gate_1_z2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/300/300_gate_1_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/300/300_gate_1_a_'+str(number)+'.txt',show_gate_4)

    gate_1_data = np.loadtxt('./gate_data/300/300_gate_1_z1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/300/300_gate_1_z2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(300):
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
    plt.subplot(212)
    for i in range(300):
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
    # plt.show()
    plt.savefig('./fig/300/300_'+str(number)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_300.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    # save the result
    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[1]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(300):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(300):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = None
    workbook.save(path)

def simulation_result_CH(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_CH.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=2000, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_CH.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[2]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(8):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(8):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = None
    workbook.save(path)

def simulation_result_BH(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.01):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_BH.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=2000, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_BH.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[3]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(13):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(13):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = None
    workbook.save(path)

def simulation_result_new_alllinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2, lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_new_alllinear.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    # save the g1 g2 A w.
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
    np.savetxt('./gate_data/new_alllinear/gate_1_z1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/new_alllinear/gate_1_z2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/new_alllinear/gate_1_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/new_alllinear/gate_1_a_'+str(number)+'.txt',show_gate_4)
    #下面是生成图片
    gate_1_data = np.loadtxt('./gate_data/new_alllinear/gate_1_z1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/new_alllinear/gate_1_z2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(30):
        if i <= 4:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
        else:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i], color='k')
    plt.subplot(212)
    for i in range(30):
        if i <= 4:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
        else:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i], color='k')
    # plt.show()
    plt.savefig('./fig/new_alllinear/30_'+str(number)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_new_alllinear.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    # save the result
    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[4]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(30):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(30):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = ''
    workbook.save(path)

def simulation_result_new_nonelinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2, lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_new_nonelinear.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    # save the g1 g2 A w.
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
    np.savetxt('./gate_data/new_nonelinear/gate_1_z1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/new_nonelinear/gate_1_z2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/new_nonelinear/gate_1_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/new_nonelinear/gate_1_a_'+str(number)+'.txt',show_gate_4)

    gate_1_data = np.loadtxt('./gate_data/new_nonelinear/gate_1_z1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/new_nonelinear/gate_1_z2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(30):
        if i <= 4:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
        else:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i], color='k')
    plt.subplot(212)
    for i in range(30):
        if i <= 4:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
        else:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i], color='k')
    # plt.show()
    plt.savefig('./fig/new_nonelinear/30_'+str(number)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_new_nonelinear.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    # save the result
    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[5]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(30):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(30):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = ''
    workbook.save(path)

def simulation_result_new_partlinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1=0.01, lambda_2=0.01, number=0,lr=0.01,lambda_3=0.0):
    # Create the NPLAM model
    NPLAM_model = NPLAM(x_train.shape[-1], lambda_1, lambda_2, lambda_3)
    NPLAM_model.build(input_shape=(None, x_train.shape[-1]))
    NPLAM_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=NPLAM_model.model_loss6, metrics=['mse'])

    # Create the address where the model is saved
    model_path = "./model/NPLAM"
    model_name = 'NPLAM_new_partlinear.ckpt'
    # set the callback
    model_file = os.path.join(model_path, model_name)
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file, monitor='val_mse', save_weights_only=False, save_best_only=True, verbose=0)
    # save the gates
    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: NPLAM_model.show_gatekernel()}))

    NPLAM_model.fit(x=x_train, y=y_train, epochs=800, batch_size=100, validation_data=(x_val, y_val), callbacks=[callback, weight_callback],verbose=1)

    # predict the result
    y1_predict = NPLAM_model.predict(x_test)
    mse_NPLAM_1 = mean_squared_error(y_test, y1_predict)
    t1_1 = NPLAM_model.getkernel()
    t2_1 = NPLAM_model.showkernel()

    # save the g1 g2 A w.
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
    np.savetxt('./gate_data/new_partlinear/gate_1_z1_'+str(number)+'.txt',show_gate_1)
    np.savetxt('./gate_data/new_partlinear/gate_1_z2_'+str(number)+'.txt',show_gate_2)
    np.savetxt('./gate_data/new_partlinear/gate_1_w_'+str(number)+'.txt',show_gate_3)
    np.savetxt('./gate_data/new_partlinear/gate_1_a_'+str(number)+'.txt',show_gate_4)
    #下面是生成图片
    gate_1_data = np.loadtxt('./gate_data/new_partlinear/gate_1_z1_'+str(number)+'.txt').T
    gate_2_data = np.loadtxt('./gate_data/new_partlinear/gate_1_z2_'+str(number)+'.txt').T
    plt.subplot(211)
    for i in range(30):
        if i <= 14:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i])
        else:
            plt.plot(range(gate_1_data.shape[-1]), gate_1_data[i], color='k')
    plt.subplot(212)
    for i in range(30):
        if i <= 14:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i])
        else:
            plt.plot(range(gate_2_data.shape[-1]), gate_2_data[i], color='k')
    # plt.show()
    plt.savefig('./fig/new_partlinear/30_'+str(number)+'.jpg')
    plt.close()

    NPLAM_model2 = NPLAM(x_train.shape[-1], lambda_1, lambda_2,lambda_3)
    NPLAM_model2.load_weights('./model/NPLAM/NPLAM_new_partlinear.ckpt')
    y1_predict = NPLAM_model2.predict(x_test)
    mse_NPLAM_2 = mean_squared_error(y_test, y1_predict)
    t1_2 = NPLAM_model.getkernel()
    t2_2 = NPLAM_model.showkernel()

    # save the result
    path = 'mse_result.xlsx'
    workbook = openpyxl.load_workbook(path)
    sheetnames = workbook.sheetnames
    table = workbook[sheetnames[6]]
    table.cell(number*7+2, 1).value = mse_NPLAM_1
    for tempi in range(30):
        table.cell(number*7+3, tempi+1).value = t1_1[tempi]
        table.cell(number*7+4, tempi+1).value = t2_1[tempi]
    table.cell(number*7+5, 1).value = mse_NPLAM_2
    for tempi in range(30):
        table.cell(number*7+6, tempi+1).value = t1_2[tempi]
        table.cell(number*7+7, tempi+1).value = t2_2[tempi]
    table.cell(number*7+8, 1).value = ''
    workbook.save(path)
