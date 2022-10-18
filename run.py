from createdata import *
import read_data as rd
from result import simulation_result



if __name__ == '__main__':
    for i in range(10):
        lambda_1 = 0.05
        lambda_2 = 0.05
        lambda_3 = 0.01
        lr_t = 0.01

        x_train, y_train, x_val, y_val, x_test, y_test = rd.simulation_data_30(128 + i)
        simulation_result(x_train, y_train, x_val, y_val, x_test, y_test,number=i, lr=lr_t, nn_node=50, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,file_name='30')
