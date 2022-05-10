from creatdata import *
from read_data import *
from result import *


if __name__ == '__main__':
    # create the dataset
    create_30_simulation_data()
    create_300_simulation_data()
    create_data_alllinear()
    create_data_nonelinear()
    create_data_partlinear()

    # lr_t = 0.01
    # i = 0
    # for lambda_1 in [1e-4,1e-3,1e-2,1e-1,1,10]:
    #     for lambda_2  in [1e-4,1e-3,1e-2,1e-1,1,10]:
    #         for lambda_3 in [1e-4,1e-3,1e-2,1e-1,1,10]:
    #             x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_30(128 + i)
    #             simulation_result_30(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
    #             i += 1

    for i in range(10):
        lambda_1 = 0.04
        lambda_2 = 0.07
        lambda_3 = 0.04
        lr_t = 0.01
        print(lambda_1, ':', lambda_2, ':', lambda_3, ':', i)

        x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_30(128 + i)
        simulation_result_30(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)

        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_300(128 + i)
        # simulation_result_300(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
        #
        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_CH(128 + i)
        # simulation_result_CH(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
        #
        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_BH(128 + i)
        # simulation_result_BH(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
        #
        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_new_nonelinear(128 + i)
        # simulation_result_new_nonelinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
        #
        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_new_alllinear(128 + i)
        # simulation_result_new_alllinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
        #
        # x_train, y_train, x_val, y_val, x_test, y_test = simulation_data_new_partlinear(128 + i)
        # simulation_result_new_partlinear(x_train, y_train, x_val, y_val, x_test, y_test, lambda_1, lambda_2, i, lr_t, lambda_3)
