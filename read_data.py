import numpy as np

def simulation_data_30(random_lambda):
    # read data
    read_data = np.load('./datasets/simulation_data_30_data.npy')
    np.random.seed(random_lambda)
    np.random.shuffle(read_data)
    read_data = read_data.T #(31.1600)
    # Max-Min
    for i in range(read_data.shape[0]):
        temp_max = np.max(read_data[i])
        temp_min = np.min(read_data[i])
        for j in range(read_data.shape[1]):
            if (temp_max - temp_min) < 0.1:
                read_data[i][j] = 0
            else:
                read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # #Z-score
    # for i in range(read_data.shape[0]):
    #     temp_max = np.max(read_data[i])
    #     temp_min = np.min(read_data[i])
    #     for j in range(read_data.shape[1]):
    #         if (temp_max - temp_min) < 0.1:
    #             read_data[i][j] = 0
    #         else:
    #             read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    read_data = read_data.T
    train_set = read_data[:960]
    val_set = read_data[960:1280]
    test_set = read_data[1280:]
    x_train = train_set.T[:-1].T
    y_train = train_set.T[-1].T
    x_val = val_set.T[:-1].T
    y_val = val_set.T[-1].T
    x_test = test_set.T[:-1].T
    y_test = test_set.T[-1].T
    return x_train,y_train,x_val,y_val,x_test,y_test

def simulation_data_300(random_lambda):
    # read data
    read_data = np.load('./datasets/simulation_data_300_data.npy')
    np.random.seed(random_lambda)
    np.random.shuffle(read_data)
    read_data = read_data.T  # (31.16000)
    # Max-Min
    for i in range(read_data.shape[0]):
        temp_max = np.max(read_data[i])
        temp_min = np.min(read_data[i])
        for j in range(read_data.shape[1]):
            if (temp_max - temp_min) < 0.1:
                read_data[i][j] = 0
            else:
                read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # #Z-score
    # for i in range(read_data.shape[0]):
    #     temp_max = np.max(read_data[i])
    #     temp_min = np.min(read_data[i])
    #     for j in range(read_data.shape[1]):
    #         if (temp_max - temp_min) < 0.1:
    #             read_data[i][j] = 0
    #         else:
    #             read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # read_data = read_data.T
    read_data = read_data.T
    train_set = read_data[:9600]
    val_set = read_data[9600:12800]
    test_set = read_data[12800:]
    x_train = train_set.T[:-1].T
    y_train = train_set.T[-1].T
    x_val = val_set.T[:-1].T
    y_val = val_set.T[-1].T
    x_test = test_set.T[:-1].T
    y_test = test_set.T[-1].T
    return x_train, y_train, x_val, y_val, x_test, y_test

def simulation_data_CH(random_lambda):
    # read data
    path_1 = r'./datasets/california_housing.data'
    file = open(path_1, 'r')
    file_data = file.readlines()
    for i in range(len(file_data)):
        file_data[i] = file_data[i].split(',')
        file_data[i][-1] = file_data[i][-1][:-1]
    data = []
    for i in range(len(file_data)):
        data.append([])
        for j in range(9):
            data[i].append(float(file_data[i][j]))
    data = np.asarray(data)
    np.random.seed(random_lambda)
    np.random.shuffle(data)
    data = data.T
    for i in range(data.shape[0]):
        data_max_train = np.max(data[i])
        data_min_train = np.min(data[i])
        for j in range(data.shape[1]):
            data[i][j] = -1 + (data[i][j] - data_min_train) * (2 / (data_max_train - data_min_train))
    x_train = data[:-1].T[:12384]
    y_train = data[-1].T[:12384].reshape(12384, 1)
    x_val = data[:-1].T[12384:16512]
    y_val = data[-1].T[12384:16512].reshape(4128, 1)
    x_test = data[:-1].T[16512:]
    y_test = data[-1].T[16512:].reshape((20640 - 16512), 1)
    return x_train, y_train, x_val, y_val, x_test, y_test

def simulation_data_BH(random_lambda):
    # read data
    path_1 = r'./datasets/boston_house_prices.csv'
    file = open(path_1, 'r')
    file_data = file.readlines()
    file_data = file_data[1:]
    for i in range(len(file_data)):
        file_data[i] = file_data[i].split(',')
        file_data[i][-1] = file_data[i][-1][:-1]
    data = []
    for i in range(len(file_data)):
        data.append([])
        for j in range(14):
            data[i].append(float(file_data[i][j]))
    data = np.asarray(data)
    np.random.seed(random_lambda)
    np.random.shuffle(data)
    data = data.T
    for i in range(data.shape[0]):
        data_max_train = np.max(data[i])
        data_min_train = np.min(data[i])
        for j in range(data.shape[1]):
            data[i][j] = -1 + (data[i][j] - data_min_train) * (2 / (data_max_train - data_min_train))
    x_train = data[:-1].T[:303]
    y_train = data[-1].T[:303].reshape(303, 1)
    x_val = data[:-1].T[303:404]
    y_val = data[-1].T[303:404].reshape(101, 1)
    x_test = data[:-1].T[404:506]
    y_test = data[-1].T[404:506].reshape(102, 1)
    return x_train, y_train, x_val, y_val, x_test, y_test

def simulation_data_new_alllinear(random_lambda):
    # read data
    read_data = np.load('./datasets/new_all_linear_30.npy')
    np.random.seed(random_lambda)
    np.random.shuffle(read_data)
    read_data = read_data.T #(31.1600)
    # Max-Min
    for i in range(read_data.shape[0]):
        temp_max = np.max(read_data[i])
        temp_min = np.min(read_data[i])
        for j in range(read_data.shape[1]):
            if (temp_max - temp_min) < 0.1:
                read_data[i][j] = 0
            else:
                read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # #Z-score
    # for i in range(read_data.shape[0]):
    #     temp_max = np.max(read_data[i])
    #     temp_min = np.min(read_data[i])
    #     for j in range(read_data.shape[1]):
    #         if (temp_max - temp_min) < 0.1:
    #             read_data[i][j] = 0
    #         else:
    #             read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    read_data = read_data.T
    train_set = read_data[:960]
    val_set = read_data[960:1280]
    test_set = read_data[1280:]
    x_train = train_set.T[:-1].T
    y_train = train_set.T[-1].T
    x_val = val_set.T[:-1].T
    y_val = val_set.T[-1].T
    x_test = test_set.T[:-1].T
    y_test = test_set.T[-1].T
    return x_train,y_train,x_val,y_val,x_test,y_test

def simulation_data_new_nonelinear(random_lambda):
    # read data
    read_data = np.load('./datasets/new_none_linear_30.npy')
    np.random.seed(random_lambda)
    np.random.shuffle(read_data)
    read_data = read_data.T #(31.1600)
    # Max-Min
    for i in range(read_data.shape[0]):
        temp_max = np.max(read_data[i])
        temp_min = np.min(read_data[i])
        for j in range(read_data.shape[1]):
            if (temp_max - temp_min) < 0.1:
                read_data[i][j] = 0
            else:
                read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # #Z-score
    # for i in range(read_data.shape[0]):
    #     temp_max = np.max(read_data[i])
    #     temp_min = np.min(read_data[i])
    #     for j in range(read_data.shape[1]):
    #         if (temp_max - temp_min) < 0.1:
    #             read_data[i][j] = 0
    #         else:
    #             read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    read_data = read_data.T
    train_set = read_data[:960]
    val_set = read_data[960:1280]
    test_set = read_data[1280:]
    x_train = train_set.T[:-1].T
    y_train = train_set.T[-1].T
    x_val = val_set.T[:-1].T
    y_val = val_set.T[-1].T
    x_test = test_set.T[:-1].T
    y_test = test_set.T[-1].T
    return x_train,y_train,x_val,y_val,x_test,y_test

def simulation_data_new_partlinear(random_lambda):
    # read data
    read_data = np.load('./datasets/new_part_linear_30.npy')
    np.random.seed(random_lambda)
    np.random.shuffle(read_data)
    read_data = read_data.T #(31.1600)
    # Max-Min
    for i in range(read_data.shape[0]):
        temp_max = np.max(read_data[i])
        temp_min = np.min(read_data[i])
        for j in range(read_data.shape[1]):
            if (temp_max - temp_min) < 0.1:
                read_data[i][j] = 0
            else:
                read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    # #Z-score
    # for i in range(read_data.shape[0]):
    #     temp_max = np.max(read_data[i])
    #     temp_min = np.min(read_data[i])
    #     for j in range(read_data.shape[1]):
    #         if (temp_max - temp_min) < 0.1:
    #             read_data[i][j] = 0
    #         else:
    #             read_data[i][j] = -1 + 2 * (read_data[i][j] - temp_min) / (temp_max - temp_min)
    read_data = read_data.T
    train_set = read_data[:960]
    val_set = read_data[960:1280]
    test_set = read_data[1280:]
    x_train = train_set.T[:-1].T
    y_train = train_set.T[-1].T
    x_val = val_set.T[:-1].T
    y_val = val_set.T[-1].T
    x_test = test_set.T[:-1].T
    y_test = test_set.T[-1].T
    return x_train,y_train,x_val,y_val,x_test,y_test
