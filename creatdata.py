import numpy as np

'''
Create the simulation dataset
'''

def create_data_alllinear():
    '''
    Create the full linear feature dataset.
    '''
    all = []
    for i in range(1600):
        temp = 0
        all.append([])
        #create data.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += (q1 * 2)
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += (q2*3)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += (q3*(-2.5))
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += (-3.5 * q4)
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (4 * q5)
        for t in range(25):
            all[i].append(np.random.uniform(-2.5, 2.5))
        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("./datasets/new_all_linear_30", all) #save to the data files.

def create_data_nonelinear():
    '''
    Create the full nonlinear feature dataset.
    '''
    all = []
    for i in range(1600):
        temp = 0
        all.append([])
        #create data.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 2
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += q2*q2
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += np.exp(-1*q3)
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp +=  np.cos(3 * q4) * 3
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (-2 * q5*q5)
        for t in range(25):
            all[i].append(np.random.uniform(-2.5, 2.5))
        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("./datasets/new_none_linear_30", all)

def create_data_partlinear():
    '''
    Create the partially linear feature dataset.
    '''
    all = []
    for i in range(1600):
        temp = 0
        all.append([])
        #create linear feature.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += (q1 * 2)
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += (q2*3)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += (q3*(-2.5))
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += (-3.5 * q4)
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (4 * q5)

        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += (q1 * (-2))
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += (q2*(-3))
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += (q3*(2.5))
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += (3.5 * q4)
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (-4 * q5)

        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += (q1 * 1.5)
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += (q2*2.5)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += (q3*(-2))
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += (-3 * q4)
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (3.5 * q5)

        # create nonlinear feature
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 2
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += q2*q2
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += np.exp(-1*q3)
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp +=  np.cos(3 * q4) * 3
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (-2 * q5*q5)

        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * -2
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += q2*q2*(-1)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += np.exp(-1*q3) *-1
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp +=  np.cos(3 * q4) * (-3)
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (2 * q5*q5)

        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 3
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += q2*q2*1.5
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += np.exp(-1*q3)*1.5
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp +=  np.cos(2 * q4) * 4
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += (1 * q5*q5)

        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("./datasets/new_part_linear_30", all)

def create_30_simulation_data():
    '''
    Create the 30-dimension simulation dataset.
    '''
    all = []
    for i in range(1600):
        temp = 0
        all.append([])
        #create data.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 4
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += np.exp(q2) * (1/1)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += q3 * q3
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += -1 * q4
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += 5 * q5
        for t in range(25):
            all[i].append(np.random.uniform(-2.5, 2.5))
        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("./datasets/simulation_data_30_data", all)

def create_300_simulation_data():
    '''
    Create the 300-dimension simulation dataset.
    '''
    all = []
    for i in range(16000):
        temp = 0
        all.append([])
        #create data.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 4
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += np.exp(q2) * (1/1)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += q3 * q3
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += -1 * q4
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += 5 * q5
        for t in range(295):
            all[i].append(np.random.uniform(-2.5, 2.5))
        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("./datasets/simulation_data_300_data", all)

def create_data_for_experiment_of_sample_size():
    '''
    This dataset for experiment of sample size.
    '''
    all = []
    for i in range(111751):
        temp = 0
        all.append([])
        #create data.
        q1 = np.random.uniform(-2.5, 2.5)
        all[i].append(q1)
        temp += np.sin(2*q1) * 4
        q2 = np.random.uniform(-2.5, 2.5)
        all[i].append(q2)
        temp += np.exp(q2) * (1/1)
        q3 = np.random.uniform(-2.5, 2.5)
        all[i].append(q3)
        temp += q3 * q3
        q4 = np.random.uniform(-2.5, 2.5)
        all[i].append(q4)
        temp += -1 * q4
        q5 = np.random.uniform(-2.5, 2.5)
        all[i].append(q5)
        temp += 5 * q5
        for t in range(25):
            all[i].append(np.random.uniform(-2.5, 2.5))
        b = np.random.normal(0, 1)
        temp += b
        all[i].append(temp)
    all = np.asarray(all)
    np.save("simulation_data_30_data_highsample", all)