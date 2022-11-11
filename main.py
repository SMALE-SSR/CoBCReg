import warnings
import math
import os
import datetime

import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD

from CoBCReg import CoBCReg
from RBF.RBFNN import InitCentersKMeans, RBFLayer

warnings.filterwarnings("ignore")
BASE_DIR = './data'
gr = 1
k = 20
beta = 2.0


def split_labeled_and_unlabeled_data(data, scale):
    """
    原文划分标准如下:
    for each data set, 25% are used as test set, while the remaining 75% are used as training examples
    where 10% of the training examples are randomly selected as the initial labeled data set L
    while the remaining 90% of the 75% of data are used as unlabeled data set U.

    自定义划分如下:
    固定 training_size 为 2000, labeled_data_split_ration 分别设置 5%, 10%， 15%， 20%
    :return:{training_data, test_data}
    """
    target_data = data
    # the article settings
    # test_data = target_data.sample(math.ceil(target_data.shape[0] * 0.25))
    # training_data = target_data.drop(test_data.index)
    # labeled_data = training_data.sample(math.ceil(training_data.shape[0] * 0.1))
    # unlabeled_data = training_data.drop(labeled_data.index)

    # our settings
    training_data = target_data.sample(2000)
    labeled_data = training_data.sample(math.ceil(2000 * scale), random_state=666)
    unlabeled_data = training_data.drop(labeled_data.index)
    test_data = target_data.drop(training_data.index)

    return labeled_data, unlabeled_data, test_data


if __name__ == '__main__':
    data_dir = os.listdir(BASE_DIR)
    data_dir.sort(key=lambda x: os.stat(os.path.join(BASE_DIR, x)).st_size)

    for run in range(20):
        print('第 {} 次实验开始'.format(run + 1))

        for scale in [.05, 0.10, .15]:
            print('------------split data size: {}---------------'.format(scale))

            for file_name in data_dir:
                file = file_name.split('.')[0]
                print('DataSet start: {}'.format(file))

                data, meta = arff.loadarff(os.path.join(BASE_DIR, file_name))
                data_labeled = pd.DataFrame(data)
                labeled_data, unlabeled_data, test_data = split_labeled_and_unlabeled_data(data_labeled, scale)

                # base_learner = SVR()
                base_learner = Sequential()
                rbflayer = RBFLayer(k,
                                    initializer=InitCentersKMeans(labeled_data.iloc[:, :-1]),
                                    betas=beta,
                                    input_shape=(labeled_data.shape[1] - 1,))
                base_learner.add(rbflayer)
                base_learner.add(Dense(1))
                base_learner.add(Activation('linear'))
                base_learner.compile(loss='mean_squared_error',
                                     optimizer=SGD(), metrics=['RootMeanSquaredError'])

                cobcreg = CoBCReg(
                    base_learner,
                    3,
                    gr,
                    labeled_data,
                    unlabeled_data,
                    omega=[0.33, 0.33, 0.33]
                )
                cobcreg.fit(epochs=25, batch_size=32)
                res = cobcreg.predict(test_data.iloc[:, :-1])

                rmse = mean_squared_error(res, test_data.iloc[:, -1], squared=False)

                print('data: {}, RMSE: {}'.format(file, rmse))

                experiment = {
                    'file': file,
                    'rmse': rmse
                }

                save_path = './experiment/{}'.format(file)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                pd.DataFrame(experiment, index=[0]).to_csv('{}/{}-{}-{}.csv'.format(
                    save_path,
                    run,
                    scale,
                    datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
