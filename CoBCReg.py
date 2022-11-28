import math

import copy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


class CoBCReg:
    def __init__(self, learner, N, gr, training_data, unlabeled_data, omega, pool_size=50, iterations=30):
        """
        :param learner: 学习器
        :param N: 学习器个数
        :param training_data: 训练数据
        :param unlabeled_data: 无标记数据
        :param pool_size: unlabeled数据选择缓冲池大小
        :param iterations: 最大迭代次数
        :param omega: 学习器预测权重
        """

        self.learner = learner
        self.learners = []
        # 学习器权重
        self.omega = omega
        # 袋内数据
        self.labeled_data_L_container = []
        # 袋外数据
        self.labeled_data_V_container = []
        # 被选择样本
        self.pi = []
        self.N = N
        self.pool_size = pool_size
        self.max_iterations = iterations
        self.gr = gr

        self.training_data = training_data
        self.unlabeled_data = unlabeled_data

        self._init_learner_and_data(learner)

    def _init_learner_and_data(self, learner):
        """
        初始化学习器及对应的数据容器
        :param learner: 学习器
        :return: null
        """
        for n in range(self.N):
            self.pi.append(None)

            per_learner = copy.deepcopy(learner)
            L, V = self._bootstrap_samples(self.training_data)
            self.labeled_data_L_container.append(L)
            self.labeled_data_V_container.append(V)
            self.learners.append(per_learner)
            self.train_inner_learner(self.learners[n], L)

    @staticmethod
    def _bootstrap_samples(labeled_data):
        """
        随机采样, 获取袋内及袋外数据, L:V=8:2
        :param labeled_data: 有标记数据
        :return: 袋内数据 L, 袋外数据 V
        """
        L = labeled_data.sample(math.ceil(labeled_data.shape[0] * 0.8))
        V = labeled_data.drop(L.index)
        return L, V

    @staticmethod
    def train_inner_learner(learner, train_data, epochs=25, batch_size=32):
        inner_learner_train_data = train_data.iloc[:, :-1]
        inner_learner_y_label = train_data.iloc[:, -1]
        learner.fit(inner_learner_train_data, inner_learner_y_label, epochs, batch_size, verbose=0)

    def fit(self, epochs, batch_size):
        for it in range(self.max_iterations):
            if self.unlabeled_data.empty:
                self.max_iterations = it - 1
                break

            for i in range(self.N):
                # 创建 unlabeled_data_pool
                self.unlabeled_data = shuffle(self.unlabeled_data).reset_index(drop=True)
                unlabeled_data_pool = self.unlabeled_data.head(self.pool_size)
                # 选择置信度样本
                pi_res = self._select_relevant_examples(i,
                                                        unlabeled_data_pool,
                                                        self.labeled_data_V_container[i],
                                                        self.gr)
                self.pi[i] = pi_res
                self.unlabeled_data = self.unlabeled_data.drop(pi_res.index)

            for i in range(self.N):
                if self.pi[i].empty:
                    continue

                # 合并 Li 和 Vi
                self.labeled_data_L_container[i] = pd.concat([self.labeled_data_L_container[i], self.pi[i]])
                # 重新训练 RBFNN
                current_labeled_data = self.labeled_data_L_container[i]
                self.train_inner_learner(self.learners[i], current_labeled_data)

            print('iter: {} has finished'.format(it))

    def _select_relevant_examples(self, j, unlabeled_data, V, gr):
        """
        选择置信度样本集合
        :param j: 当前学习器索引
        :param unlabeled_data: 无标签样本缓冲池
        :param V: 当前学习器袋外数据
        :param gr: growth rate
        :return: pi: 被选择该置信度数据
        """
        delta_x_u_result = []

        labeled_data_j = V.iloc[:, :-1]
        labeled_target_j = V.iloc[:, -1]

        # 计算当前 learner_j 在 V_j 上的 RMSE
        epsilon_j = mean_squared_error(self.learners[j].predict(labeled_data_j)[:, 0], labeled_target_j,
                                       squared=False)

        final_pred_list = []
        for index, row in unlabeled_data.iterrows():
            x_u = row.iloc[:-1].to_numpy().reshape(1, -1)

            # 计算其他学习的预测结果
            others_learner_pred_list = []
            for i in range(len(self.learners)):
                if i != j:
                    others_learner_pred_list.append(self.learners[i].predict(x_u)[:, 0])
            mean_prediction = np.mean(others_learner_pred_list)
            final_pred_list.append(mean_prediction)

            x_u = pd.DataFrame(np.hstack([x_u, np.array(mean_prediction).reshape(1, -1)]))
            x_u.columns = self.labeled_data_L_container[j].columns
            # 将当前 x_u 添加到 L_j 中, 并训练新的回归器用于计算 epsilon‘
            tmp_l_j = pd.concat([self.labeled_data_L_container[j], x_u])
            # new_learner = copy.deepcopy(self.learner)
            new_learner = self.learner
            self.train_inner_learner(new_learner, tmp_l_j)

            # 计算 x_u 置信度
            tmp_epsilon_j = mean_squared_error(new_learner.predict(labeled_data_j)[:, 0], labeled_target_j,
                                               squared=False)

            delta_x_u_result.append((epsilon_j - tmp_epsilon_j) / epsilon_j)

        pi = pd.DataFrame()

        # 获取 gr 个大于 0 的最大的 delta_x_u
        x_u_index = np.argsort(delta_x_u_result)[::-1]
        i_counts = len([_ for _ in delta_x_u_result if _ > 0])
        i_counts = i_counts if i_counts <= gr else gr

        if i_counts:
            unlabeled_data = unlabeled_data.iloc[:, :-1].loc[x_u_index[0:i_counts]]
            unlabeled_data_pseudo_label = final_pred_list[x_u_index[0:i_counts][0]]

            unlabeled_data_pseudo_label = pd.DataFrame({
                self.labeled_data_L_container[j].columns[-1]: unlabeled_data_pseudo_label
            }, index=unlabeled_data.index)
            pi = pd.concat([pi, pd.concat([unlabeled_data, unlabeled_data_pseudo_label], axis=1)])

        return pi

    def predict(self, data):
        """
        返回预测结果, 采取加权平均方式
        :param data: 待预测样本
        :return: result: 预测结果
        """
        per_pred = []
        for learner, w in zip(self.learners, self.omega):
            per_pred.append(w * learner.predict(data)[:, 0])

        result = sum(per_pred)
        return result
