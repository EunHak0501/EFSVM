from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from scipy.stats import entropy
from scipy.spatial import distance
import numpy as np
import pandas as pd

class custom_EFSVM:
    def __init__(self, C, beta, k, m, gamma = 0, type_ = 'default', method='default'):
        self.gamma = gamma
        self.type_ = type_
        self.method = method
        self.C = C
        self.beta = beta
        self.k = k
        self.m = m
        self.kernel = None
        self.X = None
        self.y = None
        self.alphas = None
        self.S = None
        self.w = None
        self.b = None
        self.support = None
        self.entr = None
        self.membership = None
        self.si_array = None

    # 커널 함수, kernel trick의 적용을 위함
    def Kernel_(self, x, y, params = 0, type_ = 'default') :
        Kernel = None
        if type_ == 'rbf' :
            Kernel = np.exp(- (np.sum(x**2, axis = 1).reshape(-1,1) + np.sum(y**2, axis = 1).reshape(1,-1) - 2 * x @ y.T)* params)
        elif type_ == 'default' :
            Kernel = np.dot(x, y.T)

        self.kernel = Kernel
        return Kernel


    # 유클리드 거리 계산
    def get_euclidean_distance(self, X):
        return distance.cdist(X, X, metric = 'euclidean')


    def cal_entropy(self, X, y, k):
        entropy_list = []
        eucli_distance = self.get_euclidean_distance(X)
        # k값에 따라 negative class에 대해 euclidean distance를 기준으로 가까운 값의 인덱스를 저장
        knn_neg = [np.argsort(eucli_distance[idx])[1:k+1] for idx, val in enumerate(y) if val < 0]
        # 자기 자신은 제외 >> index 1부터 k+1까지
        for indexs in knn_neg:
            p_cnt = len([y[idx] for idx in indexs if y[idx] > 0])
            p_pos = p_cnt / k # probability of positive
            p_neg = 1 - p_pos # probability of negative
            H_i = entropy([p_pos, p_neg]) # get entropy
            entropy_list.append(H_i)

        return entropy_list

    def cal_entropy_index(self, m, l, thrUp, thrLow, entropy_list):
        entropy_index = []
        for i, H in enumerate(entropy_list):
            if l==m:  # l이 m과 같은 경우(마지막 구간) H가 thrUp과 같을 수 있게 만들어줌
                if thrLow <= H <= thrUp:
                    entropy_index.append(i)
            else:
                if thrLow <= H < thrUp:
                    entropy_index.append(i)
        return entropy_index

    def divide_min_variance(self, list, m, method=1):
        list = np.array(list)
        n = len(list)
        if method == 1:
            # DP 테이블과 Breakpoints 테이블 초기화
            dp = np.full((n + 1, m + 1), np.inf)  # 초기값을 무한대로 설정
            breakpoints = np.zeros((n + 1, m + 1), dtype=int)

            # dp[i][1] 초기화: 첫 번째 구간의 분산 계산
            for i in range(1, n + 1):
                dp[i][1] = np.var(list[:i]) * i

                # DP 테이블 채우기
            for j in range(2, m + 1):
                for i in range(j, n + 1):
                    min_variance = float('inf')
                    best_k = -1
                    for k in range(j - 1, i):
                        current_variance = dp[k][j - 1] + np.var(list[k:i]) * (i - k)
                        if current_variance < min_variance:
                            min_variance = current_variance
                            best_k = k
                    dp[i][j] = min_variance
                    breakpoints[i][j] = best_k

                segments = []
            current = n
            for j in range(m, 0, -1):
                segments.append((breakpoints[current][j], current))
                current = breakpoints[current][j]

            segments = sorted(segments)

            bins = []
            for i, (start, end) in enumerate(segments):
                thrLow = list[start] if i == 0 else list[start - 1]
                thrUp = list[end - 1]
                bins.append((thrLow, thrUp))

        elif method == 2:
            print('Not implemented')

        return bins

    #3 Entropy-based Fuzzy Membership 계산
    def cal_fuzzy_membership(self, entropy_list, m, beta):
        # len(entropy_list) == negative의 갯수
        fuzzy_membership = {}
        H_min = min(entropy_list)
        H_max = max(entropy_list)

        if self.method == 'default': # 구간의 길이를 동일하게
            for l in range(1, m + 1):
                thrUp = H_min + (H_max - H_min) * (l / m)
                thrLow = H_min + (H_max - H_min) * (l - 1) / m

                entropy_index = self.cal_entropy_index(m, l, thrUp, thrLow, entropy_list)

                if len(entropy_index) == 0: # Low, Up 조건에 맞는 엔트로피가 없으면 FM 계산 x
                    continue

                fm = 1.0 - beta * l # cal fm
                fuzzy_membership[fm] = entropy_index # entropy index를 해당하는 fm을 key로 한 value에 넣기

        elif self.method == 'same_frequent': # 각 구간에 동일한 갯수가 포함되게
            sorted_entropy = np.sort(entropy_list)
            bins = np.array_split(sorted_entropy, m)

            for l in range(1, m + 1):
                thrUp = bins[l-1][-1]
                thrLow = bins[l-1][0]

                entropy_index = self.cal_entropy_index(m, l, thrUp, thrLow, entropy_list)

                if len(entropy_index) == 0: # Low, Up 조건에 맞는 엔트로피가 없으면 FM 계산 x
                    continue

                fm = 1.0 - beta * l # cal fm
                fuzzy_membership[fm] = entropy_index # entropy index를 해당하는 fm을 key로 한 value에 넣기

        elif self.method == 'min_variance': # 각 구간의 분산이 최소가 되도록 (비슷한 엔트로피끼리 모이도록)
            sorted_entropy = np.sort(entropy_list)
            n = len(entropy_list)
            bins = self.divide_min_variance(sorted_entropy, m)

            for l in range(1, m + 1):
                thrUp = bins[l-1][-1]
                thrLow = bins[l-1][0]

                entropy_index = self.cal_entropy_index(m, l, thrUp, thrLow, entropy_list)

                if len(entropy_index) == 0: # Low, Up 조건에 맞는 엔트로피가 없으면 FM 계산 x
                    continue

                fm = 1.0 - beta * l # cal fm
                fuzzy_membership[fm] = entropy_index # entropy index를 해당하는 fm을 key로 한 value에 넣기

        return fuzzy_membership

    # 4. Positive와 Negative class 모두에 대해 si 부여
    def cal_si(self, X, fm, y):
        si = []
        neg_class = [idx for idx, val in enumerate(y) if val < 0] # negative class 찾기

        for i in range(len(y)):
            if i in neg_class: # negative class에 대해 si 부여
                for j in fm:
                    if neg_class.index(i) in fm[j]: # y가 negative class이면서, fm에 포함되어 있는지 확인
                        si.append(j)
                        break
            else: # positive class에 대해 si 부여
                si.append(1.0)

        return si

    # convex optimization에 필요한 parameter들을 반환
    def getValue(self, X, y):
        Kernel = self.Kernel_(X, X, self.gamma, self.type_)

        entr = self.cal_entropy(X, y, self.k)
        self.entr = entr

        membership = self.cal_fuzzy_membership(entr, self.m, self.beta)
        self.membership = membership

        si_array = np.array(self.cal_si(X, self.membership, y))
        self.si_array = si_array

        y = y.reshape(-1,1)
        self.y = y
        m,n = X.shape

        H = self.kernel * (y @ y.T)

        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(np.vstack((-np.eye(m),np.eye(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * self.si_array * self.C))) # s_i를 추가적으로 이용
        A = cvxopt_matrix(y.reshape(1, -1), tc='d')
        b = cvxopt_matrix(np.zeros(1))

        return P, q, G, h, A, b

    # train data를 이용해서 convex optimization을 통해 모델을 학습
    def solver(self, X, y):
        cvxopt_solvers.options['show_progress'] = False # verbose = False
        self.X = np.array(X)
        self.y = np.array(y)
        P, q, G, h, A, b = self.getValue(self.X, self.y)
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)

        alphas = np.array(sol['x'])
        self.alphas = alphas

        if self.type_ == 'default':
            w = ((self.y * self.alphas).T @ self.X).reshape(-1, 1)
        else:
            w = None # gaussian 일 때 w는 None.
        S = ((self.alphas > 1e-4) & (self.alphas < self.C-1e-4)).flatten()
        if self.type_ == 'rbf':
            b = self.y[S] - np.sum(self.Kernel_(self.X, self.X[S], self.gamma, self.type_) * self.y * self.alphas, axis = 0).reshape(-1, 1) # gaussian
        else:
            b = self.y[S] - np.sum(self.Kernel_(self.X, self.X[S]) * self.y * self.alphas, axis = 0).reshape(-1, 1) # linear

        self.w = w
        self.S = S
        self.b = b

        # return alphas, S, w, b

    def predict(self, test_X):
        test_X = np.array(test_X)
        pred_sol = np.sign(np.sum(self.Kernel_(self.X, test_X, self.gamma, self.type_) * self.y * self.alphas, axis = 0).reshape(-1,1) + self.b[0])
        return pred_sol