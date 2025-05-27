import osqp
from scipy import sparse
from scipy.stats import entropy
from scipy.spatial import distance
import numpy as np
import pandas as pd

class custom_EFSVM:
    def __init__(self, C, beta, k, m, gamma = 0, type_ = 'default'):
        self.gamma = gamma
        self.type_ = type_
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

        return bins

    #3 Entropy-based Fuzzy Membership 계산
    def cal_fuzzy_membership(self, entropy_list, m, beta):
        # len(entropy_list) == negative의 갯수
        fuzzy_membership = {}
        H_min = min(entropy_list)
        H_max = max(entropy_list)

        # 구간의 길이를 동일하게
        for l in range(1, m + 1):
            thrUp = H_min + (H_max - H_min) * (l / m)
            thrLow = H_min + (H_max - H_min) * (l - 1) / m

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


    def getValue_osqp_dual(self, X, y_orig): # y_orig는 (m,) 형태의 원본 레이블
        # 1. 커널 행렬 계산
        _ = self.Kernel_(X, X, self.gamma, self.type_) # self.kernel에 저장됨

        # 2. 엔트로피 및 퍼지 멤버십, s_i 계산 (기존과 동일)
        entr = self.cal_entropy(X, y_orig, self.k)
        self.entr = entr
        membership = self.cal_fuzzy_membership(entr, self.m, self.beta)
        self.membership = membership
        si_array = np.array(self.cal_si(X, self.membership, y_orig)) # (m,)
        self.si_array = si_array

        y_col_vec = y_orig.reshape(-1, 1) # (m, 1) 형태의 레이블 벡터
        self.y = y_col_vec # 이후 계산을 위해 인스턴스 변수 업데이트
        m_samples, _ = X.shape

        # OSQP P matrix: Q_ij = y_i y_j K(x_i, x_j)
        P_mat_np = self.kernel * (y_col_vec @ y_col_vec.T)
        P_osqp = sparse.csc_matrix(P_mat_np)

        # OSQP q vector: -1
        q_osqp = -np.ones(m_samples)

        # OSQP A matrix and l, u vectors for constraints
        # Constraint 1: y'alpha = 0
        # Constraint 2: 0 <= alpha_i <= s_i * C
        A_osqp = sparse.vstack([
            y_col_vec.T,             # For y'alpha = 0
            sparse.eye(m_samples)    # For 0 <= alpha and alpha <= s_i*C
        ], format='csc')

        l_osqp = np.hstack([
            0,                       # Lower bound for y'alpha
            np.zeros(m_samples)      # Lower bound for alpha_i (0)
        ])
        u_osqp = np.hstack([
            0,                       # Upper bound for y'alpha
            si_array * self.C        # Upper bound for alpha_i (s_i*C)
        ])

        return P_osqp, q_osqp, A_osqp, l_osqp, u_osqp


    # train data를 이용해서 convex optimization을 통해 모델을 학습
    def solver(self, X, y):
        self.X = np.array(X)
        y_orig_shape = np.array(y)
        # self.y는 getValue_osqp_dual 내부에서 (m,1)로 설정됨

        P, q, A, l, u = self.getValue_osqp_dual(self.X, y_orig_shape)

        prob = osqp.OSQP()
        # settings = {'verbose': False, 'eps_abs': 1e-5, 'eps_rel': 1e-5}
        settings = {'verbose': False} # 기본 설정
        prob.setup(P, q, A, l, u, **settings)

        res = prob.solve()

        if res.info.status == 'solved' or res.info.status == 'solved inaccurate':
            if res.info.status == 'solved inaccurate':
                print("Warning: OSQP solved to a lower accuracy.")
            self.alphas = res.x.reshape(-1, 1)  # 결과를 (m,1) 형태로

            # w, S, b 계산 (cvxopt 버전과 매우 유사)
            if self.type_ == 'default': # 선형 커널
                # self.y는 (m,1), self.alphas는 (m,1)
                self.w = ((self.y * self.alphas).T @ self.X).reshape(-1, 1)
            else: # RBF 커널 등 비선형
                self.w = None

            # 서포트 벡터 S (alpha > epsilon 인 것들)
            epsilon = 1e-5 # 매우 작은 양수 (OSQP의 정밀도에 따라 조절)
            # S_indices = np.where(self.alphas.flatten() > epsilon)[0]
            # self.S = np.zeros(len(self.alphas), dtype=bool)
            # self.S[S_indices] = True
            # 더 간단하게:
            self.S = (self.alphas > epsilon).flatten()

            if not np.any(self.S):
                print("Warning: No support vectors found (alpha > epsilon). Model might be trivial.")
                # b를 추정하기 어려움. 모든 샘플을 사용하거나, 다른 fallback 전략 필요
                # 예시: 모든 알파가 거의 0이면 b도 0으로 설정
                if np.all(np.abs(self.alphas) < epsilon):
                    self.b = np.array([0.0])
                else:
                    # fallback: 알파가 0보다 큰 모든 샘플 사용 (S가 비었다면, epsilon 기준이 너무 엄격했을 수 있음)
                    S_fallback = (self.alphas > 1e-7).flatten() # 더 작은 epsilon
                    if not np.any(S_fallback):
                        self.b = np.array([0.0]) # 그래도 없으면 0
                    else:
                        b_candidates = self.y[S_fallback] - np.sum(self.Kernel_(self.X, self.X[S_fallback], self.gamma, self.type_) * self.y * self.alphas, axis=0).reshape(-1,1)
                        self.b = np.mean(b_candidates) if b_candidates.size > 0 else np.array([0.0])
                return

            # b 계산 (기존과 동일한 로직 사용 가능)
            # self.y는 (m,1), self.alphas는 (m,1)
            # self.S는 (m,) boolean 배열
            # self.X[self.S]는 서포트 벡터 데이터
            # self.y[self.S]는 서포트 벡터 레이블 (열벡터 형태가 될 것)
            b_terms = np.sum(self.Kernel_(self.X, self.X[self.S], self.gamma, self.type_) * self.y * self.alphas,
                             axis=0).reshape(-1, 1)
            # self.y[self.S]는 (num_S, 1)이 될 것이고 b_terms도 (num_S, 1)이 되어야 함.
            # self.Kernel_ 결과가 (m, num_S) 이고, (self.y * self.alphas)가 (m,1)이므로,
            # broadcast 후 곱하고 sum(axis=0) 하면 (num_S,)가 됨. reshape(-1,1)로 (num_S,1)

            # self.y[self.S]는 이미 올바른 shape (num_S, 1)
            b_values_at_S = self.y[self.S] - b_terms
            self.b = np.mean(b_values_at_S)

        else:
            print(f"OSQP failed to solve the problem. Status: {res.info.status}")
            self.alphas = None
            self.w = None
            self.b = None
            self.S = None
            # 오류 처리 또는 예외 발생을 고려할 수 있음


    def predict(self, test_X):
        test_X = np.array(test_X)
        b_val = self.b.item() if isinstance(self.b, np.ndarray) and self.b.size == 1 else self.b
        pred_sol = np.sign(np.sum(self.Kernel_(self.X, test_X, self.gamma, self.type_) * self.y * self.alphas, axis=0).reshape(-1,1) + b_val)

        return pred_sol