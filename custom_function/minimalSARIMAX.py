import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

class MinimalSARIMAX():
    def __init__(self, X_train, order, seasonal_order, exog=None):
        self.X_train = X_train
        self.X_train_exog = exog

        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.S = seasonal_order
        
        self.params = {}

        self.params['p'] = [random.uniform(0,0.1)]*self.p if self.p else [0]
        self.params['pX'] = [random.uniform(0,0.1)]*self.X_train_exog.shape[1] if (exog is not None) else [0]
        self.params['d'] = [random.uniform(0,0.1)]*self.d if self.d else [0]
        self.params['q'] = [random.uniform(0,0.1)]*self.q if self.q else [0]

        self.params['P'] = [random.uniform(0,0.1)]*self.P if self.P else [0]
        self.params['D'] = [random.uniform(0,0.1)]*self.D if self.D else [0]
        self.params['Q'] = [random.uniform(0,0.1)]*self.Q if self.Q else [0]
        self.params['c'] = random.uniform(0,0.1)
    
    #############################################################################
    
    def p_prediction(self, X_train, t):
        start = max(t-self.p, 0)
        X_train_t = np.array(X_train[start:t])[::-1]
        params_p = np.array(self.params['p'][:t-start])[::-1]

        p_pred = X_train_t @ params_p

        return p_pred, X_train_t[::-1]

    def pX_prediction(self, X_train_exog, t):
        if (t==0):
            return [0], [0]
        X_train_Xt = np.array(X_train_exog[t-1])
        params_pX = np.array(self.params['pX'])

        pX_pred = X_train_Xt @ params_pX

        return pX_pred, X_train_Xt

    def d_prediction(self, diff_X, t):
        if (self.d==0):
            return [0],[0] 
        start = max(t-self.p, 0)
        diff_X_t = np.array(diff_X.iloc[t-1])
        params_d = np.array(self.params['d'][:t-start])

        d_pred = diff_X_t @ params_d

        return d_pred, diff_X_t

    def q_prediction(self, X_train, Error, t):
        start = max(t-self.q, 0)
        error_t = np.array(Error[start:t])[::-1]
        params_q = np.array(self.params['q'][:t-start])[::-1]

        q_pred = error_t @ params_q

        return q_pred, error_t[::-1]
    
    #############################################################################

    def P_prediction(self, X_train, t):
        ss_c = t//self.S # season_count
        if (ss_c==0 or self.params['P']==[0]):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        X_train_ts = np.array(X_train[t-self.S::-self.S])[:ss_c]
        params_P = np.array(self.params['P'][::-1])[:ss_c]
        
        P_pred = X_train_ts @ params_P

        return P_pred, X_train_ts[::-1]

    def D_prediction(self, X_train, t):
        ss_c = t-1//self.S # season_count
        if (ss_c==0 or self.params['D']==[0]):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        diff_X_ts = np.array(X_train[t-1-self.S::-self.S])[:1]
        params_D = np.array(self.params['D'])
        
        D_pred = diff_X_ts @ params_D

        return D_pred, diff_X_ts[::-1]

    def Q_prediction(self, X_train, Error, t):
        ss_c = t//self.S # season_count
        if (ss_c==0 or len(self.params['Q'])==0):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        error_ts = np.array(Error[t-self.S::-self.S])[:ss_c]
        params_Q = np.array(self.params['Q'][::-1])[:ss_c]

        Q_pred = error_ts @ params_Q

        return Q_pred, error_ts[::-1]

    #############################################################################
    
    def update_params(self, x, error_t, lr):

        def param_pad_0(x, size):
            if (size==0): return np.zeros(1)
            x = np.array(x)
            result = np.zeros(size)
            result[:x.shape[0]] = x
            return result

        self.params['p'] += param_pad_0(x['p'], self.p) * error_t * lr

        self.params['pX'] += np.array(x['pX']) * error_t * lr

        self.params['d'] += np.array(x['d']) * error_t * lr

        self.params['q'] += param_pad_0(x['q'], self.q) * error_t * lr

        self.params['P'] += param_pad_0(x['P'], self.P) * error_t * lr

        self.params['D'] += np.array(x['D']) * error_t * lr

        self.params['Q'] += param_pad_0(x['Q'], self.Q) * error_t * lr
        
        self.params['c'] += self.params['c'] * error_t * lr

    #############################################################################
     
    def fit(self, verbose=0, lr=1e-5, lr_decay=0.999):
        X_train = self.X_train.iloc[:,0].copy()
        X_train = X_train.to_numpy()
        
        if self.X_train_exog is not None:
            X_train_exog = self.X_train_exog.copy()
            X_train_exog = X_train_exog.to_numpy()

        diff_X = self.X_train.copy()

        for i in range(1,self.d+1):
            diff_X[f'diff{i}'] = diff_X.iloc[:,[0]].diff(periods=i)

        diff_X = diff_X.fillna(0)
        diff_X = diff_X.iloc[:,1:]

        Error = [X_train[0]-X_train.mean()]

        for t in tqdm(range(1,X_train.shape[0])):
            lr *= lr_decay

            pred = {} ; x = {}

            pred['p'], x['p'] = self.p_prediction(X_train, t)
            pred['pX'], x['pX'] = self.pX_prediction(X_train_exog, t) if self.X_train_exog is not None else ([0], [0])
            pred['d'], x['d'] = self.d_prediction(diff_X, t)
            pred['q'], x['q'] = self.q_prediction(X_train, Error, t)

            pred['P'], x['P'] = self.P_prediction(X_train, t)
            pred['D'], x['D'] = self.D_prediction(X_train, t)
            pred['Q'], x['Q'] = self.Q_prediction(X_train, Error, t)

            pred['y'] = (pred['p'] + pred['pX'] + pred['d'] + pred['q'] + pred['P'] + pred['Q'] + pred['D'] + self.params['c']).sum()

            error_t = X_train[t] - pred['y']

            if (verbose):
                print(t, pred['y'], X_train[t], error_t, lr)

            Error.append(error_t)

            self.update_params(x, error_t, lr)
        
        return self.params
   
    #############################################################################
    
    def predict(self, y, y_X=None, verbose=0):
        y_t = y.iloc[:,0].copy().to_numpy()

        if y_X is not None:
            y_Xt = y_X.copy().to_numpy()

        diff_y_t = self.y.copy()
        for i in range(1,self.d+1):
            diff_y_t[f'diff{i}'] = diff_y_t.iloc[:,[0]].diff(periods=i)

        diff_y_t = diff_y_t.fillna(0)
        diff_y_t = diff_y_t.iloc[:,1:]

        Error = [y_t[0]-10]

        y_pred = [10]

        for t in range(1,len(y_t)):
            pred = {} ; x = {}
            pred['p'], x['p'] = self.p_prediction(y_t, t)
            pred['pX'], x['pX'] = self.pX_prediction(y_Xt, t) if y_X is not None else ([0], [0])
            pred['d'], x['d'] = self.d_prediction(diff_y_t, t)
            pred['q'], x['q'] = self.q_prediction(y_t, Error, t)
            
            pred['P'], x['P'] = self.P_prediction(y_t, t)
            pred['D'], x['D'] = self.D_prediction(y_t, t)
            pred['Q'], x['Q'] = self.Q_prediction(y_t, Error, t)

            pred['y'] = (pred['p'] + pred['pX'] + pred['d'] + pred['q'] + pred['P'] + pred['Q'] + pred['D'] + self.params['c']).sum()
            
            y_pred.append(pred['y'])

            error_t = y_t[t] - pred['y']

            if (verbose):
                print(t, pred['y'], y_t[t], error_t)

            Error.append(error_t)
        
        y_pred_tmp = y.iloc[:,[0]].copy()
        y_pred_tmp['PM25'] = np.array(y_pred)
        
        return y_pred_tmp, Error
    
    #############################################################################
    
    def RMSE(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f'Test on SARIMAX with RMSE: {rmse}')
    
    #############################################################################
    
    def plot(self, dataset, y_pred, title=""):
        plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(dataset, color='g',label='ground truth')
        plt.plot(y_pred, alpha=.7, color='r',label='predict')
        plt.title(title)
        plt.legend(loc="upper right")

        plt.show()