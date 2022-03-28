import numpy as np
import pandas as pd
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

        self.params['p'] = [random.uniform(0, 0.1)]*self.p if self.p else np.zeros(1)
        self.params['pX'] = [random.uniform(0, 0.1)]*self.X_train_exog.shape[1] if (exog is not None) else np.zeros(1)
        self.params['d'] = [random.uniform(0, 0.1)]*self.d if self.d else np.zeros(1)
        self.params['q'] = [random.uniform(0, 0.1)]*self.q if self.q else np.zeros(1)

        self.params['P'] = [random.uniform(0, 0.1)]*self.P if self.P else np.zeros(1)
        self.params['D'] = [random.uniform(0, 0.1)]*self.D if self.D else np.zeros(1)
        self.params['Q'] = [random.uniform(0, 0.1)]*self.Q if self.Q else np.zeros(1)
        self.params['c'] = random.uniform(0, 0.1)

        self.Error_X = None
    
    #############################################################################
    
    def p_prediction(self, X_train, t):
        if (t <= 0): return np.zeros(1), np.zeros(1)
        start = max(t-self.p, 0)
        X_train_t = np.array(X_train[start:t])[::-1]
        params_p = np.array(self.params['p'][:t-start])[::-1]

        p_pred = X_train_t @ params_p[:X_train_t.shape[0]]

        return p_pred, X_train_t[::-1]

    def pX_prediction(self, X_train_exog, t):
        if (t <= 0): return np.zeros(1), np.zeros(1)
        X_train_exog_t = np.array(X_train_exog[t-1])
        params_pX = np.array(self.params['pX'])
        
        pX_pred = X_train_exog_t @ params_pX

        return pX_pred, X_train_exog_t

    def d_prediction(self, diff_X, t):
        if (self.d == 0 or t <= 0): return np.zeros(1), np.zeros(1)
        start = max(t-self.d, 0)
        diff_X_t = np.array(diff_X[start:t])[::-1]
        params_d = np.array(self.params['d'][:t-start])[::-1]

        d_pred = diff_X_t @ params_d[:diff_X_t.shape[0]]

        return d_pred, diff_X_t[::-1]

    def q_prediction(self, Error, t):
        if (self.q == 0 or t <= 0): return np.zeros(1), np.zeros(1)
        start = max(t-self.q, 0)
        error_t = np.array(Error[start:t]).ravel()[::-1]
        params_q = np.array(self.params['q'][:t-start])[::-1]

        q_pred = error_t @ params_q

        return q_pred, error_t[::-1]
    
    #############################################################################

    def P_prediction(self, X_train, t):
        ss_c = t//self.S  # season_count
        if (ss_c == 0 or self.params['P'] == np.zeros(1) or t <= 0):
            return np.zeros(1) ,np.zeros(1)
        ss_c = min(ss_c, self.P)
        X_train_ts = np.array(X_train[t-self.S::-self.S]).reshape(-1)[:ss_c]
        params_P = np.array(self.params['P'][::-1])[:ss_c]
        
        P_pred = X_train_ts @ params_P[:X_train_ts.shape[0]]

        return P_pred, X_train_ts[::-1]

    def D_prediction(self, X_train, t):
        ss_c = t-1//self.S  # season_count
        if (ss_c == 0 or self.params['D'] == np.zeros(1) or t <= 0):
            return np.zeros(1) ,np.zeros(1)
        ss_c = min(ss_c, self.D)
        diff_X_ts = np.array(X_train[t-1-self.S::-self.S]).reshape(-1)[:1]
        params_D = np.array(self.params['D'])
        
        D_pred = diff_X_ts @ params_D[:diff_X_ts.shape[0]]

        return D_pred, diff_X_ts[::-1]

    def Q_prediction(self, Error, t):
        ss_c = t//self.S  # season_count
        if (ss_c == 0 or len(self.params['Q']) == np.zeros(1) or t <= 0):
            return np.zeros(1) ,np.zeros(1)
        ss_c = min(ss_c, self.Q)
        error_ts = np.array(Error[t-self.S::-self.S]).reshape(-1)[:ss_c]
        params_Q = np.array(self.params['Q'][::-1])[:ss_c]

        Q_pred = error_ts @ params_Q[:error_ts.shape[0]]

        return Q_pred, error_ts[::-1]
    
    #############################################################################
    
    def update_params(self, x, error_t, lr):

        def param_pad_0(x, size):
            if (size==0 or len(x)==0):
                return np.zeros(1)
            x = np.array(x).ravel()
            result = np.zeros(size)
            result[:x.shape[0]] = x
            return result


        if 'p' in x:       
            self.params['p'] += param_pad_0(x['p'], self.p) * error_t * lr

        if 'pX' in x: 
            self.params['pX'] += np.array(x['pX']) * error_t * lr

        if 'd' in x: 
            self.params['d'] += param_pad_0(x['d'], self.d) * error_t * lr

        if 'q' in x: 
            self.params['q'] += param_pad_0(x['q'], self.q) * error_t * lr

        if 'P' in x: 
            self.params['P'] += param_pad_0(x['P'], self.P) * error_t * lr

        if 'D' in x: 
            self.params['D'] += param_pad_0(x['D'], self.D) * error_t * lr

        if 'Q' in x: 
            self.params['Q'] += param_pad_0(x['Q'], self.Q) * error_t * lr
        
        self.params['c'] += 1 * error_t * lr

    #############################################################################

    def fit(self, verbose=0, lr=1e-6, lr_decay=0.999):
        tqdm_disable = False
        if (verbose==0): tqdm_disable=True

        X_train = self.X_train.iloc[:,0].copy().to_numpy()

        X_train_exog = None
        if self.X_train_exog is not None:
            X_train_exog = self.X_train_exog.copy().to_numpy()

        diff_X = self.calcDiff(self.X_train)

        Pred = [0]

        Error_X = [X_train[0]]

        last_x = {'p':[0],'pX':[0],'d':[0],'q':[0],'P':[0],'D':[0],'Q':[0]}

        for t in tqdm(range(1, X_train.shape[0]), disable=tqdm_disable):
            lr *= lr_decay

            self.update_params(last_x, Error_X[-1], lr)

            new_pred, new_x = self.predict_one(
                X_train, diff_X, t, Error_X, X_train_exog=X_train_exog)
            
            Pred.append(new_pred)
            error_X_t = X_train[t] - new_pred
            last_x = new_x

            if (verbose):
                print(t, new_pred, X_train[t], error_X_t, lr)

            Error_X.append(error_X_t)

        self.Error_X = Error_X
        
        return self.params
    
    def predict_one(self, X_train, diff_X, t, Error_X=None, X_train_exog=None):

        pred = {} ; x = {}

        pred['p'], x['p'] = self.p_prediction(X_train, t-1)

        if (X_train_exog is not None and len(X_train_exog)):
            pred['pX'], x['pX'] = self.pX_prediction(X_train_exog, t-1) 
        else:
            pred['pX'], x['pX'] = (np.zeros(1), np.zeros(1))
         

        pred['d'], x['d'] = self.d_prediction(diff_X, t-1)
        
        if (Error_X is None or len(Error_X)<2):
            pred['q'], x['q'] = (np.zeros(1), np.zeros(1))
        else:
            # print(t, len(Error_X), np.array(Error_X))
            pred['q'], x['q'] = self.q_prediction(np.array(Error_X), t-1)
            
        pred['P'], x['P'] = self.P_prediction(X_train, t-1)
        pred['D'], x['D'] = self.D_prediction(X_train, t-1)

        if (Error_X is not None and ('Q' in x) and len(x['Q'])):
            pred['Q'], x['Q'] = self.Q_prediction(np.array(Error_X), t-1)
        else:
            pred['Q'], x['Q'] =(np.zeros(1), np.zeros(1))

        pred['y'] = (pred['p'] + pred['pX'] + pred['d'] + pred['q'] + pred['P'] + pred['Q'] + pred['D'] + self.params['c']).sum()

        return pred['y'], x

   
    #############################################################################
    
    def predict(self, y, y_exog=None, init_pred=[10], init_e=[0], verbose=0, e_flag=False):
        y_t = y.copy().to_numpy()

        y_exog_t = None
        if y_exog is not None:
            y_exog_t = y_exog.copy().to_numpy()
        
        diff_y = self.calcDiff(y)

        Error = init_e

        y_pred = init_pred

        for t in range(1,len(y_t)):
            pred = {} ; x = {}

            pred['p'], x['p'] = self.p_prediction(y_t, t)
            pred['pX'], x['pX'] = self.pX_prediction(y_exog_t, t) if y_exog is not None else (np.zeros(1) ,np.zeros(1))
            pred['d'], x['d'] = self.d_prediction(diff_y, t)
            pred['q'], x['q'] = self.q_prediction(Error, t)
            
            pred['P'], x['P'] = self.P_prediction(y_t, t)
            pred['D'], x['D'] = self.D_prediction(y_t, t)
            pred['Q'], x['Q'] = self.Q_prediction(Error, t)

            pred['y'] = np.nansum(pred['p'] + pred['pX'] + pred['d'] + pred['q'] + pred['P'] + pred['Q'] + pred['D'] + self.params['c'])
            
            y_pred.append(pred['y'])

            error_t = (y_t[t] - pred['y']).sum()

            if (verbose):
                print(t, pred['y'], y_t[t], error_t)

            Error.append(error_t)

        col_name = y.columns.values[0]
        df = pd.DataFrame({col_name:y_pred[-y.shape[0]:]},index=y.index)

        if e_flag:
            return Error

        return df, Error

    def predict_step(self, val_X, y, val_X_exog=None, y_exog=None, model_exog=None, step=12, learn=False, lr=np.array([1e-7]), lr_decay=0.999):

        exog_flag = False
        if val_X_exog is not None:
            exog_flag = True
            model_exo = []
            for key in model_exog:
                model_exo.append(model_exog[key])
        

        save_time = val_X.index[0]
        start_time_y = y.index[0]

        end_time_y = y.index[-1]

        pred_sav = pd.DataFrame(columns=['Time','t','s','Predict', 'Actual'])
        y_pred_sav = pd.DataFrame(columns=['Time','t','s','Predict', 'Actual'])
        
        ## random init
        pred_init = random.randint(5,10)
        pred_init_exog = np.array([30, 25, 90])

        # initialize variables
        val_Xt = pd.concat([val_X.iloc[:,0].copy(),y.iloc[:,0].copy()]).to_numpy()
        diff_Xt = self.calcDiff(pd.concat([val_X.copy(),y.copy()]))
        Error_X = [val_Xt[0] - pred_init]
        Error_sav = [val_Xt[0] - pred_init]
        twelve = pd.DataFrame(columns=['Time','p','pX','d','q','P','D','Q','y'])

        x_update_T = {'Time':save_time,'p':np.array([0.01]),'pX':[np.array([0.01])]*3,'d':np.array([0.01]),'q':np.array([0.01]),'P':np.array([0.01]),'D':np.array([0.01]),'Q':np.array([0.01]),'y':pred_init}
        twelve = twelve.append(x_update_T, ignore_index=True)

        ## initialize variables
        val_X_exogt = None
        diff_X_exogt = []
        Error_X_exogt = []
        if exog_flag:
            val_X_exogt = pd.concat([val_X_exog.copy(),y_exog.copy()]).to_numpy()
            twelve_x = [pd.DataFrame(columns=['Time','p','pX','d','q','P','D','Q','y'])]*3
            
            for i in range(3):
                x_update_exog_T = {'Time':save_time,'p':np.array([0.01]),'pX':[],'d':np.array([0.01]),'q':np.array([0.01]),'P':np.array([0.01]),'D':np.array([0.01]),'Q':np.array([0.01]),'y':np.array(pred_init_exog[i])}
                diff_X_exogt.append(self.calcDiff(val_X_exog.iloc[:,[i]]))
                Error_X_exogt.append([val_X_exogt[0] - pred_init_exog[i]])
                if i==2:
                    dir_update_exog_T = {'Time':save_time,'p':np.array([0.001]),'pX':[],'d':np.array([0.001]),'q':np.array([0.001]),'P':np.array([0.001]),'D':np.array([0.001]),'Q':np.array([0.001]),'y':np.array(pred_init_exog[i])}
                    twelve_x[i] = twelve_x[i].append(dir_update_exog_T, ignore_index=True)
                else: twelve_x[i] = twelve_x[i].append(x_update_exog_T, ignore_index=True)
            diff_X_exogt = np.array(diff_X_exogt)


        # save to DF
        sav_item = {'Time': save_time,
                    't': 0,
                    's': 0,
                    'Predict': pred_init,
                    'Actual': val_Xt[0]}
        pred_sav = pred_sav.append(sav_item, ignore_index=True, sort=False)
        save_time += pd.Timedelta(hours=6)


        for t in tqdm(range(1, len(val_Xt))):
            if learn: lr *= lr_decay

            # limit viewing
            cur_Xt = val_Xt[:t]
            cur_diff_Xt = diff_Xt[:t]

            # do exog first
            cur_X_exogt = [[]]*3
            cur_diff_X_exogt = []
            pred_exog = [0]*3
            x_update_exog = [x_update_exog_T]*3
            error_X_exog = [0]*3
            if exog_flag:
                for i in range(3):
                    cur_X_exogt[i] = list(val_X_exogt[:, i])

                    cur_diff_X_exogt.append(diff_X_exogt[i, :t])

                    # input 't' to predict 't-1'
                    pred_exog[i], x_update_exog[i] = model_exo[i].predict_one(cur_X_exogt[i], cur_diff_X_exogt[i], t+1, Error_X_exogt[i])

                    error_X_exog[i] = 0

                    if (t >= step+1):
                        li_x_update_exo_tmp = twelve_x[i][twelve_x[i]['Time']==(save_time-pd.Timedelta(hours=6))].copy()
                        
                        li_x_update_exo = {'Time':save_time-pd.Timedelta(hours=6)}

                        # del 'this time' rows
                        twelve_x[i] = twelve_x[i][twelve_x[i]['Time']!=(save_time-pd.Timedelta(hours=6))]

                        for key in li_x_update_exo_tmp:
                            if (key=='Time'): continue
                            li_x_update_exo[key] = np.array(li_x_update_exo_tmp[key]).mean()
                        
                        error_X_exog[i] = cur_X_exogt[i][-2] - li_x_update_exo['y']
                        

                        # learning
                        if learn:
                            li_x_update_exo['pX'] = 0
                            model_exo[i].update_params(li_x_update_exo, error_X_exog[i], lr[i+1])
                    
                    # append Error
                    Error_X_exogt[i].append(error_X_exog[i])
                        

                    x_update_exog[i]['Time'] = save_time
                    x_update_exog[i]['y'] = pred_exog[i]
                    twelve_x[i] = twelve_x[i].append(x_update_exog[i], ignore_index=True)
                
                cur_X_exogt = np.array(cur_X_exogt).reshape((-1, 3))
            


            # then do main thing
            # input 't' to predict 't-1'
            pred, x_update = self.predict_one(cur_Xt, cur_diff_Xt, t+1, Error_X, X_train_exog=cur_X_exogt)

            error_X = 0
            
            if (t >= step+1):
                li_x_update_tmp = twelve[twelve['Time']==save_time-pd.Timedelta(hours=6)]

                # del 'this time' rows
                twelve = twelve[twelve['Time']!=save_time-pd.Timedelta(hours=6)]
                
                li_x_update = {'Time':save_time-pd.Timedelta(hours=6)}

                for key in li_x_update_tmp:
                    if (key=='Time'): continue
                    li_x_update[key] = np.array(li_x_update_tmp[key]).mean(axis=0)

                error_X = cur_Xt[-2] - li_x_update['y']
                
                # learning
                if learn:
                    self.update_params(li_x_update, error_X, lr[0])
                
            # append Error
            Error_X.append(error_X)
            Error_sav.append(error_X)
            
            x_update['Time'] = save_time
            x_update['y'] = pred
            twelve = twelve.append(x_update, ignore_index=True)
           
            # save to DF
            sav_item = {'Time': save_time,
                        't': t,
                        's': 0,
                        'Predict': pred,
                        'Actual': val_Xt[t]}
            if (save_time < start_time_y):
                pred_sav = pred_sav.append(sav_item, ignore_index=True, sort=False)
            elif (start_time_y <= save_time and save_time <= end_time_y):
                y_pred_sav = y_pred_sav.append(sav_item, ignore_index=True, sort=False)
            
            save_time += pd.Timedelta(hours=6)
        
            # change at a new 't'
            save_time_sub = save_time

            cur_pred = [pred]
            diff_cur_pred = [pred - cur_Xt[-1]]

            # exog too
            cur_pred_X_exogt = []
            cur_pred_diff_X_exogt = []
            if exog_flag:
                for i in range(3):
                    cur_pred_X_exogt.append([pred_exog[i]])
                    cur_pred_diff_X_exogt.append([pred_exog[i] - cur_X_exogt[-1][i]])


            for s in range(1, step):
                
                cur_X_s = list(cur_Xt) + cur_pred
                
                # append diff
                diff_Xt_s = list(cur_diff_Xt) + diff_cur_pred
                diff_Xt_s.append(cur_X_s[-1] - cur_X_s[-2])

                # do exog first
                cur_X_exog_s = [[]]*3
                cur_diff_X_exog_s = [[]]*3
                pred_exog = [0]*3 ; x_update_exog = [x_update_exog_T]*3 ; error_X_exog = [0]*3
                if exog_flag:
                    for i in range(3):
                        cur_X_exog_s[i] = list(cur_X_exogt[:, i]) + list(cur_pred_X_exogt[i])

                        # append diff
                        cur_diff_X_exog_s[i] = list(cur_diff_X_exogt[i]) + cur_pred_diff_X_exogt[i]

                        cur_diff_X_exog_s[i].append(cur_X_exog_s[i][-1] - cur_X_exog_s[i][-2])

                        # input 't' to predict 't-1'
                        pred_exog[i], x_update_exog[i] = model_exo[i].predict_one(cur_X_exog_s[i], cur_diff_X_exog_s[i], t+1)

                        cur_pred_X_exogt[i].append(pred_exog[i])
                            
                        x_update_exog[i]['Time'] = save_time_sub
                        x_update_exog[i]['y'] = pred_exog[i]
                        twelve_x[i] = twelve_x[i].append(x_update_exog[i], ignore_index=True)


                # then do main thing
                # input 't' to predict 't-1'
                pred, x_update = self.predict_one(cur_X_s, diff_Xt_s, t+s+1, X_train_exog=np.moveaxis(cur_X_exog_s, -1, 0))

                x_update['Time'] = save_time_sub
                x_update['y'] = pred
                twelve = twelve.append(x_update, ignore_index=True)

                # save to DF
                if (t+s<val_Xt.shape[0]):
                    sav_item = {'Time': save_time_sub,
                                't': t,
                                's': s,
                                'Predict': pred,
                                'Actual': val_Xt[t+s]}
                    if (save_time_sub < start_time_y):
                        pred_sav = pred_sav.append(sav_item, ignore_index=True, sort=False)
                    elif (start_time_y <= save_time_sub and save_time_sub <= end_time_y):
                        y_pred_sav = y_pred_sav.append(sav_item, ignore_index=True, sort=False)
                
                save_time_sub += pd.Timedelta(hours=6)
                
                cur_pred.append(pred)
        

        return  (pred_sav, y_pred_sav, np.array(Error_sav))


    def calcDiff(self, X):
        diff = X.copy()
        diff = diff.iloc[:,0].diff()
        diff = diff.fillna(0)
        return diff.to_numpy()
    
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