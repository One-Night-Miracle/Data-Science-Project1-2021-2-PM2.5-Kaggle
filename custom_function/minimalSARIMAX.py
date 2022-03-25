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

        self.params['p'] = [random.uniform(0, 0.1)]*self.p if self.p else [0]
        self.params['pX'] = [random.uniform(0, 0.1)]*self.X_train_exog.shape[1] if (exog is not None) else [0]
        self.params['d'] = [random.uniform(0, 0.1)]*self.d if self.d else [0]
        self.params['q'] = [random.uniform(0, 0.1)]*self.q if self.q else [0]

        self.params['P'] = [random.uniform(0, 0.1)]*self.P if self.P else [0]
        self.params['D'] = [random.uniform(0, 0.1)]*self.D if self.D else [0]
        self.params['Q'] = [random.uniform(0, 0.1)]*self.Q if self.Q else [0]
        self.params['c'] = random.uniform(0, 0.1)

        self.params['P'] = [random.uniform(0,0.1)]*self.P if self.P else [0]
        self.params['D'] = [random.uniform(0,0.1)]*self.D if self.D else [0]
        self.params['Q'] = [random.uniform(0,0.1)]*self.Q if self.Q else [0]
        self.params['c'] = random.uniform(0,0.1)

        self.Error_X = None
    
    #############################################################################
    
    def p_prediction(self, X_train, t):
        if (t <= 0): return [0], [0]
        start = max(t-self.p, 0)
        X_train_t = np.array(X_train[start:t])[::-1].reshape((-1))
        
        params_p = np.array(self.params['p'][:t-start])[::-1]
        p_pred = X_train_t @ params_p[:X_train_t.shape[0]]

        return p_pred, X_train_t[::-1]

    def pX_prediction(self, X_train_exog, t):
        if (t <= 0): return [0], [0]
        X_train_exog_t = np.array(X_train_exog[t-1]).reshape((-1))
        params_pX = np.array(self.params['pX'])
        
        pX_pred = X_train_exog_t @ params_pX

        return pX_pred, X_train_exog_t

    def d_prediction(self, diff_X, t):
        if (self.d == 0 or t <= 0):
            return [0], [0]
        start = max(t-self.d, 0)
        diff_X_t = np.array(diff_X[start:t])[::-1].reshape((-1))
        params_d = np.array(self.params['d'][:t-start])[::-1]

        d_pred = diff_X_t @ params_d[:diff_X_t.shape[0]]

        return d_pred, diff_X_t[::-1]

    def q_prediction(self, Error, t):
        if (self.q == 0 or t <= 0):
            return [0], [0]
        start = max(t-self.q, 0)

        error_t = np.array(Error[start:t])[::-1].reshape((-1))
        params_q = np.array(self.params['q'][:t-start])[::-1]

        print(error_t, params_q)
        q_pred = error_t @ params_q

        return q_pred, error_t[::-1]
    
    #############################################################################

    def P_prediction(self, X_train, t):
        ss_c = t//self.S  # season_count
        if (ss_c == 0 or self.params['P'] == [0] or t <= 0):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        X_train_ts = np.array(X_train[t-self.S::-self.S])[:ss_c].reshape((-1))
        params_P = np.array(self.params['P'][::-1])[:ss_c]
        
        P_pred = X_train_ts @ params_P[:X_train_ts.shape[0]]

        return P_pred, X_train_ts[::-1]

    def D_prediction(self, X_train, t):
        ss_c = t-1//self.S  # season_count
        if (ss_c == 0 or self.params['D'] == [0] or t <= 0):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        diff_X_ts = np.array(X_train[t-1-self.S::-self.S])[:1].reshape((-1))
        params_D = np.array(self.params['D'])
        
        D_pred = diff_X_ts @ params_D[:diff_X_ts.shape[0]]

        return D_pred, diff_X_ts[::-1]

    def Q_prediction(self, Error, t):
        ss_c = t//self.S  # season_count
        if (ss_c == 0 or len(self.params['Q']) == 0 or t <= 0):
            return [0], [0]
        ss_c = min(ss_c, self.P)
        error_ts = np.array(Error[t-self.S::-self.S])[:ss_c].ravel()
        params_Q = np.array(self.params['Q'][::-1])[:ss_c]

        Q_pred = error_ts @ params_Q[:error_ts.shape[0]]

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

        self.params['d'] += param_pad_0(x['d'], self.d) * error_t * lr

        self.params['q'] += param_pad_0(x['q'], self.q) * error_t * lr

        self.params['P'] += param_pad_0(x['P'], self.P) * error_t * lr

        self.params['D'] += param_pad_0(x['D'], self.D) * error_t * lr

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
    
    def predict_one(self, X_train, diff_X, t, Error_X=None, X_train_exog=None, model_out=None):
    
        if model_out is None:
            model_out = self

        pred = {} ; x = {}

        pred['p'], x['p'] = model_out.p_prediction(X_train, t-1)

        if (X_train_exog is None or not X_train_exog.all()):
            pred['pX'], x['pX'] = ([0], [0])
        else:
            # print(X_train_exog)
            pred['pX'], x['pX'] = model_out.pX_prediction(X_train_exog, t-1) 
            

        pred['d'], x['d'] = model_out.d_prediction(diff_X, t-1)
        pred['q'], x['q'] = model_out.q_prediction(
            Error_X, t) if Error_X is not None else ([0], [0])

        pred['P'], x['P'] = model_out.P_prediction(X_train, t-1)
        pred['D'], x['D'] = model_out.D_prediction(X_train, t-1)
        pred['Q'], x['Q'] = model_out.Q_prediction(Error_X, t-1) if Error_X is not None else ([0], [0])

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
            pred['pX'], x['pX'] = self.pX_prediction(y_exog_t, t) if y_exog is not None else ([0], [0])
            pred['d'], x['d'] = self.d_prediction(diff_y, t)
            pred['q'], x['q'] = self.q_prediction(Error, t)
            
            pred['P'], x['P'] = self.P_prediction(y_t, t)
            pred['D'], x['D'] = self.D_prediction(y_t, t)
            pred['Q'], x['Q'] = self.Q_prediction(Error, t)

            pred['y'] = (pred['p'] + pred['pX'] + pred['d'] + pred['q'] + pred['P'] + pred['Q'] + pred['D'] + self.params['c']).sum()
            
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

    def predict_step(self, val_X, y, val_X_exog=None, y_exog=None, model_exog=None, model_pd=None, step=12, learn=False, lr=np.array([1e-6]), lr_decay=0.999):
        exog_flag = False
        if val_X_exog is not None:
            exog_flag = True
            model_exo = []
            for key in model_exog:
                model_exo.append(model_exog[key])
        
        if model_pd is None:
            model_pd = self
        

        save_time_v = val_X.index[0]

        save_time_y = y.index[0]
        end_time_y = y.index[-1]

        val_pred_sav = pd.DataFrame(columns=['Time', 'Predict', 'Actual'])
        y_pred_sav = pd.DataFrame(columns=['Time', 'Predict', 'Actual'])
        
        ## random init
        pred_init = random.randint(5,20)
        pred_init_exog = [30, 25, 180]

        # initialize val variables
        val_Xt = val_X.iloc[:,0].copy().to_numpy()
        diff_Xt = self.calcDiff(val_X)
        Error_X = [val_Xt[0] - pred_init]
        Valid_Error = [val_Xt[0] - pred_init]
        twelve = pd.DataFrame(columns=['Time','p','pX','d','q','P','D','Q','y'])

        x_update_T = {'Time':save_time_v,'p':0.01,'pX':[[0.01]]*3,'d':0.01,'q':0.01,'P':0.01,'D':0.01,'Q':0.01,'y':pred_init}
        twelve = twelve.append(x_update_T, ignore_index=True)

        ## initialize val_exog variables
        val_X_exogt = None
        diff_X_exogt = []
        Error_X_exogt = []
        if exog_flag:
            val_X_exogt = val_X_exog.copy().to_numpy()
            twelve_x = [pd.DataFrame(columns=['Time','p','pX','d','q','P','D','Q','y'])]*3
            for i in range(3):
                diff_X_exogt.append(self.calcDiff(val_X_exog.iloc[:,[i]]))
                Error_X_exogt.append([val_X_exogt[0][i] - pred_init_exog[i]])
                x_update_exog_T = {'Time':save_time_v,'p':0.01,'pX':0,'d':0.01,'q':0.01,'P':0.01,'D':0.01,'Q':0.01,'y':pred_init_exog}
                twelve_x[i] = twelve_x[i].append(x_update_exog_T, ignore_index=True)
            diff_X_exogt = np.array(diff_X_exogt)
            


        # save to DF
        sav_item = {'Time': save_time_v,
                    'Predict': pred_init,
                    'Actual': val_Xt[0]}
        val_pred_sav = val_pred_sav.append(
            sav_item, ignore_index=True, sort=False)
        save_time_v += pd.Timedelta(hours=6)

        for t in tqdm(range(1, len(val_Xt))):
            if learn: lr *= lr_decay

            # limit viewing
            cur_Xt = val_Xt[:t]
            cur_diff_Xt = diff_Xt[:t]

            # do exog first
            cur_X_exogt = []
            cur_diff_X_exogt = []
            pred_exog = [0]*3
            x_update_exog = [{}]*3
            error_X_exog = [0]*3
            if exog_flag:
                for i in range(3):
                    cur_X_exogt.append(val_X_exogt[:t, i])
                    cur_diff_X_exogt.append(diff_X_exogt[i, :t])

                    # input 't' to predict 't-1'
                    pred_exog[i], x_update_exog[i] = self.predict_one(cur_X_exogt[:][i], cur_diff_X_exogt[i], t+1, Error_X_exogt[i], model_out=model_exo[i])

                    if (t >= 13):
                        li_x_update_exo = twelve_x[i][twelve_x[i]['Time']==save_time_v-pd.Timedelta(hours=6)].copy()
                        li_x_update_exo = li_x_update_exo.mean(axis=0)

                        # del 'this time' rows
                        twelve_x[i] = twelve_x[i][twelve_x[i]['Time']!=save_time_v-pd.Timedelta(hours=6)]

                        error_X_exog[i] = cur_X_exogt[-2][i] - li_x_update_exo['y']
                        # append Error
                        Error_X_exogt[i].append(error_X_exog[i])

                        # learning
                        if learn:
                            model_exo[i].update_params(li_x_update_exo.iloc[0], error_X_exog[i], lr[i+1])
                    
                    x_update_exog[i]['Time'] = save_time_v
                    x_update_exog[i]['y'] = pred_exog[i]
                    twelve_x[i] = twelve_x[i].append(x_update_exog[i], ignore_index=True)
                
                cur_X_exogt = np.array(cur_X_exogt).reshape((-1, 3))
            


            # then do main thing
            # input 't' to predict 't-1'
            pred, x_update = self.predict_one(cur_Xt, cur_diff_Xt, t+1, Error_X, X_train_exog=cur_X_exogt, model_out=model_pd)
            
            if (t >= 13):
                li_x_update = twelve[twelve['Time']==save_time_v-pd.Timedelta(hours=6)]
                li_x_update = li_x_update.mean(axis=0)

                # del 'this time' rows
                twelve = twelve[twelve['Time']!=save_time_v-pd.Timedelta(hours=6)]

                error_X = cur_Xt[-2] - li_x_update['y']
                # append Error
                Error_X.append(error_X)
                Valid_Error.append(error_X)

                # learning
                if learn:
                    self.update_params(li_x_update, error_X, lr[0])
            
            x_update['Time'] = save_time_v
            x_update['y'] = pred
            twelve = twelve.append(x_update, ignore_index=True)
           
            # save to DF
            sav_item = {'Time': save_time_v,
                        'Predict': pred,
                        'Actual': val_Xt[t]}
            val_pred_sav = val_pred_sav.append(sav_item, ignore_index=True, sort=False)
            save_time_v += pd.Timedelta(hours=6)

            # change at a new 't'
            save_time_v_sub = save_time_v

            cur_pred = [pred]
            diff_cur_pred = [pred - cur_Xt[-2]]

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
                pred_exog = [0]*3 ; x_update_exog = [0]*3 ; error_X_exog = [0]*3
                if exog_flag:
                    for i in range(3):
                        cur_X_exog_s[i] = list(cur_X_exogt[:, i]) + list(cur_pred_X_exogt[i])

                        # append diff
                        cur_diff_X_exog_s[i] = list(cur_diff_X_exogt[i]) + cur_pred_diff_X_exogt[i]

                        cur_diff_X_exog_s[i].append(cur_X_exog_s[i][-1] - cur_X_exog_s[i][-2])

                        # input 't' to predict 't-1'
                        pred_exog[i], x_update_exog[i] = model_exo[i].predict_one(cur_X_exog_s[i], cur_diff_X_exog_s[i], t+1)

                        cur_pred_X_exogt[i].append(pred_exog[i])

                        
                            # model_exo[i].update_params(x_update_exog[i], error_X_exog[i], lr[i+1])
                        if (t >= 13):
                            li_x_update_exo = twelve_x[i][twelve_x[i]['Time']==save_time_v_sub-pd.Timedelta(hours=6)].copy()
                            li_x_update_exo = li_x_update_exo.mean(axis=0)

                            # del 'this time' rows
                            twelve_x[i] = twelve_x[i][twelve_x[i]['Time']!=save_time_v_sub-pd.Timedelta(hours=6)]

                            error_X_exog[i] = cur_X_exogt[-2][i] - li_x_update_exo['y']
                            # append Error
                            Error_X_exogt[i].append(error_X_exog[i])
                            
                            # learning
                            if learn:
                                model_exo[i].update_params(li_x_update_exo.iloc[0], cur_X_exog_s[-1][i] - li_x_update_exo['y'], 1.0012*lr[i+1])
                            
                        x_update_exog[i]['Time'] = save_time_v_sub
                        x_update_exog[i]['y'] = pred_exog[i]
                        twelve_x[i] = twelve_x[i].append(x_update_exog[i], ignore_index=True)
                        twelve_x[i].save(x_update_exog[i])



                # then do main thing
                # input 't' to predict 't-1'
                pred, x_update, error_X = self.predict_one(
                    cur_X_s, diff_Xt_s, t+s, X_train_exog=np.moveaxis(cur_X_exog_s, -1, 0), model_out=model_pd)


                # learning 11 steps
                if learn:
                    if (t >= 12):
                        li_x_update = twelve.read()
                        self.update_params(li_x_update, cur_X_s[-1] - li_x_update['y'], 1.0012*lr[0])
                    twelve.save(x_update)
                    

                # append Error
                Valid_Error.append(error_X)
                
                cur_pred.append(pred['y'])
                
                ## save to DF
                if (save_time_v_sub < save_time_y):
                    sav_item = {'Time': save_time_v_sub,
                                'Predict': pred['y'],
                                'Actual': val_Xt[t+s]}
                    val_pred_sav = val_pred_sav.append(sav_item, ignore_index=True, sort=False)
                save_time_v_sub += pd.Timedelta(hours=6)
        

        #initialize y variables
        yt = y.iloc[:,0].copy().to_numpy()
        diff_yt = self.calcDiff(y)
        Error_y = Error_X + [yt[0] - list(val_Xt)[-1]]

        Test_Error = [yt[0] - list(val_Xt)[-1]]


        ## initialize y_exog variables
        y_exogt = None
        diff_y_exogt = []
        Error_y_exogt = [[]]*3
        if exog_flag:
            y_exogt = y_exog.copy().to_numpy()
            for i in range(3):
                diff_y_exogt.append(self.calcDiff(y_exog.iloc[:,[i]]))
                Error_y_exogt[i] = Error_X_exogt[i] + [y_exogt[0][i] - val_X_exogt[0][i]]
            diff_y_exogt = np.array(diff_y_exogt)


        # save to DF
        # sav_item = {'Time': save_time_v,
        #             'Predict': pred_init,
        #             'Actual': val_Xt[0]}
        # val_pred_sav = val_pred_sav.append(
        #     sav_item, ignore_index=True, sort=False)
        # save_time_v += pd.Timedelta(hours=6)

        for t in tqdm(range(val_Xt.shape[0], val_Xt.shape[0] + y.shape[0])):
            # limit viewing
            cur_yt = (list(val_Xt) + list(yt))[:t]
            cur_diff_yt = (list(diff_Xt) + list(diff_yt))[:t]

            # do exog first
            cur_y_exogt = []
            cur_diff_y_exogt = []
            pred_exog = [0]*3 ; y_update_exog = [0]*3 ; error_y_exog = [0]*3
            if exog_flag:
                for i in range(3):
                    cur_y_exogt.append(list(val_Xt) + list(y_exogt[:t-val_Xt.shape[0], i]))
                    cur_diff_y_exogt.append(list(diff_Xt) + list(diff_y_exogt[i, :t-val_Xt.shape[0]]))

                    pred_exog[i], y_update_exog[i] = model_exo[i].predict_one(cur_y_exogt[:][i], cur_diff_y_exogt[i], t+1, Error_y_exogt[i])

                    if (t >= 13):
                        li_y_update_exo = twelve_x[i][twelve_x[i]['Time']==save_time_y-pd.Timedelta(hours=6)].copy()
                        li_y_update_exo = li_y_update_exo.mean(axis=0)

                        # del 'this time' rows
                        twelve_x[i] = twelve_x[i][twelve_x[i]['Time']!=save_time_y-pd.Timedelta(hours=6)]

                        error_y_exog[i] = cur_y_exogt[-2][i] - li_y_update_exo['y']
                        # append Error
                        Error_y_exogt[i].append(error_y_exog[i])

                        # learning
                        if learn:
                            model_exo[i].update_params(li_y_update_exo.iloc[0], error_y_exog[i], lr[i+1])
                    
                    y_update_exog[i]['Time'] = save_time_y
                    y_update_exog[i]['y'] = pred_exog[i]
                    twelve_x[i] = twelve_x[i].append(y_update_exog[i], ignore_index=True)

                cur_y_exogt = np.array(cur_y_exogt).reshape((-1, 3))


            # then do main thing
            # input 't' to predict 't-1'
            pred, y_update = self.predict_one(cur_yt, cur_diff_yt, t+1, Error_y, X_train_exog=cur_y_exogt, model_out=model_pd) 
            
            if (t >= 13):
                li_y_update = twelve[twelve['Time']==save_time_y-pd.Timedelta(hours=6)]
                li_y_update = li_y_update.mean(axis=0)

                # del 'this time' rows
                twelve = twelve[twelve['Time']!=save_time_y-pd.Timedelta(hours=6)]

                error_y = cur_yt[-2] - li_y_update['y']
                # append Error
                Error_y.append(error_y)
                Test_Error.append(error_y)

                # learning
                if learn:
                    self.update_params(li_x_update, error_X, lr[0])
            
            x_update['Time'] = save_time_v
            x_update['y'] = pred
            twelve = twelve.append(x_update, ignore_index=True)
           
            # save to DF
            sav_item = {'Time': save_time_v,
                        'Predict': pred,
                        'Actual': val_Xt[t]}
            val_pred_sav = val_pred_sav.append(sav_item, ignore_index=True, sort=False)
            save_time_v += pd.Timedelta(hours=6)

            # learning
            if learn:
                if (t >= 13):
                    li_y_update = twelve.read()
                    self.update_params(li_y_update, cur_yt[-1] - li_y_update['y'], lr[0])
                twelve.save(y_update)


            # append Error
            Error_y.append(error_y)
            Test_Error.append(error_y)

            # save to DF
            sav_item = {'Time': save_time_y,
                        'Predict': pred,
                        'Actual': yt[t-val_Xt.shape[0]]}
            y_pred_sav = y_pred_sav.append(
                sav_item, ignore_index=True, sort=False)
            save_time_y += pd.Timedelta(hours=6)

            # change at a new 't'
            save_time_y_sub = save_time_y

            cur_pred = [pred]
            diff_cur_pred = [pred-cur_yt[-1]]

            ## exog too
            cur_pred_y_exogt = []
            cur_pred_diff_y_exogt = []
            if exog_flag:
                for i in range(3):
                    cur_pred_y_exogt.append([pred_exog[i]['y']])
                    cur_pred_diff_y_exogt.append([pred_exog[i]['y'] - cur_y_exogt[-1][i]])

            for s in range(1,step):
                if learn:
                    lr *= lr_decay

                cur_yt_s = list(cur_yt) + cur_pred
                
                # append diff
                diff_yt_s = list(diff_yt) + diff_cur_pred
                diff_yt_s.append(cur_yt_s[-1]-cur_yt_s[-2])


                # do exog first
                cur_y_exog_s = [[]]*3
                cur_diff_y_exog_s = [[]]*3
                pred_exog = [0]*3 ; y_update_exog = [0]*3 ; error_y_exog = [0]*3
                if exog_flag:
                    for i in range(3):
                        cur_y_exog_s[i] = list(cur_y_exogt[:,i]) + list(cur_pred_y_exogt[i])

                        # append diff
                        cur_diff_y_exog_s[i] = list(cur_diff_y_exogt[i]) + cur_pred_diff_y_exogt[i]
                        cur_diff_y_exog_s[i].append(cur_y_exog_s[i][-1] - cur_y_exog_s[i][-2])
                        
                        pred_exog[i], y_update_exog[i], error_y_exog[i] = self.predict_one(cur_y_exog_s[i], cur_diff_y_exog_s[i], t, model_out=model_exo[i])

                        cur_pred_y_exogt[i].append(pred_exog[i]['y'])
                    
                        ## learning
                        if learn:
                            if (t >= 13):
                                li_y_update_exo = twelve_x[i].read()
                                model_exo[i].update_params(li_y_update_exo, cur_y_exog_s[-1][i] - li_y_update_exo['y'], 1.0012*lr[i+1])
                            twelve_x[i].save(y_update_exog[i])


                # then do main thing
                pred, y_update, error_y = self.predict_one(
                    cur_yt_s, diff_yt_s, t+s, X_train_exog=np.moveaxis(cur_y_exog_s, -1, 0), model_out=model_pd)
                

                # learning 11 steps
                # if learn:
                #     if (t >= 12):
                #         li_y_update = twelve.read()
                #         self.update_params(li_y_update, cur_yt_s[-1] - li_y_update['y'], 1.0012*lr[0])
                #     twelve.save(x_update)

                # learning 11 steps
                if learn:
                    self.update_params(y_update, error_y, lr[0])

                # append Error
                Test_Error.append(error_y)
                
                cur_pred.append(pred['y'])
                
                # save to DF
                if (save_time_y_sub <= end_time_y and
                        t+s-val_Xt.shape[0] < y.shape[0]):
                    sav_item = {'Time': save_time_y_sub,
                                'Predict': pred['y'],
                                'Actual': yt[t+s-val_Xt.shape[0]]}
                    y_pred_sav = y_pred_sav.append(sav_item, ignore_index=True, sort=False)
                    save_time_y_sub += pd.Timedelta(hours=6)

        return  (val_pred_sav, y_pred_sav, np.array(Valid_Error), np.array(Test_Error))


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