import numpy as np
import pandas as pd
import xgboost as xgb
import math
import matplotlib.pyplot as plt
import csv
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import model_from_json
from itertools import chain
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import pearsonr
import statsmodels.api as sm
global mean_ts, sd_ts, res_sbs, res_notsbs
import os
from itertools import chain
np.random.seed(1234)


def load_data(reverse, path_to_dataset, index = 1, limit = None, data_sep = ";"):
    global data_mean, data_sd
    num_errors = 0
    with open(path_to_dataset) as f:
        dataread = csv.reader(f, delimiter=data_sep)
        data = []
        for line in dataread:
            try:
                data.append(float(line[index]))
            except ValueError:
                num_errors += num_errors
                pass
    print("There were", num_errors, "conversion errors while reading the input file!")            
    if limit is None:
        limit = len(data)
    data = data[:limit]
    
    if reverse == True:
        data = list(reversed(data)) # data must be sorted from old to new
     
    # Standardize (Z-tranfo)
    holdout_len = round(len(data)*0.15)
    data_holdout = data[(len(data)-holdout_len):len(data)]
    data = data[0:(len(data)-holdout_len)]
    data_mean = np.mean(data)
    data_sd = np.std(data)
    data = [x-data_mean for x in data] # subtract mean
    data = [x/data_sd for x in data] # divide by sd
    data_holdout = [x-data_mean for x in data_holdout] # subtract mean
    data_holdout = [x/data_sd for x in data_holdout] # divide by sd       
    return data, data_holdout
	
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
	
def run_network(data, model_filename_base, num_epochs = 15, seqlens = range(50, 105, 5), dropout_rates = [0.3],
               num_hl = [2], layers = ["standard"], batch = 32, loss_function = "mse", lrs = [0.001]):
    for current_seq_length in seqlens:
        for current_dropout in dropout_rates:
            for current_num_hl in num_hl:
                for current_layer in layers:
                    for current_lr in lrs:
                        current_name = "models/ipy/"+model_filename_base+"_"+str(current_seq_length)+"_"+str(current_dropout)+"_hl_"+str(current_num_hl)+"_"+str(current_layer)+"_"+str(current_lr)
                        if os.path.isfile(current_name+".json"):
                            print("File "+current_name+" exists already. Skipping.")
                            continue
                        train = []
                        num_fc = current_seq_length
                        # create sequences of length sequence_length
                        for index in range(len(data)-current_seq_length-num_fc):
                            train.append(data[index:index+current_seq_length+num_fc])
                        train = np.array(train) 
                        np.random.shuffle(train)
                        X_train = train[:, :-num_fc]
                        y_train = train[:, -num_fc:]
                        #  (#examples, #values in sequences, dim. of each value)
                        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(timestamp, "- Building model now with ", len(X_train), 
                              " training sequences, dr="+str(current_dropout), ", numHL="+str(current_num_hl), ", nodes="+str(current_layer),
                              ", lr="+str(current_lr), ", seqlen="+str(current_seq_length))

                        model = Sequential()
                        if current_layer is "standard":
                            layers_size = [1]
                            for hl in range(current_num_hl+1): # +1 because first layer is input layer, not hidden layer
                                layers_size.extend([current_seq_length])
                            layers_size.extend([1])
                        if current_layer is "half":
                            layers_size = [1]
                            for hl in range(current_num_hl+1):
                                layers_size.extend([int(current_seq_length/2)])
                            layers_size.extend([1])
                        if current_layer is "quarter": # TODO!!!
                            layers_size = [1]
                            for hl in range(current_num_hl+1):
                                layers_size.extend([int(current_seq_length/4)])
                            layers_size.extend([1])

                        model.add(LSTM(input_shape=(None, layers_size[0]), units=layers_size[1], return_sequences=True))
                        model.add(Dropout(current_dropout))
                        # hidden layers
                        for hl in range(current_num_hl+1): # +1 because first layer is input layer, not hidden layer
                            model.add(LSTM(layers_size[hl+1], return_sequences=True))
                            model.add(Dropout(current_dropout))
                        # output layer
                        model.add(TimeDistributed(Dense(units=layers_size[-1])))
                        model.add(Activation("linear"))

                        rms = RMSprop(lr=current_lr) # default 0.001
                        model.compile(loss=loss_function, optimizer=rms)
                        model.fit(X_train, y_train, batch_size=batch, epochs=num_epochs, validation_split=0.1, verbose = False)
                        # store model to disk as JSON
                        model_json = model.to_json()
                        with open(current_name+".json", "w") as json_file:
                            json_file.write(model_json)
                        model.save_weights(current_name+".h5")
						
def get_average_pairwise_correlation(forecasts):
    forecasts = np.reshape(forecasts, (forecasts.shape[0], forecasts.shape[1]*forecasts.shape[2]))
    correlations = []
    for index in range(len(forecasts)):
        for index2 in range(len(forecasts)):
            if index < index2:
                correlations.append(pearsonr(forecasts[index], forecasts[index2])[0])
    return np.mean(correlations)
	
def simple_mean_forecast(meta_test_seqs, meta_test_actual):
    i = 0
    preds = []
    actuals = []
    for test_seq in meta_test_seqs:
        #seq_mean = test_seq[-1] # this is LOCF
        input_vals = test_seq
        curr_fc = 0
        for step in range(50):
            curr_fc = np.mean(input_vals)
            preds.append(curr_fc)
            input_vals = np.append(input_vals, curr_fc) # add last known (estimated) value
            input_vals = input_vals[1:] # remove oldest value
        #seq_mean = np.mean(test_seq)
        #preds.extend(np.repeat(seq_mean, 50))
        actuals.extend(meta_test_actual[i])
        i+=1
    return rmse(np.array(preds)*data_sd+data_mean, np.array(actuals)*data_sd+data_mean)

def exp_smoothing_forecast(meta_test_seqs, meta_test_actual, alphas):
    best_rmse = None
    best_alpha = None
    for alpha in alphas:
        i = 0
        preds = []
        actuals = []
        for test_seq in meta_test_seqs:
            input_vals = test_seq
            curr_fc = 0
            for step in range(50):
                curr_fc += alpha*(1-alpha)**step*input_vals[-step-1]
                preds.append(curr_fc)
                input_vals = np.append(input_vals, curr_fc) # add last known (estimated) value
                input_vals = input_vals[1:] # remove oldest value
            actuals.extend(meta_test_actual[i])
            i+=1
        curr_rmse = rmse(np.array(preds)*data_sd+data_mean, np.array(actuals)*data_sd+data_mean)
        if best_rmse is None or curr_rmse < best_rmse:
            best_rmse = curr_rmse
            best_alpha = alpha
    print("Exponential Smoothing best alpha:", best_alpha)
    return best_rmse

def arima_forecast(meta_test_seqs, meta_test_actual):
    p_values = [2, 5, 10]
    q_values = [2, 5, 10]
    d_values = [0, 1, 2]
    best_rmse = None
    best_params = None
    for p in p_values:
        for q in q_values:
            for d in d_values:
                preds = []
                actuals = []
                i=0
                # one ARIMA model for each sequence to be forecasted
                for test_seq in meta_test_seqs[:10]:
                    try:
                        model = ARIMA(test_seq, order=(p, d, q)) #p,d,q
                        results_AR = model.fit(disp=-1)
                        forecasts = results_AR.predict(len(test_seq)-1, len(test_seq)+50-1)
                        forecasts = forecasts[1:] # omit first entry in the array since it is the forecast for the last value of the training set  
                        if len(forecasts)==50:
                            preds.extend(forecasts)
                            actuals.extend(meta_test_actual[i])
                    except:
                        pass
                    i+=1
                if len(preds) >= 50:
                    curr_rmse = rmse(np.array(preds)*data_sd+data_mean, np.array(actuals)*data_sd+data_mean)
                    if best_rmse is None or curr_rmse < best_rmse:
                        best_rmse = curr_rmse
                        best_params = [p, d, q]
                        
    # apply the best arima model to entire test data
    preds = []
    actuals = []
    i=0
    # one ARIMA model for each sequence to be forecasted
    for test_seq in meta_test_seqs:
        try:
            model = ARIMA(test_seq, order=(best_params[0], best_params[1], best_params[2])) #p,d,q
            results_AR = model.fit(disp=-1)
            forecasts = results_AR.predict(len(test_seq)-1, len(test_seq)+50-1)
            forecasts = forecasts[1:] # omit first entry in the array since it is the forecast for the last value of the training set  
            preds.extend(forecasts)
            actuals.extend(meta_test_actual[i])
        except:
            pass
        i+=1
    arima_rmse = rmse(np.array(preds)*data_sd+data_mean, np.array(actuals)*data_sd+data_mean)
    
    
    print("Best ARIMA model", best_params, "with RMSE =", arima_rmse)
    return arima_rmse 


def xgboost_forecast(meta_train, meta_test):
    xgb_train = meta_train
    xgb_test = meta_test
    # training data
    df_cols = []
    for index in range(len(xgb_train)-2*50):
        if len(df_cols) == 0: # init
            for i in range(50):
                df_cols.append(list(np.repeat(xgb_train[index+i], 50)))
            df_cols.append(list(range(1, 51, 1))) # feature "fc_step"
            df_cols.append(xgb_train[index+50:index+2*50]) # target y

        else:
            for i in range(50):
                df_cols[i].extend(list(np.repeat(xgb_train[index+i], 50)))
            df_cols[-2].extend(list(range(1, 51, 1))) # feature "fc_step"
            df_cols[-1].extend(xgb_train[index+50:index+2*50]) # target y
        
    # fill DF according to Sec. 4.1
    xgb_colnames = []
    for i in range(50):
        xgb_colnames.append("t-"+str(50-i-1))
    xgb_colnames.append("fc_step")
    xgb_colnames.append("actual")
    xgb_train_df = pd.DataFrame(columns = xgb_colnames, data=list(map(list, zip(*df_cols))))
    
    # testing data
    df_cols = []
    for index in range(len(xgb_test)-2*50):
        if len(df_cols) == 0: # init
            for i in range(50):
                df_cols.append(list(np.repeat(xgb_test[index+i], 50)))
            df_cols.append(list(range(1, 51, 1))) # feature "fc_step"
            df_cols.append(xgb_test[index+50:index+2*50]) # target y

        else:
            for i in range(50):
                df_cols[i].extend(list(np.repeat(xgb_test[index+i], 50)))
            df_cols[-2].extend(list(range(1, 51, 1))) # feature "fc_step"
            df_cols[-1].extend(xgb_test[index+50:index+2*50]) # target y
    
    # fill DF according to Sec. 4.1
    xgb_test_df = pd.DataFrame(columns = xgb_colnames, data=list(map(list, zip(*df_cols))))
    del(df_cols)

    # xgboost
    y_train = xgb_train_df['actual'].values
    y_mean = np.mean(y_train)
    xgb_params = {
        'n_trees': 200, 
        'eta': 0.0045,
        'max_depth': 4,
        'subsample': 0.9,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean, # base prediction = mean(target)
        'silent': 1
    }
    dtrain = xgb.DMatrix(xgb_train_df.drop('actual', axis=1), y_train)
    dtest = xgb.DMatrix(xgb_test_df.drop('actual', axis=1))

    num_boost_rounds = 200
    # train model
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, verbose_eval=1)
    y_pred = model.predict(dtest)
    rmse_xg = rmse(y_pred*data_sd+data_mean, xgb_test_df['actual'].values*data_sd+data_mean)
    return rmse_xg
	
def meta_forecast(meta_train_forecasts, meta_train_actual, meta_test_forecasts, meta_test_actual, num_trees = 100,
                  ridge_alphas = [20, 15, 10, 8, 6, 4, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 
                                  0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
                  meta_learner = "rf"):
    meta_train_X = None
    meta_train_y = []
    meta_test_X = None
    meta_test_y = []

    for model in meta_train_forecasts:
        model_seqs = []
        for seq in model:
            model_seqs.extend(seq)
        if meta_train_X is None:
            meta_train_X = model_seqs
        else:
            meta_train_X = np.vstack((meta_train_X, model_seqs))
    # add extra column indicating the respective forecast step 1, 2, ..., 50
    #num_repeat_steps =  int(meta_train_X.shape[1]/50)
    #feature_fc_steps = np.array(np.tile(list(range(1, 51)), num_repeat_steps))
    #feature_fc_steps = (feature_fc_steps-np.mean(feature_fc_steps))/np.std(feature_fc_steps)
    #meta_train_X = np.vstack((meta_train_X, feature_fc_steps))
        
    meta_train_X = meta_train_X.transpose()    

    for seq_actual in meta_train_actual:
        meta_train_y.extend(seq_actual)
    y = np.array(meta_train_y)
    #return meta_train_X
    
    
    # generate test data
    for model in meta_test_forecasts:
        model_seqs = []
        for seq in model:
            model_seqs.extend(seq)
        if meta_test_X is None:
            meta_test_X = model_seqs
        else:
            meta_test_X = np.vstack((meta_test_X, model_seqs))
    # add extra column indicating the respective forecast step 1, 2, ..., 50
    #num_repeat_steps =  int(meta_test_X.shape[1]/50)
    #feature_fc_steps = np.array(np.tile(list(range(1, 51)), num_repeat_steps))
    #feature_fc_steps = (feature_fc_steps-np.mean(feature_fc_steps))/np.std(feature_fc_steps)
    #meta_test_X = np.vstack((meta_test_X, feature_fc_steps))
    
    meta_test_X = meta_test_X.transpose()  
    
    for seq_actual in meta_test_actual:
        meta_test_y.extend(seq_actual)
        
    # Learn meta model
    if meta_learner is "rf":
        print("Now learning random forest with", num_trees, "trees...")
        meta_learner = RandomForestRegressor(n_estimators = num_trees)
        meta_learner.fit(meta_train_X, meta_train_y)
        meta_forecasts = meta_learner.predict(meta_test_X)
        # evaluate
        meta_rmse = rmse(np.array(meta_forecasts)*data_sd+data_mean, np.array(meta_test_y)*data_sd+data_mean)
        return meta_forecasts, meta_rmse
    
    if meta_learner is "xgb":
        print("Now learning xgboost meta-learner...")
        y_mean = np.mean(meta_train_y)
        
        xgb_params = {
        'n_trees': 520,
        'eta': 0.0045,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean, # base prediction = mean(target)
        'silent': 1
        }
        
        dtrain = xgb.DMatrix(meta_train_X, meta_train_y)
        dtest = xgb.DMatrix(meta_test_X)
        num_boost_rounds = 1000
        # train model
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds, verbose_eval=1)
        meta_forecasts = model.predict(dtest)
        # evaluate
        meta_rmse = rmse(np.array(meta_forecasts)*data_sd+data_mean, np.array(meta_test_y)*data_sd+data_mean)
        return meta_forecasts, meta_rmse
    
    if meta_learner is "ridge":
        print("Now learning ridge regressor with alphas:", ridge_alphas, "...")
        best_ridge_rmse = np.Inf
        best_alpha = None
        best_forecasts = None
        for ridge_alpha in ridge_alphas:
            meta_learner = Ridge(alpha=ridge_alpha, normalize=True)
            meta_learner.fit(meta_train_X, meta_train_y)
            meta_forecasts = meta_learner.predict(meta_test_X)
            # evaluate
            meta_rmse = rmse(np.array(meta_forecasts)*data_sd+data_mean, np.array(meta_test_y)*data_sd+data_mean)
            if meta_rmse < best_ridge_rmse:
                best_ridge_rmse = meta_rmse
                best_alpha = ridge_alpha
                best_forecasts = meta_forecasts
        print("Best Ridge result with alpha="+str(best_alpha)+": RMSE="+str(best_ridge_rmse))
        return best_forecasts, best_ridge_rmse
		
def fc_lstm_ensemble(holdoutset, model_filename_base, seqlens = range(50, 105, 5), dropout_rates = [0.3],
               num_hl = [2], layers = ["standard"], batch = 32, loss_function = "mse", lrs = [0.001]):
    rmse_all = {}
    rmses_per_point_allModels = []
    num_fc = min(seqlens) # this is an important assumption of this approach!
    offset = max(seqlens)+num_fc
    upper_limit = round(0.7*len(holdoutset))
    
    meta_train_input = []
    meta_train_forecasts = []
    meta_train_actual = []
    meta_test_forecasts = []
    meta_test_actual = []
    for current_seq_length in seqlens:
        for current_dropout in dropout_rates:
            for current_num_hl in num_hl:
                for current_layer in layers:
                    for current_lr in lrs:
                        print("Seqlen:", current_seq_length, ", Dropout:", current_dropout, ", numHL:", current_num_hl,
                             "Layer:", current_layer, ", LR:", current_lr)
                        current_name = "models/ipy/"+model_filename_base+"_"+str(current_seq_length)+"_"+str(current_dropout)+"_hl_"+str(current_num_hl)+"_"+str(current_layer)+"_"+str(current_lr)
                        
                        json_file = open(current_name+".json", "r")
                        loaded_model_json = json_file.read()
                        json_file.close()
                        model = model_from_json(loaded_model_json)
                        # load weights into model
                        model.load_weights(current_name+".h5")

                        meta_train = holdoutset[:upper_limit]
                        meta_test = holdoutset[upper_limit:]
                        # generate training data for stacking model
                        meta_train_seqs = []
                        for index in range(len(meta_train)-offset):
                            meta_train_seqs.append(meta_train[index:index+offset])
                        meta_train_seqs = np.array(meta_train_seqs)
                        meta_train_seqs_X = meta_train_seqs[:, len(meta_train_seqs[0])-num_fc-current_seq_length:len(meta_train_seqs[0])-num_fc]
                        meta_train_seqs_y = meta_train_seqs[:, -num_fc:]
                        meta_train_input.append(meta_train_seqs_X[:, -num_fc:]) # additional features for meta-learner. num_fc = min(sequence_lengths)!

                        meta_train_seqs_X = np.reshape(meta_train_seqs_X, (meta_train_seqs_X.shape[0], meta_train_seqs_X.shape[1], 1))
                        meta_train_seqs_y = np.reshape(meta_train_seqs_y, (meta_train_seqs_y.shape[0], meta_train_seqs_y.shape[1], 1))
                        meta_train_seqs_forecasts = model.predict(meta_train_seqs_X)[:, :num_fc]
                        
                        meta_train_forecasts.append(np.reshape(meta_train_seqs_forecasts, (len(meta_train_seqs_forecasts), len(meta_train_seqs_forecasts[0]))))
                        meta_train_actual = np.reshape(meta_train_seqs_y, (len(meta_train_seqs_y), len(meta_train_seqs_y[0]))) # is identical for each seqlen 


                        # compute model forecasts and errors on the actual test data
                        meta_test_seqs = []
                        for index in range(len(meta_test)-offset):
                            #meta_test_seqs.append(meta_test[index:index+current_seq_length+num_fc])
                            meta_test_seqs.append(meta_test[index:index+offset])
                        meta_test_seqs = np.array(meta_test_seqs)
                        print("Evaluating on", len(meta_test_seqs), "test sequences.")
                        meta_test_seqs_X = meta_test_seqs[:, len(meta_test_seqs[0])-num_fc-current_seq_length:len(meta_test_seqs[0])-num_fc]
                        meta_test_seqs_y = meta_test_seqs[:, -num_fc:]

                        meta_test_seqs_X = np.reshape(meta_test_seqs_X, (meta_test_seqs_X.shape[0], meta_test_seqs_X.shape[1], 1))
                        meta_test_seqs_y = np.reshape(meta_test_seqs_y, (meta_test_seqs_y.shape[0], meta_test_seqs_y.shape[1], 1))

                        meta_test_seqs_forecasts = model.predict(meta_test_seqs_X)[:, :num_fc]
                        meta_test_forecasts.append(np.reshape(meta_test_seqs_forecasts, (len(meta_test_seqs_forecasts), len(meta_test_seqs_forecasts[0]))))
                        meta_test_actual = np.reshape(meta_test_seqs_y, (len(meta_test_seqs_y), len(meta_test_seqs_y[0]))) # is identical for each seqlen 
                        curr_rmse = rmse(meta_test_seqs_forecasts*data_sd+data_mean, meta_test_seqs_y*data_sd+data_mean)
                        rmse_all["SeqLen-"+str(current_seq_length)+"_"+"DO-"+str(current_dropout)+"_HL-"+str(current_num_hl)] = curr_rmse
                        print("RMSE:", curr_rmse)
    meta_test_forecasts = np.array(meta_test_forecasts)# dim: (# models, # test sequences, # forecasts)
    meta_train_forecasts = np.array(meta_train_forecasts)
    meta_train_actual = np.array(meta_train_actual)
    meta_test_forecasts = np.array(meta_test_forecasts)
    print("Avg. pairwise correlation between base forecasts:", get_average_pairwise_correlation(meta_test_forecasts))
    
    # Mean Forecast
    # Important: Assuming len(rmse_all) == Number of individual base learners!
    meta_test_forecast_mean = np.mean(meta_test_forecasts, axis = 0)
    # meta_test_actual contains the same values for every LSTM
    fc_mean_rmse = rmse(meta_test_forecast_mean*data_sd+data_mean, 
                                     meta_test_actual*data_sd+data_mean)
    num_baseLearners = len(rmse_all)
    rmses_baseLearners = rmse_all.copy()
    rmse_sum = 0
    for r in rmses_baseLearners.values():
        rmse_sum+=r
    print("Baselearner avg. RMSE:", round(rmse_sum/len(rmses_baseLearners), 2))
    percent_better_than_mean = len({k: v for k, v in rmses_baseLearners.items() if v < fc_mean_rmse}.values())/num_baseLearners
    rmse_all["Mean Forecast"] = fc_mean_rmse
    print("Mean Forecast RMSE:", fc_mean_rmse, ", which is better than", str(round((1-percent_better_than_mean)*100, 2)), "% of the base learners.")
    
    # Ridge Regression Forecast
    #for curr_alpha in [10, 5, 3, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.00001]:
    fc_ridge = meta_forecast(meta_train_forecasts, meta_train_actual, meta_test_forecasts, meta_test_actual, 
                             meta_learner="ridge")
    percent_better_than_ridge = len({k: v for k, v in rmses_baseLearners.items() if v < fc_ridge[1]}.values())/num_baseLearners
    rmse_all["Ridge Forecast"] = fc_ridge[1]
    print("RidgeReg RMSE:", fc_ridge[1], ", which is better than", str(round((1-percent_better_than_ridge)*100, 2)), "% of the base learners.")


    # Random Forest forecast
    fc_rf = meta_forecast(meta_train_forecasts, meta_train_actual, meta_test_forecasts, meta_test_actual, 
                          num_trees = 250, meta_learner="rf")
    percent_better_than_rf = len({k: v for k, v in rmses_baseLearners.items() if v < fc_rf[1]}.values())/num_baseLearners
    rmse_all["RF Forecast"] = fc_rf[1]
    print("RF RMSE:", fc_rf[1], ", which is better than", str(round((1-percent_better_than_rf)*100, 2)), "% of the base learners.")
    
    
    return meta_train_forecasts, meta_train_actual, meta_test_forecasts, meta_test_actual, meta_test_forecast_mean, fc_rf, fc_ridge, rmse_all
	
