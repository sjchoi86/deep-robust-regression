import os
import numpy as np
import sklearn.model_selection
import scipy.io
import tensorflow as tf
from sklearn.utils import shuffle

class nzr(object):
    def __init__(self,_rawdata,_eps=1e-8,_VERBOSE=False):
        self.rawdata = _rawdata
        self.eps     = _eps
        self.mu      = np.mean(self.rawdata,axis=0)
        self.std     = np.std(self.rawdata,axis=0)
        self.nzd_data = self.get_nzdval(self.rawdata)
        self.org_data = self.get_orgval(self.nzd_data)
        self.maxerr = np.max(self.rawdata-self.org_data)
        if _VERBOSE:
            print (self.mu)
            print (self.std)
    def get_nzdval(self,_data):
        _n = _data.shape[0]
        _nzddata = (_data - np.tile(self.mu,(_n,1))) / np.tile(self.std+self.eps,(_n,1))
        return _nzddata
    def get_orgval(self,_data):
        _n = _data.shape[0]
        _orgdata = _data*np.tile(self.std+self.eps,(_n,1))+np.tile(self.mu,(_n,1))
        return _orgdata

def get_train_test_datasets(dataset,dataset_name,
        _test_size=0.333,_outlier_rate=0,
        _seed=0,_SAVE_MAT=True,_VERBOSE=False):

    if _VERBOSE:
        print(dataset.keys())
        print(dataset.data.shape)
        print(dataset.target.shape)
        print(dataset.feature_names)
        print(dataset.DESCR)
    
    x_total,y_total = np.copy(dataset.data),np.copy(dataset.target) # copy the training data
    x_total,y_total = shuffle(x_total,y_total,random_state=_seed) # shuffle with a fixed seed
    skl_tts = sklearn.model_selection.train_test_split # abbreviate
    x_train,x_test,y_train,y_test = skl_tts(x_total,y_total,test_size=_test_size,random_state=_seed) 
    if len(np.shape(y_train)) == 1: # make output rank two
        y_train = np.reshape(y_train,newshape=[-1,1])
    if len(np.shape(y_test)) == 1: # make output rank two
        y_test = np.reshape(y_test,newshape=[-1,1])        

    if _outlier_rate > 0: # add outlier to y_train
        n_train,y_dim = y_train.shape[0],y_train.shape[1]
        n_outlier = (int)(n_train*_outlier_rate) 
        for _d_idx in range(y_dim): # add dimension-wise outliers
            curr_y_train = y_train[:,_d_idx]
            y_min,y_max = np.min(curr_y_train),np.max(curr_y_train)
            rand_idx = np.random.permutation(n_train)[:n_outlier]
            y_train[rand_idx,_d_idx] = y_min + (y_max-y_min)*np.random.rand(n_outlier)
            


    if _SAVE_MAT:
        dict_val = {'x_total':x_total,'y_total':y_total,'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test}
        mat_name = 'data/%s.mat'%(dataset_name)
        scipy.io.savemat(mat_name, mdict=dict_val) # save to a mat file
        if _VERBOSE:
            print ("[%s] saved (size is [%.3fMB])."%(mat_name,os.path.getsize(mat_name)/1000./1000.))

    return x_train,x_test,y_train,y_test 


def gpusession(): 
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

def remove_file_if_exists(_file_name,_VERBOSE=True):
    if os.path.exists(_file_name):
        os.remove(_file_name)
        if _VERBOSE:
            print ("[%s] removed."%(_file_name))

def create_gradient_clipping(loss,optm,vars,clipVal=1.0):
    grads, vars = zip(*optm.compute_gradients(loss, var_list=vars))
    grads = [None if grad is None else tf.clip_by_value(grad,-clipVal,clipVal) for grad in grads]
    op = optm.apply_gradients(zip(grads, vars))
    train_op = tf.tuple([loss], control_inputs=[op])
    return train_op[0]