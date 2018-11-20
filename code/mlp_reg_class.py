import os,warnings
warnings.filterwarnings("ignore") 
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.contrib.slim as slim
import scipy.io as sio
from sklearn.utils import shuffle
from util import gpusession,nzr,print_n_txt,remove_file_if_exists
import matplotlib.pyplot as plt

class mlp_reg_class(object):
    def __init__(self,_name='mlp_reg',_x_dim=1,_y_dim=1,_h_dims=[64, 64],_actv=tf.nn.tanh,_bn=slim.batch_norm,
                _l2_reg_coef=1e-5,_GPU_ID=0,_VERBOSE=True):
        self.name = _name
        self.x_dim = _x_dim
        self.y_dim = _y_dim
        self.h_dims = _h_dims
        self.actv = _actv
        self.bn = _bn
        self.l2_reg_coef = _l2_reg_coef
        self.GPU_ID = _GPU_ID
        self.VERBOSE = _VERBOSE
        
        if _GPU_ID < 0:  # with CPU only (no GPU)
            # Build model
            self.build_model()
            # Build graph
            self.build_graph()
            # Check params
            self.check_params()
        else:  # with GPU
            with tf.device('/device:GPU:%d' % (self.GPU_ID)):
                # Build model
                self.build_model()
                # Build graph
                self.build_graph()
                # Check params
                self.check_params()
    
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.x_dim]) # Input [N x xdim]
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,self.y_dim]) # Output [N x ydim]
        self.kp = tf.placeholder(dtype=tf.float32,shape=[]) # Keep probability 
        self.lr = tf.placeholder(dtype=tf.float32,shape=[]) # Learning rate
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) # Training flag
        self.fully_init  = tf.random_normal_initializer(stddev=0.01)
        self.bias_init   = tf.constant_initializer(0.)
        self.bn_init     = {'beta': tf.constant_initializer(0.),
                           'gamma': tf.random_normal_initializer(1., 0.01)}
        self.bn_params   = {'is_training':self.is_training,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bn_init,'updates_collections':None}
        # Build graph
        with tf.variable_scope(self.name,reuse=False) as scope:
            with slim.arg_scope([slim.fully_connected]
                                ,activation_fn=self.actv
                                ,weights_initializer=self.fully_init
                                ,biases_initializer=self.bias_init
                                ,normalizer_fn=self.bn,normalizer_params=self.bn_params
                                ,weights_regularizer=None):
                _net = self.x # Input [N x xdim]
                for h_idx in range(len(self.h_dims)): # Loop over hidden layers
                    _h_dim = self.h_dims[h_idx]
                    _net = slim.fully_connected(_net,_h_dim,scope='lin'+str(h_idx))
                    _net = slim.dropout(_net,keep_prob=self.kp,is_training=self.is_training
                                        ,scope='dr'+str(h_idx))  
                self.feat = _net # Feature [N x Q]
                self.out = slim.fully_connected(self.feat,self.y_dim,activation_fn=None
                                                ,scope='out') # [N x D]
    
    def build_graph(self):
        # L2 fitting loss
        self._loss_fit = tf.reduce_sum(tf.pow(self.out-self.y,2),axis=1) # [N x 1]
        self.loss_fit = tf.reduce_mean(self._loss_fit) # [1]
        # Weight decay
        _t_vars = tf.trainable_variables()
        self.c_vars = [var for var in _t_vars if '%s/'%(self.name) in var.name]
        self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in self.c_vars])) # [1]
        self.loss_total = self.loss_fit + self.l2_reg # [1]
        # Optimizer
        USE_ADAM = True
        if USE_ADAM:
            self.optm = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.9,beta2=0.999
                                               ,epsilon=1e-0).minimize(self.loss_total)
        else:
            self.optm = tf.train.MomentumOptimizer(learning_rate=self.lr
                                                   ,momentum=0.0).minimize(self.loss_total)

    def check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print (" [%02d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
    
    def sampler(self,_sess,_x):
        outVal = _sess.run(self.out,feed_dict={self.x:_x,self.kp:1.0,self.is_training:False})
        return outVal

    def save2npz(self,_sess,_save_name=None):
        """ Save name """
        if _save_name==None:
            _save_name='net/net_%s.npz'%(self.name)
        """ Get global variables """
        self.g_wnames,self.g_wvals,self.g_wshapes = [],[],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            curr_wval_sqz  = curr_wval.squeeze()
            self.g_wnames.append(curr_wname)
            self.g_wvals.append(curr_wval_sqz)
            self.g_wshapes.append(curr_wval.shape)
        """ Save """
        np.savez(_save_name,g_wnames=self.g_wnames,g_wvals=self.g_wvals,g_wshapes=self.g_wshapes)
        if self.VERBOSE:
            print ("[%s] saved. Size is [%.4f]MB" % 
                   (_save_name,os.path.getsize(_save_name)/1000./1000.))
    
    def restore_from_npz(self,_sess,_loadname=None):
        if _loadname==None:
            _loadname='net/net_%s_final.npz'%(self.name)
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            _sess.run(tf.assign(curr_wvar,g_wvals[widx].reshape(g_wshapes[widx])))
        if self.VERBOSE:
            print ("Weight restored from [%s] Size is [%.4f]MB" % 
                   (_loadname,os.path.getsize(_loadname)/1000./1000.))

    def save2mat_from_npz(self,_x_train='',_y_train='',_save_name=None,_npz_path=None):
        # Save weights to mat file so that MATLAB can use it.
        if _npz_path == None:
            _npz_path = 'net/net_%s.npz'%(self.name)
        l = np.load(_npz_path)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        D = {}
        for w_idx,w_name in enumerate(g_wnames):
            curr_name = w_name.replace(':0','')
            curr_name = curr_name.replace(self.name+'/','')
            curr_name = curr_name.replace('/','_')
            curr_val = g_wvals[w_idx].reshape(g_wshapes[w_idx])
            D[curr_name] = curr_val
        # Save train data
        if _x_train!='': D['x_train'] = _x_train
        if _y_train!='': D['y_train'] = _y_train
        # Save dictionary D to the mat file
        if _save_name == None:
            _save_name = 'net/net_%s.mat'%(self.name)
        sio.savemat(_save_name,D)
        if self.VERBOSE: 
            print ("[%s] saved. Size is [%.4f]MB" % 
                   (_save_name,os.path.getsize(_save_name)/1000./1000.))
    
    def train(self,_sess,_x_train,_y_train,_lr=1e-3,_batch_size=512,_max_epoch=1e4,_kp=1.0,
                _LR_SCHEDULE=True,_PRINT_EVERY=20,_PLOT_EVERY=20,
                _SAVE_TXT=True,_SAVE_BEST_NET=True,_SAVE_FINAL=True,_REMOVE_PREVS=True,
                _x_dim4plot=0,_x_name4plot=None):
        
        self.x_dim4plot = _x_dim4plot
        self.x_name4plot = _x_name4plot

        # Remove existing files
        if _REMOVE_PREVS:
            remove_file_if_exists('net/net_%s_best.npz'%(self.name),_VERBOSE=self.VERBOSE)
            remove_file_if_exists('net/net_%s_best.mat'%(self.name),_VERBOSE=self.VERBOSE)
            remove_file_if_exists('net/net_%s_final.npz'%(self.name),_VERBOSE=self.VERBOSE)
            remove_file_if_exists('net/net_%s_final.mat'%(self.name),_VERBOSE=self.VERBOSE)
            remove_file_if_exists('res/res_%s.txt'%(self.name),_VERBOSE=self.VERBOSE)

        # Reference training data
        x_train,y_train = _x_train,_y_train
        if len(np.shape(y_train)) == 1: # if y is a vector
            y_train = np.reshape(y_train,newshape=[-1,1]) # make it rank two
        self.nzr_x,self.nzr_y = nzr(x_train),nzr(y_train) # get normalizer
        
        # Iterate
        if _PRINT_EVERY == 0: print_period = 0
        else: print_period = _max_epoch//_PRINT_EVERY
        if _PLOT_EVERY == 0: plot_period = 0
        else: plot_period = _max_epoch//_PLOT_EVERY
        
        max_iter = max(x_train.shape[0]//_batch_size, 1)
        best_loss_val = np.inf
        if _SAVE_TXT:
            txt_name = ('res/res_%s.txt'%(self.name));f = open(txt_name,'w') # Open txt file
            print_n_txt(_f=f,_chars='Text name: '+txt_name,_DO_PRINT=True)
        for epoch in range((int)(_max_epoch)+1): # For every epoch
            x_train,y_train = shuffle(x_train,y_train)
            nzd_x_train,nzd_y_train = self.nzr_x.get_nzdval(x_train),self.nzr_y.get_nzdval(y_train)
            for iter in range(max_iter): # For every iteration
                start,end = iter*_batch_size,(iter+1)*_batch_size
                if _LR_SCHEDULE:
                    if epoch < 0.5*_max_epoch:
                        lr_use = _lr
                    elif epoch < 0.75*_max_epoch:
                        lr_use = _lr/5.
                    else:
                        lr_use = _lr/10.
                else:
                    lr_use = _lr
                feeds = {self.x:nzd_x_train[start:end,:],self.y:nzd_y_train[start:end,:]
                         ,self.kp:_kp,self.lr:lr_use,self.is_training:True}
                # Optimize 
                _sess.run(self.optm,feeds)

            # Track the Best result
            BEST_FLAG = False
            check_period = _max_epoch//100
            if (epoch % check_period)==0:
                # Feed total dataset 
                feeds = {self.x:nzd_x_train,self.y:nzd_y_train,self.kp:1.0,self.is_training:False}
                opers = [self.loss_total,self.loss_fit,self.l2_reg]
                loss_val,loss_fit,l2_reg = _sess.run(opers,feeds)
                if (loss_val < best_loss_val) & (epoch >= 3):
                    best_loss_val = loss_val
                    BEST_FLAG = True
                    if _SAVE_BEST_NET: # Save the current best model 
                        if self.VERBOSE:
                            print ("Epoch:[%d] saving current network (best loss:[%.3f])"%(epoch,best_loss_val))
                        self.save2npz(_sess,_save_name='net/net_%s_best.npz'%(self.name)) 
                        self.save2mat_from_npz(_x_train=x_train,_y_train=y_train,
                                    _save_name='net/net_%s_best.mat'%(self.name),
                                    _npz_path='net/net_%s_best.npz'%(self.name))
            
            # Print current result 
            if (print_period!=0) and ((epoch%print_period)==0 or (epoch==(_max_epoch-1))): # Print 
                feeds = {self.x:nzd_x_train,self.y:nzd_y_train,self.kp:1.0,self.is_training:False}
                opers = [self.loss_total,self.loss_fit,self.l2_reg]
                loss_val,loss_fit,l2_reg = _sess.run(opers,feeds)
                if _SAVE_TXT:
                    str_temp = ("[%d/%d] loss:%.3f(fit:%.3f+l2:%.3f) bestLoss:%.3f"
                               %(epoch,_max_epoch,loss_val,loss_fit,l2_reg,best_loss_val))
                    print_n_txt(_f=f,_chars=str_temp,_DO_PRINT=self.VERBOSE)
                else:
                    if self.VERBOSE:
                        print ("[%d/%d] loss:%.3f(fit:%.3f+l2:%.3f) bestLoss:%.3f"
                                   %(epoch,_max_epoch,lossVal,loss_fit,l2_reg,best_loss_val))

            # Plot current result 
            if (plot_period!=0) and ((epoch%plot_period)==0 or (epoch==(_max_epoch-1))): # Plot
                # Get loss vals
                feeds = {self.x:nzd_x_train,self.y:nzd_y_train,self.kp:1.0,self.is_training:False}
                opers = [self.loss_total,self.loss_fit,self.l2_reg]
                lossVal,loss_fit,l2_reg = _sess.run(opers,feeds)
                # Output
                nzd_y_test = self.sampler(_sess=_sess,_x=nzd_x_train)
                y_pred = self.nzr_y.get_orgval(nzd_y_test)[:,0]
                # Plot one dimensions of both input and output
                x_plot,y_plot = x_train[:,self.x_dim4plot],y_train[:,0] # Traning data 
                plt.figure(figsize=(8,4))
                plt.axis([np.min(x_plot),np.max(x_plot),np.min(y_plot)-0.1,np.max(y_plot)+0.1])
                h_tr,=plt.plot(x_plot,y_plot,'k.') # Plot training data
                h_pr,=plt.plot(x_plot,y_pred,'b.') # Plot prediction
                plt.title("[%d/%d] name:[%s] loss_val:[%.3e]"%(epoch,_max_epoch,self.name,loss_val),fontsize=13); 
                plt.legend([h_tr,h_pr],['Train data','Predictions'],fontsize=13,loc='upper left')
                if self.x_name4plot != None:
                    plt.xlabel(self.x_name4plot,fontsize=13)
                plt.show()

        # Save final results
        if _SAVE_FINAL:
            self.save2npz(_sess,_save_name='net/net_%s_final.npz'%(self.name)) 
            self.save2mat_from_npz(_x_train=x_train,_y_train=y_train,
                        _save_name='net/net_%s_final.mat'%(self.name),
                        _npz_path='net/net_%s_final.npz'%(self.name))
        
        if self.VERBOSE:
            print ("Train done.")
        
    def test(self,_sess,_x_train,_y_train,_x_test=None,_y_test=None,
            _title_str4data=None,_title_str4test=None,
            _PLOT_TRAIN=False,_PLOT_TEST=False,_SAVE_FIG=False,
            _x_dim4plot=0,_x_name4plot=None):
        
        self.x_dim4plot = _x_dim4plot
        self.x_name4plot = _x_name4plot

        # Get normalizer 
        if len(np.shape(_y_train)) == 1: # if y is a vector
            _y_train = np.reshape(_y_train,newshape=[-1,1]) # make it rank two
        self.nzr_x,self.nzr_y = nzr(_x_train),nzr(_y_train) # get normalizer
        
        # Plot train data and predictions
        if _PLOT_TRAIN:
            if len(np.shape(_y_train)) == 1: # if y is a vector
                _y_train = np.reshape(_y_train,newshape=[-1,1]) # make it rank two
            x_train4plot,y_train4plot = _x_train[:,self.x_dim4plot],_y_train[:,0] # traning data 
            nzd_y_pred = self.sampler(_sess=_sess,_x=self.nzr_x.get_nzdval(_x_train))
            y_pred_train = self.nzr_y.get_orgval(nzd_y_pred)[:,0]
            plt.figure(figsize=(8,4))
            plt.axis([np.min(x_train4plot),np.max(x_train4plot),np.min(y_train4plot)-0.1,np.max(y_train4plot)+0.1])
            h_tr,=plt.plot(x_train4plot,y_train4plot,'k.') # plot train data
            h_pr,=plt.plot(x_train4plot,y_pred_train,'b.') # plot prediction for train data
            plt.legend([h_tr,h_pr],['Train data','Train predictions'],fontsize=13,loc='upper left')
            if self.x_name4plot != None:
                plt.xlabel(self.x_name4plot,fontsize=13)
            plt.ylabel('Output',fontsize=13)
            if _title_str4data != None:
                plt.title(_title_str4data,fontsize=15); 
            if _SAVE_FIG: 
                plt.savefig('fig/fig_%s_data.png'%(self.name))
            plt.show()

        # Plot test data and predictions
        if len(np.shape(_y_train)) == 1: # if y is a vector
            _y_train = np.reshape(_y_train,newshape=[-1,1]) # make it rank two
        if len(np.shape(_y_test)) == 1: # if y is a vector
            _y_test = np.reshape(_y_test,newshape=[-1,1]) # make it rank two
        x_data4plot,y_data4plot = _x_train[:,self.x_dim4plot],_y_train[:,0] # traning data 
        x_test4plot,y_test4plot = _x_test[:,self.x_dim4plot],_y_test[:,0] # test data 
        nzd_y_test = self.sampler(_sess=_sess,_x=self.nzr_x.get_nzdval(_x_test))
        y_pred_test = self.nzr_y.get_orgval(nzd_y_test)[:,0]

        if _PLOT_TEST:            
            fig = plt.figure(figsize=(8,4))
            plt.axis([np.min(x_data4plot),np.max(x_data4plot),np.min(y_data4plot)-0.1,np.max(y_data4plot)+0.1])

            # h_tr,=plt.plot(x_data4plot,y_data4plot,'k.') # plot train data
            h_pr,=plt.plot(x_test4plot,y_pred_test,'b.') # plot prediction for the test data
            h_te,=plt.plot(x_test4plot,y_test4plot,'r.') # plot test data

            plt.legend([h_pr,h_te],['Test data','Test predictions'],fontsize=13,loc='upper left')
            if self.x_name4plot != None:
                plt.xlabel(self.x_name4plot,fontsize=13)
            plt.ylabel('Output',fontsize=13)           
            if _title_str4test != None:
                plt.title(_title_str4test,fontsize=15); 
            if _SAVE_FIG: 
                plt.savefig('fig/fig_%s_test.png'%(self.name))
            plt.show()

        rmse = np.sqrt(np.mean((y_pred_test-y_test4plot)**2))
        return rmse
