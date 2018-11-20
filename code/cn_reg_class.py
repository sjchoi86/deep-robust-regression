import os,warnings
warnings.filterwarnings("ignore") 
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.contrib.slim as slim
import scipy.io as sio
from sklearn.utils import shuffle
from util import gpusession,nzr,print_n_txt,remove_file_if_exists,create_gradient_clipping
import matplotlib.pyplot as plt

class cn_reg_class(object): # ChoiceNet
    def __init__(self,_name='cn_reg',_x_dim=1,_y_dim=1,_h_dims=[64,64]
                 ,_k_mix=5,_actv=tf.nn.relu,_bn=slim.batch_norm
                 ,_rho_ref_train=0.95,_tau_inv=1e-2,_var_eps=1e-2
                 ,_pi1_bias=0.0,_log_sigma_Z_val=0
                 ,_kl_reg_coef=1e-5,_l2_reg_coef=1e-5
                 ,_SCHEDULE_MDN_REG=False
                 ,_GPU_ID=0,_VERBOSE=True):
        self.name = _name
        self.x_dim = _x_dim
        self.y_dim = _y_dim
        self.h_dims = _h_dims
        self.k_mix = _k_mix
        self.actv = _actv 
        self.bn = _bn # slim.batch_norm / None
        self.rho_ref_train = _rho_ref_train # Rho for training 
        self.tau_inv = _tau_inv
        self.var_eps = _var_eps # This will be used for the loss function (var+var_eps)
        self.pi1_bias = _pi1_bias
        self.log_sigma_Z_val = _log_sigma_Z_val
        self.kl_reg_coef = _kl_reg_coef
        self.l2_reg_coef = _l2_reg_coef # L2 regularizer 
        self.SCHEDULE_MDN_REG = _SCHEDULE_MDN_REG
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

    # Build model
    def build_model(self):
        # Placeholders 
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,self.x_dim],name='x') # Input [None x xdim]
        self.t = tf.placeholder(dtype=tf.float32,shape=[None,self.y_dim],name='t') # Output [None x ydim]
        self.kp = tf.placeholder(dtype=tf.float32,shape=[],name='kp') # Keep probability 
        self.lr = tf.placeholder(dtype=tf.float32,shape=[],name='lr') # Learning rate
        self.is_training = tf.placeholder(dtype=tf.bool,shape=[]) # Training flag
        self.rho_ref = tf.placeholder(dtype=tf.float32,shape=[],name='rho_ref') # Training flag
        self.train_rate = tf.placeholder(dtype=tf.float32,shape=[],name='train_rate') # from 0.0~1.0
        # Initializers
        trni = tf.random_normal_initializer
        tci = tf.constant_initializer
        self.fully_init = trni(stddev=0.01)
        self.bias_init = tci(0.)
        self.bn_init = {'beta':tci(0.),'gamma':trni(1.,0.01)}
        self.bn_params = {'is_training':self.is_training,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bn_init,'updates_collections':None}
        # Build graph
        with tf.variable_scope(self.name,reuse=False) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                                weights_initializer=self.fully_init,biases_initializer=self.bias_init,
                                normalizer_fn=self.bn,normalizer_params=self.bn_params,
                                weights_regularizer=None):
                _net = self.x # Now we have an input
                self.N = tf.shape(self.x)[0] # Input dimension
                for h_idx in range(len(self.h_dims)): # Loop over hidden layers
                    _hdim = self.h_dims[h_idx]
                    _net = slim.fully_connected(_net,_hdim,scope='lin'+str(h_idx))
                    _net = slim.dropout(_net,keep_prob=self.kp,is_training=self.is_training
                                        ,scope='dr'+str(h_idx))
                self.feat = _net # Feature [N x Q]
                self.Q = self.feat.get_shape().as_list()[1] # Feature dimension
                # Feature to K rhos (NO ACTIVATION !!!)
                _rho_raw = slim.fully_connected(self.feat,self.k_mix,activation_fn=None
                                                ,scope='rho_raw')
                
                # self.rho_temp = tf.nn.tanh(_rho_raw) # [N x K] between -1.0~1.0 for regression
                self.rho_temp = tf.nn.sigmoid(_rho_raw) # [N x K] between 0.0~1.0 for classification
                
                # Maker sure the first mixture to have 'self.rho_ref' correlation
                self.rho = tf.concat([self.rho_temp[:,0:1]*0.0+self.rho_ref,self.rho_temp[:,1:]]
                                     ,axis=1) # [N x K] 
                # Variabels for the sampler 
                self.muW = tf.get_variable(name='muW',shape=[self.Q,self.y_dim],
                                           initializer=tf.random_normal_initializer(stddev=0.1),
                                           dtype=tf.float32) # [Q x D]
                self.logSigmaW = tf.get_variable(name='logSigmaW'
                                        ,shape=[self.Q,self.y_dim]
                                        ,initializer=tf.constant_initializer(-2.0)
                                        ,dtype=tf.float32) # [Q x D]
                self.muZ = tf.constant(np.zeros((self.Q,self.y_dim))
                                        ,name='muZ',dtype=tf.float32) # [Q x D]
                self.logSigmaZ = tf.constant(self.log_sigma_Z_val*np.ones((self.Q,self.y_dim)) 
                                        ,name='logSigmaZ',dtype=tf.float32) # [Q x D]
                # Reparametrization track (THIS PART IS COMPLICATED, I KNOW)
                _muW_tile = tf.tile(self.muW[tf.newaxis,:,:]
                                    ,multiples=[self.N,1,1]) # [N x Q x D]
                _sigmaW_tile = tf.exp(tf.tile(self.logSigmaW[tf.newaxis,:,:]
                                              ,multiples=[self.N,1,1])) # [N x Q x D]
                _muZ_tile = tf.tile(self.muZ[tf.newaxis,:,:]
                                    ,multiples=[self.N,1,1]) # [N x Q x D]
                _sigmaZ_tile = tf.exp(tf.tile(self.logSigmaZ[tf.newaxis,:,:]
                                              ,multiples=[self.N,1,1])) # [N x Q x D]
                _samplerList = []
                for jIdx in range(self.k_mix): # For all K mixtures
                    _rho_j = self.rho[:,jIdx:jIdx+1] # [N x 1] 
                    _rho_tile = tf.tile(_rho_j[:,:,tf.newaxis]
                                        ,multiples=[1,self.Q,self.y_dim]) # [N x Q x D]
                    _epsW = tf.random_normal(shape=[self.N,self.Q,self.y_dim],mean=0,stddev=1
                                             ,dtype=tf.float32) # [N x Q x D]
                    _W = _muW_tile + tf.sqrt(_sigmaW_tile)*_epsW # [N x Q x D]
                    _epsZ = tf.random_normal(shape=[self.N,self.Q,self.y_dim]
                                            ,mean=0,stddev=1,dtype=tf.float32) # [N x Q x D]
                    _Z = _muZ_tile + tf.sqrt(_sigmaZ_tile)*_epsZ # [N x Q x D]
                    _Y = _rho_tile*_muW_tile + (1.0-_rho_tile**2) \
                        *(_rho_tile*tf.sqrt(_sigmaZ_tile)/tf.sqrt(_sigmaW_tile) \
                              *(_W-_muW_tile)+tf.sqrt(1-_rho_tile**2)*_Z)
                    _samplerList.append(_Y) # Append 
                WlistConcat = tf.convert_to_tensor(_samplerList) # K*[N x Q x D] => [K x N x Q x D]
                self.wSample = tf.transpose(WlistConcat,perm=[1,3,0,2]) # [N x D x K x Q]
                # K mean mixtures [N x D x K]
                _wTemp = tf.reshape(self.wSample
                                ,shape=[self.N,self.k_mix*self.y_dim,self.Q]) # [N x KD x Q]
                _featRsh = tf.reshape(self.feat,shape=[self.N,self.Q,1]) # [N x Q x 1]
                _mu = tf.matmul(_wTemp,_featRsh) # [N x KD x Q] x [N x Q x 1] => [N x KD x 1]
                self.mu = tf.reshape(_mu,shape=[self.N,self.y_dim,self.k_mix]) # [N x D x K]
                # K variance mixtures [N x D x K]
                _logvar_raw = slim.fully_connected(self.feat,self.y_dim,scope='var_raw') # [N x D]
                _var_raw = tf.exp(_logvar_raw) # [N x D]
                _var_tile = tf.tile(_var_raw[:,:,tf.newaxis]
                                    ,multiples=[1,1,self.k_mix]) # [N x D x K]
                _rho_tile = tf.tile(self.rho[:,tf.newaxis,:]
                                    ,multiples=[1,self.y_dim,1]) # [N x D x K]
                _tau_inv = self.tau_inv
                self.var = (1.0-_rho_tile**2)*_var_tile + _tau_inv # [N x D x K]
                # Weight allocation probability pi [N x K]
                _pi_logits = slim.fully_connected(self.feat,self.k_mix
                                                  ,scope='pi_logits') # [N x K]
                self.pi_temp = tf.nn.softmax(_pi_logits,dim=1) # [N x K]
                # Some heuristics to ensure that pi_1(x) is high enough
                if self.pi1_bias != 0:
                    self.pi_temp = tf.concat([self.pi_temp[:,0:1]+self.pi1_bias
                                              ,self.pi_temp[:,1:]],axis=1) # [N x K]
                    self.pi = tf.nn.softmax(self.pi_temp,dim=1) # [N x K]
                else: self.pi = self.pi_temp # [N x K]
    
    # Build graph
    def build_graph(self):
        # Parse
        _M = tf.shape(self.x)[0] # Current batch size
        t,pi,mu,var = self.t,self.pi,self.mu,self.var
        
        # Mixture density network loss 
        trepeat = tf.tile(t[:,:,tf.newaxis],[1,1,self.k_mix]) # (N x D x K)
        self.quadratics = -0.5*tf.reduce_sum(((trepeat-mu)**2)/(var+self.var_eps),axis=1) # (N x K)
        self.logdet = -0.5*tf.reduce_sum(tf.log(var+self.var_eps),axis=1) # (N x K)
        self.logconstant = - 0.5*self.y_dim*tf.log(2*np.pi) # (1)
        self.logpi = tf.log(pi) # (N x K)
        self.exponents = self.quadratics + self.logdet + self.logpi # + self.logconstant 
        self.logprobs = tf.reduce_logsumexp(self.exponents,axis=1) # (N)
        self.gmm_prob = tf.exp(self.logprobs) # (N)
        self.gmm_nll  = -tf.reduce_mean(self.logprobs) # (1)
        
        # Regression loss 
        maxIdx = tf.argmax(input=pi,axis=1, output_type=tf.int32) # Argmax Index [N]
        maxIdx = 0*tf.ones_like(maxIdx)
        coords = tf.stack([tf.transpose(gv) for gv in tf.meshgrid(tf.range(self.N),tf.range(self.y_dim))] + 
                          [tf.reshape(tf.tile(maxIdx[:,tf.newaxis],[1,self.y_dim]),shape=(self.N,self.y_dim))]
                          ,axis=2) # [N x D x 3]
        self.mu_bar = tf.gather_nd(mu,coords) # [N x D]
        fit_mse_coef = 1e-2
        self.fit_mse = fit_mse_coef*tf.maximum((1.0-2.0*self.train_rate),0.0) \
            *tf.reduce_sum(tf.pow(self.mu_bar-self.t,2))/(tf.cast(self.N,tf.float32)) # (1)
        
        # KL-divergence
        _eps = 1e-2
        self.rho_pos = self.rho+1.0 # Make it positive
        self._kl_reg = self.kl_reg_coef*tf.reduce_sum(-self.rho_pos
                        *(tf.log(self.pi+_eps)-tf.log(self.rho_pos+_eps)),axis=1) # (N)
        self.kl_reg = tf.reduce_mean(self._kl_reg) # (1)

        # Weight decay
        _g_vars = tf.trainable_variables()
        self.c_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        self.l2_reg = self.l2_reg_coef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in self.c_vars])) # [1]

        # Schedule MDN loss and regression loss 
        if self.SCHEDULE_MDN_REG:
            self.gmm_nll = tf.minimum((2.0*self.train_rate+0.1),1.0)*self.gmm_nll
            self.fit_mse = tf.maximum((1.0-2.0*self.train_rate),0.0)*self.fit_mse
            self.loss_total = self.gmm_nll+self.kl_reg+self.l2_reg+self.fit_mse # [1]
        else:
            self.gmm_nll = self.gmm_nll
            self.fit_mse = tf.constant(0.0)
            self.loss_total = self.gmm_nll+self.kl_reg+self.l2_reg
        # Optimizer
        USE_ADAM = True
        GRAD_CLIP = True
        if GRAD_CLIP: # Gradient clipping
            if USE_ADAM:
                _optm = tf.train.AdamOptimizer(learning_rate=self.lr
                                               ,beta1=0.9,beta2=0.999,epsilon=1e-1) # 1e-4
            else:
                _optm = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.0)
            self.optm = create_gradient_clipping(self.loss_total
                                           ,_optm,tf.trainable_variables(),clipVal=1.0)
        else:
            if USE_ADAM:
                self.optm = tf.train.AdamOptimizer(learning_rate=self.lr
                            ,beta1=0.9,beta2=0.999,epsilon=1e-1).minimize(self.loss_total) 
            else:
                self.optm = tf.train.MomentumOptimizer(learning_rate=self.lr
                                                       ,momentum=0.0).minimize(self.loss_total)
                
    # Check parameters
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
    
    # Sampler 
    def sampler(self,_sess,_x,n_samples=1,_deterministic=True):
        pi, mu, var = _sess.run([self.pi, self.mu, self.var],
                                feed_dict={self.x:_x,self.kp:1.0,self.is_training:False
                                          ,self.rho_ref:1.0}) #
        n_points = _x.shape[0]
        _y_sampled = np.zeros([n_points,self.y_dim,n_samples])
        for i in range(n_points):
            for j in range(n_samples):
                if _deterministic: 
                    k = 0
                else: 
                    k = np.random.choice(self.k_mix,p=pi[i,:])
                _y_sampled[i,:,j] = mu[i,:,k] # + np.random.randn(1,self.y_dim)*np.sqrt(var[i,:,k])
        return _y_sampled 
    
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
    

    # Train
    def train(self,_sess,_x_train,_y_train,_lr=1e-3,_batch_size=512,_max_epoch=1e4,_kp=1.0
              ,_LR_SCHEDULE=True,_PRINT_EVERY=20,_PLOT_EVERY=20
              ,_SAVE_TXT=True,_SAVE_BEST_NET=True,_SAVE_FINAL=True,_REMOVE_PREVS=True
              ,_x_dim4plot=0,_x_name4plot=None):
        
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
            train_rate = (float)(epoch/_max_epoch)
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
                feeds = {self.x:nzd_x_train[start:end,:],self.t:nzd_y_train[start:end,:]
                         ,self.kp:_kp,self.lr:lr_use,self.train_rate:(float)(epoch/_max_epoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:True}
                # Optimize 
                _sess.run(self.optm,feeds)

            # Track the Best result
            BEST_FLAG = False
            check_period = _max_epoch//100
            if (epoch%check_period)==0:
                feeds = {self.x:nzd_x_train,self.t:nzd_y_train,self.kp:1.0,self.train_rate:train_rate
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                loss_val,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)
                if (loss_val < best_loss_val) & (train_rate >= 0.5):
                    best_loss_val = loss_val
                    BEST_FLAG = True
                    if _SAVE_BEST_NET:
                        if self.VERBOSE:
                            print ("Epoch:[%d] saving current network (best loss:[%.3f])"%(epoch,best_loss_val))
                        self.save2npz(_sess,'net/net_%s_best.npz'%(self.name)) # Save the current best model 
                        self.save2mat_from_npz(_x_train=x_train,_y_train=y_train,
                                _save_name='net/net_%s_best.mat'%(self.name),
                                _npz_path='net/net_%s_best.npz'%(self.name))
            
            # Print current result 
            if (print_period!=0) and ((epoch%print_period)==0 or (epoch==(_max_epoch-1))): # Print 
                # Feed total dataset 
                feeds = {self.x:nzd_x_train,self.t:nzd_y_train,self.kp:1.0,self.train_rate:(float)(epoch/_max_epoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                loss_val,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)
                if _SAVE_TXT:
                    strTemp = ("[%d/%d] loss:%.3f(gmm:%.3f+kl:%.3f+l2:%.3f+fit:%.3f) bestLoss:%.3f"
                               %(epoch,_max_epoch,loss_val,gmm_nll,kl_reg,l2_reg,fit_mse,best_loss_val))
                    print_n_txt(_f=f,_chars=strTemp,_DO_PRINT=self.VERBOSE)
                else:
                    if self.VERBOSE:
                        print ("[%d/%d] loss:%.3f(gmm:%.3f+kl:%.3f+l2:%.3f+fit:%.3f) bestLoss:%.3f"
                                   %(epoch,_max_epoch,loss_val,gmm_nll,kl_reg,l2_reg,fit_mse,best_loss_val))

            # Plot current result
            if (plot_period!=0) and ((epoch%plot_period)==0 or (epoch==(_max_epoch-1))): # Plot
                # Get loss values
                feeds = {self.x:nzd_x_train,self.t:nzd_y_train,self.kp:1.0,self.train_rate:(float)(epoch/_max_epoch)
                         ,self.rho_ref:self.rho_ref_train,self.is_training:False}
                opers = [self.loss_total,self.gmm_nll,self.kl_reg,self.l2_reg,self.fit_mse]
                loss_val,gmm_nll,kl_reg,l2_reg,fit_mse = _sess.run(opers,feeds)
                # Sampling
                n_sample = 1
                nzd_y_pred = self.sampler(_sess=_sess,_x=nzd_x_train,n_samples=n_sample)
                # Plot first dimensions of both input and output
                x_plot,y_plot = x_train[:,self.x_dim4plot],y_train[:,0] # Traning data 
                plt.figure(figsize=(8,4))
                plt.axis([np.min(x_plot),np.max(x_plot),np.min(y_plot)-0.1,np.max(y_plot)+0.1])
                h_tr,=plt.plot(x_plot,y_plot,'k.') # Plot training data
                for i in range(n_sample): 
                    ith_nzd_y_pred = nzd_y_pred[:,0:1,i]
                    h_pr,=plt.plot(x_plot,self.nzr_y.get_orgval(ith_nzd_y_pred),'b.') # Plot prediction
                plt.title("[%d/%d] name:[%s] loss_val:[%.3e]"%(epoch,_max_epoch,self.name,loss_val))
                plt.legend([h_tr,h_pr],['Train data','Predictions'],fontsize=13,loc='upper left')
                if self.x_name4plot != None:
                    plt.xlabel(self.x_name4plot,fontsize=13)
                plt.show()

        # Save final weights 
        if _SAVE_FINAL:
            self.save2npz(_sess,'net/net_%s_final.npz'%(self.name)) # Save the current best model 
            self.save2mat_from_npz(_x_train=x_train,_y_train=y_train,
                    _save_name='net/net_%s_final.mat'%(self.name),
                    _npz_path='net/net_%s_final.npz'%(self.name))

    # Test
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
        nzd_y_pred = self.sampler(_sess=_sess,_x=self.nzr_x.get_nzdval(_x_test))
        y_pred_test = self.nzr_y.get_orgval(nzd_y_pred[:,0:1,0])

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

        # Compute the test prediction error
        rmse = np.sqrt(np.mean((y_pred_test.squeeze()-y_test4plot.squeeze())**2))
        return rmse