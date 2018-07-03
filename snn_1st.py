#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Weinrot
#
# Created:     07/03/2015
# Copyright:   (c) Weinrot 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import theano
import theano.tensor as T
import time
import numpy as np
import os
from Choose_frame import create_dataset, read_image
import math
from matplotlib import pyplot as plt
import sys
# !!!!!
# dt is changed manually into 1ms

class SNNgroup(object):
    def __init__(self,Ne,Ni,n_inp,W_inp=None,W_inner=None):
        '''class SNNgroup's self Parameters:
            self.A:       update matrix
            self.S:       neuron state varaibles
            self.W_inner: inner-connect weights in the group
            self.W_inp:   input weights
            self.spikes:  the spikes matrix in the time t
            self.SpkC  :  spike containers
            input : '''
        self.number = Ne+Ni
        self.Ne = Ne
        self.Ni = Ni
        self.mV=self.ms=1e-3    # units
        dt=1*self.ms     # timestep
        self.dt = dt
        taum=20*self.ms    # membrane time constant
        taue=5*self.ms
        taui=10*self.ms
        #self.Vt=-1*self.mV      # threshold = -50+49
        self.Vt = 15*self.mV     #threshold = -55+70
        #self.Vr=-11*self.mV     # reset = -60+49
        self.Vr = 0*self.mV      # reset = -70+70
        self.Vi = -10*self.mV    # VI = -80+70
        self.dApre = .0001
        #self.dApre = .95 #changed into .95
        self.dApost = -self.dApre*1.05
        self.tauP = 20*self.ms
        #self.input = input
        self.n_inp = n_inp
        self.weight = .001
        self.weightIn = 1.
        self.wmax = 200*self.weight
        zero = np.array([0]).astype(theano.config.floatX)
        self.zero = theano.shared(zero,name='zero',borrow=True)
        """
        Equations
        ---------
        eqs='''
        dv/dt = (ge*70mV-gi*10-(v+70*mV))/(20*ms) : volt
        dge/dt = -ge/(5*self.ms) : volt
        dgi/dt = -gi/(10*self.ms) : volt
        '''
        """
        # Update matrix
        A = np.array([[np.exp(-dt/taum),0,0],
                      [taue/(taum-taue)*(np.exp(-dt/taum)-np.exp(-dt/taue)),np.exp(-dt/taue),0],
                      [-taui/(taum-taui)*(np.exp(-dt/taum)-np.exp(-dt/taui)),0,np.exp(-dt/taui)]
                      ],dtype=theano.config.floatX).T
        A = theano.shared(value=A,name='A',borrow=True)
        self.A = A
        # State varible : [v;ge;gi] (size=3*self.number)
        S = np.ones((1,self.number),dtype=theano.config.floatX)*self.Vr
        S = np.vstack((S,np.zeros((2,self.number),dtype=theano.config.floatX)))
        self.S_init = S
        S = theano.shared(value=S,name='S',borrow=True)
        self.S = S
        if W_inner == None:
        # weights of inner connections (size= self.number*self.number)
            self.W_inner_ini = np.ones((self.number,self.number),dtype=theano.config.floatX)*self.weight
            #self.W_inner_ini[Ne:,:] = self.weightIn
            self.W_inner_ini[Ne:,:] = self.weight
            wtmp = np.eye(self.number)
            ind = wtmp.nonzero()
            self.W_inner_ini[ind]=0
            W_inner = theano.shared(value=self.W_inner_ini,name='W_inner',borrow=True)
            self.W_inner = W_inner
        else:
            self.W_inner = theano.shared(W_inner,name='W_inner',borrow=True)
        # weights of input connections (size=n_inp*self.number
        rng = np.random.RandomState(1234)
        if W_inp ==None:
            #W_inp = np.ones((self.n_inp,self.number)).astype(theano.config.floatX) #needs specification later
            #W_inp = np.random.rand(self.n_inp,self.number).astype(theano.config.floatX)*.00001*self.ms #needs specification later
            self.W_inp_ini = np.ones((self.n_inp,self.number)).astype(theano.config.floatX)*self.weight
            self.W_inp_ini[:,self.Ne:] = self.weightIn
            W_inp = theano.shared(self.W_inp_ini,name='W_inp',borrow=True)
            self.W_inp = W_inp
        else:
            self.W_inp = theano.shared(W_inp,name='W_inp',borrow=True)
        # Spike Container
        #spkC = theano.shared(value=np.empty((1,self.number)).astype(theano.config.floatX),name='spkC',borrow=True)
        spkC = np.empty((1,self.number)).astype(theano.config.floatX)
        self.spkC = spkC
        #spikes=np.empty((self.number,1),dtype=theano.config.floatX)
        #self.spikes = theano.shared(value=spikes,name='spikes',borrow=True)
        # not sure the dtype of sp_history
        self.sp_history = np.array([])
        #output = np.empty(self.number,dtype=theano.config.floatX)
        #self.output = theano.shared(value=output,name='output',borrow=True)
        self.V_record = np.empty((1,self.number))
        self.ge_record = np.empty((1,self.number))
        self.gi_record = np.empty((1,self.number))
        #================================================
        # Process Function Initial
        # input:: 0-1 vector

        '''Update Schedule:
        1.Update state variables of SNNgroup: dot(A,S)
        1.Update state variables of Synapses: dot(exp(-dt/tau),Ssynapse), including W_inp and W_inner
        2.Call thresholding function: S[0,:]>Vt
        3.Push spikes into SpikeContainer
        4.Propagate spikes via Connection(possibly with delays)
        5.Update state variables of Synapses (STDP)
        6.Call reset function on neurons which has spiked'''
        Ne = self.Ne
        Ni = self.Ni
        m = T.fmatrix(name='m')
        #self.Vt = T.as_tensor_variable(self.Vt,'Vt')

        # "Update state function:: stat()"
        # return np array
        # shape(stat()) = shape(self.S)
        S_update = T.dot(self.A,self.S)
        self.stat = theano.function(
            inputs = [],
            outputs = [],
            updates = {self.S : S_update})
        #============================================================
        # Update state of Synapses
        # Update matrix of Synapse
        A_STDP = np.array([[np.exp(-self.dt/self.tauP),0],[0,np.exp(-self.dt/self.tauP)]],dtype=theano.config.floatX)
        # Spre_inner :: pre  synapse of inner connections
        # Spost_inner:: post synapse of inner connections
        # Spre_inp   :: pre  synapse of input conenctions
        # Spost_inp  :: post synapse of input connections
        self.Spre_inner_ini = np.zeros((self.number,self.number),dtype=theano.config.floatX)
        Spre_inner = theano.shared(self.Spre_inner_ini,name='Spre_inner',borrow=True)
        self.Spre_inner = Spre_inner
        self.Spost_inner_ini = np.zeros((self.number,self.number),dtype=theano.config.floatX)
        Spost_inner = theano.shared(value=self.Spost_inner_ini,name='Spost_inner',borrow=True)
        self.Spost_inner = Spost_inner
        self.Spre_inp_ini = np.zeros((self.n_inp,self.number)).astype(theano.config.floatX) #needs specification later
        Spre_inp = theano.shared(value=self.Spre_inp_ini,name='Spre_inp',borrow=True)
        self.Spre_inp = Spre_inp
        self.Spost_inp_ini = np.zeros((self.n_inp,self.number)).astype(theano.config.floatX) #needs specification later
        Spost_inp = theano.shared(value=self.Spost_inp_ini,name='Spost_inp',borrow=True)
        self.Spost_inp = Spost_inp
        U = T.fscalar('U')
        UM = T.fmatrix('UM')
        #UpreV = theano.shared(A_STDP[0,0],name='UpreV',borrow=True) # Wpre = UpreV*Wpre
        #UpostV = theano.shared(A_STDP[1,1],name='UpostV',borrow=True)
        self.tmp = np.array(np.exp(-self.dt/self.tauP).astype(theano.config.floatX))
        self.SynFresh = theano.shared(self.tmp,name='SynFresh',borrow=True)
        self.UpdateSpre_inner = theano.function(inputs=[],outputs=None,updates={self.Spre_inner:T.dot(self.SynFresh,self.Spre_inner)},allow_input_downcast=True)
        self.UpdateSpost_inner = theano.function(inputs=[],outputs=None,updates={self.Spost_inner:T.dot(self.SynFresh,self.Spost_inner)},allow_input_downcast=True)
        self.UpdateSpre_inp = theano.function(inputs=[],outputs=None,updates={self.Spre_inp:T.dot(self.SynFresh,self.Spre_inp)},allow_input_downcast=True)
        self.UpdateSpost_inp = theano.function(inputs=[],outputs=None,updates={self.Spost_inp:T.dot(self.SynFresh,self.Spost_inp)},allow_input_downcast=True)
        #------------------------------------------

        #tmp = math.exp(-self.dt/self.tauP)
        #tmp = T.as_tensor(0.95122945)
        #================================================================
        #------------------------------------------
        # "thresholding function:: spike_fun()"
        # type return :: np.ndarray list
        # shape return:: shape(spike_fun()) = (self.number,)
        self.spike_fun = theano.function(
            inputs = [U], #[self.S]
            outputs = (T.gt(self.S[0,:],U))) #type outputs: np.ndarray,shape::(nL,)
            #'outputs = (self.S[0,:]>Vt).astype(theano.config.floatX)), #type outputs: list'
            #'updates={self.spikes:(self.S[0,:]>Vt).astype(theano.config.floatX)}'

        #------------------------------------
        #------------------------------------
        #=================================================================
        # "Push spike into Container function:: spCfun(vector)"
        # type vector :: np.array([],dtype=theano.config.floatX)!!!
        # type return :: np array
        # shape return:: shape(spCfun()) = ( shape(self.spkC)[0]+1 , shape(self.spkC)[1] )
            #updates={self.spkC:T.stack(self.spkC,sp)})
        '''spike_prop = theano.function( #wrong
            inputs = [],
            outputs =[],
            updates = {self.S:np.dot(self.W_inner,self.spikes)+self.S})#wrong'''
        #-------------------------------
        #--------------------------------
        #====================================================================
        # Propagate spikes
          # inner connection:
          # S_inner = f(inputs, outputs, updates)
          #   Param:: inputs: spike 0-1 vector
          #   Param:: inputs: spike is from function-> spike_fun
          #   S_inner(spk)::-> for i in spk[0:Ne].nonzero()[0]:
          #                        S[1,:] = Winner[i,:]+S[1,:]  (excitatory conenction)
          #                    for j in spk[Ne,:].nonzero()[0]:
          #                        S[2,:] = Winner[j,:]+S[2,:]  (inhibitory connection)
        vinner = T.fvector(name='vinner') # vinner = spk :: np.array((1,self.number)
        def add_f1(i,p,q):
            np = T.inc_subtensor(p[1,:],q[i,:]) #ge
            return {p:np}
        def add_f2(i,p,q):
            np = T.inc_subtensor(p[2,:],q[i,:]) #gi
            return {p:np}
        #deltaWinner1,updates1 = theano.scan(fn=lambda i: self.W_inner[i,:]*i+self.S[1,:], sequences=vinner[0:Ne])
        deltaWinner1,updates1 = theano.scan(fn=add_f1, sequences=vinner[0:Ne].nonzero()[0],non_sequences=[self.S,self.W_inner])
        #deltaWinner2,updates2 = theano.scan(fn=lambda i: self.W_inner[i,:]*i+self.S[2,:], sequences=vinner[Ne:])
        deltaWinner2,updates2 = theano.scan(fn=add_f2, sequences=vinner[Ne:].nonzero()[0]+self.Ne,non_sequences=[self.S,self.W_inner])
        # S = S+W
        self.S_inner1 = theano.function(inputs=[vinner],outputs=None,updates=updates1,allow_input_downcast=True)
        self.S_inner2 = theano.function(inputs=[vinner],outputs=None,updates=updates2,allow_input_downcast=True)
        #------------------------------------------
        #------------------------------------------
         # outter connection (input spikes):
         # type input: index list
        voutter = T.fvector(name='voutter')
        #deltaWoutter = theano.scan(fn=lambda j: self.W_inp[j,:]+self.S[1,:],sequences=voutter)
        deltaWoutter,updatesout1 = theano.scan(fn=add_f1,sequences=voutter.nonzero()[0],non_sequences=[self.S,self.W_inp])
        self.S_inp = theano.function(inputs=[voutter],outputs=None,updates=updatesout1,allow_input_downcast=True)
        #------------------------------------
        #-------------------------------------
        #=====================================================================

        # Update Synapses (STDP | STDC)

        # Pre::  Apre += self.dApre, w+=Apost
        # Post:: Apost+=self.dApost, w+=Apre
        #
        # USpreInner :: Perform Pre function No.1 in inner connections
        # UWInner    :: Perform Pre function No.2 in inner connections
        # UpreInner  :: Function
        def add_synap_pre(i,p,po,s,q):
            # i :: sequence
            # p :: pre | post
            # s :: dApre | dApost
            # q :: W
            index = T.nonzero(q[i,:self.Ne])
            np = T.inc_subtensor(p[i,index],s)
##            tmp = p[i,:]
##            tmp=T.inc_subtensor(tmp[index],s)
##            np=T.set_subtensor(p[i,:],tmp)
            #np = T.inc_subtensor(p[i,:],s)
            nw = T.inc_subtensor(q[i,:],po[i,:])
            nw=T.clip(nw,0,self.wmax)
            return {p:np,q:nw}

        def add_synap_pre_inp(i,p,po,s,q):
            # i :: sequence
            # p :: pre | post
            # s :: dApre | dApost
            # q :: W
            index = T.nonzero(q[i,:self.Ne])
            np = T.inc_subtensor(p[i,index],s)
##            tmp = p[i,:]
##            tmp=T.inc_subtensor(tmp[index],s)
##            np=T.set_subtensor(p[i,:],tmp)
            #np = T.inc_subtensor(p[i,:],s)
            nw = T.inc_subtensor(q[i,:],po[i,:])
            nw=T.clip(nw,0,self.wmax)
            return {p:np,q:nw}

        def add_synap_post(i,po,p,s,q):
            # i:: sequence
            # po:: post
            # p:: pre
            # s:: dA
            # q:: W
            index = T.nonzero(q[:self.Ne,i])
            npo = T.inc_subtensor(po[index,i],s)
            nw = T.inc_subtensor(q[:,i],p[:,i])
            nw = T.clip(nw,0,self.wmax)
            return {po:npo,q:nw}

        def add_synap_post_inp(i,po,p,s,q):
            # i:: sequence
            # po:: post
            # p:: pre
            # s:: dA
            # q:: W
            index = T.nonzero(q[:self.Ne,i])
            npo = T.inc_subtensor(po[index,i],s)
            nw = T.inc_subtensor(q[:,i],p[:,i])
            nw = T.clip(nw,0,self.wmax)
            return {po:npo,q:nw}

        add_dA = T.fscalar('add_dA')
        add_p,add_po,add_q = T.fmatrices('add_p','add_po','add_q')
        #-------------------------------------------------------------------------
        #USinner,updatesUinner = theano.scan(fn=add_synap_pre,sequences=vinner,non_sequences=[self.Spre_inner,self.Spost_inp,self.dApre,self.W_inner])
        'USinner,updatesUinner = theano.scan(fn=add_synap_pre,sequences=vinner.nonzero()[0],non_sequences=[add_p,add_po,add_dA,add_q])'
        #USinner1,updatesUinner1 = theano.scan(fn=add_synap_pre,sequences=vinner,non_sequences=[self.Spost_inner,self.Spre_inner,self.dApost,self.W_inner])
        #-------------------------------------------------------------------------
        #UpostInner = theano.function(inputs[vinner],updates={self.Spost_inner:USpostInner})
        #UpostInp = theano.function(inputs=[vinner],updates={self.W_inner:UWInnerpost})
        'USinner_f = theano.function(inputs=[vinner,add_p,add_po,add_dA,add_q],outputs=None,updates=updatesUinner)'
        #USinner_step2 = theano.function(inputs=[vinner,add_p,add_po,add_dA,add_q],outputs=None,updates=updatesUinner)
        USinner_inner_pre,updatesUinner_inner_pre = theano.scan(fn=add_synap_pre,sequences=vinner[:self.Ne].nonzero()[0],non_sequences=[self.Spre_inner,self.Spost_inner,add_dA,self.W_inner])
        self.USinner_f_inner_pre = theano.function(inputs=[vinner,add_dA],outputs=None,updates=updatesUinner_inner_pre,allow_input_downcast=True)

        USinner_innerpost,updatesUinner_inner_post = theano.scan(fn=add_synap_post,sequences=vinner[:self.Ne].nonzero()[0],non_sequences=[self.Spost_inner,self.Spre_inner,add_dA,self.W_inner])
        self.USinner_f_inner_post = theano.function(inputs=[vinner,add_dA],outputs=None,updates=updatesUinner_inner_post,allow_input_downcast=True)

        USinner_inp_pre,updatesUSinner_inp_pre =theano.scan(fn=add_synap_pre_inp,sequences=vinner.nonzero()[0],non_sequences=[self.Spre_inp,self.Spost_inp,add_dA,self.W_inp])
        self.USinner_f_inp_pre = theano.function(inputs=[vinner,add_dA],outputs=None,updates=updatesUSinner_inp_pre,allow_input_downcast=True)

        USinner_inp_post,updatesUSinner_inp_post =theano.scan(fn=add_synap_post_inp,sequences=vinner[:self.Ne].nonzero()[0],non_sequences=[self.Spost_inp,self.Spre_inp,add_dA,self.W_inp])
        self.USinner_f_inp_post = theano.function(inputs=[vinner,add_dA],outputs=None,updates=updatesUSinner_inp_post,allow_input_downcast=True)
        # Call reset function
        def reset_v(index,vr):
            nv = T.set_subtensor(self.S[0,index],vr)
            return{self.S:nv}
        resetV,resetV_update = theano.scan(fn=reset_v,sequences=vinner.nonzero()[0],non_sequences=[U])
        self.resetV_f = theano.function(inputs=[vinner,U],outputs=None,updates=resetV_update,allow_input_downcast=True)

        setvalue = T.fscalar('setvalue')
        iv = T.ivector('iv')
        def reset_state(i,value,state):
            nstate = T.set_subtensor(state[i,:],value)
            return {state:nstate}
        reset_S_state,Upreset_S_state = theano.scan(fn=reset_state,sequences=iv,non_sequences=[setvalue,self.S])
        self.reset_S_fn = theano.function(inputs=[iv,setvalue],outputs=None,updates=Upreset_S_state)


    def reset_syn(self):

        self.Spost_inner.set_value(self.Spost_inner_ini)
        self.Spost_inp.set_value(self.Spost_inp_ini)
        self.Spre_inner.set_value(self.Spre_inner_ini)
        self.Spre_inp.set_value(self.Spre_inp_ini)

        #self.reset_S_fn([1,2],0.)
        self.S.set_value(self.S_init)



    def state_update(self,input,itera,fired=None,stdp=False):
        Ne = self.Ne
        Ni = self.Ni
        #self.Spre_record = np.empty()
        if itera ==0:

            #=====================================================================
            #===================================================================
            # main process
            '''Update Schedule:
            1.Update state variables of SNNgroup: dot(A,S)
            1.Update state variables of Synapses: dot(exp(-dt/tau),Ssynapse), including W_inp and W_inner
            2.Call thresholding function: S[0,:]>Vt
            3.Push spikes into SpikeContainer
            4.Propagate spikes via Connection(possibly with delays)
            5.Update state variables of Synapses (STDP)
            6.Call reset function on neurons which has spiked'''

            #---------------------------------------------
            #----------------------------------------------
            #---------------
            #self.V_record = np.vstack((self.V_record,self.S.get_value()[0,:]))
            self.stat()
            # Record
##            self.V_record = np.vstack((self.V_record,self.S.get_value()[0,:]))
##            self.V_record = self.V_record[1:,:]
##            self.ge_record = np.vstack((self.ge_record,self.S.get_value()[1,:]))
##            self.ge_record = self.ge_record[1:,:]
##            self.gi_record=np.vstack((self.gi_record,self.S.get_value()[2,:]))
##            self.gi_record = self.gi_record[1:,:]
            # update Synapses
            if stdp!=False:
                self.UpdateSpre_inner()
                self.UpdateSpost_inner()
                self.UpdateSpre_inp()
                self.UpdateSpost_inp()
            # Thresholding
            spktmp = self.spike_fun(self.Vt) #(1,n), 0-1 vector, int8 np.array
            self.out = spktmp
            # supervised term
            if fired != None:
                # fired is a vector
                #fired = np.concatenate([fired,spktmp[self.Ne:]])
                self.error = len(np.nonzero(spktmp[:Ne] != fired)[0])
                spktmp = fired
            # spike containing
##            self.spkC = np.vstack((self.spkC,spktmp))
##            self.spkC = self.spkC[1:,:]
            # Propagate Spike
            if len(spktmp[0:Ne].nonzero()[0])!=0:
                self.S_inner1(spktmp)
            if len(spktmp[Ne:].nonzero()[0])!=0:
                self.S_inner2(spktmp)
            if len(input[:].nonzero()[0])!=0: # input is a 0-1 vector
                self.S_inp(input)
            # STDP
            if stdp!=False:
                if len(input[:].nonzero()[0])!=0:
                    #Pre  (inp)
                    self.USinner_f_inp_pre(input,self.dApre)

                if len(spktmp[:self.Ne].nonzero()[0])!=0:
                    #Pre (inner)
                    self.USinner_f_inner_pre(spktmp,self.dApre)
                    #Post (inner)
                    self.USinner_f_inner_post(spktmp,self.dApost)
                     #Post (inp)
                    self.USinner_f_inp_post(spktmp,self.dApost)


            # Reset
            if len(spktmp[:].nonzero()[0])!=0:
                self.resetV_f(spktmp,self.Vr)
            #-----------------
        #==========================================================================
        else:
##            if itera ==47:
##                print 'iteration is 47'
            #self.V_record = np.vstack((self.V_record,self.S.get_value()[0,:]))
            # updata state S
            self.stat()
            # Record
##            self.V_record = np.vstack((self.V_record,self.S.get_value()[0,:]))
##            self.ge_record = np.vstack((self.ge_record,self.S.get_value()[1,:]))
##            self.gi_record=np.vstack((self.gi_record,self.S.get_value()[2,:]))
            #update state of Synapses
            if stdp!=False:
                self.UpdateSpre_inner()
                self.UpdateSpost_inner()
                self.UpdateSpre_inp()
                self.UpdateSpost_inp()

            #threshold function
            spktmp = self.spike_fun(self.Vt) #(1,n), 0-1 vector, float32 np.array
            self.out = spktmp
            #supervised term
            if fired != None:
                #fired = np.concatenate([fired,spktmp[self.Ne:]])
                self.error = len(np.nonzero(spktmp[:Ne] != fired[:Ne])[0])
                spktmp = fired
    ##        else:
    ##            spktmp = self.spike_fun() #(1,n), 0-1 vector, float32 np.array

            # push spike into Container
##            self.spkC = np.vstack((self.spkC,spktmp))

            # propagate spikes
            if len(spktmp[0:self.Ne].nonzero()[0])!=0:
                self.S_inner1(spktmp)
            if len(spktmp[self.Ne:].nonzero()[0])!=0:
                self.S_inner2(spktmp)

            if len(input[:].nonzero()[0])!=0: # input is a 0-1 vector
                self.S_inp(input)

            # Update Synapses(STDP|STDC)
            if stdp!=False:
                if len(input[:].nonzero()[0])!=0:
                    self.USinner_f_inp_pre(input,self.dApre)

                if len(spktmp[:self.Ne].nonzero()[0])!=0:
                    self.USinner_f_inner_pre(spktmp,self.dApre)
                    self.USinner_f_inner_post(spktmp,self.dApost)
                    self.USinner_f_inp_post(spktmp,self.dApost)



            # Reset S
##            if itera ==47:
##                print 'iteration is 47'
            if len(spktmp[:].nonzero()[0])!=0:
                self.resetV_f(spktmp,self.Vr)



    def run(self):
        # run the network
        self.state_update()
def load_data(dataDir):
    # return img_data
    # shape:: (n_img, all_pixel)
    imageList = os.listdir(dataDir)
    dataset = []
    for i in imageList:
        tmp = read_image(dataDir+i).flatten()
        tmp = tmp/255.*4.
        dataset.append(tmp)
    return dataset

