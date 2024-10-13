#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:15:53 2022

@author: mberar
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

#the state representation is the histogram of the face values
def get_states(nb_dices,nb_faces):
    #return states and associated probability
    Roll = [ [] for _ in range(nb_dices) ]
    Roll_P = []
    
    for it in itertools.product(range(nb_faces),repeat=nb_dices):
        for d in range(nb_dices):
            # each state is seen as a product of nb_dice 
            s = np.zeros((nb_faces),dtype='int')
            for f in range(nb_faces) :
                    s[f] = it[d:].count(f)
                    if s.sum() == nb_dices :
                        break
            Roll[d].append(s) 
      
    for  d in range(nb_dices):
        S,counts = np.unique(Roll[d],axis=0,return_counts=True)
        Roll[d] = list(S)
        Roll_P.append(counts/counts.sum())
    return Roll, Roll_P

def get_state_index(s):
    return np.argwhere((s == S).sum(axis=1)==nb_faces)[0][0]

def get_actions_from_state(s) :
    # action are described as keeped dices histogram (the reroll number is implicit)
    nb_actions = np.prod(s+1)
    A = np.zeros((nb_actions,nb_faces),dtype='int')   
    # first step is to generate the list of iterables
    l = []
    for f in range(nb_faces) :
        l.append(list(np.arange(s[f]+1)))
    k=0
    # then generates all possible actions
    for i in itertools.product(*l): 
       A[k,:]=np.array(i) 
       k = k+1
    return A

def get_actions_list(S):
    Aa = []
    for s in S:
        Aa = Aa + get_actions_from_state(s).tolist()
        Aa = list(np.unique(Aa,axis=0))
    return Aa

def get_action_index(a):
    return np.argwhere((a == Aa).sum(axis=1)==nb_faces)[0][0]



def get_states_from_action(a) :
    if a.sum() == nb_dices :
        return [a],np.array([1.])
    return  np.array(Roll[a.sum()])+a,Roll_P[a.sum()]


# BACKWARD RECURSION
def one_step_backward(v_out):
    v_in = np.zeros(len(S))
    a_in = np.zeros(len(Aa))
    Q_in = np.zeros((len(S),len(Aa)))

    for a in Aa:
        i_a = get_action_index(a)
        Sr,Pr = get_states_from_action(a)
        k = 0
        for k in range(len(Pr)) :
            i_sr = get_state_index(Sr[k])
            a_in[i_a]+=  Pr[k]*v_out[i_sr]
            k=k+1
        #########################"
    for s in S :
        A = get_actions_from_state(s)
        i_s = get_state_index(s)
        for a in A :
            i_a = get_action_index(a)
            Q_in[i_s][i_a] = a_in[i_a]
        v_in[i_s] = Q_in[i_s].max()
    
    return v_in, Q_in 


nb_faces = 4
nb_dices = 3
Roll, Roll_P = get_states(nb_dices,nb_faces)
S = Roll[0]
Aa = get_actions_list(S)

# false rewards for computation purpose
v_3 = np.arange(len(S))
# Recursion : getting v_2 and Q_2 from v_3
v_2,Q_2 =  one_step_backward(v_3)
v_1,Q_1 =  one_step_backward(v_2)    

#########################################################
#MC Control

def get_state_from_action(a) :
    #reroll number
    r = nb_dices - a.sum()
    dice_view = list(np.random.randint(0,high=nb_faces,size=r))
    s = np.zeros((nb_faces),dtype='int')
    for f in range(nb_faces) :
        s[f] = dice_view.count(f)
        if s.sum() == r :
            break
    return a+s

def choose_action(s,p):
    i_a = np.random.choice(np.arange(len(Aa)), p=p)
    return Aa[i_a],i_a


def create_uniform_policy():
    Policy = np.zeros((len(S),len(Aa)))
    for s in S:
        i_s = get_state_index(s)
        A = get_actions_from_state(s) 
        for a in A :
            i_a = get_action_index(a)
            Policy[i_s][i_a] = 1./len(A)
            
    return Policy


Policy_1 = create_uniform_policy()
Policy_2 = Policy_1.copy()


def generate_episode(Policy_1,Policy_2):
    # generate first state
    s = get_state_from_action(np.zeros((nb_faces),dtype='int'))
    i_s = get_state_index(s)
    a,i_a = choose_action(s,Policy_1[i_s])
    E =[(i_s,i_a,0)]
    
    #######################################""
    s = get_state_from_action(a)
    i_s = get_state_index(s)
    a,i_a = choose_action(s,Policy_2[i_s])
    ############################################"""
    s = get_state_from_action(a)
    r_s = get_state_index(s)
    E.append((i_s,i_a,v_3[r_s]))
    return E
    

def MCControl(Policy_1,Policy_2):
    epsilon = 0.1
    itermax = 10000
    
    Q_MC_1  = np.zeros((len(S),len(Aa)))
    Q_MC_2  = np.zeros((len(S),len(Aa)))
    
    # 
    Returns_1 =  [ [] for _ in range(len(S)*len(Aa)) ]
    Returns_2 =  [ [] for _ in range(len(S)*len(Aa)) ]
    
    for i in range(itermax) :
        E = generate_episode(Policy_1,Policy_2)
        # first policy
        i_s,i_a,r = E[0]
        index = np.ravel_multi_index((i_s,i_a), (len(S), len(Aa)))
        Returns_1[index].append(E[1][2])
        Q_MC_1[i_s][i_a] = np.asarray(Returns_1[index]).mean()
        
        support = np.argwhere(Policy_1[i_s] != 0)
        argmax = np.argmax(Q_MC_1[i_s])
        Policy_1[i_s][support] =  epsilon / len(support)
        Policy_1[i_s][argmax] += 1-epsilon     
        
        # second policy
        i_s,i_a,r = E[1]
        index = np.ravel_multi_index((i_s,i_a), (len(S), len(Aa)))
        Returns_2[index].append(E[1][2])
        Q_MC_2[i_s][i_a] = np.asarray(Returns_2[index]).mean()
        
    return Q_MC_1, Q_MC_2
        
        
# THe MC part is done
Q_MC_1, Q_MC_2 = MCControl(Policy_1,Policy_2)

print(Q_MC_1)



















