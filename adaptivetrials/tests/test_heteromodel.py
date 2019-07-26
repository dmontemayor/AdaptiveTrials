"""Test Heterogeneous Disease model
"""
import numpy as np
from math import ceil
#from adaptivetrials.hetero_model import *

#set random Seed
np.random.seed(1199167)

#define short and long term constants
short_term = 0.5
mid_term = 1.5
long_term = 3.0

def test_q_decline():
    """assert Q decline is statistically the same
    between type1, and type2 disease subtype populations and faster than in
    healthy population.
    """
    #define pilot cohort size
    #npatients = 10

    #simulate for long term in short term increments
    #nstep = int(long_term/short_term)

    #init rewards per patient per time
    #phi0_reward = np.zeros((nstep,npatients)) #healthy patients
    #phi1_reward = np.zeros((nstep,npatients)) #type 1 patients
    #phi1_reward = np.zeros((nstep,npatients)) #type 2 patients

    ##init healthy simulation
    #simulator = abbc_environment(patients=npatients, phenotype_prob=[1,0,0])
    ##define action taken
    #action = np.repeat(0, npatients)
    ##main simulation loop
    #for step in range(nstep):
    #    _, phi0_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #init type1 simulation
    #simulator = HeteroEnvironment(patients=npatients, phenotype_prob=[0,1,0])
    #define action taken
    #action = np.repeat(0, npatients)
    #main simulation loop
    #for step in range(nstep):
    #    _, phi1_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #init type2 simulation
    #simulator = HeteroEnvironment(patients=npatients, phenotype_prob=[0,0,1])
    #define action taken
    #action = np.repeat(0, npatients)
    #main simulation loop
    #for step in range(nstep):
    #    _, phi2_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    assert False

    #calculate the Q decline at each step
    #phi1_rate = phi1_reward[-1:1,:]-phi1_reward[-2:0,:]
    #phi2_rate = phi2_reward[-1:1,:]-phi2_reward[-2:0,:]

    ##assert type1 and type2 Q decline averaged over all steps is statistically the same
    #_, pvalue = stats.ttest_ind(np.mean(phi1_rate, axis=0), np.mean(phi2_rate, axis=0))
    #assert pvalue > .05

    ##calculate sample size required to resolve effect size for each simulation step.
    #zalpha = 1.96 #critical zvalue for p=1-alpha/2
    #zbeta = 0.8416 #critical zvalue for p=1-beta
    #zsquared = (zalpha + zbeta)**2
    ##calculate the diference in effect size
    #delta = (np.mean(phi1_rate, axis=1) - np.mean(phi2_rate, axis=1))
    #samplesize = np.divide((np.var(phi1_rate, axis=1) + np.var(phi2_rate, axis=1))*zsquared,delta**2)
    #print(samplesize)
    #print(delta)

    #assert max sample size is greater than 100
    #assert np.max(samplesize)>100


    """assert metabolites A, B, and Bprime are correlated with Q decline similarly
    between healthy, type1, and type2 disease subtype populations.
    """

    """assert drug1 slows Q decline in type1 patients and not (or less so) in
    type2 patients and healthy patients.
    """

    """assert drug2 slows Q decline in type2 patients and not (or less so) in
    type1 patients and healthly patients.
    """
