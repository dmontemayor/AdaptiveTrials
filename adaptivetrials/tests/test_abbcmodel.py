"""Test ABBC model"""
import numpy as np
from math import ceil
from adaptivetrials.abbc_model import *

#set random Seed
np.random.seed(1199167)

#define short and long term constants
short_term = 0.5
mid_term = 1.5
long_term = 3.0

def test_power_and_reward_crossover():
    """test sample size increases as effect size between drugx and drugy crossover with time.
    Samplesize is calculated assuming 80% power and 0.05 significance.
    Assert max samplesize is greater than 100 patients.
    Assert DrugX has greater effect than DrugY at short term while DrugY has
    greater effect at long term.
    """
    #define pilot cohort size
    npatients = 10

    #simulate drugX and drugY policies for long term in short term increments
    nstep = int(long_term/short_term)

    #init rewards per patient per time for drugx and drugy policies
    drugx_reward = np.zeros((nstep,npatients))
    drugy_reward = np.zeros((nstep,npatients))

    #init drugX simulation
    simulator = AbbcEnvironment(patients=npatients)
    #define action taken
    action = np.repeat(1, npatients)
    #main simulation loop
    for step in range(nstep):
        _, drugx_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #init drugY simulation
    simulator = AbbcEnvironment(patients=npatients)
    #define action taken
    action = np.repeat(2, npatients)
    #main simulation loop
    for step in range(nstep):
        _, drugy_reward[step,:] = simulator.take_action(action=action, simtime=short_term)

    #calculate sample size required to resolve effect size for each simulation step.
    zalpha = 1.96 #critical zvalue for p=1-alpha/2
    zbeta = 0.8416 #critical zvalue for p=1-beta
    zsquared = (zalpha + zbeta)**2
    #calculate the diference in effect size
    delta = (np.mean(drugx_reward, axis=1) - np.mean(drugy_reward, axis=1))
    samplesize = np.divide((np.var(drugx_reward, axis=1) + np.var(drugx_reward, axis=1))*zsquared,delta**2)
    print(samplesize)
    print(delta)

    #assert max sample size is greater than 100
    assert np.max(samplesize)>100

    #assert max sample size at is at least 50 times the mins at short and long terms
    assert np.max(samplesize)>50*samplesize[0]
    assert np.max(samplesize)>50*samplesize[-1]

    #assert DrugX is better short term than DrugY
    assert delta[0]>0
    #assert DrugY is better in long term than DrugX
    assert delta[-1]<0
