"""Test deep Q learning on abbc model"""
import numpy as np
from scipy import stats
from adaptivetrials.abbc_model import *

#set random Seed
np.random.seed(1199167)

#def main q learning loop
def run_simulation(agent, simulator, maxstep, update=True, enroll_rate=0):
    """
    Define Q learning orchestration

    Args:
        agent : determines policy, at every simulation step will decide which
            action to take for every patient in the simulator.
        simulator: determines the evolution of multiple patients in a simulation
            (cohort). Simulator evolves the state of each patient for one
            simulation step given the action dermined by the agent and calculates
            the reward of that state-action pair.
        maxstep: total integer number of simulation steps
        update: Boolean flag when true will allow agent to update the policy.
            When false, the agent will apply its current policy for the duration
            of the simulation.
        enroll_rate: integer number of new patients to add to the
            simulator at every simulation step.
    ChangeLog:
        + Randomly enroll new patients - Let simulator determine dropout rate
        + Have training, validation, and tesing patients
            - testing and validation patients enrolled together at 7:3 ratio
                + validation patients used for naive early stopping (overfitting)
            - testing patients enrolled once policy is set - used for reporting
    """

    #simulation step time increment
    stepsize = 0.5

    #minibatch size
    minibatchsize = 10

    #scaling factor
    factor = .001

    #create action and reward logs
    actionlog = np.full([maxstep, simulator.state.shape[1]], np.nan)
    rewardlog = np.full([maxstep, simulator.state.shape[1]], 0.0)
    for step in range(maxstep):

        print("- - - - S T E P - - - -  "+str(step))

        #store current state
        state = np.copy(simulator.state)
        state *= factor

        #get withdrawn npatients
        withdrawn = np.copy(simulator.withdrawn)

        #query agent for the next action
        action = agent.get_next_action(state, withdrawn)

        #take action, get new state and reward
        new_state, reward = simulator.take_action(action, simtime=stepsize)
        new_state *= factor
        reward *= factor

        #log actions and rewards
        actionlog[step, :] = action
        rewardlog[step, :] = reward

        #get patients with valid actions (technically should be defined by withdrawn)
        patients = np.where(np.isnan(action), False, True)

        #init experience log at first simulation step
        #if step == 0:
        #    replaybuffer = {"state": state[:,patients],
        #                    "action": action[patients],
        #                    "reward": reward[patients],
        #                    "new_state": new_state[:,patients]}
        #else:#accumulate experience log
        #    replaybuffer["state"] =  np.hstack((replaybuffer["state"],state[:,patients]))
        #    replaybuffer["action"] =  np.hstack((replaybuffer["action"],action[:,patients]))
        #    replaybuffer["reward"] =  np.hstack((replaybuffer["reward"],reward[:,patients]))
        #    replaybuffer["new_state"] =  np.hstack((replaybuffer["new_state"],new_state[:,patients]))

        #let agent update policy
        if update:
            #prepare mini-batch
            #Select random experiences
            #get validation identifier
            validation = np.copy(simulator.validation)
            agent.update(state, action, reward, new_state, validation)

        #drop out patients who need to leave study
        simulator.dropout()

        #periodically enroll new patients
        if enroll_rate > 0:
            simulator.enroll(enroll_rate)
            #add new action and reward columns for every newly enrolled patient
            actionlog = np.hstack((actionlog, np.full([maxstep, enroll_rate], np.nan)))
            rewardlog = np.hstack((rewardlog, np.full([maxstep, enroll_rate], 0.0)))

    return rewardlog, actionlog


def test_benchmark_policies_for_short_and_long_term_rewards():
    """test a drug2 policy has higher long term rewards than the drug2 policy
        and drug1 policy has higher short term rewards

        We will use a chort of 10 patients to decide if rewards are different based
        on previous power analysis.
    """
    from scipy import stats
    #constants
    maxstep = 6 #max simulation step
    cohort_size = 5

    #benchmark simulation with drug1 agent
    agent = Drug1Policy()
    simulator = AbbcEnvironment(patients=cohort_size)
    rewardlog, actionlog = run_simulation(agent, simulator, maxstep, update=False)
    drug1_short_reward = rewardlog[0, :] #np.sum(rewardlog, axis=0)
    drug1_long_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug1 rewards")
    print(rewardlog)
    print(actionlog)
    #assert all actions were drug 1
    #assert(all(action == 1 for action in actionlog))
    assert(np.all(actionlog == 1 ))

    #benchmark simulation with drug2 agent
    agent = Drug2Policy()
    simulator = AbbcEnvironment(patients=cohort_size)
    rewardlog, actionlog = run_simulation(agent, simulator, maxstep, update=False)
    drug2_short_reward = rewardlog[0, :] #np.sum(rewardlog, axis=0)
    drug2_long_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug2 rewards")
    print(rewardlog)
    print(actionlog)
    #assert all actions were drug 2
    assert(np.all(actionlog == 2 ))
    #assert(all(action == 2 for action in actionlog))

    #assert drug2 rewards are better in long run on average
    assert drug2_long_reward.mean() > drug1_long_reward.mean()

    #assert long rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_long_reward, drug2_long_reward)
    assert pvalue < .05

    #assert drug1 rewards are better in short run on average
    assert drug1_short_reward.mean() > drug2_short_reward.mean()

    #assert short rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_short_reward, drug2_short_reward)
    assert pvalue < .05


def test_dql_agent_randomwalk():
    """test dql agent will take random drug with no exploration decay rate.
    """
    #constants
    nstep = 100 #max q learning steps
    factor = .001 #reward and state scaling factor

    #conduct deep q learning
    agent = DqlAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=1.0)
    simulator = AbbcEnvironment()
    #store current state
    state = np.copy(simulator.state)
    state *= factor
    #get withdrawn npatients
    withdrawn = np.copy(simulator.withdrawn)
    #init action statisitcs
    action = np.zeros((nstep,1))
    #ask for next action many times to get statistics on action chosen
    for step in range(nstep):
        action[step,] = agent.get_next_action(state, withdrawn)
    print("randomw walk")
    print(action)

    #assert distribution of actions are statistically the same
    margin = 1.5/np.sqrt(nstep)
    assert np.mean(action==0) < 1/3 + margin
    assert np.mean(action==1) < 1/3 + margin
    assert np.mean(action==2) < 1/3 + margin
    assert np.mean(action==0) > 1/3 - margin
    assert np.mean(action==1) > 1/3 - margin
    assert np.mean(action==2) > 1/3 - margin

def test_dql_agent_updates_Q_properly():
    """test dql agent tends to prefer to take any drug over no treatment.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6
    testing_cohort = 10

    #conduct deep q learning
    agent = DqlAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=0.9)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate=1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #Takes any drug more than 80% of the time
    assert np.mean(actionlog>0) > .80


def test_dql_agent_selects_drug1():
    """test dql agent will preferentially select drug 1 with discount rate = 0.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6
    testing_cohort = 10

    #conduct deep q learning
    agent = DqlAgent(discount=0.0, exploration_rate=1.0, exploration_rate_decay=0.9)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate=1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #Takes drug1 more than 50% of the time
    assert np.mean(actionlog == 1) > .50

    assert False


def test_dql_policy_against_naive_short_term_solution():
    """test a dql policy has higher long term rewards than the drug2 policy.
    """
    #constants
    training_steps = 24 #max q learning steps
    training_cohort = 3
    testing_steps = 6   #simulation steps of 0.5 time units
    testing_cohort = 10

    #conduct deep q learning
    agent = DqlAgent(discount=0.95, exploration_rate=1.0, exploration_rate_decay=.90)
    simulator = AbbcEnvironment(patients=training_cohort, validation_fraction=.3)
    rewardlog, actionlog = run_simulation(agent, simulator, training_steps, enroll_rate = 1)
    print("dql training")
    print(actionlog)
    print(rewardlog)

    #simulate trained dql agent with fixed policy
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    dql_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("dql testing")
    print(actionlog)
    print(rewardlog)

    #benchmark simulation with drug1 agent (will always take drug1)
    agent = Drug1Policy()
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, drug1_actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    drug1_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug1 rewardlog")
    print(rewardlog)

    #assert trained dql policy rewards are better in long run than drug1 policy
    assert drug1_reward.mean() < dql_reward.mean()

    #assert two rewards are significantly different
    _, pvalue = stats.ttest_ind(drug1_reward, dql_reward)
    assert pvalue < .05
    print (pvalue)

    #benchmark simulation with drug2 agent (will always take drug2)
    agent = Drug2Policy()
    simulator = AbbcEnvironment(patients=testing_cohort)
    rewardlog, actionlog = run_simulation(agent, simulator, testing_steps, update=False)
    drug2_reward = rewardlog[-1, :] #np.sum(rewardlog, axis=0)
    print("drug2 rewardlog")
    print(rewardlog)

    #assert trained dql policy rewards statistically the same as drug2 policy rewards
    _, pvalue = stats.ttest_ind(drug2_reward, dql_reward)
    assert pvalue > .05
    print (pvalue)

    assert False
