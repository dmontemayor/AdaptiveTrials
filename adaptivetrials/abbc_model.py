"""Define ABBC model - agent and environment"""

import numpy as np
from adaptivetrials.ssa import ssa
from crpm.ffn_bodyplan import read_bodyplan
from crpm.ffn_bodyplan import init_ffn
from crpm.ffn_bodyplan import copy_ffn
from crpm.fwdprop import fwdprop
from crpm.gradientdecent import gradientdecent

#from enums import *
#import random

class AbbcEnvironment:
    """ ABBC Emulator """

    def __init__(self, patients=1, validation_fraction=0, dropout_rate=0):
        """init population with 1000 units of metabolite X
            N=5 chemical species A, B, B_prime, C, and X
            interact by M=9 chemical reactions
            expressed as follows
            -----------
            rxn1: X -> A
            rxn2: A -> X
            rxn3: X -> C
            rxn4: C -> X
            rxn5: A -> C
            rxn6: A -> B
            rxn7: A -> B_prime
            rxn8: B -> C
            rxn9: B_prime -> C
            -----------
            Species X is held constant
            """
        #declare number of chemical species
        self.nspec = 5

        #clamp species population (1/0 = true/false)
        self.__clamp = [0, 0, 0, 0, 1]

        #init population condition
        self.istate = [0, 0, 0, 0, 1000]

        #rate vectors (M)
        self.__kvec_healthy = [1.0, 0.5, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 5.0]
        self.__kvec_disease = [1.0, 0.5, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 5.0]
        self.__kvec_drug1 = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 5.0]
        self.__kvec_drug2 = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 5.0]

        #Full stoichiometry matrix (2N x M)
        self.__nmat = [[0, 1, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0, 0, 0, 0],

                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 0, 0, 1, 1],
                       [0, 1, 0, 1, 0, 0, 0, 0, 0]]

        #define health contribution for each species
        # X is not in scope
        # A is begnign
        # C is good
        # B is bad
        # B_prime is worse
        #self.__weight = np.array([0, -1, -1.7, 1, 0])
        self.__weight = np.array([0, -1, -2, 1, 0])
        self.__weight = np.reshape(self.__weight, (self.nspec, 1))

        #if less than 1 patient throw warning and set patients to 1
        if patients < 1: #throw warning and set patients to 1
            print("Number of patients must be positive integer.")
            print("Will recover by setting number of patients to default value.")
            patients = 1

        #ensure validation fraction is non negative
        if validation_fraction < 0:
            validation_fraction = 0
        #set the validation fraction
        self.validation_fraction = validation_fraction

        #ensure the dropout rate is not negative
        if dropout_rate < 0:
            dropout_rate = 0
        #set the dropout_rate
        self.dropout_rate = dropout_rate

        #init at least one new patient
        self.state = self.newpatient()
        #this patient is automatically not in the validation group.
        self.validation = np.array([False])
        #this patient is has not yet withdrawn because just enrolled
        self.withdrawn = np.array([False])
        #init this patient's visit counter
        self.visit = np.array([0])

        #enroll more patients if needed
        if patients > 1:
            self.enroll(patients=(patients-1))


    def newpatient(self):
        """returns state of newly diagnosed patient"""
        #Equilibrate healthy patient
        trajectory, _ = ssa(state=self.istate, nmat=self.__nmat,
                            kvec=self.__kvec_healthy, runtime=10,
                            clamp=self.__clamp)
        #simulate acute onset of disease until diagnosis
        undiagnosed = True
        while undiagnosed:
            state = np.copy(trajectory[0:self.nspec, -1])
            trajectory, _ = ssa(state=state, nmat=self.__nmat,
                                kvec=self.__kvec_disease, runtime=.5,
                                clamp=self.__clamp)
            qvalue = self.__weight.T.dot(trajectory[0:self.nspec, :])
            #check for diagnosis
            if np.any(qvalue < 250):
                #Retrieve state upon diagnosis defined when health measure drops below 250
                idx = np.where(qvalue.T < 250)[0][0]
                undiagnosed = False
        return trajectory[0:self.nspec, idx].reshape((self.nspec, 1))

    def enroll(self, patients=1):
        """enroll new diagnosed patients."""
        istart = self.state.shape[1]
        #for every new patient, calculate patient state at diagnosis
        for _ in range(istart, (istart+patients)):
            #add new patient to state
            self.state = np.hstack((self.state, self.newpatient()))
            #assign patient to validation group if ratio is under the validation_fraction
            isval = False
            if np.mean(self.validation) < self.validation_fraction:
                isval = True
            self.validation = np.append(self.validation, isval)
            #this patient is has not yet withdrawn because just enrolled
            self.withdrawn = np.append(self.withdrawn, False)
            #init this patient's visit counter
            self.visit = np.append(self.visit, 0)

    def dropout(self):
        """keep track of patients who have withdrawn"""
        patients = self.state.shape[1]

        #track patients for only 6 visits
        self.withdrawn = np.where(self.visit > 6, True, self.withdrawn)

        #randomly drop patients with fixed probability
        self.withdrawn = np.where(np.random.random_sample(patients) < self.dropout_rate, True, self.withdrawn)

    def take_action(self, action, simtime=.5):
        """simulate drug treatment for 0.1 units of time"""
        #create reward vector
        patients = self.state.shape[1]
        reward = np.zeros(patients)
        #simulate action for each patient
        for patient in range(patients):
            kvec = None #default action is take no medication
            if action[patient] == 0:
                # No Treatment: use disease kvector
                kvec = self.__kvec_disease
            if action[patient] == 1:
                # drug1 treatment
                kvec = self.__kvec_drug1
            if action[patient] == 2:
                # drug2 treatment
                kvec = self.__kvec_drug2

            #conduct simulation
            if kvec is not None:
                trajectory, _ = ssa(state=self.state[:, patient], nmat=self.__nmat, kvec=kvec, runtime=simtime, clamp=self.__clamp)
                self.state[:, patient] = trajectory[0:self.nspec, -1]
                #increment patient's visit counter
                self.visit[patient] += 1
                #calculate reward for given action
                reward[patient] = self.__weight.T.dot(self.state[:, patient])
            else:
                self.state[:, patient] = self.istate

        return np.copy(self.state), reward

class Drug1Policy:
    """ This agent will always say to take drug 1. """
    def __init__(self):
        """ create drug1 policy agent """
        pass #nothing to initialize

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        # always take drug1: action == 1 unless patient is withdrawn
        return np.where(withdrawn, np.nan, 1)
        #return np.repeat(1, state.shape[1])

    def update(self, state, action, reward, new_state, validation=None):
        pass # nothing to update! policy never changes!!

class Drug2Policy:
    """ This agent will always say to take drug 2. """
    def __init__(self):
        """ create drug2 policy agent """
        pass #nothing to initialize

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        # always take drug2: action == 2
        return np.where(withdrawn, np.nan, 2)
        #return np.repeat(2, state.shape[1])

    def update(self, state, action, reward, new_state, validation=None):
        pass # nothing to update! policy never changes!!

class DqlAgent:
    def __init__(self, discount=0.95, exploration_rate=1.0, exploration_rate_decay=.99, target_every=1):
        """ define deep network hyperparameters"""
        self.discount = discount # how much future rewards are valued w.r.t. current
        self.exploration_rate = exploration_rate # initial exploration rate
        self.exploration_rate_decay = exploration_rate_decay # transition from exploration to expliotation
        self.target_every = target_every #how many iterations to skip before we copy prediciton network to target network

        #retrieve the body plan
        #input has 5 neurons, one for each metabolite conc.
        #output has 3 neurons, representing the Q values for each of the 3 actions
        #   action 0 is no treatment, action 1 is drug1 Tx, and and action 2 is for drug2 Tx
        self.bodyplan = read_bodyplan("adaptivetrials/models/abbc_bodyplan.csv")

        #define prediction network
        self.prednet = init_ffn(self.bodyplan)
        self.loss = None #current prediction error

        #init the target network with the prediciton network
        self.targetnet = copy_ffn(self.prednet)

        #init counter used to determine when to update target network with prediction network
        self.iteration = 0

   # Ask model to estimate Q value for current state (inference)
    def get_qvalue(self, state):
        # prediction network input: array of 5 values representing metabolite conc.
        #  output: Array of 3 Q values given state
        prediction, _ = fwdprop(state, self.prednet)
        return prediction

   # Ask model to calcualte Q value keeping current policy
    def get_target_qvalue(self, state):
        # target network input: array of 5 values representing metabolite conc.
        #  output: Array of 3 Q values given state
        prediction, _ = fwdprop(state, self.targetnet)
        return prediction

    def get_next_action(self, state, withdrawn):
        """ returns an array of actions with size of the number of columns in state"""
        #get number of patients
        patients = state.shape[1]
        greedy_actions = self.greedy_action(state)
        random_actions = self.random_action(patients)
        actions = np.where(np.random.random_sample(patients) > self.exploration_rate, greedy_actions, random_actions)
        return np.where(withdrawn, np.nan, actions)

    # Which action has bigger Q-value, estimated by our model (inference).
    def greedy_action(self, state):
        # argmax picks the higher Q-value and returns the index for every patient
        return np.argmax(self.get_qvalue(state), axis=0)

    def random_action(self, size):
        return np.random.randint(self.bodyplan[-1]["n"], size=size)

    def train(self, state, action, reward, new_state, validation=None):
        """ will train deep network model with the log of the state to predict Qvalues"""
        # Ask the model for the Q values of the current state (inference)
        state_q_values = self.get_qvalue(state)
        print("state_q_values")
        print(state_q_values)

        # Ask the model for the Q values of the new state (target)
        new_state_q_values = self.get_target_qvalue(new_state)
        print("new_state_q_values")
        print(new_state_q_values)

        print("actions")
        print(action)

        # Real Q value for the action we took. This is what we will train towards.
        #loop over valid actions
        for iact in range(state_q_values.shape[0]):
            #get patients who took this action
            patients = np.where(action == iact, True, False)
            #update Q Values if any patients took this action
            if np.sum(patients) > 0:
                state_q_values[iact, patients] = reward[patients] + self.discount * np.amax(new_state_q_values[:, patients])

        # Train prediciton network
        nfeat = state.shape[0] #network input size
        nlabels = new_state_q_values.shape[0] #network output size
        if validation is None or np.sum(validation) < 1:
            #get data from patients that participated
            intrain = np.squeeze(~np.isnan(action))
            nobv = np.sum(intrain)
            #exit if too few participated
            if nobv < 1:
                print("too few participants found for training")
                return
            #otherwise proceed with training
            traindata = state[:, intrain].reshape((nfeat, nobv))
            print("training data")
            print(traindata)
            print("training labels")
            print(state_q_values[:, intrain].reshape((nlabels, nobv)))
            _, self.loss, _ = gradientdecent(self.prednet,
                                             traindata,
                                             state_q_values[:, intrain].reshape((nlabels, nobv)),
                                             "mse")
        else:
            #partition out validation patients from dataset
            intrain = np.logical_and(~validation, ~np.isnan(action))
            invalid = np.logical_and(validation, ~np.isnan(action))
            nobv = np.sum(intrain)
            nobv_v = np.sum(invalid)
            #intrain = np.squeeze(np.where(np.logical_and(~validation, ~np.isnan(action))))
            #invalid = np.squeeze(np.where(np.logical_and(validation, ~np.isnan(action))))
            #exit if too few participated
            if nobv < 1:
                print("too few participants found for training")
                return
            if nobv_v < 1:
                print("too few participants found for validation")
                return
            #otherwise proceed with training
            traindata = state[:, intrain].reshape((nfeat, nobv))
            validata = state[:, invalid].reshape((nfeat, nobv_v))
            _, self.loss, _ = gradientdecent(self.prednet,
                                             traindata,
                                             state_q_values[:, intrain].reshape((nlabels, nobv)),
                                             "mse",
                                             validata=validata,
                                             valitargets=state_q_values[:, invalid].reshape((nlabels, nobv_v)),
                                             earlystop=True)
            print("loss")
            print(self.loss)

    def update(self, state, action, reward, new_state, validation=None):

        # Train our model with new data
        self.train(state, action, reward, new_state, validation)

        # Periodically, copy the prediction network to the target network
        if self.iteration % self.target_every == 0:
            self.targetnet = copy_ffn(self.prednet)

        # Finally shift our exploration_rate toward zero (less gambling)
        self.exploration_rate *= self.exploration_rate_decay

        #increment iteration counter
        self.iteration += 1
