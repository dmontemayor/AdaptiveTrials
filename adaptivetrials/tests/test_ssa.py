"""Test Stochastic Simulation Algorithm
"""
import numpy as np

#set random Seed
np.random.seed(1199167)

def test_a2b():
    """test N is conserved for A->B reaction.
    """
    from adaptivetrials.ssa import ssa
    #from adaptivetrials.ssa import plottrajectory

    #N=2 chemical species
    #interact by M=1 chemical reactions
    #this reaction is expressed A->B

    #init population condition (N)
    #NA(t=0) = 100
    #NB(t=0) = 0
    state = [100, 0]

    #rate vectors (M)
    #k_AB = 1
    kvec = [1]

    #Full stoichiometry matrix (2N x M)
    nmat = [[1],[0],[0],[1]]

    #run Simulation - returns a trajectory of poulations (N x npt)
    #where npt is some arbitrary number of incremented time points
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec, runtime=10)

    #assert sum of A and B populations at last time point = initial A population
    assert trajectory[-1,0]==sum(trajectory[0:1,-1])

    #assert time has passed
    assert trajectory.shape[1]>1

    #plottrajectory(trajectory)

    return

def test_fixed_schloglmodel():
    """test species A and B are fixed in schloglmodel.
    """
    from adaptivetrials.ssa import ssa
    #from adaptivetrials.ssa import plottrajectory

    #N=3 chemical species
    #interact by M=4 chemical reactions
    #expressed as follows
    # -----------
    # rxn1: A+2X -> 3X
    # rxn2: 3X -> A+2X
    # rxn3: B -> X
    # rxn4: X -> B
    # -----------
    # Species A and B are held constant

    #init population condition (N)
    #NA(t=0) = 1
    #NB(t=0) = 1
    #NX(t=0) = 0
    state = [1, 1, 0]

    #rate vectors (M)
    #k 1-4 => 3, .6, .25, 2.95
    kvec = [3.00, 0.60, 0.25, 2.95]

    #Full stoichiometry matrix (2N x M)
    nmat = [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [2, 3, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [3, 2, 1, 0]]

    #clamp species population (1/0 = true/false)
    clamp = [1, 1, 0]

    #run Simulation - returns a trajectory of poulations (N x npt)
    #where npt is some arbitrary number of incremented time points
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec, runtime=100, clamp=clamp)

    #plottrajectory(trajectory)

    #assert species A is const
    assert trajectory[0,0]==trajectory[0,-1]

    #assert species B is const
    assert trajectory[1,0]==trajectory[1,-1]

    return

def test_abbcmodel():
    """test DrugX is better short term than DrugY yet DrugY is better in long run.
    """
    import numpy as np

    from adaptivetrials.ssa import ssa
    from adaptivetrials.ssa import plottrajectory
    from adaptivetrials.ssa import plotweightedvalue

    #N=5 chemical species A, B, B_prime, C, and X
    #interact by M=9 chemical reactions
    #expressed as follows
    # -----------
    # rxn1: X -> A
    # rxn2: A -> X
    # rxn3: X -> C
    # rxn4: C -> X
    # rxn5: A -> C
    # rxn6: A -> B
    # rxn7: A -> B_prime
    # rxn8: B -> C
    # rxn9: B_prime -> C
    # -----------
    # Species X is held constant

    #init population condition (N)
    state = [0, 0, 0, 0, 1000]

    #clamp species population (1/0 = true/false)
    clamp = [0, 0, 0, 0, 1]

    #rate vectors (M)
    kvec_healthy = [1.0, 0.5, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 5.0]
    kvec_disease = [1.0, 0.5, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 5.0]
    kvec_drugx = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 5.0]
    kvec_drugy = [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 5.0]

    #Full stoichiometry matrix (2N x M)
    nmat = [[0, 1, 0, 0, 1, 1, 1, 0, 0],
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
    #weight = [0, -1, -1.7, 1, 0]
    weight = [0, -1, -2, 1, 0]
    weight = np.array(weight)
    nspec = np.size(weight)
    weight = np.reshape(weight, (nspec, 1))

    #average over n=5 patients
    pshort = []
    xshort = []
    yshort = []
    plong = []
    xlong = []
    ylong = []

    #Equilibrate healthy patient
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_healthy, runtime=10, clamp=clamp)
    # plot healthy species equilibration
    #plottrajectory(trajectory)
    #plot health trajectory
    #plotweightedvalue(trajectory,weight)

    #simulate acute onset of disease
    state = trajectory[0:5, -1]
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_disease, runtime=5, clamp=clamp)
    # plot onset species evolution
    #plottrajectory(trajectory)
    #plot health trajectory after onset until diagnosis
    #plotweightedvalue(trajectory,weight)

    #Retrieve state upon diagnosis defined when health measure drops below 250
    idx = np.where(weight.T.dot(trajectory[0:nspec, :]).T < 250)[0][0]
    istate = state = trajectory[0:5, idx]

    patients = range(5)
    for patient in patients:

        #simulate no treatment for 2 years after diagnosis
        placebo_trajectory, _ = ssa(state=istate, nmat=nmat, kvec=kvec_disease, runtime=2, clamp=clamp)
        trajectory = placebo_trajectory
        idx = np.squeeze(np.where(trajectory[-1, :] < .5))[-1]
        pshort.append(weight.T.dot(trajectory[0:nspec, idx]))
        idx = np.squeeze(np.where(trajectory[-1, :] < 2))[-1]
        plong.append(weight.T.dot(trajectory[0:nspec, idx]))

        #simulate drugX treatment for 2 years after diagnosis
        drugx_trajectory, _ = ssa(state=istate, nmat=nmat, kvec=kvec_drugx, runtime=2, clamp=clamp)
        trajectory = drugx_trajectory
        idx = np.squeeze(np.where(trajectory[-1, :] < .5))[-1]
        xshort.append(weight.T.dot(trajectory[0:nspec, idx]))
        idx = np.squeeze(np.where(trajectory[-1, :] < 2))[-1]
        xlong.append(weight.T.dot(trajectory[0:nspec, idx]))


        #simulate drugY treatment for 2 years after diagnosis
        drugy_trajectory, _ = ssa(state=istate, nmat=nmat, kvec=kvec_drugy, runtime=2, clamp=clamp)
        trajectory = drugy_trajectory
        idx = np.squeeze(np.where(trajectory[-1, :] < .5))[-1]
        yshort.append(weight.T.dot(trajectory[0:nspec, idx]))
        idx = np.squeeze(np.where(trajectory[-1, :] < 2))[-1]
        ylong.append(weight.T.dot(trajectory[0:nspec, idx]))

        # plot species evolution under no treatment
        #plottrajectory(placebo_trajectory)
        # plot species evolution under drugX treatment
        #plottrajectory(drugx_trajectory)
        # plot species evolution under drugY treatment
        #plottrajectory(drugy_trajectory)
        #plot health trajectory under drugY treatment
        #plotweightedvalue([placebo_trajectory, drugx_trajectory, drugy_trajectory], weight)

    #assert DrugX is better than DrugY in short term t=.5 by mean
    assert(np.mean(xshort) > np.mean(yshort))

    #assert DrugY is better than DrugX in long term at t=2 by mean
    assert(np.mean(ylong) > np.mean(xlong))

    return

def test_heteromodel():
    """test health decline is statistically the same
    between type1, and type2 disease subtype populations and faster than in
    healthy population.
    """
    import numpy as np

    from adaptivetrials.ssa import ssa
    from adaptivetrials.ssa import plottrajectory
    from adaptivetrials.ssa import plotweightedvalue

    #N=5 chemical species A, B, B_prime, C, and X
    #interact by M=9 chemical reactions
    #expressed as follows
    # -----------
    # rxn1: X -> A (slow)
    # rxn2: A -> X (slow)
    # rxn3: X -> C (very slow)
    # rxn4: C -> X (very slow)
    # rxn5: A -> B (fast)
    # rxn6: A -> B_prime (fast)
    # rxn7: B_prime -> A (fast)
    # rxn8: B -> C (slow)
    # rxn9: B_prime -> C (slow)
    # -----------
    # Species X is held constant

    #init population condition (N)
    istate = [0, 0, 0, 0, 200]

    #clamp species population (1/0 = true/false)
    clamp = [0, 0, 0, 0, 1]

    #rate vectors   X:A, A:X, X:C, C:X, A:B, B:b, b:A, B:C, b:C
    kvec_healthy = [0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0, 1.0, 1.0]
    kvec_type1 =   [0.5, 0.5, 0.5, 0.5, 6.0, 5.0, 5.0, 1.0, 1.0]
    kvec_type2 =   [0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 4.0, 1.0, 1.0]

    #Full stoichiometry matrix (2N x M)
    nmat = [[0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],

            [1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0]]

    #define health contribution for each species
    # A is begnign
    # B is begnign
    # B_prime begnign
    # C is bad
    # X is not in scope
    weight = [0, 0, 0, -1, 0]
    weight = np.array(weight)
    nspec = np.size(weight)
    weight = np.reshape(weight, (nspec, 1))

    #average over n=5 patients
    ncohort = 50
    eq_time = 5.0
    short_term = 1.0
    long_term = 10.0
    npt = int(long_term/short_term)
    healthy_data = np.zeros((npt,ncohort))
    type1_data = np.zeros((npt,ncohort))
    type2_data = np.zeros((npt,ncohort))
    for patient in range(ncohort):

        #Equilibrate healthy patient
        trajectory, _ = ssa(state=istate, nmat=nmat, kvec=kvec_healthy, runtime=eq_time, clamp=clamp)
        # plot healthy patient equilibration
        #plottrajectory(trajectory)
        #plot health trajectory
        #plotweightedvalue(trajectory,weight)

        #save equilibrated state
        eqstate = trajectory[0:nspec, -1]

        #simulate healthy patient
        state = eqstate
        trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_healthy, runtime=long_term, clamp=clamp)
        # plot onset species evolution
        #plottrajectory(trajectory)
        #plot health trajectory after onset until diagnosis
        #plotweightedvalue(trajectory,weight)
        #loop over short_term increments
        for ipt in range(npt):
            idx = np.squeeze(np.where(trajectory[-1, :] < short_term*(ipt+1)))[-1]
            healthy_data[ipt, patient] = weight.T.dot(trajectory[0:nspec, idx])

        #simulate type1 patient
        state = eqstate
        trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_type1, runtime=long_term, clamp=clamp)
        # plot onset species evolution
        #plottrajectory(trajectory)
        #plot health trajectory after onset until diagnosis
        #plotweightedvalue(trajectory,weight)
        for ipt in range(npt):
            idx = np.squeeze(np.where(trajectory[-1, :] < short_term*(ipt+1)))[-1]
            type1_data[ipt, patient] = weight.T.dot(trajectory[0:nspec, idx])

        #simulate type2 patient
        state = eqstate
        trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_type2, runtime=long_term, clamp=clamp)
        # plot onset species evolution
        #plottrajectory(trajectory)
        #plot health trajectory after onset until diagnosis
        #plotweightedvalue(trajectory,weight)
        for ipt in range(npt):
            idx = np.squeeze(np.where(trajectory[-1, :] < short_term*(ipt+1)))[-1]
            type2_data[ipt, patient] = weight.T.dot(trajectory[0:nspec, idx])


    print("healthy data")
    #print(healthy_data)
    print(np.vstack((np.mean(healthy_data, axis=1),np.std(healthy_data, axis=1))).T)

    print("type1 data")
    #print(type1_data)
    print(np.vstack((np.mean(type1_data, axis=1),np.std(type1_data, axis=1))).T)
    #print(np.mean(type1_data, axis=1))
    #print(np.std(type1_data, axis=1))

    print("type2 data")
    #print(type2_data)
    print(np.vstack((np.mean(type2_data, axis=1),np.std(type2_data, axis=1))).T)
    #print(np.mean(type2_data, axis=1))
    #print(np.std(type2_data, axis=1))

    #assert(np.mean(xshort) > np.mean(yshort))

    ##assert DrugY is better than DrugX in long term at t=2 by mean
    #assert(np.mean(ylong) > np.mean(xlong))

    assert False

def test_oscmodel():
    """assert oscillating state can be observed.
    """
    import numpy as np

    from adaptivetrials.ssa import ssa
    from adaptivetrials.ssa import plottrajectory
    from adaptivetrials.ssa import plotweightedvalue

    #N=5 chemical species A, B, C, D, X
    #interact by M=4 chemical reactions
    # modeled after Bray–Liebhafsky reaction
    #expressed as follows
    # -----------
    # rxn1: X + A -> B
    # rxn2: X + B -> A
    # rxn1: X + C -> D
    # rxn2: X + D -> C
    # -----------
    # Z is consumed

    #init population condition (N)
    state = [10, 10, 10, 10, 1]

    #clamp species population (1/0 = true/false)
    clamp = [0, 0, 0, 0, 1]

    #rate vectors (M)
    kvec = [1.0, 1.0, 1.0, 1.0]

    #Full stoichiometry matrix (2N x M)
    nmat = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],

            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 0]]

    #define health contribution for each species
    weight = [1, 0, 1, 0, 0]
    weight = np.array(weight)
    nspec = np.size(weight)
    weight = np.reshape(weight, (nspec, 1))

    #Simulate
    trajectory, ierr = ssa(state=state, nmat=nmat, kvec=kvec, runtime=20, clamp=clamp)
    print(ierr)
    # plot trajectory
    #plottrajectory(trajectory[[0,1,2,-1],:])
    plottrajectory(trajectory)
    plotweightedvalue(trajectory,weight)

    assert False
    return
