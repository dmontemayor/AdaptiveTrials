"""Test Stochastic Simulation Algorithm
"""

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
    #from adaptivetrials.ssa import plottrajectory
    #from adaptivetrials.ssa import plotweightedvalue

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
    kvec_drugx =   [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 5.0]
    kvec_drugy =   [1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 5.0]

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
    weight = [0, -1, -1.7, 1, 0]
    weight = np.array(weight)
    nspec = np.size(weight)
    weight = np.reshape(weight, (nspec, 1))

    #Equilibrate healthy patient
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_healthy, runtime=10, clamp=clamp)
    # plot healthy species equilibration
    #plottrajectory(trajectory)
    #plot health trajectory
    #plotweightedvalue(trajectory,weight)

    #simulate acute onset of disease
    state = trajectory[0:5,-1]
    trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_disease, runtime=10, clamp=clamp)
    # plot onset species evolution
    #plottrajectory(trajectory)
    #plot health trajectory after onset until diagnosis
    #plotweightedvalue(trajectory,weight)

    #Retrieve state upon diagnosis defined when health measure drops below 250
    idx = np.where(weight.T.dot(trajectory[0:nspec,:]).T<250)[0][0]

    #simulate no treatment for 5 years after diagnosis
    state = trajectory[0:5,idx]
    placebo_trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_disease, runtime=10, clamp=clamp)

    #simulate drugX treatment for 5 years after diagnosis
    state = trajectory[0:5,idx]
    drugx_trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_drugx, runtime=10, clamp=clamp)

    #simulate drugY treatment for 5 years after diagnosis
    state = trajectory[0:5,idx]
    drugy_trajectory, _ = ssa(state=state, nmat=nmat, kvec=kvec_drugy, runtime=10, clamp=clamp)

    # plot species evolution under no treatment
    #plottrajectory(placebo_trajectory)
    # plot species evolution under drugX treatment
    #plottrajectory(drugx_trajectory)
    # plot species evolution under drugY treatment
    #plottrajectory(drugy_trajectory)
    #plot health trajectory under drugY treatment
    #plotweightedvalue([placebo_trajectory,drugx_trajectory,drugy_trajectory],weight)

    #assert DrugX is better than DrugY in short term t=.1
    idx = np.squeeze(np.where(drugx_trajectory[-1,:]<.1))[-1]
    idy = np.squeeze(np.where(drugy_trajectory[-1,:]<.1))[-1]
    Qx = weight.T.dot(drugx_trajectory[0:nspec,idx])
    Qy = weight.T.dot(drugy_trajectory[0:nspec,idy])
    assert(Qx>Qy)

    #assert DrugY is better than DrugX in long term t=9
    idx = np.squeeze(np.where(drugx_trajectory[-1,:]<9))[-1]
    idy = np.squeeze(np.where(drugy_trajectory[-1,:]<9))[-1]
    Qx = weight.T.dot(drugx_trajectory[0:nspec,idx])
    Qy = weight.T.dot(drugy_trajectory[0:nspec,idy])
    assert(Qy>Qx)

    return
