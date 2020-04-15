import numpy as np
from audit_lab_procedure import *

def bayes_procedure(pop, initial_sample_size, escalation_factor, alpha,\
                    n_trials, pseudocount_base, reported_winners = ['Alice'],\
                    verbose=False):
    """
    simulate the audit-lab Bayesian audit and return the upset probability
    at the conclusion of the audit.
    either the upset probability is below alpha and the audit stops, or the audit ends 
    up being a full hand count
    
    
    Inputs
    ------
    pop : array-like
        a list of actual votes (names of candidates) in a random order
        Invalid votes, undervotes, and similar should be represented by 'invalid.'
        It is assumed that every candidate received at least one vote (the list of candidates
        is assumed to be the distinct names in pop).
    initial_sample_size : int
        initial sample size
    escalation_factor : float
        multiplier to increase the size of the audit sample as the sample expands. 
        Should be >1 or the sample will not grow.
    alpha : float
        Bayesian upset probability limit
    n_trials : int
        number of times to sample from the posterior
    pseudocount_base : float
        initial value of the flat Dirichlet-multinomial prior
    reported_winners : list
        name(s) of the candidate(s) reported to have won. The social choice function is
        assumed to be plurality: whoever gets the most votes wins. The number of winners
        the social choice function allows is inferred from the cardinality of reported_winners
    verbose : bool
        print sample size and the observed posterior at each step? Defaults to False
        
    Returns
    -------
    float : posterior probability when the audit stops
    
    """

    N = len(pop)
    assert initial_sample_size <= N, 'initial sample size exceeds population size'
    assert escalation_factor > 1, 'escalation factor too small'
    candidates = np.unique(pop)
    reported_losers = [c for c in candidates if not(np.isin(c, reported_winners))]
    n_winners = len(reported_winners)
    sample_tally = {c : 0 for c in candidates}
    prior = initialize_dirichlet(candidates,\
                    prior_parameters = [pseudocount_base]*len(candidates))
    posterior = 1
    n_iter = 0
    incremental_sample_size = initial_sample_size
    total_sample_size = 0

    pop_c = pop.copy()

    while (posterior > alpha and total_sample_size < N):
        wrong = 0
        n_iter += 1
    
        # Augment the sample
        new_sam = pop_c[:incremental_sample_size]
        pop_c = pop_c[incremental_sample_size:]
        new_sample_tally = tally_sample(candidates, new_sam)
        prior = update_dirichlet(prior, new_sample_tally)
        sample_tally = update_sample_tally(sample_tally, new_sample_tally)
    
        # Estimate posterior probability that any of the reported winners lost
        for i in range(n_trials):
            pseudodata = generate_pseudo_pop(sample_tally, prior, N)
            least_winner = np.min([pseudodata[c] for c in reported_winners])
            greatest_loser = np.max([pseudodata[c] for c in reported_losers])
            if least_winner <= greatest_loser:
                wrong += 1
        posterior = wrong/n_trials

        if verbose:
            print(n_iter, total_sample_size, posterior)

        total_sample_size = total_sample_size + incremental_sample_size
        incremental_sample_size = int(total_sample_size*(escalation_factor-1))
        
    return posterior


# Maine simulation
np.random.seed(2683587038)
alpha = 0.05
pseudocount_base = 0.5

trials = 10000  # replications to estimate risk - 10000
n_trials = 1000  # simulations in Bayesian audit - 1000
initial_sample_size = 25 
escalation_factor = 1.2

# Population size
pop_size = int(281371)
# Margin
m = 0.01247
margin = int(m*pop_size)
pop = ['Poliquin']*int(pop_size/2-margin/2)+ ['Golden']*int(pop_size/2+margin/2)
bayes_risk = np.zeros(trials)
print('\nballots:', pop_size, 'margin:', m)
for i in range(trials):
            np.random.shuffle(pop)
            bayes_risk[i] = bayes_procedure(pop=pop, 
                                initial_sample_size=initial_sample_size,\
                                escalation_factor=escalation_factor, \
                                alpha=alpha,\
                                n_trials=n_trials,\
                                reported_winners = ["Poliquin"], \
                                pseudocount_base=pseudocount_base)
            if i % 1000 == 0:
                print('iteration', i, 'estimated risk', np.mean(bayes_risk[:i+1]<=alpha))
freq_risk = np.mean(bayes_risk <= alpha)
print('margin', margin, 'votes', 100*m, '%; estimated risk', freq_risk, freq_risk/alpha)