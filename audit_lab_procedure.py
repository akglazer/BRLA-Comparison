import numpy as np

def initialize_dirichlet(candidates, prior_parameters=None):
    """
    Create the initial prior for the Dirichlet distribution
    
    Parameters
    ----------
    candidates : array-like
        names of all candidates in the contest
    prior_parameters : array-like
        parameter values for the Dirichlet distribution, corresponding to each candidate
        size of the values corresponds to confidence (?)
        if None supplied, then use a flat prior
        
    Returns
    -------
    dict of parameter values, indexed by candidate
        
    """
    if prior_parameters == None:
        return {cand : len(candidates)/2 for cand in candidates}
    else:
        assert len(prior_parameters)==len(candidates)
        return {candidates[i] : prior_parameters[i] for i in range(len(candidates))}
    
    
def test_initialize_dirichlet():
    candidates = ['A', 'B']
    prior = initialize_dirichlet(candidates, [0]*2)
    assert prior == {'A':0, 'B':0}
    prior = initialize_dirichlet(candidates)
    assert prior == {'A':1, 'B':1}
    
    
def tally_sample(candidates, sample):
    """
    Tally the observed sample
    
    Parameters
    ----------
    candidates : array-like
        names of all candidates in the contest
    sample : array-like
        observed sample, with labels that match values in candidate list
        
    Returns
    -------
    dict of the tallied sample, indexed by candidate
        
    """
    sample_tally = {candidates[i] : 0 for i in range(len(candidates))}
    for s in sample:
        sample_tally[s] += 1
    return sample_tally
    
    
def test_tally_sample():
    candidates = ['A', 'B']
    sample = ['A']*10 + ['B']*5
    assert tally_sample(candidates, sample) == {'A':10, 'B':5}
    
    
def update_dirichlet(prior_parameters, sample_tally):
    """
    Update the prior for the Dirichlet distribution after observing the sample
    
    Parameters
    ----------
    prior_parameters : dict
        prior parameter values for the Dirichlet distribution, indexed by candidate
    sample_tally : dict
        observed number of votes in the sample, indexed by candidate

    Returns
    -------
    dict of parameter values, indexed by candidate
        
    """
    updated_param = {cand: prior_parameters[cand] + sample_tally[cand] for cand in prior_parameters}
    return updated_param
    
    
def update_sample_tally(sample_tally, new_sample_tally):
    """
    Update the sample tally after observing a new sample
    
    Parameters
    ----------
    sample_tally : dict
        observed number of votes in the original sample, indexed by candidate    
    new_sample_tally : dict
        observed number of votes in the new sample, indexed by candidate
        
    Returns
    -------
    dict of the tallied sample, indexed by candidate
        
    """
    sample_tally = {cand : sample_tally[cand] + new_sample_tally[cand] for cand in sample_tally}
    return sample_tally


# Step 3 (don't call this fun directly - it gets called in Step 4)
def sample_from_dirichlet(prior_parameters):
    """
    Draw a sample from the prior
    
    Parameters
    ----------
    prior_parameters : dict
        prior parameter values for the Dirichlet distribution, indexed by candidate

    Returns
    -------
    dict of draws from the prior, indexed by candidate
    These correspond to "probabilities" that add up to 1.
        
    """
    rvs = {cand : np.random.gamma(prior_parameters[cand]) for cand in prior_parameters}
    tot = sum(rvs.values())
    rvs = {cand : rvs[cand]/tot for cand in rvs}
    return rvs
    
    
def test_sample_from_dirichlet():
    np.random.seed(2345)
    prior = {'A':1, 'B':1}
    np_vals = [np.random.dirichlet([1, 1])[0] for i in range(1000)]
    our_vals = [sample_from_dirichlet(prior)['A'] for i in range(1000)]
    np.testing.assert_almost_equal(np.mean(np_vals), np.mean(our_vals), 2)
    np.testing.assert_almost_equal(np.std(np_vals), np.std(our_vals), 2)
    
    prior = {'A':5, 'B':1}
    np_vals = [np.random.dirichlet([5, 1])[0] for i in range(5000)]
    our_vals = [sample_from_dirichlet(prior)['A'] for i in range(5000)]
    np.testing.assert_almost_equal(np.mean(np_vals), np.mean(our_vals), 2)
    np.testing.assert_almost_equal(np.std(np_vals), np.std(our_vals), 2)
    

# Step 3.5 (don't call this fun directly - it gets called in Step 4)

def sample_from_multinomial(probs, sample_size):
    """
    Draw a sample from the Dirichlet-multinomial, where a sample from the
    Dirichlet prior sets the probabilities for the multinomial distribution.
    
    Parameters
    ----------
    probs : dict
        parameter values for the multinomial distribution, indexed by candidate
    sample_size : int
        number of samples to draw

    Returns
    -------
    dict with a tally of the draws, indexed by candidate
        
    """
    cand_sorted = sorted(probs)
    ps_sorted = [probs[cand] for cand in cand_sorted]
    multinomial_freqs_sorted = np.random.multinomial(sample_size, ps_sorted)
    freq = {vote: vote_freq for (vote, vote_freq) in zip(cand_sorted, multinomial_freqs_sorted)}
    return freq


def test_sample_from_multinomial():
    np.random.seed(2345)
    probs = {'A':0.5, 'B':0.5}
    np_vals = [np.random.multinomial(n=5, pvals=[0.5, 0.5])[0] for i in range(5000)]
    our_vals = [sample_from_multinomial(probs, sample_size=5)['A'] for i in range(5000)]
    np.testing.assert_almost_equal(np.mean(np_vals), np.mean(our_vals), 1)
    np.testing.assert_almost_equal(np.std(np_vals), np.std(our_vals), 1)
    
    probs = {'A':0.9, 'B':0.1}
    np_vals = [np.random.multinomial(n=5, pvals=[0.9, 0.1])[0] for i in range(5000)]
    our_vals = [sample_from_multinomial(probs, sample_size=5)['A'] for i in range(5000)]
    np.testing.assert_almost_equal(np.mean(np_vals), np.mean(our_vals), 2)
    np.testing.assert_almost_equal(np.std(np_vals), np.std(our_vals), 1)
    
    

# Step 4
def sample_dirichlet_multinomial(prior_parameters, sample_size):
    """
    Draw a sample from the Dirichlet-multinomial, where a sample from the
    Dirichlet prior sets the probabilities for the multinomial distribution.
    
    Parameters
    ----------
    prior_parameters : dict
        prior parameter values for the Dirichlet distribution, indexed by candidate
    sample_size : int
        number of samples to draw

    Returns
    -------
    dict with a tally of the draws, indexed by candidate
        
    """
    multinomial_parameters = sample_from_dirichlet(prior_parameters)
    freq = sample_from_multinomial(multinomial_parameters, sample_size)
    return freq


# Putting it all together: Combine observed sample with fake one
def generate_pseudo_pop(sample_tally, prior, N):
    n = sum(sample_tally.values())
    unsample_size = N-n
    fake_data = sample_dirichlet_multinomial(prior, unsample_size)
    total_votes = {cand : sample_tally[cand] + fake_data[cand] for cand in fake_data}
    return total_votes