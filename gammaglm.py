import numpy as np
from scipy.special import gammaln

ITER = 1000
EPS = 0.2
SIGMA = 0.1
BURNIN = 100
INTERVAL = 10

def likelihood(Wa, Wb, y, X):
    a = np.exp(np.dot(X, Wa))
    b = np.exp(np.dot(X, Wb))
    return float(((a - 1.) * np.log(y) - gammaln(a) - a * np.log(b) - y / b).sum(axis=0))

def update_Wa(index, Wa, Wb, y, X):
    lik0 = likelihood(Wa, Wb, y, X) - (Wa[index]**2 / (2 * SIGMA**2))
    new_Wa = Wa.copy()
    new_Wa[index] += EPS * np.random.normal()
    lik1 = likelihood(new_Wa, Wb, y, X) - (new_Wa[index]**2 / (2 * SIGMA**2))
    if lik1 > lik0: # to avoid overflow caused by exp function ...
        # accept
        return new_Wa,1
    elif np.random.random() < np.exp(lik1 - lik0):
        # accept
        return new_Wa,1
    else:
        # revert
        return Wa,0

def update_Wb(index, Wa, Wb, y, X):
    lik0 = likelihood(Wa, Wb, y, X) - (Wb[index]**2 / 2 * SIGMA**2)
    new_Wb = Wb.copy()
    new_Wb[index] += EPS * np.random.normal()
    lik1 = likelihood(Wa, new_Wb, y, X) - (new_Wb[index]**2 / 2 * SIGMA**2)
    if lik1 > lik0:
        # accept
        return new_Wb,1
    elif np.random.random() < np.exp(lik1 - lik0):
        # accept
        return new_Wb,1
    else:
        # revert
        return Wb,0

def MCMC(data):
    y = np.array(data[:,0],ndmin=2).T
    X = data[:,1:]
    n_samples, n_features = X.shape
    print 'n_samples: ',n_samples,'n_features: ',n_features,'\n'
    X_c = np.hstack((X, np.array([[1.]]*n_samples))) # add constant
    Wa = np.array([[0.]]*(n_features+1))
    Wb = np.array([[0.]]*(n_features+1))
    randomized_index = range(2*(n_features+1))
    np.random.shuffle(randomized_index)
    ofp_wa = open('Wa.sample','w')
    ofp_wb = open('Wb.sample','w')
    ofp = open('mcmc.log','w')
    for i in range(ITER):
        print 'iteration :',i
        total_accept = 0
        for index in randomized_index:
            if index < (n_features+1):
                Wa,accept = update_Wa(index, Wa, Wb, y, X_c)
            else:
                Wb,accept = update_Wb(index-(n_features+1), Wa, Wb, y, X_c)
            total_accept += accept
        Likelihood = likelihood(Wa, Wb, y, X_c)
        AcceptanceRatio = float(total_accept) / float(len(randomized_index)) * 100.
        print '\t LogLikelihood ... %.8f \t Acceptance Ratio ... %.2f%%\n'%(Likelihood,AcceptanceRatio)
        ofp.write('\t'.join([str(i),str(Likelihood),str(AcceptanceRatio)])+'\n')
        if (i > BURNIN) and (i % INTERVAL == 0):
            ofp_wa.write('\t'.join([str(x) for x in Wa.T.ravel()])+'\n')
            ofp_wb.write('\t'.join([str(x) for x in Wb.T.ravel()])+'\n')
    ofp.close()
    ofp_wa.close()
    ofp_wb.close()
    return Wa, Wb

if __name__ == '__main__':
    data = np.loadtxt('./data/data.tsv')
    Wa,Wb = MCMC(data)
    outmodel = open('model.tsv','w')
    for (a,b) in zip(Wa.ravel(),Wb.ravel()):
        outmodel.write(str(a)+'\t'+str(b)+'\n')
    outmodel.close()
