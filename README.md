GAMMA_GLM
=========

The Python implementation of Gamma generalised linear model with MCMC parameter optimization.

###Model
Each data (1..N) has binary feature vetor f[i] = {0,1}^K (K = number of features)and response variable y[i] whichi is a positive real number.
Model assumes that response variable y[i] are taken from Gamma distributioin.
y[i] ~ Gamma(a[i],b[i])
where,
a[i] = exp(W_a^T*f[i])
b[i] = exp(W_b^T*f[i])

For huge number of features, ordinary optimization method based on gradient method would not usually work well. This implementation employs Metropolis-Hastings MCMC optimization instead.
