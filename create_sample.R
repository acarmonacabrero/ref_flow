args = commandArgs(trailingOnly=TRUE)  # n_gsa_runs, gsa_order, random_state
# args = c('16000', '1', '1')  # arguments are string when passed from the Python code
library(sensitivity)
options(scipen=99999)

empirical_RSsobolSample = function(data, n, order=1, type=sobolEff){
  # Type can be sobol, sobol2007 or sobolEff
  X1 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  X2 = data.frame(matrix(nrow=n, ncol=dim(data)[2]))
  var = 0
  for (i in 1:(dim(data)[2])){
    var = var + 1
    # These two matrices need to be different
    set.seed(random_state*var)
    X1[, var] = sample(data[, var], n, replace=TRUE)
    set.seed(random_state*var*2 + 1)  # *2 + 1, so the matrices are different 
    X2[, var] = sample(data[, var], n, replace=TRUE)
    write.csv(X1, 'X1.csv', row.names = FALSE)  # These files are read by the analysis .R code later
    write.csv(X2, 'X2.csv', row.names = FALSE)
  }
  xx = sobolEff(model=NULL, X1, X2, order=order, nboot=100, conf=0.95)
  return(xx)
} 

# n_gsa_runs, gsa_order, random_state
n = strtoi(args[1])
order = strtoi(args[2])
random_state = strtoi(args[3])

X = read.csv('X.csv')
k = dim(X)[2]

xx = empirical_RSsobolSample(data=X, order=order, n=n, type=sobolEff)

gsa_sample = xx$X
colnames(gsa_sample) = colnames(X)
write.csv(gsa_sample, row.names=FALSE, 'gsa_sample.csv')

