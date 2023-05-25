args = commandArgs(trailingOnly=TRUE)  # gsa_order
# args = c('1')
library(sensitivity)

order = args[1]
X1 = read.csv('X1.csv')
X2 = read.csv('X2.csv')
xx = sobolEff(model=NULL, X1, X2, order=order, nboot=100, conf=0.95)
gsa_runs = read.csv('gsa_runs.csv')
gsa_runs2 = as.matrix(gsa_runs)
tell(xx, gsa_runs2)
Sindices = data.frame(xx$S)

Sindices2 = data.frame(t(Sindices))
write.csv(Sindices2, 'S_indices.csv')
