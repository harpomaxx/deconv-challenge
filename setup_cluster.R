## Setup Cluster
#Sys.setenv(PATH = paste0(Sys.getenv('PATH'), ':/usr/lib/rstudio-server/bin/postback'))
require("foreach")
require("doParallel") # install doParallel in all the hosts

primary <- '10.64.10.37' # remember to use IP or FQDN
machineAddresses <- list(
  list(host=primary,user='harpo',
       ncore=4),
  list(host='10.64.10.36',user='harpo',
       ncore=4)
)

spec <- lapply(machineAddresses,
               function(machine) {
                 rep(list(list(host=machine$host,
                               user=machine$user)),
                     machine$ncore)
               })
spec <- unlist(spec,recursive=FALSE)

parallelCluster <- parallel::makePSOCKcluster(
                                         spec,
                                         master=primary,
                                         homogeneous=T,manual=F)


registerDoParallel(parallelCluster)

print(parallelCluster)
getDoParWorkers()

x <- iris[which(iris[,5] != "setosa"), c(1,5)]
trials <- 10000
ptime <- system.time({
r <- foreach(icount(trials), .combine=cbind) %dopar% {
    ind <- sample(10000, 10000, replace=TRUE)
    result1 <- glm(x[ind,2]~x[ind,1], family=binomial(logit))
    coefficients(result1)
    }
  })[3]
ptime
print(paste(ptime))
# Shutdown cluster neatly
if(!is.null(parallelCluster)) {
  parallel::stopCluster(parallelCluster)
  parallelCluster <- c()
}
