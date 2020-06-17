## DEVCON RF  TUNNING SCRIPT
## 30/04/2020

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
#load("./deconv_cgdata_cps_less_feat.RData")

# new features 20000
#load("deconv_cgdata_cps_new_feat.RData")

# newmix (25/05/2020) finegrain with Ben's samples
#load("deconv_fgdata_cps_new_feat_last3.RData")

# newmix (26/05/2020) coarse with Ben's samples
load("deconv_cgdata_cps_new_feat_last3.RData")

library(caret)
library(randomForest)
library(foreach)
require(doParallel)
#library(doMC)
library(dplyr)
#registerDoMC(cores=7)

# SETUP SNOW CLUSTER
primary <- '10.64.10.37' # SAMSON
machineAddresses <- list(
  list(host=primary,user='harpo',
       ncore=7),
  list(host='10.64.10.36',user='harpo', # KERRRIGAN
       ncore=8)
#  list(host='10.64.10.39',user='harpo', # JOKER
#       ncore=16)
)

spec <- lapply(machineAddresses,
               function(machine) {
                 rep(list(list(host=machine$host,
                               user=machine$user)),
                     machine$ncore)
               })
spec <- unlist(spec,recursive=FALSE)



mtry_range <- c(2,3,4,5,6,7,8,9,10,50,100,150,200)
ntree_range <- c(500)

parallelCluster <- parallel::makePSOCKcluster(
  spec,
  master=primary,
  homogeneous=T,manual=F)
registerDoParallel(parallelCluster)
print(paste("Workers: ",getDoParWorkers()))




parms <- expand.grid(mtry = mtry_range,ntree=ntree_range)
results_final <-c()
results_final_models <- list()
# Starting training  ---------
for (label_number in rownames(trainprop)) {
# create datasets  ----------
  labels <- trainprop[label_number, ]
  trainset <- t(train)
  #trainset <- scale(trainset,center=TRUE,scale=TRUE)
  #data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(test)
  #data_test <- cbind(label = labels_test, testset)
  # start parallel fine tuning  -----------
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    mtry <- parms[i,]$mtry
    ntree <- parms[i,]$ntree
  
    model <- randomForest::randomForest(
      y=labels,
      x= trainset,
      mtry = mtry,
      ntree = ntree,
      na.action=na.omit
      
    )
    
    preds <- predict(model, testset)
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
  # select bes model ---------
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$mtry, ", ", best_model$ntree," Pearson value : ", best_model$pearson,sep=""))
  model <- randomForest::randomForest(
    y = labels,
    x = trainset,
    mtry = best_model$mtry,
    ntree = best_model$ntree,
    na.action=na.omit
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = "results_rf_devcon_bestmodels_coarsegrain_data_cps_20000_newmix_last3.rdata",compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path="results_rf_devcon_coarsegrain_data_cps_20000_newmix_last3.csv")
}

# Shutdown cluster neatly
if(!is.null(parallelCluster)) {
  parallel::stopCluster(parallelCluster)
  parallelCluster <- c()
}
