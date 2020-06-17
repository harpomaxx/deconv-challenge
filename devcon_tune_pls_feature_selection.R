## DEVCON PLS TUNNING SCRIPT
## 05/15/2020

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
# Load RF models
load("results_rf_devcon_bestmodels_finegrain_data_cps_20000_newmix_last3.rdata")
# Load dataset with full set of reatures
load("deconv_fgdata_cps_new_feat_last3.RData")
#library(caret)
require(tibble)
require(randomForest)
require(pls)
require(foreach)
require(doParallel)
require(dplyr)
require(optparse)
#registerDoMC(cores=6)


# SETUP SNOW CLUSTER
primary <- '10.64.10.37' # SAMSON
primary <- 'localhost' # cabildo
machineAddresses <- list(
  list(host=primary,user='harpo',
       ncore=7)#,
#  list(host='10.64.10.36',user='harpo', # KERRRIGAN
#       ncore=8),
#  list(host='10.64.10.39',user='harpo', # KERRRIGAN
#       ncore=16)
)

spec <- lapply(machineAddresses,
               function(machine) {
                 rep(list(list(host=machine$host,
                               user=machine$user)),
                     machine$ncore)
               })
spec <- unlist(spec,recursive=FALSE)




ncomp_range<-c(3,4,5,6,7,8,9,10,11,12,13,14)
nop_range<-c(1)

# Intuitively, the gamma parameter defines how far the influence of a single training example reaches, 
# with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as 
# the inverse of the radius of influence of samples 
# selected by the model as support vectors.

#### MAIN 

option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--nfeature", action="store", type="numeric", default=500, help = "Set the maximun number of features ")
)

opt <- parse_args(OptionParser(option_list=option_list))


## Select features using RF
rf_models<-results_final_models
select_features <-function(data,label){
  
  varimp<-randomForest::importance(rf_models[[label]]) 
  varimp<- varimp %>% as.data.frame() %>% tibble::add_column(feature=rownames(varimp))# %>% 
  
  best_features<-(varimp %>% arrange(desc(IncNodePurity)))[1:opt$nfeature,] %>% select(feature) %>% unlist() %>% unname()
  data[which(data %>% rownames() %in% best_features),] 
}

parallelCluster <- parallel::makePSOCKcluster(
  spec,
  master=primary,
  homogeneous=T,manual=F)
registerDoParallel(parallelCluster)
print(paste("Workers: ",getDoParWorkers()))



parms <- expand.grid(ncomp = ncomp_range,nop=nop_range)
results_final <-c()
results_final_models <- list()

for (label_number in rownames(trainprop)) {
  
  labels <- trainprop[label_number, ]
  trainset <- t(select_features(train,label_number))
  data_train <- cbind(label = labels, trainset) %>% as.data.frame()
  
  labels_test <- testprop[label_number, ]
  testset <- t(select_features(test,label_number))
  data_test <- cbind(label = labels_test, testset) %>% as.data.frame()
  
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
      ncomp  <- parms[i,]$ncomp
      #ncomp <- parms$ncomp
    model <- pls::plsr(
      label ~ .,
      data = data_train,
      scale = FALSE,
      center = FALSE,
      ncomp = ncomp
    
    )
    
    preds <- predict(model, data_test,ncomp = ncomp)
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    #readr::write_csv(partial_results,path=paste(e,"_",c,"_partial_results_svm_devcon.csv"))
    partial_results 
    
  }
  best_model <- results %>% mutate(spearson=(pearson+spearman)/2) %>% arrange(desc(spearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ", best_model$label_number," : ",best_model$ncomp,
              " Pearson value : ", best_model$pearson %>% round(digits = 4),
              " Spearman value : ", best_model$spearman %>% round(digits = 4),
              " Spearson value : ", best_model$spearson %>% round(digits = 4),
              sep=""))
  data_train<-rbind(data_train,data_test)
  labels<-rbind(labels,labels_test)
  model <- pls::plsr(
    label ~ .,
    data = as.data.frame(data_train),
    scale = FALSE,
    center = FALSE,
    ncomp = best_model$ncomp
    )
  results_final_models[[label_number]]<-list(model=model,ncomp=best_model$ncomp)
  save(results_final_models,file = paste0("results_pls_devcon_bestmodels_finegrain_data_cps_20000_2_noscale",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("results_pls_devcon_finegrain_cps_20000_2_noscale_center",opt$experimenttag,".csv"))
}

# Shutdown cluster neatly
if(!is.null(parallelCluster)) {
  parallel::stopCluster(parallelCluster)
  parallelCluster <- c()
}
