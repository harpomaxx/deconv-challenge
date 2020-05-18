## DEVCON SVM TUNNING SCRIPT
## 02/12/2019

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
# Load RF models
load("results_rf_devcon_bestmodels_fgdata_cps_20000_feat_2.rdata")
# Load dataset with full set of reatures
load("deconv_cgdata_cps_new_feat.RData")
library(caret)
require(tibble)
require(randomForest)
require(e1071)
require(foreach)
require(doParallel)
require(dplyr)
require(optparse)
#registerDoMC(cores=6)


# SETUP SNOW CLUSTER
primary <- '10.64.10.37' # SAMSON
machineAddresses <- list(
  list(host=primary,user='harpo',
       ncore=7),
  list(host='10.64.10.36',user='harpo', # KERRRIGAN
       ncore=8),
  list(host='10.64.10.39',user='harpo', # KERRRIGAN
       ncore=16)
)

spec <- lapply(machineAddresses,
               function(machine) {
                 rep(list(list(host=machine$host,
                               user=machine$user)),
                     machine$ncore)
               })
spec <- unlist(spec,recursive=FALSE)





epsilon_range <- c(5e-3, 5e-2, 5e-5,5e-4,5e-6)
cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32,64,128,256)
#cost_range <- c(0.1, 0.5, 1, 2, 4,8,16,32,64,128,256,384,512,768)
#cost_range <- c(256,384,512,768)
gamma_range <- c(1e-8,1e-7,1e-6,1e-5,1e-04)

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
select_features_rf <-function(data,label){
  
  varimp<-randomForest::importance(rf_models[[label]]) 
  varimp<- varimp %>% as.data.frame() %>% tibble::add_column(feature=rownames(varimp))# %>% 
  
  best_features<-(varimp %>% arrange(desc(IncNodePurity)))[1:opt$nfeature,] %>% select(feature) %>% unlist() %>% unname()
  data <- data[which(data %>% rownames() %in% best_features),] 
}

## select features using Cor Matrix
select_features_cor <- function(data){
  cor_idx<- data %>% cor() %>% findCorrelation(cutoff = 0.70)
  best_features<-colnames(data)[-cor_idx]
}

parallelCluster <- parallel::makePSOCKcluster(
  spec,
  master=primary,
  homogeneous=T,manual=F)
registerDoParallel(parallelCluster)
print(paste("Workers: ",getDoParWorkers()))



parms <- expand.grid(cost = cost_range, epsilon = epsilon_range, gamma= gamma_range)
results_final <-c()
results_final_models <- list()

for (label_number in rownames(trainprop)) {
  
  labels <- trainprop[label_number, ]
  trainset <- t(select_features_rf(train,label_number))
  
  best_features_cor<-select_features_cor(trainset)
 
  trainset <- trainset[,best_features_cor]
  data_train <- cbind(label = labels, trainset)
  print(trainset %>% ncol)
  print(trainset %>% nrow)
  
  labels_test <- testprop[label_number, ]
  testset <- t(select_features_rf(test,label_number))
  testset <-testset[,best_features_cor]
  data_test <- cbind(label = labels_test, testset)
  print(testset %>% ncol)
  print(testset %>% nrow)
  
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    c <- parms[i,]$cost
    e <- parms[i,]$epsilon
    g  <- parms[i,]$gamma
    model <- e1071::svm(
      label ~ .,
      data = data_train,
      type = "eps-regression",
      kernel = "radial",
      scale = FALSE,
      cost = c,
      epsilon = e,
      gamma = g,
      probability = F
    )
  
    preds <- predict(model, data_test)
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    #readr::write_csv(partial_results,path=paste(e,"_",c,"_partial_results_svm_devcon.csv"))
    partial_results 
    
  }
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ", best_model$label_number," : ",best_model$gamma,",", best_model$cost, ", ", 
              best_model$epsilon," Pearson value : ", best_model$pearson %>% round(digits = 4)," Spearman value : ", best_model$spearman %>% round(digits = 4),
              sep=""))
  data_train<-rbind(data_train,data_test)
  labels<-rbind(labels,labels_test)
  model <- svm(
    label ~ .,
    data = data_train,
    type = "eps-regression",
    kernel = "radial",
    scale = FALSE,
    gamma = best_model$gamma,
    cost = best_model$cost,
    epsilon = best_model$epsilon,
    probability = F
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = paste0("results_svr_radial_devcon_bestmodels_fgdata_cps_20000_2_noscale",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("results_svr_radial_devcon_fgdata_cps_20000_2_noscale",opt$experimenttag,".csv"))
}

# Shutdown cluster neatly
if(!is.null(parallelCluster)) {
  parallel::stopCluster(parallelCluster)
  parallelCluster <- c()
}
