## DEVCON LIBNEAR + RF (kernelab) TUNNING SCRIPT
## 06/05/2020

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
# Load RF models
#load("results_rf_devcon_bestmodels_fgdata_cps_less_feat.rdata")
load("results_rf_devcon_bestmodels_fgdata_cps_20000_feat_2.rdata")
# Load dataset with full set of features
load("deconv_cgdata_cps_new_feat.RData")
library(caret)
library(LiblineaR)
library(foreach)
library(tibble)
library(randomForest)
library(doMC)
library(dplyr)
library(optparse)
registerDoMC(cores=6)

epsilon_range <- c(0.005, 0.05, 0.1, 0.00005,0.0005)
cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32)

#### MAIN 

option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--nfeature", action="store", type="numeric", default=500, help = "Set the maximun number of features ")
)
opt <- parse_args(OptionParser(option_list=option_list))



#epsilon_range <- c(0.005)
#cost_range <- c(0.0001)
rf_models<-results_final_models

select_features <-function(data,label){
  
  varimp<-randomForest::importance(rf_models[[label]]) 
  varimp<- varimp %>% as.data.frame() %>% tibble::add_column(feature=rownames(varimp))# %>% 
  
  best_features<-(varimp %>% arrange(desc(IncNodePurity)))[1:opt$nfeature,] %>% select(feature) %>% unlist() %>% unname()
  data[which(data %>% rownames() %in% best_features),] 
}



parms <- expand.grid(cost = cost_range, epsilon = epsilon_range)
results_final <-c()
results_final_models <- list()
# Starting training  ---------
for (label_number in rownames(trainprop)) {
  # Create datasets  ----------
  labels <- trainprop[label_number, ]
  trainset <- t(select_features(train,label_number))
  #trainset <- scale(trainset,center=TRUE,scale=TRUE)
  #data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(select_features(test,label_number))
  #data_test <- cbind(label = labels_test, testset)
  # Start parallel fine tuning  -----------
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    c <- parms[i,]$cost
    e <- parms[i,]$epsilon
    model <- LiblineaR(
      target=labels,
      data = trainset,
      type = 13,
      cost = c,
      svr_eps = e,
    )
    
    preds <- predict(model, testset)
    spear <- cor(x = preds$predictions, y = labels_test, method = "spearman")
    pears <- cor(x = preds$predictions, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
  # Select best model ---------
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$cost, ", ", best_model$epsilon," Pearson value : ", best_model$pearson,sep=""))
  model <- LiblineaR(
    target=labels,
    data = trainset,
    type = 13,
    cost = best_model$cost,
    svr_eps = best_model$epsilon,
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = paste0("results_liblinear_devcon_bestmodels_fgdata_cps_20000_2",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("results_liblinear_devcon_fgdata_cps_20000_2",opt$experimenttag,".csv"))
}
