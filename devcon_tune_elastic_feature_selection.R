## DEVCON ELASTIC + RF (kernelab) TUNNING SCRIPT
## 08/05/2020

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
# Load RF models
#load("results_rf_devcon_bestmodels_finegrain_data_cps_20000_newmix_last3.rdata")
load("results_rf_devcon_bestmodels_coarsegrain_data_cps_20000_newmix_last3.rdata")

# Load dataset with full set of reatures
load("deconv_cgdata_cps_new_feat_last2.RData") #coarse grain
#load("deconv_fgdata_cps_new_feat_last3.RData") # fine grain
library(caret)
library(glmnet)
library(foreach)
library(tibble)
library(randomForest)
library(doMC)
library(dplyr)
library(optparse)
registerDoMC(cores=6)



alpha_range <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
lambda_ratio_range <- c(10e-8,10e-7,10e-6,10e-5,10e-4,10e-3)

#### MAIN 

option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--nfeature", action="store", type="numeric", default=500, help = "Set the maximun number of features ")
)
opt <- parse_args(OptionParser(option_list=option_list))



rf_models<-results_final_models

select_features <-function(data,label){
  
  varimp<-randomForest::importance(rf_models[[label]]) 
  varimp<- varimp %>% as.data.frame() %>% tibble::add_column(feature=rownames(varimp))# %>% 
  
  best_features<-(varimp %>% arrange(desc(IncNodePurity)))[1:opt$nfeature,] %>% select(feature) %>% unlist() %>% unname()
  data[which(data %>% rownames() %in% best_features),] 
}


parms <- expand.grid(alpha = alpha_range, lambda_ratio = lambda_ratio_range)
results_final <-c()
results_final_models <- list()
# Starting training  ---------
for (label_number in rownames(trainprop)) {
  # Create datasets  ----------
  labels <- trainprop[label_number, ]
  trainset <- t(select_features(train,label_number))
  print(trainset %>% nrow)
  print(trainset %>% ncol)
  #trainset <- scale(trainset,center=TRUE,scale=TRUE)
  #data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(select_features(test,label_number))
  print(testset %>% nrow)
  print(testset %>% ncol)
  #data_test <- cbind(label = labels_test, testset)
  # Start parallel fine tuning  -----------
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    alpha <- parms[i,]$alpha
    lambda_ratio <- parms[i,]$lambda_ratio
    model <- cv.glmnet(
      type.measure = "mse",
      nfolds = 5,
      y = labels,
      x = trainset,
      alpha = alpha,
      standardize = FALSE,
      nlambda = 100,
      lambda.min.ratio = lambda_ratio,
      family = "gaussian",
      lower.limit = -Inf
      )
    
    preds <- predict(model, testset,s = "lambda.1se")
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
  # Select best model ---------
  best_model <- results %>% mutate(spearson=(pearson+spearman)/2) %>% arrange(desc(spearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$alpha, ", ", 
              best_model$lambda_ratio," Pearson value : ", best_model$pearson  %>% round(digits = 4),
              " Spearman value : ", best_model$spearman %>% round(digits = 4),
              " Spearson value : ", best_model$spearson %>% round(digits = 4),
              sep=""))
   trainset <- rbind(trainset,testset)
   labels <- c(labels,labels_test)
   model <- cv.glmnet(
    type.measure = "mse",
    nfolds = 5,
    y = labels,
    x = trainset,
    alpha = best_model$alpha,
    standardize = FALSE,
    nlambda = 100,
    lambda.min.ratio = best_model$lambda_ratio,
    family = "gaussian",
    lower.limit = -Inf
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = paste0("results_glmnet_devcon_bestmodels_coarsegrain_data_cps_20000_2",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("results_glmnet_devcon_coarsegrain_data_cps_20000_2",opt$experimenttag,".csv"))
}

