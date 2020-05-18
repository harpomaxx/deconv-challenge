## DEVCON RF  TUNNING SCRIPT
## 30/04/2020

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
#load("./deconv_cgdata_cps_less_feat.RData")
load("deconv_cgdata_cps_new_feat.RData")
library(caret)
library(randomForest)
library(foreach)
library(doMC)
library(dplyr)
registerDoMC(cores=7)

mtry_range <- c(2,3,4,5,6,7,8,9,10,50,100,150,200)
ntree_range <- c(500)
#cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32)




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
  
    model <- randomForest(
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
  model <- randomForest(
    y = labels,
    x = trainset,
    mtry = best_model$mtry,
    ntree = best_model$ntree,
    na.action=na.omit
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = "results_rf_devcon_bestmodels_fgdata_cps_20000_feat_2.rdata",compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path="results_rf_devcon_fgdata_cps_20000_feat_2.csv")
}
