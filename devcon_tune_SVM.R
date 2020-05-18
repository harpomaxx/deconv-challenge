## DEVCON SVM TUNNING SCRIPT
## 02/12/2019

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
load("./deconv_cgdata_cps_less_feat.RData")
library(caret)
library(e1071)
library(foreach)
library(doMC)
library(dplyr)
registerDoMC(cores=8)

epsilon_range <- c(0.005, 0.05, 0.1, 0.00005,0.0005)
cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32)

#epsilon_range <- c(0.005, 0.05)
#cost_range <- c(0.0001, 0.001)


parms <- expand.grid(cost = cost_range, epsilon = epsilon_range)
results_final <-c()
results_final_models <- list()
# Starting training  -------
for (label_number in rownames(trainprop)) {
# create datasets  -------
  labels <- trainprop[label_number, ]
  trainset <- t(train)
  data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(test)
  data_test <- cbind(label = labels_test, testset)
# start parallel fine tuning  -------
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    c <- parms[i,]$cost
    e <- parms[i,]$epsilon
    model <- svm(
      label ~ .,
      data = data_train,
      type = "eps-regression",
      kernel = "linear",
      cost = c,
      epsilon = e,
      probability = F
    )
  
    preds <- predict(model, data_test)
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
# select bes model ------
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$cost, ", ", best_model$epsilon," Pearson value : ", best_model$pearson,sep=""))
  model <- svm(
    label ~ .,
    data = data_train,
    type = "eps-regression",
    kernel = "linear",
    cost = best_model$cost,
    epsilon = best_model$epsilon,
    probability = F
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = "results_svm_devcon_bestmodels_fgdata_cps_less_feat.rdata",compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path="results_svm_devcon_fgdata_cps_less_feat.csv")
}
