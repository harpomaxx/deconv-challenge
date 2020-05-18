## DEVCON SVM TUNNING SCRIPT
## 02/12/2019

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/")
load("datasets/deconv_data_cps-2.RData")
library(caret)
library(e1071)
library(foreach)
library(doMC)
registerDoMC(cores=6)

nu_range <- c(0.005, 0.05, 0.1, 0.00005,0.0005)
cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32)
parms <- expand.grid(cost = cost_range, nu = nu_range)
results_final <-c()

for (label_number in rownames(trainprop)) {
  
  labels <- trainprop[label_number, ]
  trainset <- t(train)
  data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(test)
  data_test <- cbind(label = labels_test, testset)
  
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    c <- parms[i,]$cost
    n <- parms[i,]$nu
    model <- svm(
      label ~ .,
      data = data_train,
      type = "nu-regression",
      kernel = "linear",
      cost = c,
      nu  = n,
      probability = F
    )
  
    preds <- predict(model, data_test)
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    #readr::write_csv(partial_results,path=paste(e,"_",c,"_partial_results_svm_devcon.csv"))
    partial_results 
    
  }
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path="results_nu_svm_devcon.csv")
}
