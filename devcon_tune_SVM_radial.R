## DEVCON SVM TUNNING SCRIPT
## 02/12/2019

setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")
#load("deconv_cgdata_cps.RData")
load("deconv_cgdata_cps_less_feat.RData")
library(caret)
library(e1071)
library(foreach)
library(doMC)
library(dplyr)
registerDoMC(cores=6)

epsilon_range <- c(5e-3, 5e-2, 5e-5,5e-4,5e-6)
#cost_range <- c(0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 4,8,16,32,64,128,256)
cost_range <- c(256,384,512,768)

# Intuitively, the gamma parameter defines how far the influence of a single training example reaches, 
# with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as 
# the inverse of the radius of influence of samples 
# selected by the model as support vectors.

gamma_range <- c(1e-7,1e-6,1e-5,1e-04)

#epsilon_range <- c(0.005, 0.05)
#cost_range <- c(0.0001, 0.001)
#gamma_range <- c(1e-8)


parms <- expand.grid(cost = cost_range, epsilon = epsilon_range, gamma= gamma_range)
results_final <-c()
results_final_models <- list()

for (label_number in rownames(trainprop)) {
  
  labels <- trainprop[label_number, ]
  trainset <- t(train)
  data_train <- cbind(label = labels, trainset)
  
  labels_test <- testprop[label_number, ]
  testset <- t(test)
  data_test <- cbind(label = labels_test, testset)
  
  results <- foreach(i = 1:nrow(parms), .combine = rbind) %dopar% {
    c <- parms[i,]$cost
    e <- parms[i,]$epsilon
    g  <- parms[i,]$gamma
    model <- svm(
      label ~ .,
      data = data_train,
      type = "eps-regression",
      kernel = "radial",
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
  print(paste("selecting best model for ", best_model$label_number," : ",best_model$gamma,",", best_model$cost, ", ", best_model$epsilon," Pearson value : ", best_model$pearson,sep=""))
  model <- svm(
    label ~ .,
    data = data_train,
    type = "eps-regression",
    kernel = "radial",
    gamma = best_model$gamma,
    cost = best_model$cost,
    epsilon = best_model$epsilon,
    probability = F
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = "results_svm_radial_devcon_bestmodels_less_features.rdata",compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path="results_svm_radial_devcon_less_features.csv")
}
