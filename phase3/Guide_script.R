#Deconv data
library(dplyr)

##Validation set##
#Estos son los sets que dieron en la competencia
setwd("/home/harpo/Dropbox/ongoing-work/git-repos/devcon/phase3/")

load("deconv_cgdata_cps_new_feat.RData") 
load("featureMaster.Rdata") #object "inALL" with the features shared in all the datasets
inALL<-train %>% rownames()
#Normalization function

mynorm4= function(l){
  
  lr= rank(l*-1,ties.method="average")
  lr=log2(lr)
  lr= lr*-1
  lr= lr+ log2(length(lr))
  
  
  lr[lr==min(lr)]=0
  res=lr
  return(res)
}




testsetdir1 = "./phase1_data"
testsetdir2 = "./phase2_data"
testsetdir3 = "./phase3_data"

goldstand = "./gold_standards"

goldCoarse = read.csv(list.files(goldstand, full.names = TRUE)[1])
goldCoarse2 = read.csv(list.files(goldstand, full.names = TRUE)[2])
goldCoarse3 = read.csv(list.files(goldstand, full.names = TRUE)[3])
goldCoarse = rbind(rbind(goldCoarse, goldCoarse2),goldCoarse3)
rm(goldCoarse2)
rm(goldCoarse3)


files = list.files(testsetdir1, full.names = TRUE)
files = c(files, list.files(testsetdir2, full.names = TRUE))
files = c(files, list.files(testsetdir3, full.names = TRUE))
DATASETS = NULL
for (i in files) {
  name = substr(i, 15, nchar(i) - 19)
  print(name)
  DATASETS = c(DATASETS, name)
  fl = read.csv(i, row.names = 1)
  assign(name, fl)
  
}


for (i in DATASETS) {
  print(i)
  DS = get(i)
  DS[is.na(DS)]=min(DS,na.rm=TRUE)
  expression_matrix = DS - min(DS)
  if(any(!inALL %in% rownames(expression_matrix))){
    Fs=inALL[!inALL %in% rownames(expression_matrix)]
    Add_matrix= matrix(NA,nrow=length(Fs),ncol=ncol(expression_matrix))
    colnames(Add_matrix)=colnames(expression_matrix)
    rownames(Add_matrix)=Fs
    expression_matrix=rbind(expression_matrix,Add_matrix)
    
  }
  
  expression_matrix = expression_matrix[inALL, ]
  expression_matrix[is.na(expression_matrix)] = log2(1)
  expression_matrix= apply(expression_matrix,2,mynorm4)
  
  rownames(expression_matrix) = inALL
  assign(i, expression_matrix)
  print("Done!")
  
}

#Ejemplo de testeo de los modelos contra el validation set
library(LiblineaR)
library(e1071)
library(glmnet)
library(randomForest)
#load("results_liblinear_devcon_bestmodels_fgdata_cps_less_feat.rdata")
#load("results_svm_radial_devcon_bestmodels_3.rdata")
#load("results_svm_devcon_bestmodels_fgdata_cps.rdata")
#load("results_svm_devcon_bestmodels_fgdata_cps_less_feat.rdata")
#load("results_svm_radial_devcon_bestmodels_less_features.rdata")

#load("results_liblinear_devcon_bestmodels_fgdata_cps_12_noscale_rf_selected_features.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_12_noscale_rf_selected_features_1500.csv.rdata")

#load("results_liblinear_devcon_bestmodels_fgdata_cps_12_noscale_rf_selected_features_2000.csv.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_12_noscale_rf_selected_features_2500.csv.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_113noscale_rf_selected_features_1500.csv.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_13_noscale_rf_selected_features_1000.csv.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_13_noscale_rf_selected_features_2000.csv.rdata")
#load("results_liblinear_devcon_bestmodels_fgdata_cps_113noscale_rf_selected_features_250.csv.rdata")
#load("results_glmnet_devcon_bestmodels_fgdata_cps_glmnet_rf_selected_features_250.rdata")
#load("results_glmnet_devcon_bestmodels_fgdata_cps_glmnet_rf_20000_selected_features_2500.rdata")

#load("results_liblinear_devcon_bestmodels_fgdata_cps_12_noscale.rdata")
load("results_rf_devcon_bestmodels_fgdata_cps_20000_feat.rdata")
RESULTS=list()
for(i in DATASETS){
  if(any(goldCoarse$dataset.name==i & !is.na(goldCoarse$measured))){
    cells= as.character(unique(goldCoarse[goldCoarse$dataset.name==i & !is.na(goldCoarse$measured),"cell.type"]))
    RESULTS[[i]]= data.frame(datasets=i,cells=cells,pearson=NA,spearman=NA)
  }
}

RESULTS=do.call(rbind,RESULTS)
for (label_number in  rownames(trainprop) %>% setdiff("noise")) {
cell = label_number
model = results_final_models[[cell]]
print(cell)
sets= as.character(RESULTS[RESULTS$cells==cell,"datasets"])
for (i in sets) {
#for (i in c("DS446395","DS500","CIAS4","CIVA3")) {
  Dset = i
  print(paste("  ",Dset))
  expression_matrix = get(Dset)
  # select only the features presen in the liblinear model
  #expression_matrix<-expression_matrix[which(expression_matrix %>% rownames() %in% (model$W %>% colnames())),]
  #expression_matrix<-select_features(expression_matrix,cell)
  #prediction = predict(model, newx = t(expression_matrix), s = "lambda.1se")
  
  #prediction = predict(model, newx = t(expression_matrix),s="lambda.1se")
  prediction = predict(model,  t(expression_matrix))
  
  #prediction <- prediction$predictions
  
  prediction[prediction < 0] = 0
  true = goldCoarse[goldCoarse$dataset.name == Dset &
                      goldCoarse$cell.type == cell, ]
  trueM = true[match(colnames(expression_matrix), true$sample.id), "measured"]
  pearson= cor(prediction, trueM,use="pairwise.complete.obs")
  spearman= cor(prediction, trueM,method="spearman",use="pairwise.complete.obs")
  RESULTS[RESULTS$cells==cell & RESULTS$datasets==Dset,"pearson"]=pearson
  RESULTS[RESULTS$cells==cell & RESULTS$datasets==Dset,"spearman"]=spearman
  print(paste("pearson", pearson, sep = ": "))
  print(paste("spearman", spearman, sep = ": "))
  #if (Dset %in% c("DS446395","DS500","CIAS4","CIVA3")){
  #  dev.copy(png,paste0('plot_',label_number,"_",i,".png"))
  #  
  #  plot(prediction, trueM,main = paste(Dset,cell))
  #  lines(x=c(0,1),y=c(0,1))
  #  dev.off()
  #}
}
}

#Espero que sea claro jeje, es medio enquilombado el tema de los validation por que son varios sets diferentes y las mediciones de cada set pueden variar de escalas por eso no puedo juntarlos todos en una

