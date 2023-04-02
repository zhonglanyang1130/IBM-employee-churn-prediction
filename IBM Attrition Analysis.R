
installpkg("readxl")
installpkg("ggplot2")
installpkg("randomForest")
installpkg("libcoin")
installpkg("partykit")
installpkg("glmnet")
installpkg("tree")
installpkg("VIM")
installpkg("heatmaply")

library(readxl)
library(ggplot2)
library(randomForest)
library(libcoin)
library(partykit)
library(glmnet)
library(tree)
library(VIM)
library(heatmaply)
source("DataAnalyticsFunctions.R")
source("PerformanceCurves.R")

attrition <- read.csv("HR_attrition_orig_proc.csv")

summary(attrition)
str(attrition)

## Data Understanding Part

## Missing Value
aggr(attrition, cex.axis=0.5)

aggr(attrition,combined = TRUE, numbers=TRUE, cex.numbers=0.5)

## Summary
length(attrition$Attrition[attrition$Attrition == 0])
length(attrition$Attrition[attrition$Attrition == 1])

## Attrition vs Job Level
plot(factor(Attrition) ~ factor(JobLevel), data=attrition, col=c(8,2), ylab="Attrition", xlab="Job Level",
     legend.text = c("Yes", "No"), main = "Attrition Spread in different Job Levels")

## Attrition vs StockOption Level
plot(factor(Attrition) ~ factor(StockOptionLevel), data=attrition, col=c(8,2), ylab="Attrition", xlab="Stock Option Level",
     legend.text = c("Yes", "No")) 

## Attrition vs Relationship Satisfaction
plot(factor(Attrition) ~ factor(RelationshipSatisfaction), data=attrition, col=c(8,2), ylab="Attrition", xlab="Job Satisfaction",
     legend.text = c("Yes", "No")) 

# Attrition vs Years in Current Role
plot(factor(Attrition) ~ factor(YearsInCurrentRole), data=attrition, col=c(8,2), ylab="Attrition", xlab="Years In Current Role",
     legend.text = c("Yes", "No"), main = "Attrition Spread among Years stay in Current Role") 

# Attrition vs YearsIncurrentRole
plot(factor(Attrition) ~ factor(YearsInCurrentRole), data=attrition, col=c(8,2), ylab="Attrition", xlab="Years In Current Role",
     legend.text = c("Yes", "No"), main = "Attrition Spread among Years stay in Current Role") 

# Job Level vs YearsInCurrentRole
boxplot(JobLevel ~ YearsInCurrentRole, data=attrition,ylab="Jobs level", xlab="Years in current Role",
        legend.text = c("Yes", "No"), main = "Is job levels affecting years in current role?") 

#JobLevel vs Years At Company
boxplot(JobLevel ~ YearsAtCompany, data=attrition,ylab="Jobs level", xlab="Years at Company",
        legend.text = c("Yes", "No"), main = "IBM Career Promotion Path in years") 


## Monthly income vs Attrition 
boxplot(MonthlyIncome ~ Attrition, data=attrition, col=c(8,2), ylab="Monthly Income", xlab="Attrition",
        names = c("Yes", "No")) 

## Correlation Heatmap
heatmaply_cor(x = cor(attrition), xlab = "Features",
              ylab = "Features", k_col = 2, k_row = 2, title = "Correlation Heatmap")

## Modelling

### create a vector of fold memberships (random order)
nfold <- 10
n = nrow(attrition)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
### create an empty dataframe of results
OOS <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
### First we establish logistic, logistic interaction, tree and a benchmark Null model
for(k in 1:nfold){ 
  train <- which(foldid!=k)
  #fit the models
  model.logistic.interaction <-glm(Attrition~.^2, data=attrition, subset=train, family="binomial")
  model.logistic <-glm(Attrition~., data=attrition, subset=train,family="binomial")
  model.tree <- tree(factor(Attrition)~ ., data=attrition, subset=train) 
  model.nulll <-glm(Attrition~1, data=attrition, subset=train,family="binomial")
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=attrition[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=attrition[-train,], type="response")
  pred.tree                 <- predict(model.tree, newdata=attrition[-train,], type="vector")
  pred.tree <- pred.tree[,2]
  pred.null <- predict(model.nulll, newdata=attrition[-train,], type="response")
  ## calculate and log R2
  # Logistic Interaction
  OOS$logistic.interaction[k] <- R2(y=attrition$Attrition[-train], pred=pred.logistic.interaction, family="binomial")
  OOS$logistic.interaction[k]
  # Logistic
  OOS$logistic[k] <- R2(y=attrition$Attrition[-train], pred=pred.logistic, family="binomial")
  OOS$logistic[k]
  # Tree
  OOS$tree[k] <- R2(y=attrition$Attrition[-train], pred=pred.tree, family="binomial")
  OOS$tree[k]
  #Null
  OOS$null[k] <- R2(y=attrition$Attrition[-train], pred=pred.null, family="binomial")
  OOS$null[k]
  #Null Model guess
  sum(attrition$Attrition[train])/length(train)
  
  print(paste("Iteration",k,"of",nfold,"(thank you for your patience)"))
}

### we have nfold values in OOS for each model, this computes the mean of them
colMeans(OOS)
m.OOS <- as.matrix(OOS)
rownames(m.OOS) <- c(1:nfold)
barplot(t(as.matrix(OOS)), beside=TRUE, legend=TRUE, args.legend=c(xjust=1, yjust=0.5),
        ylab= bquote( "Out of Sample " ~ R^2), xlab="Fold", names.arg = c(1:10))


## based on interaction - regularization
Mx<- model.matrix(Attrition ~ .^2, data=attrition)[,-1]
My<- attrition$Attrition

## regularization - theory
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.churn <- sum(My)
w <- (num.churn/num.n)*(1-(num.churn/num.n))
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory)
summary(lassoTheory)
length(support(lassoTheory$beta))


lasso <- glmnet(Mx,My, family="binomial")
lassoCV <- cv.glmnet(Mx,My, family="binomial")
plot(lasso, xvar="lambda", main="# of non-zero coefficients", ylab ="Coefficient values", xlab = expression(paste("log(",lambda,")")))

PL.OOS <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
features.1se <- support(lasso$beta[,which.min( (lassoCV$lambda-lassoCV$lambda.1se)^2)])
length(features.1se) 
features.theory <- support(lassoTheory$beta)
length(features.theory)

data.min <- data.frame(Mx[,features.min],"Attrition"=My)
data.1se <- data.frame(Mx[,features.1se],"Attrition"=My)
data.theory <- data.frame(Mx[,features.theory],"Attrition"=My)


for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'

### This is the CV for the Post Lasso Estimates
  rmin <- glm(Attrition~., data=data.min, subset=train, family="binomial")
  if ( length(features.1se) == 0){  r1se <- glm(Attrition~1, data=attrition, subset=train, family="binomial") 
  } else {r1se <- glm(Attrition~., data=data.1se, subset=train, family="binomial")
  }

  if ( length(features.theory) == 0){ 
    rtheory <- glm(Attrition~1, data=attrition, subset=train, family="binomial") 
  } else {rtheory <- glm(Attrition~., data=data.theory, subset=train, family="binomial") }


  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  PL.OOS$PL.min[k] <- R2(y=My[-train], pred=predmin, family="binomial")
  PL.OOS$PL.1se[k] <- R2(y=My[-train], pred=pred1se, family="binomial")
  PL.OOS$PL.theory[k] <- R2(y=My[-train], pred=predtheory, family="binomial")

### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)

  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  L.OOS$L.min[k] <- R2(y=My[-train], pred=predlassomin, family="binomial")
  L.OOS$L.1se[k] <- R2(y=My[-train], pred=predlasso1se, family="binomial")
  L.OOS$L.theory[k] <- R2(y=My[-train], pred=predlassotheory, family="binomial")

  print(paste("Iteration",k,"of",nfold,"completed"))
}

R2performance <- cbind(PL.OOS,L.OOS,OOS)


PL.OOS.TPR <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.TPR <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
PL.OOS.FPR <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.FPR <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 
PL.OOS.ACC <- data.frame(PL.min=rep(NA,nfold), PL.1se=rep(NA,nfold), PL.theory=rep(NA,nfold)) 
L.OOS.ACC <- data.frame(L.min=rep(NA,nfold), L.1se=rep(NA,nfold), L.theory=rep(NA,nfold)) 

OOS.TPR <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.FPR <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.ACC <- data.frame(logistic.interaction=rep(NA,nfold), logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 


val <- .5
for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### This is the CV for the Post Lasso Estimates
  rmin <- glm(Attrition~., data=data.min, subset=train, family="binomial")
  
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  pred1se  <- predict(r1se, newdata=data.1se[-train,], type="response")
  predtheory <- predict(rtheory, newdata=data.theory[-train,], type="response")
  
  values <- FPR_TPR( (predmin >= val) , My[-train] )
  PL.OOS.ACC$PL.min[k] <- values$ACC
  PL.OOS.TPR$PL.min[k] <- values$TPR
  PL.OOS.FPR$PL.min[k] <- values$FPR
  
  values <- FPR_TPR( (pred1se >= val) , My[-train] )
  PL.OOS.ACC$PL.1se[k] <- values$ACC
  PL.OOS.TPR$PL.1se[k] <- values$TPR
  PL.OOS.FPR$PL.1se[k] <- values$FPR
  
  values <- FPR_TPR( (predtheory >= val) , My[-train] )
  PL.OOS.ACC$PL.theory[k] <- values$ACC
  PL.OOS.FPR$PL.theory[k] <- values$FPR
  PL.OOS.TPR$PL.theory[k] <- values$TPR
  
  ### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  lasso1se  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.1se)
  lassoTheory <- glmnet(Mx[train,],My[train], family="binomial",lambda = lambda.theory)
  
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  predlasso1se  <- predict(lasso1se, newx=Mx[-train,], type="response")
  predlassotheory <- predict(lassoTheory, newx=Mx[-train,], type="response")
  values <- FPR_TPR( (predlassomin >= val) , My[-train] )
  L.OOS.ACC$L.min[k] <- values$ACC
  L.OOS.TPR$L.min[k] <- values$TPR
  L.OOS.FPR$L.min[k] <- values$FPR
  values <- FPR_TPR( (predlasso1se >= val) , My[-train] )
  L.OOS.ACC$L.1se[k] <- values$ACC
  L.OOS.TPR$L.1se[k] <- values$TPR
  L.OOS.FPR$L.1se[k] <- values$FPR
  values <- FPR_TPR( (predlassotheory >= val) , My[-train] )
  L.OOS.ACC$L.theory[k] <- values$ACC
  L.OOS.TPR$L.theory[k] <- values$TPR
  L.OOS.FPR$L.theory[k] <- values$FPR
  
  
  ## fit the two regressions and null model
  ##### full model uses all 200 signals
  model.logistic.interaction <-glm(Attrition~.^2, data=attrition, subset=train, family="binomial")
  model.logistic <-glm(Attrition~., data=attrition, subset=train,family="binomial")
  model.tree <- tree(factor(Attrition)~ ., data=attrition, subset=train) 
  model.nulll <-glm(Attrition~1, data=attrition, subset=train,family="binomial")
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=attrition[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=attrition[-train,], type="response")
  pred.tree                 <- predict(model.tree, newdata=attrition[-train,], type="vector")
  pred.tree <- pred.tree[,2]
  pred.null <- predict(model.nulll, newdata=attrition[-train,], type="response")
  
  ## calculate and log R2
  # Logistic Interaction
  values <- FPR_TPR( (pred.logistic.interaction >= val) , My[-train] )
  
  OOS.ACC$logistic.interaction[k] <- values$ACC
  OOS.FPR$logistic.interaction[k] <- values$FPR
  OOS.TPR$logistic.interaction[k] <- values$TPR
  # Logistic
  values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
  
  OOS.ACC$logistic[k] <- values$ACC
  OOS.FPR$logistic[k] <- values$FPR
  OOS.TPR$logistic[k] <- values$TPR
  # Tree
  values <- FPR_TPR( (pred.tree >= val) , My[-train] )

  OOS.ACC$tree[k] <- values$ACC
  OOS.FPR$tree[k] <- values$FPR
  OOS.TPR$tree[k] <- values$TPR
  
  #Null
  values <- FPR_TPR( (pred.null >= val) , My[-train] )

  OOS.ACC$null[k] <- values$ACC
  OOS.FPR$null[k] <- values$FPR
  OOS.TPR$null[k] <- values$TPR
  
  print(paste("Iteration",k,"of",nfold,"completed"))
}

ACCperformance <- cbind(PL.OOS.ACC,L.OOS.ACC,OOS.ACC)
TPRperformance <- cbind(PL.OOS.TPR,L.OOS.TPR,OOS.TPR)
FPRperformance <- cbind(PL.OOS.FPR,L.OOS.FPR,OOS.FPR)

acc <- colMeans(ACCperformance)
tpr <- colMeans(TPRperformance)
fpr <- colMeans(FPRperformance)
performance <- as.data.frame(cbind(R2 = colMeans(R2performance), acc, tpr, fpr))


###visualization
barplot(performance$R2, names.arg = row.names(performance))
barplot(performance$acc,  names.arg = row.names(performance))
barplot(performance$tpr,  names.arg = row.names(performance))
barplot(performance$fpr,  names.arg = row.names(performance))

### in this business context, we pay more attention to the low FPR, because when we think an employee will leave, these is cost.
### based on this, we first choose PL.1se, PL.min, Lasso.interaction from the 10 models
### for this part, we use test/training split
### train/test split
permuted_rows <- sample(nrow(attrition))
attrition_shuffled <- attrition[permuted_rows,]
split_num <- round(nrow(attrition)*0.8)
train <- attrition_shuffled[1:split_num,]
test <- attrition_shuffled[(split_num+1):nrow(attrition),]

### logistic
m_log <- glm(Attrition~., data = train, family = "binomial")
p_log <- predict(m_log, test, type="response")

### logistic_interaction
m_int <- glm(Attrition~.^2, data=train, family="binomial")
p_int <- predict(m_int, test, type="response")

## based on interaction - regularization
Mx<- model.matrix(Attrition ~ .^2, data=train)[,-1]
My<- train$Attrition
Mx_test <- model.matrix(Attrition~.^2, test)[,-1]
My_test <- test$Attrition
lasso <- glmnet(Mx,My, family="binomial")
lassoCV <- cv.glmnet(Mx,My, family="binomial")

## regularization - theory
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.churn <- sum(My)
w <- (num.churn/num.n)*(1-(num.churn/num.n))
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory)
summary(lassoTheory)
length(support(lassoTheory$beta))
p_theory <- predict(lassoTheory, newx=Mx_test, type="response")
p_theory <- as.numeric(p_theory)

## min
lassomin  <- glmnet(Mx,My, family="binomial",lambda = lassoCV$lambda.min)
p_min <- predict(lassomin, newx=Mx_test, type = 'response')
p_min <- as.numeric(p_min)
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
data.min <- data.frame(Mx[,features.min],"Attrition"=My)   # dataset after dimension reduction

## 1se
lasso1se <- glmnet(Mx, My, family="binomial", lambda=lassoCV$lambda.1se)
p_1se <- predict(lasso1se, newx=Mx_test, type="response")
p_1se <- as.numeric(p_1se)
features.1se <- support(lasso$beta[,which.min((lassoCV$lambda-lassoCV$lambda.1se)^2)])
length(features.1se) 
data.1se <- data.frame(Mx[,features.1se],"Attrition"=My)

## post lasso
m_log_1se <- glm(Attrition~., data=data.1se, family="binomial")
Mx_test_1se <- data.frame(Mx_test[, features.1se])
p_post_1se <- predict(m_log_1se, Mx_test_1se, type="response")

m_log_min <-glm(Attrition~., data=data.min, family="binomial")
Mx_test_min <- data.frame(Mx_test[, features.min])
p_post_min <- predict(m_log_min, Mx_test_min, type="response")

p_total <- as.data.frame(cbind(p_post_1se, p_post_min, p_int))
### Confusion matrix under 0.5 threshold
th <- 0.5
performance <- as.data.frame(matrix(nrow=3, ncol=3),row.names=colnames(p_total))
names(performance) <- c("FPR", "TPR", "ACC")
for (i in 1:ncol(p_total)) {
  p_class <- ifelse(p_total[,i]>th, 1, 0)
  performance[i,1] <- FPR_TPR(p_class, My_test)$FPR
  performance[i,2] <- FPR_TPR(p_class, My_test)$TPR
  performance[i,3] <- FPR_TPR(p_class, My_test)$ACC
  print(colnames(p_total)[i])
  print(table(p_class, test[["Attrition"]]))
  print("------------------------------------")
}

barplot(performance$ACC, names.arg = row.names(performance))
barplot(performance$FPR)
barplot(performance$TPR)

### However, confusion matrix can only reflect situation under certain threshold.
### Next we will use ROC to display the whole picture.
install.packages("caTools")
library(caTools)
for (i in 1:ncol(p_total)) {
  colAUC(p_total[,i], My_test, plotROC = TRUE)
}

### according to roc curve, the pl.1se pl.min performs better (close to (0,1)).

### Lastly, we need to bring the problem back into the business environment
### We now conduct a cost-benefit analysis.
profit_total <- c()
for (i in 1:ncol(p_total)) {
  p_class <- ifelse(p_total[,i]>0.5, 1, 0)
  t <- table(p_class,test[["Attrition"]])
  if (nrow(t)>1) {
    profit <- t["1","1"]*16000-t["1","0"]*10000-t["0","1"]*16000+t["0","0"]*0}
    else {profit <- t["1","1"]*16000-t["1","0"]*10000
  }
  profit_total <- c(profit_total, profit)
}
profit_total <- as.data.frame(cbind(row.names(performance), profit_total))
best_model_business <- profit_total$V1[which.max(profit_total$profit_total)]

### Finally, we utilize the whole data set to train a ready-to-use model based on our choice of model.
Mx_full <- model.matrix(Attrition ~ .^2, data=attrition)[,-1]
My_full <-attrition$Attrition
data.1se_full <- data.frame(Mx_full[,features.1se],"Attrition"=My_full)
Model_full <- glm(Attrition~., data=data.1se_full, family="binomial")
Model_full
