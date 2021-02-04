rm(list=ls())

library(prodlim)
library(dummies)

getPseudoConditional <- function(t, d, qt){
#     Conditional pseudo values for CIF given the risk set.
#     Arguments:
#       t: survival time
#       d: event status
#       qt: Evaluation times

#     Returns:
#       Conditional pseudo values for CIF given the risk set for all subjects for all causes at different intervals. 
    s <- c(0, qt)  
    n=length(t)
    ns=length(s)-1  # the number of intervals
    D1 <- do.call(cbind, lapply(1:ns, function(j)  (s[j] < t) * (t <= s[j+1]) * (d == 2)))
    D1[D1==1]=2
    D2 <- do.call(cbind, lapply(1:ns, function(j)  (s[j] < t) * (t <= s[j+1]) * (d == 1)))
    D3=D2+D1
    R <- do.call(cbind, lapply(1:ns, function(j) ifelse(s[j] < t, 1, 0)))
    Delta<-do.call(cbind, lapply(1:ns, function(j) pmin(t,s[j+1])-s[j]))

    # format into long formate
    dd.tmp=cbind.data.frame(id=rep(1:n,ns),s=rep(c(0,qt[-length(qt)]), each=n), y=c(R*Delta),d=c(D3))

    dd=dd.tmp[dd.tmp$y>0,]
    pseudost1=rep(NA, nrow(dd))
    pseudost2=rep(NA, nrow(dd))

    for (j in 1:ns){
        index= (dd$s==s[j])
        dds=dd[index,]
        time<-dds$y
        event<-dds$d
        f =prodlim(Hist(time, event)~1) #call function for estimating CIF from right censored data using the Aalen-Johansen estimator 
        pseudost1[index]= jackknife(f,times=s[j+1]-s[j],cause=1) #pseudo values for cause 1
        pseudost2[index]= jackknife(f,times=s[j+1]-s[j],cause=2) #pseudo values for cause 2
        }
        dd$pseudost1=pseudost1
        dd$pseudost2=pseudost2
  
    return(dd[,c(1,2,5, 6)])
}
                                 
                                
get_conditional_pseudo_data<-function(data, evalTime){
#   Get data that will be used to train cause-specific conditional_DeepPseudo model  
#   Arguments:
#       - data: A competing risk dataframe containing survival time and event status, which will be used to estimate pseudo values for CIF.
#       - evalTime: Evaluation times at which pseudo values are calculated
    
#   Returns:
#        - A dataframe with dummy variable of intervals as covariates and pseudo values.
    
    time<-data[, 'time']
    status<-data[,'status']
    
    drops <- c("time","status")  
    features<-data[ , !(names(data) %in% drops)]
    # get the pseudo conditinal CIF
    pseudoCond <- getPseudoConditional(time, status, evalTime)    
    
    # covaraite
    x <- features[pseudoCond$id,]
    # create dummy variables for the time intervals
    smatrix=model.matrix(~as.factor(pseudoCond$s)+0)
    
    #create input predictors 
    x <- cbind(x, smatrix)
    y1 <- pseudoCond$pseudost1
    y2 <- pseudoCond$pseudost2
    data.all<-cbind(x, y1, y2)
    return(data.all)
 }   

get_conditional_test_data <-function(test_data, evalTime){  
#   Get data that will be used to test the cause-specific conditional_DeepPseudo model  
#   Arguments:
#       - data: A test data 
#       - evalTime: Evaluation times at which pseudo values are calculated
    
#   Returns:
#        - A dataframe with dummy variable of intervals that will be used to test the cause-specific conditional_DeepPseudo model.
    drops <- c("time","status")  
    test_features<-test_data[ , !(names(test_data) %in% drops)]
 
    # format the test data 
    s<-c(0,12)
    x_test=do.call(rbind, replicate(length(evalTime), test_features, simplify=FALSE))
    s_test=rep(s,each=nrow(test_features))
    smatrix.test=model.matrix(~as.factor(s_test)+0)
    x_test=cbind(x_test,smatrix.test)
    return(x_test)
    
 }   
