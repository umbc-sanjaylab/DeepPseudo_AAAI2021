#Import library
library(prodlim)

             
pseudo<-function(data, evaltime){
#      The Aalen-Johansen estimator is used to estimate the absolute risk of the competing causes, i.e., the cumulative incidence functions.  
#      Pseudo values can be estimated at a pre-specified grid of time points. We consider 12 and 60 months as evaluation times and 2 causes of the event in our paper. 
#      It can be generalized for more than 2 evaluation times and more than 2 causes of the event.

#     Arguments:
#       data: A competing risk dataframe containing survival time and event status, which will be used to estimate pseudo values for CIF.
#       evaltime: Evaluation times

#     Returns:
#       A dataframe of pseudo values for all subjects for all causes at the evaluation time points. 
    
    
    # Extract time and event status from the competing risk data
    time<-data[, 'time']
    status<-data[,'status']
    
    # Call function for estimating CIF from right censored data using the Aalen-Johansen estimator 
    f=prodlim(Hist(time, status)~1)                    
    
    #Compute pseudo values for cause 1
    pseudo_c1<-jackknife(f, times=evaltime, cause=1)   #times: Time points at which to compute pseudo values
                                                       #cause: Cause of failure
    
    #Compute pseudo values for cause 2
    pseudo_c2<-jackknife(f, times=evaltime, cause=2)  
    
    # Each row of pseudo values for cause2 are arranged just after each row of pseudo values for cause1 so that for each
    # observation, we get an array with shape = (No. of events, No. of evaluation times)
    pseudo_value<-rbind(pseudo_c1[1,], pseudo_c2[1,])
    for (i in 2:length(time)){ 
         pseudo_value<-rbind(pseudo_value, pseudo_c1[i,], pseudo_c2[i,])
      }
      return(pseudo_value) 

}
