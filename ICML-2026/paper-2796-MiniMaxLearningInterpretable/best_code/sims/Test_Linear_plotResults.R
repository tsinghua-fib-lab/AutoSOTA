# plot results
if(FINAL_RESULTS == T){ name_key <- optimType }

vals2gray <- function(x,start = 0, end = 0.9){
  return( gray.colors(n=length(x),start = start, end = end)[ order(  f2n(x) ,decreasing = T ) ] ) }

{
  pdf(PDF_NAME <- sprintf("%s/VarBoundEst1_%s.pdf",FiguresLoc, name_key))
  {
    theoreticalVarMat_meaned <- do.call(cbind, tapply(1:nrow(res_df),unlist( res_df$kFactors ),
           function(zer){
        tapply(abs(unlist( res_df$theoretical_varB[zer] )-
          unlist( res_df$SE_Q_exact[zer] )),
          unlist( res_df$nObs[zer] ),
          mean) }))
    par(mar=c(5,8,3,1))
    plot(f2n(row.names(theoreticalVarMat_meaned)),
         theoreticalVarMat_meaned[,1],
         cex=0, type = "p",ylim=c(0,1.25*max(theoreticalVarMat_meaned)),
         log = 'x', xlab = "Number of Observations",
         main = "Linear Model with Interactions",
         ylab = "Mean Absolute Deviation \n True vs. Mean Estimated Variance Bound",
         cex.lab = 2, cex.main = 2)
    for(i in 1:ncol(theoreticalVarMat_meaned)){
      points(f2n(row.names(theoreticalVarMat_meaned)),
           theoreticalVarMat_meaned[,i],
           cex=2, type = "b",pch=" ")
      text(f2n(row.names(theoreticalVarMat_meaned)),
          theoreticalVarMat_meaned[,i],
          cex = 2, label =  rep(colnames(theoreticalVarMat_meaned)[i],nrow(theoreticalVarMat_meaned)))
      #points(c(theoreticalVarMat_meaned[i,3],theoreticalVarMat_meaned[i,3]),
             #c(theoreticalVarMat_meaned[i,1] - 1.96*theoreticalVarMat_meaned[i,2],
               #theoreticalVarMat_meaned[i,1] + 1.96*theoreticalVarMat_meaned[i,2] ), type="l",ltw=2)
    }
  }
  dev.off()

  nK = length(kFactors_seq);
  iterSeq <- 1:nK
  if(nK == 1){iterSeq <- rep(1,times=3)}
  pdf(sprintf("%s/varianceFigQ_%s.pdf",FiguresLoc,name_key),width = 3*length(iterSeq),height=5)
  {
    layersM = matrix(1:length(iterSeq),byrow=T,ncol=length(iterSeq))
    layersM = cbind(length(iterSeq)+1,layersM)
    layersM = rbind(layersM,length(iterSeq)+2)
    layersM = rbind(length(iterSeq)+(2:(2+length(iterSeq))),layersM)
    layersM[1,1] <- layersM[nrow(layersM),1] <- 0
    innerHeights <- innerWidths <- 5
    layout(layersM, widths =  c(innerWidths/2,rep(innerWidths,times=length(iterSeq)),1),
           heights = c(innerWidths/4,innerHeights,innerWidths/4))
    for(ija in iterSeq){
      atValue = 1+ija
      par(mar=c(1,4,1,1))
      unlist( res_df$estSE_Qhat_exact )
      atValue <- unlist(res_df$kFactors) == sort(unique(unlist(res_df$kFactors)))[ija]
      trueSE_Qhat_vec <- tapply(unlist( res_df[atValue,]$Q_value_hat_withTruePi),
                                unlist( res_df[atValue,]$nObs),
                                sd)
      estSE_Qhat_MEst_vec <- tapply(unlist( res_df[atValue,]$estSE_Qhat_MEst),
                                unlist( res_df[atValue,]$nObs),
                                mean)
      estSE_Qhat_exact_vec <- tapply(unlist( res_df[atValue,]$estSE_Qhat_exact),
                                unlist( res_df[atValue,]$nObs),
                                mean)
      axisLims = summary(c(estSE_Qhat_MEst_vec,
                           #estSE_Qhat_exact_vec,
                           trueSE_Qhat_vec
                           ))[c(1,6)]
      axisLims_ = axisLims
      axisLims_[1] <- axisLims_[1] - mean(base::diff(axisLims_))/5
      plot(trueSE_Qhat_vec,
           estSE_Qhat_MEst_vec,type = "b",
           cex.axis = 2, pch = "",xlab = "",ylab = "",yaxt = 'n',xaxt ="n",
           xlim = axisLims_, ylim = axisLims_);
      axLabels = seq(axisLims[1],axisLims[2],length.out=4)
      axLabels = round(axLabels,2)
      axis(1,at = axLabels, labels = axLabels,cex.axis=1.5)
      axis(2,at = axLabels, labels = axLabels,cex.axis=1.5)
      points(trueSE_Qhat_vec, estSE_Qhat_MEst_vec,
             type = "b",
             col = vals2gray( names(trueSE_Qhat_vec) ),
             pch = 19, lty = 1, cex = 2)
      #text(trueSE_Qhat_vec, estSE_Qhat_MEst_vec,
           #labels = names(trueSE_Qhat_vec),cex=1.5)
      abline(a=0,b=1,lty=2,col='gray',lwd=2)
      if(T == F){
        points(trueSE_Qhat_vec, estSE_Qhat_exact_vec,
               type = "b",
               col = vals2gray( names(trueSE_Qhat_vec) ),
               pch = "B", lty = 2, cex = 2)
      }
    }
    par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "Mean Estimated S.E.",cex=3,srt=90)
    #fancyLabel = paste("True Variance of",expression(hat(Q)(pi^*)))
    #fancyLabel = expression(hat(Q)(pi^*))
    fancyLabel = expression(paste("True S.E. of ", hat(Q)(hat(pi)^'*')))
    par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = fancyLabel,cex=3)
    for(k__ in kFactors_seq){par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = sprintf("%s Factors",k__),cex=3)}
  }
  dev.off()

  pdf(sprintf("%s/varianceFig_%s_.pdf",FiguresLoc,name_key),width = 3*length(iterSeq),height=5)
  {
    layersM = matrix(1:length(iterSeq),byrow=T,ncol=length(iterSeq))
    layersM = cbind(length(iterSeq)+1,layersM)
    layersM = rbind(layersM,length(iterSeq)+2)
    layersM = rbind(length(iterSeq)+(2:(2+length(iterSeq))),layersM)
    layersM[1,1] <- layersM[nrow(layersM),1] <- 0
    innerHeights <- innerWidths <- 5
    layout(layersM, widths =  c(innerWidths/2,rep(innerWidths,times=length(iterSeq)),1),
           heights = c(innerWidths/4,innerHeights,innerWidths/4))
    for(ija in iterSeq){
      par(mar=c(1,4,1,1))
      atValue <- unlist(res_df$kFactors) == sort(unique(unlist(res_df$kFactors)))[ija]
      estSE_pi_ <-  do.call(rbind,tapply(( res_df[atValue,]$pi_star_hat_se),
                            unlist( res_df[atValue,]$nObs),
                             function(zer){apply(do.call(rbind, zer),2,mean)}))
      trueSE_pi_ <- do.call(rbind,tapply(( res_df[atValue,]$pi_star_hat),
                          unlist( res_df[atValue,]$nObs),
                          function(zer){apply(do.call(rbind, zer),2,sd)}))
      pi_star_true_ <- tapply(( res_df[atValue,]$pi_star_true), unlist( res_df[atValue,]$nObs), c)[[1]][[1]]
      axisLims_ = axisLims <- summary(c(estSE_pi_[,-1], trueSE_pi_[,-1]))[c(1,6)]
      axisLims_[1] <- axisLims_[1] - mean(base::diff(axisLims_))/5
      axisLims_[1] <- 0#max(0,axisLims_[1])
      plot(1,type = "b",
           cex.axis = 2, pch = "",xlab = "",ylab = "", xlim = axisLims_,
           xaxt = "n",yaxt = "n",
           ylim = axisLims_, log="");
      atValue = 1+ija
      axLabels = seq(axisLims[1],axisLims[2],length.out=5)
      axLabels = round(axLabels,2)
      #color_vec <- rev(rank(nObs_seq))
      #color_vec <- c("lightgray","gray","darkgray", "black")
      #color_vec <- color_vec[(length(color_vec)-length(nObs_seq)+1):length(color_vec)]
      color_vec <- c("black","black","black")
      for(k__ in 1:(ncol(trueSE_pi_))){
        print(k__)
        #pch_vec <- 1*(min(abs(pi_star_true_[k__] - c(1-my_ep,my_ep))) < 0.10)+1
        pch_vec <- 1
        #points(trueSE_pi_[,k__], estSE_pi_[,k__],type = "p", col = color_vec, cex = 2,pch=pch_vec)
        #points(trueSE_pi_[,k__], estSE_pi_[,k__],type="b",lty = 2,
               #pch = 1, lwd=0.6,col = vals2gray( row.names(trueSE_pi_) ), cex = 1.5)
        points(trueSE_pi_[,k__], estSE_pi_[,k__],type="b",pch = " ",lwd=0.6,col="gray")
        text(trueSE_pi_[,k__], estSE_pi_[,k__], labels = "o",
             col = vals2gray( row.names(trueSE_pi_) ), cex = 2)
        #text(trueSE_pi_[,k__], estSE_pi_[,k__], labels = row.names(trueSE_pi_),
             #col = color_vec, cex = 1)
      }
      abline(a=0,b=1,lty=2,col='gray',lwd=2)
      library( sfsmisc )
      sfsmisc::eaxis(1, cex.axis=1.15,log=F,draw.between.ticks=F,max.at = 2, equidist.at.tol	= 0.000001)
      sfsmisc::eaxis(2, cex.axis=1.15,log=F,draw.between.ticks=F,max.at = 2, equidist.at.tol	= 0.000001)
      #legend("bottomright",col=c("black","darkgray"),lty=c(1,2),legend=c("Conservative","M Estimation"))
    }
    par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "Mean Estimated S.E.",cex=3,srt=90)
    #fancyLabel = paste("True Variance of",expression(hat(Q)(pi^*)))
    #fancyLabel = expression(hat(Q)(pi^*))
    fancyLabel = expression(paste("True Sampling Variability of ", hat(pi)^'*', "\n (SD scale)"))
    par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = fancyLabel,cex=3)
    for(k__ in kFactors_seq){par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = sprintf("%s Factors",k__),cex=3)}
  }
  dev.off()

  for(outer__ in 1:3){
    if(outer__ == 1){ PDF_NAME = sprintf("%s/fullCoverageQCon_%s.pdf",FiguresLoc,name_key) }
    if(outer__ == 2){ PDF_NAME = sprintf("%s/fullCoverage_%s.pdf",FiguresLoc,name_key) }
    if(outer__ == 3){ PDF_NAME = sprintf("%s/fullCoverageQMest_%s.pdf",FiguresLoc,name_key) }
    pdf(PDF_NAME,width = 3*nK,height=5)
    {
      layersM = matrix(1:nK,byrow=T,ncol=nK)
      layersM = cbind(nK+1,layersM)
      layersM = rbind(layersM,nK+2)
      layersM = rbind(nK+(2:(2+nK)),layersM)
      layersM[1,1] <- layersM[nrow(layersM),1] <- 0
      innerHeights <- innerWidths <- 5
      layout(layersM, widths =  c(innerWidths/4,rep(innerWidths,times=nK),1),
             heights = c(innerWidths/4,innerHeights,innerWidths/4))
      cexAx = 1.5
      for(jaa in 1:length(kFactors_seq)){
        par(mar=c(1,3,1,1.85))
        kFactors_indices <- unlist(res_df$kFactors) == (k__<-sort(unique(unlist(res_df$kFactors)))[jaa])
        if(outer__ == 1){
          type_<-"Q"; startAt = jaa+1;
          inCI_matUse <-  tapply(unlist( res_df[kFactors_indices,]$inCIQ ),
                                      unlist( res_df[kFactors_indices,]$nObs),
                                      mean)
        }
        if(outer__ == 2){
          type_<-"pi";startAt = 2;
          tmp_ <- do.call(rbind, res_df[kFactors_indices,]$inCI_pi_star)
          inCI_matUse <- do.call(rbind, tapply(1:nrow(tmp_),
                                     unlist( res_df[kFactors_indices,]$nObs),
                                     function(zer){colMeans(tmp_[zer,])}))
        }
        if(outer__ == 3){
          type_<-"Q";startAt = jaa+1;
          inCI_matUse <- abs( tapply(unlist( res_df[kFactors_indices,]$inCIQ_MEst ),
                                  unlist( res_df[kFactors_indices,]$nObs),
                                  mean))
        }
        nObs_values <- abs( tapply(unlist( res_df[kFactors_indices,]$nObs),
                                   unlist( res_df[kFactors_indices,]$nObs),
                                   unique))
        plot(nObs_values,
             unlist(ifelse(type_ == "pi",
                           yes = list(rowMeans(inCI_matUse)),
                           no = list(inCI_matUse))),
             #ylim = c(0,1),
             ylim = (ylim_cov <- c(min(unlist(inCI_matUse))*0.75,1)),
             cex= 1*(type_=="pi")+2*(type_=="Q"),
             yaxt = "n", log = "x",
             xaxt = "n", ylab = "", xlab = "",
             cex.lab = 1*(type_=="pi")+2*(type_=="Q"),
             type = "b",cex.axis=cexAx)
        if(jaa %in% c(1)){ axis(2,vals2<-round(seq(round(ylim_cov[1],2L),1,length=5),2L),labels=vals2,cex.axis=cexAx) }
        axis(1,nObs_seq,labels = nObs_seq,cex.axis = 1.5)
        abline(h = ConfLevel,lty= 2,col="gray",lwd = 2)
        if(type_ == "pi"){
          points(nObs_values,rowMeans(inCI_matUse),type = "b",lwd=5,pch=19,cex=2)#,pch=16)
          for(jaa in 1:ncol(inCI_matUse)){ points(nObs_values,inCI_matUse[,jaa],type = "b") }
        }
      }
      par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "Coverage",cex=3,srt=90)
      par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "Number of Observations",cex=3)
      for(k__ in kFactors_seq){par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = sprintf("%s Factors",k__),cex=3)}
    }
    dev.off()
  }

  for(outer_ in 1:2){
    pow_ <- 0.5
    if(outer_ == 1){pdf_name = sprintf('%s/BIAS_RMSE_%s.pdf',FiguresLoc,name_key)}
    if(outer_ == 2){pdf_name =  sprintf('%s/BIAS_RMSE_Q_%s.pdf',FiguresLoc,name_key)}
    pdf(pdf_name,width = 7, height = 12)
    {
      logX <- "x";
      innerWidths <- 8; innerHeights=8
      zones <- matrix(c(1,0,2,3,0,4,5,0,6),nrow = 3, byrow = T)
      zones = rbind(c(7,0,8),zones)
      zones = rbind(zones,c(9,9,9))
      layout(zones, widths =  c(innerWidths,innerWidths/5,innerWidths),
             heights = c(innerHeights/4,innerHeights,innerHeights,innerHeights,innerHeights/4))
      par(mar=c(5, 6, 3, 1) )
      loop_counter <- 0
      for(ija in 1:length(kFactors_seq)){
        loop_counter = loop_counter + 1
        kFactors_indices <- unlist(res_df$kFactors) == (k_<-sort(unique(unlist(res_df$kFactors)))[ija])
        if(outer_ == 1){
          pi_hat_ija <- do.call(rbind, res_df[kFactors_indices,]$pi_star_hat)
          pi_true_ija <- do.call(rbind, res_df[kFactors_indices,]$pi_star_true)
          bias_mat_hajek <- abs(do.call(rbind, tapply(1:nrow(pi_true_ija),
                                    unlist( res_df[kFactors_indices,]$nObs),
                                    function(fi){
                      colMeans( pi_hat_ija[fi,] - pi_true_ija[fi,] ) })))
          discrep_mat_hajek <- do.call(rbind, tapply(1:nrow(pi_true_ija),
                          unlist( res_df[kFactors_indices,]$nObs),
                                     function(fi){
                             colMeans( (pi_hat_ija[fi,] - pi_true_ija[fi,])^2 ) }))^pow_
          ylim_ = c(0,eval(parse(text=sprintf("max(discrep_mat_hajek)"))) )
        }
        if(outer_ == 2){
          bias_mat_hajek <- abs( tapply(unlist( res_df[kFactors_indices,]$Qest_MEst ) -
                                          unlist( res_df[kFactors_indices,]$trueQ ),
                                                      unlist( res_df[kFactors_indices,]$nObs),
                                                      mean))
          discrep_mat_hajek <- abs( tapply((unlist( res_df[kFactors_indices,]$Qest_MEst ) -
                                             unlist( res_df[kFactors_indices,]$trueQ ))^2,
                                           unlist( res_df[kFactors_indices,]$nObs),
                                           mean))^pow_
          ylim_ = c(0,eval(parse(text=sprintf("1.1*max(c(unlist(discrep_mat_hajek)) )"))) )
        }

        xlim_ <- c(min(unlist(res_df$nObs))*0.9,max(unlist(res_df$nObs))*1.25)
        #bias
        {
          mar_vec <- c(2,6,2,0)
          par(mar=mar_vec)
          plot( nObs_seq, unlist(ifelse(outer_ == 1,
                                 yes = list(rowMeans(bias_mat_hajek)),
                                 no = list(bias_mat_hajek))),
                main = sprintf("%i Factors", k_),
                ylab = "",xaxt = "n", type = "b", yaxt = "n",
                pch=19*(outer_%in%c(1,3))+1*(outer_%in%c(2,4))  ,
                ylim =ylim_, xlim = xlim_,
                xlab = "",log = logX,lwd=3,
                cex = 4,cex.axis = 2,cex.lab = 2,cex.main = 2)
          sfsmisc::eaxis(2,  #at = seq(ylim_[1],ylim_[2],length.out=5),
                         cex.axis=1.5,log=F,n.axp = NULL, draw.between.ticks=F,drop.1 = F,max.at = 10, equidist.at.tol	= 0.0001)
          if(loop_counter == length(kFactors_seq)){axis(1,nObs_seq,labels = nObs_seq,cex.axis = 1.5)}
          if(loop_counter != length(kFactors_seq)){axis(1,nObs_seq,labels = rep("",times=length(nObs_seq)),cex.axis = 1.5)}
          if(outer_ %in% c(1)){
            for(jao in 1:ncol(bias_mat_hajek)){ points( nObs_seq,abs(bias_mat_hajek[,jao]),type = "b") }
          }
        }

        #total
        {
          mar_vec_total <- mar_vec
          mar_vec_total[2] <- 0
          mar_vec_total[4] <- (mar_vec[4]+mar_vec[2])
          par(mar=mar_vec_total)
          plot( nObs_seq,
                unlist(ifelse(outer_ == 1,
                              yes = list(rowMeans(discrep_mat_hajek)),
                              no = list(discrep_mat_hajek))),
                main = sprintf("%i Factors", k_),
                ylab = "",xaxt ="n", type = "b", lwd = 3, xlim = xlim_,
                ylim =ylim_, xlab = "",log = logX,yaxt = "n",
                pch=19*(outer_ %in% c(1,3))+1*(outer_%in%c(2,4))  ,
                cex = 4, cex.axis = 2,cex.lab = 2,cex.main = 2)
          #sfsmisc::eaxis(2, cex.axis=1.5,log=F,draw.between.ticks=F,max.at = 2, equidist.at.tol	= 0.000001)
          if(loop_counter == length(kFactors_seq)){axis(1,nObs_seq,labels = nObs_seq,cex.axis = 1.5)}
          if(loop_counter != length(kFactors_seq)){axis(1,nObs_seq,labels = rep("",times=length(nObs_seq)),cex.axis = 1.5)}
          if(outer_ %in% c(1)){
            for(jao in 1:ncol(discrep_mat_hajek)){ points( nObs_seq,discrep_mat_hajek[,jao],type = "b") }
          }
        }
      }
      par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "   Absolute Bias",cex=3)
      par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "RMSE  ",cex=3)
      par(mar=c(0,0,0,0));plot(0,0,xaxt = "n",yaxt = 'n',cex=1,xlab = "",ylab = "",bty="n",col="white"); text(0,0,labels = "Number of Observations",cex=3)
    }
    dev.off()
  }
}
