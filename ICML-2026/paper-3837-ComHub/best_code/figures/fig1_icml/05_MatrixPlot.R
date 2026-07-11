plot.pdm = function(data, main, breaks = 21, diagonal = FALSE,
                    cex = 0.05, entryrange = NULL){
  .d = dim(data)[1]
  
  
  if( is.null(entryrange)){
    entryrange = 0
    if(diagonal){
      entryrange = max(abs(data))
    }
    else{
      for(.i in 1:.d){
        for(.j in (1:.d)[-.i]){
          entryrange = max(entryrange, abs(data[.i,.j]))
        }
      }
    }
  }
  
  rbPal <- colorRampPalette(c('red',"gray95",'blue'))
  color <- rbPal(breaks)
  plot(1,1,
       col = "white",
       xlim = c(1, .d),
       ylim = c(1, .d),
       xlab = "Row Index",
       ylab = "Column Index",
       xaxt = "none",
       yaxt = "none",
       main = main)
  axis(2, at = c(1, seq(from = 0, to = .d, length.out = 6)[-1]), 
       labels = c(seq(from = .d, to = 0, length.out = 6)[-6], 1))
  axis(1, at = c(1, seq(from = 0, to = .d, length.out = 6)[-1]), 
       labels = c(1, seq(from = 0, to = .d, length.out = 6)[-1]))
  
  for(.i in 1:.d){
    if(diagonal){
      for(.j in 1:.d){
        colorindex = floor( breaks*(data[.i,.j] + entryrange)/(2*entryrange) + 0.99 )
        points(x= c(.j), y= c(.d-.i+1), col = color[colorindex], 
               pch = 19, cex = cex)
      }
    } else {
      for(.j in (1:.d)[-.i]){
        colorindex = floor( breaks*(data[.i,.j] + entryrange)/(2*entryrange) + 0.99 )
        points(x= c(.j), y= c(.d-.i+1), col = color[colorindex], 
               pch = 19, cex = cex)
      }
    }
  }
}