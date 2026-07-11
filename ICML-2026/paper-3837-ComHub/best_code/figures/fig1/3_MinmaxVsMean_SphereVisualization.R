###################### Creating folders:
wd <- getwd()
req_lib_dir <- paste0(wd,"/req_lib")
subfolder_new        <- paste0("req_lib/")
if (!dir.exists(subfolder_new)) {
       dir.create(subfolder_new)
}
.libPaths(req_lib_dir)
print("Installing packages in directories:")
print(.libPaths())

## For 02_BIC.R
if(!require(cooltools)){
  .libPaths(req_lib_dir)
  print(.libPaths)
  install.packages("cooltools_2.4.tar.gz", repos = NULL, type="source")
  library(cooltools)
}
library(cooltools)

x <- 7.5
Sigma1 <- diag(c(1, 1, x))
Sigma2 <- diag(c(1, x, x))
dim(Sigma1)
dim(Sigma2)
dim(t(c(1,2,3)))

f_sum <- function (theta, phi)  {
  
  ## Conversion to cartesian:
  x <- sin(theta) * cos(phi)
  y <- sin(theta) * sin(phi)
  z <- cos(theta)
  
  x_vec <- cbind(x,y,z)
  
  val1 <- diag(x_vec %*% Sigma1 %*% t(x_vec))
  val2 <- diag(x_vec %*% Sigma2 %*% t(x_vec))
  
  vals <- apply(cbind(val1, val2), MARGIN = 1, mean)
  
  return(vals)
}
f_max <- function (theta, phi)  {
  
  ## Conversion to cartesian:
  x <- sin(theta) * cos(phi)
  y <- sin(theta) * sin(phi)
  z <- cos(theta)
  
  x_vec <- cbind(x,y,z)
  
  val1 <- diag(x_vec %*% Sigma1 %*% t(x_vec))
  val2 <- diag(x_vec %*% Sigma2 %*% t(x_vec))
  
  vals <- apply(cbind(val1, val2), MARGIN = 1, max)
  
  return(vals)
}



par(mar = c(0,0,0,0))
nplot(xlim=c(0,3),ylim=c(0,2),asp=1)
x <- 10
yc1 <- 2
yi1 <- 1
yc2 <- 2
phi <- 0.4
theta <- pi/3 + 0.1
Sigma1 <- diag(c(yc1, yi1, x))
Sigma2 <- diag(c(yc1, yi1, x))
sphereplot(f_sum, 50, col=planckcolors(100), 
           phi0 = phi, theta0 = theta, 
           add=TRUE, clim=c(1,x),
           center=c(0.5,1), radius=0.4)
text(0.5,1.5,sprintf("Matrix 1"), cex = 1.8)

Sigma1 <- diag(c(yc2, x, x))
Sigma2 <- diag(c(yc2, x, x))
sphereplot(f_sum, 50, col=planckcolors(100), 
           phi0 = phi, theta0 = theta, 
           add=TRUE, clim=c(1,x),
           center=c(1.5,1), radius=0.4)
text(1.5,1.5,sprintf("Matrix 2"), cex = 1.8)

Sigma1 <- diag(c(yc1, yi1, x))
Sigma2 <- diag(c(yc2, x, x))
sphereplot(f_sum, 50, col=planckcolors(100), 
           phi0 = phi, theta0 = theta, 
           add=TRUE, clim=c(1,x),
           center=c(2.5,1.5), radius=0.4)
text(2.5,2,sprintf("Additive Aggregate"), cex = 1.8)


sphereplot(f_max, 100, col=planckcolors(100), 
           phi0 = phi, theta0 = theta, 
           add=TRUE, clim=c(1,x),
           center=c(2.5,0.5), radius=0.4)
text(2.5,1,sprintf("Minimax Aggregate"), cex = 1.8)


dev.off()
par(mar = c(0.5, 2, 0.5, 0.5))
.length <- 1000
planckcolors(100)
rbPal <- planckcolors(.length)
color <- planckcolors(.length)
plot(x = rep(0,.length),
     y = seq(1, x, length.out = .length),
     col = color,
     ylab = "Scale",
     xlab = "",
     xaxt = "none")



