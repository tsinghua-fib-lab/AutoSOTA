# Create a flag to toggle between datasets
use_lr_data <- FALSE  # Set to TRUE to use logistic regression data
use_chd_data <- TRUE  # Set to TRUE to use CHD data, FALSE for stroke data

# Create both datasets
data_rf_chd <- data.frame(
  signal = rep(c("ρ(b,Y)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 3),
  mean = c(
    # SE_Y1
    0.031962306338244276, 0.08155455977715628,
    # SE_A
    -0.045141905376108835, -0.00046874829840122534,
    # SE_S
    0.09035409569932758, -0.04041617086693901
  ),
  lower = c(
    # SE_Y1
    0.018978037521950257, 0.04182262526377176,
    # SE_A
    -0.06519967161461451, -0.03538322535950635,
    # SE_S
    0.0016548700740890432, -0.05848961985472129
  ),
  upper = c(
    # SE_Y1
    0.044946575154538294, 0.12128649429054081,
    # SE_A
    -0.025084139137603153, 0.0344457287627039,
    # SE_S
    0.1790533213245661, -0.022342721879156723
  )
)

data_lr_chd <- data.frame(
  signal = rep(c("ρ(b,Y)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 3),
  mean = c(
    # SE_Y1
    0.06578256016820236, 0.17178690085946174,
    # SE_A
    -0.06477697292852276, -0.020646935286137445,
    # SE_S
    0.0427577687558696, -0.049773484602441395
  ),
  lower = c(
    # SE_Y1
    0.017334797599937314, 0.07362633418081539,
    # SE_A
    -0.09471456694756249, -0.03191580229964409,
    # SE_S
    -0.00811790510670165, -0.07461592316540933
  ),
  upper = c(
    # SE_Y1
    0.11423032273646741, 0.2699474675381081,
    # SE_A
    -0.034839378909483035, -0.009378068272630801,
    # SE_S
    0.09363344261844084, -0.02493104603947347
  )
)

# Stroke data (new data frames)
data_rf_stroke <- data.frame(
  signal = rep(c("ρ(b,Y)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 3),
  mean = c(
    # SE_Y1
    0.0338352020699568, 0.12333890353858519,
    # SE_A
    -0.046607697023839614, -0.020340027887138595,
    # SE_S
    0.2266931752223158, -0.04512527821289835
  ),
  lower = c(
    # SE_Y1
    0.018925002701517687, 0.04620130949930053,
    # SE_A
    -0.06725212552717151, -0.030514933854793706,
    # SE_S
    0.08999307763062633, -0.06500784326160651
  ),
  upper = c(
    # SE_Y1
    0.04874540143839591, 0.20047649757786984,
    # SE_A
    -0.025963268520507718, -0.010165121919483485,
    # SE_S
    0.36339327281400524, -0.025242713164190186
  )
)

data_lr_stroke <- data.frame(
  signal = rep(c("ρ(b,Y)", "ρ(b,A)", "ρ(b,S)"), each = 2),
  setup = rep(c("Biased", "Unbiased"), 3),
  mean = c(
    # SE_Y1
    0.039067805894640525, 0.10108212953120048,
    # SE_A
    -0.10025319850777348, -0.013340700980222506,
    # SE_S
    0.018520911355769033, -0.05232157262760371
  ),
  lower = c(
    # SE_Y1
    0.02288427626772275, 0.052673158987189224,
    # SE_A
    -0.1445638165558293, -0.03312052658621503,
    # SE_S
    -0.037518258899853434, -0.07575297624064037
  ),
  upper = c(
    # SE_Y1
    0.0552513355215583, 0.14949110007521174,
    # SE_A
    -0.05594258045971765, 0.006439124625770018,
    # SE_S
    0.0745600816113915, -0.028890169014567056
  )
)

# Select the appropriate dataset based on both flags
data <- if(use_lr_data) {
  if(use_chd_data) data_lr_chd else data_lr_stroke
} else {
  if(use_chd_data) data_rf_chd else data_rf_stroke
}

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(grid)
library(ggthemes)
library(scales)

# Define the custom theme and scale functions
theme_Publication <- function(base_size=14, base_family="sans") {
  (theme_foundation(base_size=base_size, base_family=base_family)
   + theme(plot.title = element_text(face = "bold",
                                     size = rel(1.2), hjust = 0.5),
           text = element_text(),
           panel.background = element_rect(colour = NA),
           plot.background = element_rect(colour = NA),
           panel.border = element_rect(colour = NA),
           axis.title = element_text(face = "bold",size = rel(1)),
           axis.title.y = element_text(angle=90,vjust =2),
           axis.title.x = element_text(vjust = -0.2),
           axis.text = element_text(), 
           axis.line = element_line(colour="black"),
           axis.ticks = element_line(),
           panel.grid.major = element_line(colour="#f0f0f0"),
           panel.grid.minor = element_blank(),
           legend.key = element_rect(colour = NA),
           legend.position = "bottom",
           legend.direction = "horizontal",
           legend.key.size= unit(0.2, "cm"),
           legend.spacing = unit(0, "cm"),
           legend.title = element_text(face="italic"),
           plot.margin=unit(c(10,5,5,5),"mm"),
           strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
           strip.text = element_text(face="bold")
      ))
}
# "#386cb0","#ef3b2c"
scale_fill_Publication <- function(...){
  discrete_scale("fill","Publication",manual_pal(values = c("#0f7ba2", "#dd5129","#7fc97f","#fdb462","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}
# "#386cb0","#ef3b2c"
scale_colour_Publication <- function(...){
  discrete_scale("colour","Publication",manual_pal(values = c("#0f7ba2", "#dd5129","#7fc97f","#fdb462","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

# Create the plot
# title <- if(use_lr_data) { 
#     "Alignment w/ Predictive Performance using LogReg (stroke)"
# } else {
#     "Alignment w/ Predictive Performance using RF (stroke)"
# }
data$signal <- factor(data$signal, 
                     levels = c("ρ(b,S)", "ρ(b,A)", "ρ(b,Y)"))

p <- ggplot(data, aes(x = signal, y = mean, fill = setup)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
                position = position_dodge(width = 0.8),
                width = 0.25,
                size=1.) +
  labs(x = "",
       y = "",
       fill = "Setup") +
  scale_y_continuous(limits = c(-0.15, 0.2)) +
  theme_Publication() +
  scale_fill_Publication() +
#   ggtitle(title) +
  theme(
    plot.title = element_text(size=25, face="bold"),
    axis.text = element_text(size = 22),
    axis.title = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 22),
    legend.text = element_text(size = 21),
    legend.key.size = unit(1,"cm")
  )

# Print the plot
print(p)

# Save the plot with a name that reflects which dataset was used
filename <- if(use_lr_data) "covariance_comparison_lr_v3_STROKE.svg" else "covariance_comparison_rf_v3_STROKE.svg"
ggsave(filename, p, width = 12, height = 8, dpi = 300)