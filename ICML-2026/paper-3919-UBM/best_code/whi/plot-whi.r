# # Function to install and load required packages
# install_and_load_packages <- function(packages) {
#   for (package in packages) {
#     if (!require(package, character.only = TRUE)) {
#       cat(sprintf("Installing package: %s\n", package))
#       install.packages(package, repos = "https://cran.rstudio.com", quiet = TRUE)
#       library(package, character.only = TRUE)
#     }
#   }
# }

# # List of required packages
# required_packages <- c(
#   "ggplot2",
#   "dplyr",
#   "tidyr",
#   "gridExtra",
#   "grid",
#   "ggthemes",
#   "scales",
#   "argparse",
#   "svglite"
# )

# # Install and load all required packages
# install_and_load_packages(required_packages)
library(argparse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(grid)
library(ggthemes)
library(scales)

# Create argument parser
parser <- ArgumentParser()
parser$add_argument("--biased_file", type="character", help="Path to biased results CSV")
parser$add_argument("--unbiased_file", type="character", help="Path to unbiased results CSV")
parser$add_argument("--manual_biased_file", type="character", help="Path to manually biased results CSV")
parser$add_argument("--output_file", type="character", help="Name of output file (including extension)")

args <- parser$parse_args()

path <- "/Users//Documents/research/benchmarking-OS/whi"
args$biased_file <- file.path(path, "results/DROPSOME_biased_includecensored_False_STROKE_RF.csv")
args$unbiased_file <- file.path(path, "results/DROPSOME_unbiased_includecensored_False_STROKE_RF.csv")
args$manual_biased_file <- file.path(path, "results/DROPSOME_manually_biased_includecensored_False_STROKE_RF.csv")
args$output_file <- file.path(path, "figures/dropsome_censored_False_STROKE_RF.png")

# Read the CSV files
biased_data <- read.csv(args$biased_file)
unbiased_data <- read.csv(args$unbiased_file)
manual_biased_data <- read.csv(args$manual_biased_file)

# Create the combined data frame
data <- data.frame(
  signal = rep(c("ρ(b,S)", "ρ(b,A)", "ρ(b,Y)"), each = 3),  # each signal appears 3 times
  setup = rep(c("Biased", "Unbiased", "Manual"), times = 3), # three setups for each signal
  mean = c(
    # S values
    biased_data$mean[biased_data$X == "SE_S"],
    unbiased_data$mean[unbiased_data$X == "SE_S"],
    manual_biased_data$mean[manual_biased_data$X == "SE_S"],
    # A values
    biased_data$mean[biased_data$X == "SE_A"],
    unbiased_data$mean[unbiased_data$X == "SE_A"],
    manual_biased_data$mean[manual_biased_data$X == "SE_A"],
    # Y values (using SE_Y1)
    biased_data$mean[biased_data$X == "SE_Y1"],
    unbiased_data$mean[unbiased_data$X == "SE_Y1"],
    manual_biased_data$mean[manual_biased_data$X == "SE_Y1"]
  ),
  lower = c(
    # S values
    biased_data$lower[biased_data$X == "SE_S"],
    unbiased_data$lower[unbiased_data$X == "SE_S"],
    manual_biased_data$lower[manual_biased_data$X == "SE_S"],
    # A values
    biased_data$lower[biased_data$X == "SE_A"],
    unbiased_data$lower[unbiased_data$X == "SE_A"],
    manual_biased_data$lower[manual_biased_data$X == "SE_A"],
    # Y values (using SE_Y1)
    biased_data$lower[biased_data$X == "SE_Y1"],
    unbiased_data$lower[unbiased_data$X == "SE_Y1"],
    manual_biased_data$lower[manual_biased_data$X == "SE_Y1"]
  ),
  upper = c(
    # S values
    biased_data$upper[biased_data$X == "SE_S"],
    unbiased_data$upper[unbiased_data$X == "SE_S"],
    manual_biased_data$upper[manual_biased_data$X == "SE_S"],
    # A values
    biased_data$upper[biased_data$X == "SE_A"],
    unbiased_data$upper[unbiased_data$X == "SE_A"],
    manual_biased_data$upper[manual_biased_data$X == "SE_A"],
    # Y values (using SE_Y1)
    biased_data$upper[biased_data$X == "SE_Y1"],
    unbiased_data$upper[unbiased_data$X == "SE_Y1"],
    manual_biased_data$upper[manual_biased_data$X == "SE_Y1"]
  )
)


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
  discrete_scale("fill","Publication",manual_pal(values = c("#0f7ba2","#7fc97f", "#dd5129","#fdb462","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}
# "#386cb0","#ef3b2c"
scale_colour_Publication <- function(...){
  discrete_scale("colour","Publication",manual_pal(values = c("#0f7ba2","#7fc97f","#dd5129","#fdb462","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

data$signal <- factor(data$signal, 
                      levels = c("ρ(b,S)", "ρ(b,A)", "ρ(b,Y)"))
data$setup <- factor(data$setup, 
                      levels = c("Biased", "Unbiased", "Manual"))

# Create the plot
p <- ggplot(data, aes(x = signal, y = mean, fill = setup)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
                position = position_dodge(width = 0.8),
                width = 0.25,
                size = 1.) +
  labs(x = "",
       y = "",
       fill = "Setup") +
  scale_y_continuous(limits = c(-0.15, 0.4)) +
  theme_Publication() +
  scale_fill_Publication() +
  theme(
    plot.title = element_text(size = 25, face = "bold"),
    axis.text = element_text(size = 22),
    axis.title = element_text(size = 24, face = "bold"),
    legend.title = element_text(size = 22),
    legend.text = element_text(size = 21),
    legend.key.size = unit(1, "cm")
  )


print(p)

# Save the plot
ggsave(args$output_file, p, width = 12, height = 8, dpi = 300)
