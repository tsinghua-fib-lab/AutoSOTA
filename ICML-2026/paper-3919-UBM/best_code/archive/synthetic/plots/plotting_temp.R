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

scale_fill_Publication <- function(...){
  discrete_scale("fill","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

scale_colour_Publication <- function(...){
  discrete_scale("colour","Publication",manual_pal(values = c("#386cb0","#fdb462","#7fc97f","#ef3b2c","#662506","#a6cee3","#fb9a99","#984ea3","#ffff33")), ...)
}

## unbiased results
# Results for X1 = -1:
# Outcome covariance: 0.0006 ± 0.0009
# Treatment covariance: 0.0059 ± 0.0055
# Selection covariance: 0.0000 ± 0.0000

# Results for X1 = 1:
# Outcome covariance: 0.0001 ± 0.0001
# Treatment covariance: 0.0014 ± 0.0003
# Selection covariance: -0.0000 ± 0.0000

# Results for X1 = all:
# Outcome covariance: 0.0012 ± 0.0002
# Treatment covariance: 0.0011 ± 0.0011
# Selection covariance: 0.1246 ± 0.0029

## selection type 1/2 results
# Results for X1 = -1:
# Outcome covariance: -0.0000 ± 0.0005
# Treatment covariance: 0.0000 ± 0.0000
# Selection covariance: 0.0000 ± 0.0000

# Results for X1 = 1:
# Outcome covariance: 0.0013 ± 0.0003
# Treatment covariance: 0.0608 ± 0.0067
# Selection covariance: -0.0000 ± 0.0000

# Results for X1 = all:
# Outcome covariance: 0.0022 ± 0.0003
# Treatment covariance: 0.0762 ± 0.0061
# Selection covariance: 0.1093 ± 0.0036

## selection type 3 results
# Results for X1 = -1:
# Outcome covariance: 0.0006 ± 0.0009
# Treatment covariance: 0.0059 ± 0.0055
# Selection covariance: 0.0000 ± 0.0000

# Results for X1 = 1:
# Outcome covariance: 1.2067 ± 0.0594
# Treatment covariance: -0.0092 ± 0.0016
# Selection covariance: 0.0000 ± 0.0000

# Results for X1 = all:
# Outcome covariance: 0.7791 ± 0.0421
# Treatment covariance: -0.0085 ± 0.0018
# Selection covariance: 0.1767 ± 0.0030


# Create a data frame with all results
data <- data.frame(
  bias_setup = rep(c("Unbiased", "Selection Type 2", "Selection Type 3"), each = 9),
  X1 = rep(rep(c("All", "X[1]==1", "X[1]==-1"), each = 3), 3),
  signal = rep(c("Outcome", "Treatment", "Selection"), 9),
  mean = c(
    # Unbiased
    0.0012, 0.0011, 0.1246,  # All
    0.0001, 0.0014, -0.0000, # X1 = 1
    0.0006, 0.0059, 0.0000,  # X1 = -1
    # Selection Type 1/2
    0.0022, 0.0762, 0.1093,  # All
    0.0013, 0.0608, -0.0000, # X1 = 1
    -0.0000, 0.0000, 0.0000, # X1 = -1
    # Selection Type 3
    0.7791, -0.0085, 0.1767, # All
    1.2067, -0.0092, 0.0000, # X1 = 1
    0.0006, 0.0059, 0.0000   # X1 = -1
  ),
  ci = c(
    # Unbiased
    0.0002, 0.0011, 0.0029,  # All
    0.0001, 0.0003, 0.0000,  # X1 = 1
    0.0009, 0.0055, 0.0000,  # X1 = -1
    # Selection Type 1/2
    0.0003, 0.0061, 0.0036,  # All
    0.0003, 0.0067, 0.0000,  # X1 = 1
    0.0005, 0.0000, 0.0000,  # X1 = -1
    # Selection Type 3
    0.0421, 0.0018, 0.0030,  # All
    0.0594, 0.0016, 0.0000,  # X1 = 1
    0.0009, 0.0055, 0.0000   # X1 = -1
  )
)

# Calculate confidence interval bounds directly
data$ci_lower <- data$mean - data$ci
data$ci_upper <- data$mean + data$ci

# Make signal a factor with specific order
data$signal <- factor(data$signal, levels = c("Treatment", "Outcome", "Selection"))

# Create the plot with modified aesthetics
p <- ggplot(data, aes(x = signal, y = mean, fill = bias_setup)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                position = position_dodge(width = 0.8),
                width = 0.25) +
  facet_wrap(~X1, labeller=label_parsed) +
  labs(x = "Covariance Signal Type",
       y = "Covariance",
       fill = "Bias Setup") +
  theme_Publication() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold"),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 11),
    strip.text = element_text(size = 12, face = "bold"),
    panel.spacing = unit(2, "lines")
  ) +
  scale_fill_Publication()

# Print the plot
print(p)

# Save the plot
ggsave("covariance_results.png", p, width = 12, height = 6, dpi = 300)