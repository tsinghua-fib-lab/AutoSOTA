library(ggplot2)
library(tidyr)
library(dplyr)
library(gridExtra)

# Get the directory of the current script
current_dir <- dirname(rstudioapi::getSourceEditorContext()$path)

# Read the CSV files from the script's directory
data_rct <- read.csv(file.path(current_dir, "data/data_rct_biased.csv"))
data_obs <- read.csv(file.path(current_dir, "data/data_obs_biased.csv"))
cate_obs <- read.csv(file.path(current_dir, "data/cate_obs_biased.csv"))

# Add a 'Source' column to both datasets
data_rct <- data_rct %>% mutate(Source = "RCT")
data_obs <- data_obs %>% mutate(Source = "OBS")

# Combine the datasets
combined_data <- rbind(data_rct, data_obs)

# Define Tableau colors
tableau_colors <- c("RCT" = "#4E79A7", "OBS" = "#E15759")

# 1. Scatter plot (both A = 1 and A = 0)
p_scatter <- ggplot(combined_data, aes(x = X, y = Y, color = Source, shape = factor(A))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = tableau_colors) +
  scale_shape_manual(values = c("0" = 4, "1" = 16), labels = c("0" = "Control", "1" = "Treatment")) +
  labs(title = "Y vs X (Both Groups)",
       x = "X", y = "Y", shape = "Group") +
  theme_bw() +
  theme(legend.position = "bottom")

# 2. CATE plot
p_cate <- ggplot() +
  geom_point(data = cate_obs, aes(x = X, y = CATE-10), color = "red", alpha = 0.6) +
  labs(title = "CATE Signal Difference",
       x = "X", y = "CATE Signal Difference (phi1 - phi0)") +
  theme_bw() +
  theme(legend.position = "none")

# 3. Density plot for Y
p_density <- ggplot(combined_data, aes(x = Y, fill = Source)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = tableau_colors) +
  labs(title = "Distribution of Y",
       x = "Y", y = "Density") +
  theme_bw() +
  theme(legend.position = "bottom")

# 4. Bar plot for A
p_bar <- ggplot(combined_data, aes(x = factor(A), fill = Source)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = tableau_colors) +
  labs(title = "Distribution of A",
       x = "A", y = "Count") +
  theme_bw() +
  theme(legend.position = "bottom")

# 5. Scatter plot of A vs Y
p_scatter_ay <- ggplot(combined_data, aes(x = factor(A), y = Y, color = Source)) +
  geom_jitter(alpha = 0.6, width = 0.2) +
  scale_color_manual(values = tableau_colors) +
  labs(title = "Y vs A",
       x = "A", y = "Y") +
  theme_bw() +
  theme(legend.position = "bottom")

# Combine the plots
combined_plot <- grid.arrange(p_scatter, p_cate, p_density, p_bar, p_scatter_ay, ncol = 2, nrow = 3)

# Save the combined plot
ggsave(
  file.path(current_dir, "plots/combined_analysis_plot_berksons_cate.png"),
  plot = combined_plot, width = 12, height = 10, dpi = 300
)

# Print the combined plot
print(combined_plot)
