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

# Get the directory of the current script
current_dir <- dirname(rstudioapi::getSourceEditorContext()$path)

# Read the data from CSV
df_merged <- read.csv(file.path(current_dir, "df_merged_selection.csv"))

# Calculate witness function
wf <- c(mean(df_merged$psi[df_merged$X1 == -1], na.rm = TRUE),
         mean(df_merged$psi[df_merged$X1 == 1], na.rm = TRUE))

# Calculate BCE Loss
bceloss <- c(mean(df_merged$BCELoss_T[df_merged$S == 0 & df_merged$X1 == -1], na.rm = TRUE),
              mean(df_merged$BCELoss_T[df_merged$S == 0 & df_merged$X1 == 1], na.rm = TRUE))

# Calculate RMSE Loss for T=0
mseloss_t0 <- c(mean(df_merged$SE_Y0[df_merged$S == 0 & df_merged$X1 == -1 & df_merged$T == 0], na.rm = TRUE),
                 mean(df_merged$SE_Y0[df_merged$S == 0 & df_merged$X1 == 1 & df_merged$T == 0], na.rm = TRUE))

# Calculate RMSE Loss for T=1
mseloss_t1 <- c(mean(df_merged$SE_Y1[df_merged$S == 0 & df_merged$X1 == -1 & df_merged$T == 1], na.rm = TRUE),
                 mean(df_merged$SE_Y1[df_merged$S == 0 & df_merged$X1 == 1 & df_merged$T == 1], na.rm = TRUE))

# Create a plot with a single legend for colors
plot_data_long <- data.frame(X1 = c(-1, 1), 
                              wf = wf, 
                              rmse_t1 = sqrt(mseloss_t1), 
                            #   rmse_t0 = sqrt(mseloss_t0),
                              bceloss = bceloss) %>%
  pivot_longer(cols = c(wf, rmse_t1, bceloss), 
               names_to = "Loss_Type", 
               values_to = "Value")

# Modify the Loss_Type labels
plot_data_long <- plot_data_long %>%
  mutate(Loss_Type = case_when(
    Loss_Type == "wf" ~ "Witness Function",
    Loss_Type == "rmse_t1" ~ "RMSE Loss (T=1)",
    # Loss_Type == "rmse_t0" ~ "RMSE Loss (T=0)",
    Loss_Type == "bceloss" ~ "BCE Loss"
  ))

# Create the plot with filled shapes and larger text
p <- ggplot(data = plot_data_long) +
  geom_line(aes(x = X1, y = Value, color = Loss_Type, linetype = Loss_Type), size=1) +
  geom_point(aes(x = X1, y = Value, shape = Loss_Type, fill = Loss_Type), color = "black", size = 3) +
  labs(title = "Losses of fitted models for regions of X1 (selection bias)", 
       x = expression(paste(X[1])), 
       y = expression("Loss Values")) +
  theme_Publication() +
  theme(
    legend.position = "top",
    plot.title = element_text(size = 20),  # Increase title size
    axis.title = element_text(size = 16),  # Increase axis label size
    axis.text = element_text(size = 14)    # Optionally increase axis text size
  ) +
  scale_shape_manual(values = c(21, 22, 24)) +  # Use filled shape values
  scale_fill_Publication() +  # Use fill instead of colour for points
  scale_colour_Publication() +  # Keep this for line colors
  scale_linetype_manual(values = c("solid", "solid", "dashed")) +  # Adjust as needed
  guides(color = guide_legend(title = "Loss Type"),
         shape = "none",
         linetype = "none",
         fill = "none")  # Hide fill from legend

# Print the plot
print(p)

# Save the plot as a PNG file
ggsave(filename = file.path(current_dir, "toy-experiment.png"), plot = p, width = 10, height = 8, dpi = 300)

# Print a message to confirm the file has been saved
cat("Plot saved as 'toy-experiment.png' in the current directory.\n")
