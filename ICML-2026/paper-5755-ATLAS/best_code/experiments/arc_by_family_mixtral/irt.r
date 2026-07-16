# usage: 
# for i in $(seq 106 105 737) 841; do
#   Rscript irt.r $i
# done
library(mirt)

args <- commandArgs(trailingOnly = TRUE)
index_name <- args[1]
i <- as.numeric(args[1])
print(i)
data<-read.csv("data/arc_response_matrix_other.csv")

# Get dimensions of the original data
cat("Dimensions of origial data:", dim(data), "\n")

data_clean <- na.omit(data)              # remove NA rows
data <- data_clean[, colSums(is.na(data_clean)) == 0]  # remove NA columns

# Drop constant columns (no variance)
constant_cols <- apply(data, 2, function(x) length(unique(x)) == 1)
clean_data <- data[, !constant_cols]
cat("Dropped", sum(constant_cols), "constant columns.\n")

# Optionally drop constant rows
constant_rows <- apply(clean_data, 1, function(x) length(unique(x)) == 1)
clean_data <- clean_data[!constant_rows, ]
cat("Dropped", sum(constant_rows), "constant rows.\n")

# Get dimensions of the cleaned data
cat("Dimensions of cleaned data:", dim(clean_data), "\n")
 
# data_for_id <- clean_data
# Dimensions of cleaned data: 3296 845 

#dat<-clean_data[, (i-112):i]
# dat<-clean_data[, 2:106]

# dat<-clean_data[, (i-111):i] # start at i=113
# dat<-clean_data[, (i-103):i]

#head(dat)
if (i < 841) {
  dat<-clean_data[, (i-104):i]
} else {
  dat<-clean_data[, 737:i]
}

model <- mirt(dat, model = 1, method="EM", itemtype = "3PL", technical = list(NCYCLES = 100000))

cat("===MODEL===\n")
print(model)

cat("====M2====\n")
m2<-M2(model)
print(m2)
summary(model, rotate = "oblimin", suppress = 0.20)
cat("===Q3===\n")
# Suppose your Q3 matrix is called q3mat
q3mat <- residuals(model, type = "Q3")
print(q3mat)
# Get upper triangle only (to avoid duplicates)
q3_upper <- q3mat
q3_upper[lower.tri(q3_upper, diag = TRUE)] <- NA

# Find pairs with Q3 > 0.20
threshold <- 0.20
bad_pairs <- which(q3_upper > threshold, arr.ind = TRUE)

# Format results nicely
results <- data.frame(
  Item1 = rownames(q3_upper)[bad_pairs[,1]],
  Item2 = colnames(q3_upper)[bad_pairs[,2]],
  Q3    = q3_upper[bad_pairs]
)

# Sort by strongest dependence
q3 <- results[order(-results$Q3), ]
print(q3)

item_fit <- itemfit(model, fit_stats = "S_X2")
bad <- item_fit[order(-item_fit$RMSEA.S_X2), ]
head(bad, 10)
plot(model, type = "trace", which.items = c(1,6))  # 1= X0, 6= X5（按顺序号）


# # 4. Person fit statistics
cat("\n4. Person Fit Statistics (first 10 persons):\n")
# Calculate factor scores first and pass them to personfit
theta_scores <- fscores(model, method = "EAP", full.scores = TRUE, full.scores.SE = TRUE, quadpts = 61)
# person_fit <- personfit(model, Theta = theta_scores, na.rm = TRUE)
# print(head(person_fit, 10))
# Item parameter estimates
cat("\nItem Parameters:\n")
item_params <- coef(model, simplify = TRUE)$items
print(item_params)

# # Save results to files
# path <- "mmlu_math_112_8b_only/"
# path <- "mmlu_gaussian_sampled_300/"
path <- "./"
if (!dir.exists(path)) dir.create(path)

output_params_file <- paste0(path,"irt_item_parameters_", index_name, ".csv")
output_scores_file <- paste0(path, "irt_person_scores_", index_name, ".csv")
# output_fit_file <- paste0(path, "irt_model_fit_", index_name, ".csv")
output_itemfit_file <- paste0(path, "irt_item_fit_", index_name, ".csv")
output_m2_file<- paste0(path, "m2_", index_name, ".csv")
output_q3_file<- paste0(path, "q3_", index_name, ".csv")
write.csv(item_params, output_params_file, row.names = TRUE)
write.csv(theta_scores, output_scores_file, row.names = FALSE)
# write.csv(person_fit, output_fit_file, row.names = FALSE)
write.csv(bad, output_itemfit_file, row.names = TRUE)
write.csv(m2, output_m2_file, row.names = TRUE)
write.csv(q3, output_q3_file, row.names = TRUE)

