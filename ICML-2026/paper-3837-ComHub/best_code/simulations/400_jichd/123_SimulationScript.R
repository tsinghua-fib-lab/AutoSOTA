#################################################
#################################################
#################################################
##
## In the following document, we introduce the
## script that has to be run to get the 
## simulation outputs.
##
#################################################
#################################################
#################################################


#################################################
## Sourcing:
#################################################
source("001_requirements.R"); search()
source("002_GeneratingMultipleMatrixSparse.R")
source("003_UsefulMatrixTransforms.R")

source("041_Method_JICHD.R")
source("042_Method_MatrixThresholding.R")

source("061_Simulation_CreatingParameters.R")

index  <- 1
main_folder <- "400_jichd/"
source(paste0(main_folder, index, "22_SimulationFunction.R"))

#################################################
#################################################
## Step 1: Read imputs:
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Reading inputs")

input                <- commandArgs(trailingOnly = TRUE)
id_task              <- as.numeric(input[1]) %/% 10
id_mircorun          <- as.numeric(input[1]) %% 10
runtype              <- as.numeric(input[2])


args <- CreateParameters(id_task, runtype)
print("#----------------------Verifying Inputs")
print(args)

#################################################
#################################################
## Step 2: Setup run numbers and create folders
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Creating Folders")

## runtype = 1 (experiment), runtype = 2 (full simulation)
runtype_name         <- c("pretrainings", "experiments", "outputs")[runtype]
subfolder            <- paste0(main_folder, runtype_name, index, "/")
subfolder_data       <- paste0(subfolder, "data/")
subfolder_logs       <- paste0(subfolder, "logs/")
subfolder_plots      <- paste0(subfolder, "plots/")
subfolder_bic_data   <- paste0(subfolder, "bic_data/")
subfolder_bic_plots  <- paste0(subfolder, "bic_plots/")

if (!dir.exists(subfolder_data)) {
       dir.create(subfolder_data)
}
if (!dir.exists(subfolder_logs)) {
       dir.create(subfolder_logs)
}
if (!dir.exists(subfolder_plots)) {
       dir.create(subfolder_plots)
}
if (!dir.exists(subfolder_bic_data)) {
       dir.create(subfolder_bic_data)
}
if (!dir.exists(subfolder_bic_plots)) {
       dir.create(subfolder_bic_plots)
}

#################################################
#################################################
## Step 2: Run simulation scenarios:
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#------------------Running simulations")
assign(paste0("output_full", id_task), FullSimulation(args = args, index = index))

#################################################
#################################################
## Step 3: Save SIM outputs
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Saving Outputs")
tmp_env_sim <- new.env()
assign(paste0("output", id_task,"_", id_mircorun),
       get(paste0("output_full", id_task))$output_sim,
       pos = tmp_env_sim)
assign(paste0("args", id_task),
       args,
       pos = tmp_env_sim)
assign(paste0("rho_min", id_task,"_", id_mircorun),
       get(paste0("output_full", id_task))$rho_min_vec,
       pos = tmp_env_sim)

print(paste0(subfolder_data, "output", id_task,"_", id_mircorun, ".RData"))
save(
       list = ls(all.names = TRUE, pos = tmp_env_sim), envir = tmp_env_sim,
       file = paste0(subfolder_data, "output", id_task,"_", id_mircorun, ".RData"))

warnings()