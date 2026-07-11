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

source("011_Tuning_BicGlasso.R")
source("021_Method_HWGLASSO.R")
source("022_Pretuning_HWGLASSO.R")

source("061_Simulation_CreatingParameters.R")

index  <- 1
main_folder <- "000_pretraining/"
source(paste0(main_folder, index, "02_PretrainingFunction.R"))




#################################################
#################################################
## Step 1: Read imputs:
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Reading inputs")

input                <- commandArgs(trailingOnly=TRUE)
id_task              <- as.numeric(input[1])
runtype              <- as.numeric(input[2])

args <- CreateParameters(id_task, runtype)
print("#----------------------Verifying Inputs")
print(args)

#################################################
#################################################
## Step 2: Setup run numbers and create folders
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Creating Folders")

print(getwd())
## Runtype: pretrainings.
runtype_name         <- c("pretrainings", "experiments", "outputs")[runtype]
subfolder            <- paste0(main_folder, runtype_name, index, "/")
subfolder_data       <- paste0(subfolder, "data/")
subfolder_logs       <- paste0(subfolder, "logs/")
subfolder_plots      <- paste0(subfolder, "plots/")
if (!dir.exists(subfolder)) {
       dir.create(subfolder)
}
if (!dir.exists(subfolder_data)) {
       dir.create(subfolder_data)
}
if (!dir.exists(subfolder_logs)) {
       dir.create(subfolder_logs)
}
if (!dir.exists(subfolder_plots)) {
       dir.create(subfolder_plots)
}

## Creating pretraining folders in 200_SimWithHglMethod
main_folder200       <- "200_hwgl/"
subfolder200         <- paste0(main_folder200, runtype_name, index, "/")
subfolder_data200    <- paste0(subfolder200, "data/")
subfolder_logs200    <- paste0(subfolder200, "logs/")
subfolder_plots200   <- paste0(subfolder200, "plots/")
if (!dir.exists(subfolder200)) {
       dir.create(subfolder200)
}
if (!dir.exists(subfolder_data200)) {
       dir.create(subfolder_data200)
}
if (!dir.exists(subfolder_logs200)) {
       dir.create(subfolder_logs200)
}
if (!dir.exists(subfolder_plots200)) {
       dir.create(subfolder_plots200)
}


#################################################
#################################################
## Step 2: Run simulation scenarios:
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#------------------Running simulations")
assign(paste0("output", id_task), pretraining_fun(args = args))

#################################################
#################################################
## Step 3: Save all the outputs
print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("#-----------------------Saving Outputs")
tmp.env <- new.env()
assign(paste0("output", id_task),
       get(paste0("output", id_task)),
       pos = tmp.env)
assign(paste0("args", id_task),
       args,
       pos = tmp.env)

print(paste0(subfolder_data, "output", id_task, ".RData"))
## Saving results in 000_pretraining
save(
       list = ls(all.names = TRUE, pos = tmp.env), envir = tmp.env, 
       file = paste0(subfolder_data, "output", id_task, ".RData"))
## Saving results in 200_SimWithHglMethod
save(
       list = ls(all.names = TRUE, pos = tmp.env), envir = tmp.env, 
       file = paste0(subfolder_data200, "output", id_task, ".RData"))
warnings()
