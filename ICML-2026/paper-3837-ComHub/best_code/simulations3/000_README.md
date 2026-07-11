# simulations

This folder contains the code corresponding to the numerical simulations results found in Section 5 of the main paper, and multiple sections in our Supplementary Materials. First, we here provide a description of the overall structure of the files. Second, we describe in detail the steps required to replicate the simulations, 

## Overall Structure of Files/Directories:

- Directory <code>req_lib</code>: directory where the required packages will be saved as a local library. 

- Files 001-061: these files contain all the required functions to perform simulations. They are divided into specific tasks, such as importing required packages (<code>001_requirements.R</code>), calculating our JIC-HD method (<code>044_JICHD.R</code>), calculating the GLASSO, and HWGL (<code>012_Method_GLASSO.R</code>, <code>021_Method_HWGLASSO.R</code>) or generating the parameters for our simulations (<code>061_Simulation_CreatingParameters.R</code>).

- Files in directory 000: The HWGL method requires of pretraining to reduce the computational cost of their simulations. The files inside of the directory <code>000_pretraining</code> contain the calls for such pretraining procedures.

- Files in directories 100-400: These directories contain the calls for the numerical simulations. Each folder corresponds to one method. The files can be used to run (A) a small debugging example to ensure things run smoothly, (B) reduced experiments with a total of 20 simulation replicates, (C) full simulations with 50 simulation replicates.

- Files/Directories 500: After reduced experiments are run, the file <code>511_Outcomes_DataAggregationExperiments.R</code> aggregates the experiments data, and saves it to the directory <code>500_AggregatedDataExperiments</code>. Then, files <code>521_MainPlots_Experiment.R</code> and <code>531_TimePlots_Experiments.R</code> generate plots with the aggregated data.

- Files/Directories 600: After full sized simulations with 50 replicates are run, the file <code>611_Outcomes_DataAggregationFull.R</code> aggregates the full simulation data, and saves it to the folder <code>600_AggregatedDataFull</code>. Then, files <code>621_MainPlots_Full.R</code> and <code>631_TimePlots_Full.R</code> generate plots with the aggregated data.

## Simulation Rerunning Instructions:

The simulations are run in a linux cluster with SLURM scheduling system. 


0. **Clone Repository:** First, clone the github repository <em>MultiHubGgmSupplementalCode</em> to a linux cluster with SLURM scheduling system.

1. **Install Requirements:** Install all packages required for our simulations by running

    - <code>Rscript 001_requirements.R</code>

    Ensure that all packages are properly installed by reviewing the directory <code>req_lib/</code>.

2. **Debugging Pretraining of HGL and HWGL:** the HWGL method is designed specially for hub recovery, but requires a careful fine-tuning of its hyperparameters, more demanding than the GLASSO method. To obtain these pre-training parameters, you must first run the pretraining scripts. 

    In the linux cluster terminal, enter the directory <code>./simulations/</code>. To ensure that things run smoothly, run the command,

    - <code>sbatch 000_pretraining/110_ClusterPassPretraining0.sh</code>

    This generates the pretraining of the HWGL method for a toy example of index 0. Verify that directories <code>/pretrainings1/</code> are created inside of the folders <code>000_pretraining/</code> and <code>200_hwgl/</code>. In each of the  <code>/pretrainings1/</code> folders, you should find the directory <code>data/</code> containing the pretuning parameters in RData format, and <code>logs/</code> with the command-line logs helping with debugging.

3. **Pretraining the HWGL:** To pre-train all simulation scenarios considered, run the following commands:

    - <code>sbatch 000_pretraining/111_ClusterPassPretraining100.sh</code>
    - <code>sbatch 000_pretraining/112_ClusterPassPretraining200.sh</code>
    - <code>sbatch 000_pretraining/113_ClusterPassPretraining500.sh</code>

    This should save the pre-trained tuning parameters in the <code>/pretrainings1/</code> folders of each of the directories <code>000_pretraining/</code> and <code>200_hwgl/</code>. Check the log files in <code>/pretrainings1/logs/</code> for any additional debugging needed.

4. **Debugging Systematic Simulations:** Once pretraining is completed, in the linux cluster terminal, and enter the directory  <code>./simulations/</code>. To ensure that things run smoothly, run the commands,

    - <code>sbatch 100_glasso/130_ClusterPassExperiment0.sh</code>
    - <code>sbatch 200_hwgl/130_ClusterPassExperiment0.sh</code>
    - <code>sbatch 300_ipchd/130_ClusterPassExperiment0.sh</code>
    - <code>sbatch 400_jichd/130_ClusterPassExperiment0.sh</code>
    
    Check the log file <code>100_glasso/experiments1/logs/output0.out</code> to verify that the glasso method ran properly. Often, warnings related to the packages may occur, but check for errors. Check whether the data file <code>100_glasso/experiments1/data/output0_0.RData</code> is created, which ensures that the simulation was completed and successful. You can explore the log and data files in the 200, 300 and 400 directories to ensure the HWGL, IPC-HD and JIC-HD methods also ran successfully. 


3. **Running Simulation Experiments:** Once debugging experiments ran succesfully, you can run preliminary simulation experiments. We have a total of 432 simulation scenarios, corresponding to different choices of dimension p, sample size n, common and individual hub strengths pC and pI, among other parameters. For each of these simulation scenarios, we perform 20 simulation replicates. To do this, we request 10 cluster nodes, and ask each to perform 2 simulation replicates per simulation scenarios. To run this, in a cluster terminal, enter the directory <code>./simulations_4/</code>, and run the lines,

    - <code>sbatch 100_glasso/131_ClusterPassExperiment100.sh</code>
    - <code>sbatch 100_glasso/132_ClusterPassExperiment200.sh</code>
    - <code>sbatch 100_glasso/133_ClusterPassExperiment500.sh</code>
    - <code>sbatch 200_hwgl/131_ClusterPassExperiment100.sh</code>
    - <code>sbatch 200_hwgl/132_ClusterPassExperiment200.sh</code>
    - <code>sbatch 200_hwgl/133_ClusterPassExperiment500.sh</code>
    - <code>sbatch 300_ipchd/131_ClusterPassExperiment100.sh</code>
    - <code>sbatch 300_ipchd/132_ClusterPassExperiment200.sh</code>
    - <code>sbatch 300_ipchd/133_ClusterPassExperiment500.sh</code>
    - <code>sbatch 400_jichd/131_ClusterPassExperiment100.sh</code>
    - <code>sbatch 400_jichd/132_ClusterPassExperiment200.sh</code>
    - <code>sbatch 400_jichd/133_ClusterPassExperiment500.sh</code>
    
    The logs for the GLASSO reduced experiments are saved in <code>100_glasso/experiments1/logs/</code>, indexed from 10 to 4329. The simulation data is saved in RData files in <code>100_glasso/experiments1/data/</code> 1-432. Similar log and data directories for the HWGL, IPC-HD and JIC-HD can be found in the folders 200, 300, 400, respectively. 

4. **Generating Simulation Experiment Plots:** Once all preliminary simulation experiments are complete and the data is saved, you can generate simulation plots. For this, in the command line terminal enter the directory <code>./simulations/</code>, and run

    - <code>Rscript 511_Outcomes_DataAggregationExperiments.R</code> 
    
    This saves aggregated data in the directory <code>500_AggregatedDataExperiments/data_all</code>. Plots that verify performance in terms of true positive rate (TPR) and false positive rate (FPR) and computational running time. The results are saved in the directory <code>500_AggregatedDataExperiments/plots_all/</code> by running
    
    - <code>Rscript 521_MainPlots_Experiment.R 2 1 </code>
    - <code>Rscript 521_MainPlots_Experiment.R 5 1 </code>
    
    Here, we are passing the script <code>521_MainPlots_Experiment.R</code> with two diagonal shift parameter 2 or 5, and we pass 1 to indicate that no variable screening is needed in the model.

    Plots that verify performance in terms of computational time are saved in the directory <code>500_AggregatedDataExperiments/time_all/</code> by running the script 
    
    - <code>Rscript 531_TimePlots_Experiment.R 2 1</code>
    - <code>Rscript 531_TimePlots_Experiment.R 5 1</code>
    
    Finally, plots that compare the degrees of hubs and non-hubs by method are saved in the directory <code>500_AggregatedDataExperiments/bymethod_all/</code> generated by running,

    - <code>Rscript 514_Outcomes_DegreeByMethodPlotsExperiments.R</code> TODO: create file. 


5. **Running Full Simulations:** You can also run full numerical simulations. For each of the 432 simulation scenarios, we perform 50 simulation replicates. To do this, we request 10 cluster nodes, and ask each to perform 10 simulation replicates per simulation scenarios. To run this, in a cluster terminal, enter the directory <code>./simulations/</code>, and run the lines,
    
    - <code>sbatch 100_glasso/141_ClusterPassFull100.sh</code>
    - <code>sbatch 100_glasso/142_ClusterPassFull200.sh</code>
    - <code>sbatch 100_glasso/143_ClusterPassFull500.sh</code>
    - <code>sbatch 200_hwgl/141_ClusterPassFull100.sh</code>
    - <code>sbatch 200_hwgl/142_ClusterPassFull200.sh</code>
    - <code>sbatch 200_hwgl/143_ClusterPassFull500.sh</code>
    - <code>sbatch 300_ipchd/141_ClusterPassFull100.sh</code>
    - <code>sbatch 300_ipchd/142_ClusterPassFull200.sh</code>
    - <code>sbatch 300_ipchd/143_ClusterPassFull500.sh</code>
    - <code>sbatch 400_jichd/141_ClusterPassFull100.sh</code>
    - <code>sbatch 400_jichd/142_ClusterPassFull200.sh</code>
    - <code>sbatch 400_jichd/143_ClusterPassFull500.sh</code>
    
    The logs for the GLASSO full simulations are saved in <code>100_glasso/outputs1/logs/</code>, indexed from 10 to 4329. The simulation data is saved in RData files in <code>100_glasso/outputs1/data/</code> 1-432. Similar log and data directories for the HWGL, IPC-HD and JIC-HD can be found in the folders 200, 300, 400, respectively. 

6. **Generating Simulation Experiment Plots:** Once all full simulations are complete, you can generate simulation plots. For this, in the command line terminal enter the directory <code>./simulations/</code>, and run
    
    - <code>Rscript 611_Outcomes_DataAggregationFull.R</code>
    
    This saves aggregated data in the directory <code>600_AggregatedDataFull/data_all</code>. Plots that verify performance in terms of true positive rate (TPR) and false positive rate (FPR) of hub detection are generated by running
    
    - <code>Rscript 621_MainPlots_Full.R 2 1 </code>
    - <code>Rscript 621_MainPlots_Full.R 5 1 </code>

    Here, we are passing the script <code>621_MainPlots_Full.R</code> with two diagonal shift parameter 2 or 5, and we pass 1 to indicate that no variable screening is needed in the model.
    
    Plots are saved in the directory <code>600_AggregatedDataFull/plots_all</code>. Plots that verify performance in terms of computational time are generated by running the lines:
    
    - <code>Rscript 631_TimePlots_Full.R 2 1</code>
    - <code>Rscript 631_TimePlots_Full.R 5 1</code>
    
    Plots are saved in the directory <code>600_AggregatedDataFull/time_all</code>. Finally, plots that compare the degrees of hubs and non-hubs by method are generated by running,

    - <code>Rscript 614_Outcomes_DegreeByMethodPlotsFull.R</code>  TODO: create file, correct file name.
