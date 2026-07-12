"""
MRI Meta-Learning Training Script
Based on Reptile algorithm and SIREN network
Supports multi-threaded data loading, intelligent caching, and adaptive learning rate
"""

from SIREN_IPOD_utils import *

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training function"""
    # Set random seed
    set_seed(35236)
    
    # Configuration parameters
    data_dir = ''
    save_dir = ''
    
    # TCNN encoding and network configuration
    encoding_config = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 12,
        "per_level_scale": 2,
        "interpolation": "Linear"
    }
    
    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 16,
        "n_hidden_layers": 2
    }
    
    # Training parameters
    base_lr = 5e-4
    inner_lr = 2e-4
    inner_steps = 300
    epochs = 3000
    tasks_per_epoch = 15
    samples_per_task = 1
    meta_lr = base_lr * tasks_per_epoch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create optimized dataset
    dataset = OptimizedMRIMetaDataset(
        data_dir,
        cache_size=999,
        preload_samples=True,
        num_workers=12,
        memory_limit_gb=5
    )
    
    # Create Reptile trainer
    trainer = ReptileTrainer(
        encoding_config=encoding_config,
        network_config=network_config,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        inner_steps=inner_steps,
        samples_per_task=samples_per_task,
        device=device,
        load_index=0, # load checkpoint
    )
    
    # Train model
    training_stats = trainer.train(
        dataset=dataset,
        epochs=epochs,
        save_dir=save_dir,
        eval_interval=20,
        save_visuals=True,
        tasks_per_epoch=tasks_per_epoch,
        samples_per_task=samples_per_task
    )
    
    print("\nReptile Training Completed!")
    
    # Print final performance statistics
    if training_stats['data_loading_times'] and training_stats['training_times']:
        avg_data_time = np.mean(training_stats['data_loading_times'])
        avg_train_time = np.mean(training_stats['training_times'])
        avg_total_time = avg_data_time + avg_train_time
        avg_gpu_util = (avg_train_time / avg_total_time) * 100
        
        print(f"\nFinal Performance Statistics:")
        print(f"Average data loading time: {avg_data_time:.2f}s")
        print(f"Average training time: {avg_train_time:.2f}s")
        print(f"Average GPU utilization: {avg_gpu_util:.1f}%")


if __name__ == "__main__":
    main()