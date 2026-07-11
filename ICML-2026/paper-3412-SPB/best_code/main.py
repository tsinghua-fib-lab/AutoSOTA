import run_dataset
import run_training
import run_plot

if __name__ == "__main__":
    run_dataset.run_all()
    run_training.run_all()
    run_plot.run_all()
    print("All tasks complete.")