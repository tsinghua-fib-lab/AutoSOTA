import sys
import math

def compute_mean_and_std(num1, num2, num3):
    # Ensure the numbers are positive
    if num1 <= 0 or num2 <= 0 or num3 <= 0:
        raise ValueError("All numbers must be positive.")
    
    # Compute the mean
    mean = (num1 + num2 + num3) / 3
    
    # Compute the standard deviation
    variance = ((num1 - mean) ** 2 + (num2 - mean) ** 2 + (num3 - mean) ** 2) / 3
    std_dev = math.sqrt(variance)
    
    return mean, std_dev

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <num1> <num2> <num3>")
        sys.exit(1)
    
    try:
        # Parse command-line arguments as floats
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[2])
        num3 = float(sys.argv[3])
        
        mean, std_dev = compute_mean_and_std(num1, num2, num3)
        print(f"Mean: {mean:.2f}, Standard Deviation: {std_dev:.2f}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
