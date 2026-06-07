#!/usr/bin/env python3
"""
Binary voxel optimization runner.
Offers training, conversion to meshes, and a tuned Taubin smoothing workflow
that removes marching-cubes staircase artifacts.

Usage:
    python run.py --train                    # train with default args
    python run.py --train --convert          # train then convert to mesh
    python run.py --convert                  # convert the latest model only
    python run.py --demo                     # run a quick demonstration

Supervised training:
    python run.py --train --supervision_image1 path1.png --supervision_image2 path2.png

Surface smoothing controls:
    python run.py --convert                           # all smoothing enabled
    python run.py --convert --no_enhanced_smoothing   # Taubin only
    python run.py --convert --no_volume_preprocessing # skip voxel preprocessing
    python run.py --convert --no_smoothing            # disable smoothing
    python run.py --convert --smooth_iterations 20    # custom iteration count
"""

import os
import sys
import glob
import argparse
import torch
import subprocess
import json  

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_latest_model_path(base_dir="results"):
    """Return the most recent model checkpoint in the results tree."""
    pattern = os.path.join(base_dir, "*", "models", "binary_voxel_model_*.pt")
    model_files = glob.glob(pattern)
    if not model_files:
        return None
    return max(model_files, key=os.path.getmtime)

def list_gpus():
    """List visible CUDA devices."""
    if not torch.cuda.is_available():
        print("CUDA is unavailable, falling back to CPU.")
        return []
    
    gpu_count = torch.cuda.device_count()
    print(f"Detected GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({memory_gb:.1f}GB)")
    
    return list(range(gpu_count))

def train_model(args):
    """Train the binary voxel optimizer."""
    print("Starting binary voxel training...")
    

    if args.supervision_image1 and args.supervision_image2:
        if not os.path.exists(args.supervision_image1):
            print(f"Error: supervision image 1 not found: {args.supervision_image1}")
            return False
        if not os.path.exists(args.supervision_image2):
            print(f"Error: supervision image 2 not found: {args.supervision_image2}")
            return False
        print("Supervised mode:")
        print(f"  - View 1 (0 deg): {args.supervision_image1}")
        print(f"  - View 2 (180 deg): {args.supervision_image2}")
    else:
        print("Unsupervised mode (centered cube initialization).")
    

    try:
        from main import binary_voxel_train
        
        # Select device
        device = 'cpu'
        if args.gpu is not None and torch.cuda.is_available():
            device = f'cuda:{args.gpu}'
        elif torch.cuda.is_available():
            device = 'cuda'
        
        print(f"Using device: {device}")
        
        # Resolve render size overrides
        render_size_arg = None
        if getattr(args, 'render_width', None) is not None and getattr(args, 'render_height', None) is not None:
            render_size_arg = (args.render_height, args.render_width)


        model, output_path = binary_voxel_train(
            supervision_image1=args.supervision_image1,
            supervision_image2=args.supervision_image2,
            volume_size=args.volume_size,
            n_iter=args.n_iter,
            lr=args.lr,
            device=device,
            output_dir=args.output_dir,
            render_scale=args.render_scale,
            render_size=render_size_arg,
            n_pts_per_ray=args.pts_per_ray,
            azim1=args.azim1,
            azim2=args.azim2,

            elev1=args.elev1,
            elev2=args.elev2,
            orthographic=args.orthographic,

            decouple_training=getattr(args, 'decouple_training', True),
            shape_ratio=getattr(args, 'shape_ratio', 0.6),
            freeze_density_mapping=getattr(args, 'freeze_density_mapping', True),
            disable_pruning_after_boundary=getattr(args, 'disable_pruning_after_boundary', True)
        )
        
        print(f"Training finished; artifacts saved to: {output_path}")
        

        print("Attempting automatic mesh conversion...")
        latest_model = get_latest_model_path(args.output_dir)
        if latest_model:

            convert_args = argparse.Namespace()
            convert_args.model_path = latest_model
            convert_args.threshold = getattr(args, 'threshold', 0.5)
            convert_args.voxel_size = getattr(args, 'voxel_size', 0.1)
            convert_args.gpu = args.gpu
            convert_args.output_dir = args.output_dir


            convert_args.no_smoothing = getattr(args, 'no_smoothing', False)
            convert_args.smooth_iterations = getattr(args, 'smooth_iterations', 15)  # higher default for smoother meshes
            convert_args.smooth_lambda = getattr(args, 'smooth_lambda', 0.5)
            convert_args.smooth_nu = getattr(args, 'smooth_nu', 0.53)

            convert_args.no_enhanced_smoothing = getattr(args, 'no_enhanced_smoothing', False)
            convert_args.no_volume_preprocessing = getattr(args, 'no_volume_preprocessing', False)
            
            # Perform conversion
            mesh_success = convert_to_mesh(convert_args)
            if mesh_success:
                print("Model converted to mesh.")
            else:
                print("Mesh conversion failed, but training succeeded.")
        else:
            print("Warning: trained model not found; skipping conversion.")
        
        return True
        
    except ImportError:
        print("Falling back to subprocess-based training...")
        training_success = train_with_subprocess(args)
        

        if training_success:
            print("Attempting automatic mesh conversion...")
            latest_model = get_latest_model_path(args.output_dir)
            if latest_model:
                convert_args = argparse.Namespace()
                convert_args.model_path = latest_model
                convert_args.threshold = getattr(args, 'threshold', 0.5)
                convert_args.voxel_size = getattr(args, 'voxel_size', 0.1)
                convert_args.gpu = args.gpu
                convert_args.output_dir = args.output_dir

                convert_args.no_smoothing = getattr(args, 'no_smoothing', False)
                convert_args.smooth_iterations = getattr(args, 'smooth_iterations', 15)  # higher default for smoother meshes
                convert_args.smooth_lambda = getattr(args, 'smooth_lambda', 0.5)
                convert_args.smooth_nu = getattr(args, 'smooth_nu', 0.53)

                convert_args.no_enhanced_smoothing = getattr(args, 'no_enhanced_smoothing', False)
                convert_args.no_volume_preprocessing = getattr(args, 'no_volume_preprocessing', False)
                
                mesh_success = convert_to_mesh(convert_args)
                if mesh_success:
                    print("Model converted to mesh.")
                else:
                    print("Mesh conversion failed, but training succeeded.")
            else:
                print("Warning: trained model not found; skipping conversion.")
        
        return training_success
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def train_with_subprocess(args):
    """Train by shelling out to main.py."""
    cmd = ["python", "main.py"]
    

    if args.supervision_image1:
        cmd.extend(["--supervision_image1", args.supervision_image1])
    if args.supervision_image2:
        cmd.extend(["--supervision_image2", args.supervision_image2])
    
    cmd.extend([
        "--volume_size", str(args.volume_size),
        "--n_iter", str(args.n_iter),
        "--lr", str(args.lr),
        "--render_scale", str(args.render_scale),
        "--pts_per_ray", str(args.pts_per_ray),
        "--output_dir", args.output_dir
    ])

    cmd.extend(["--azim1", str(args.azim1), "--azim2", str(args.azim2)])

    cmd.extend(["--elev1", str(args.elev1), "--elev2", str(args.elev2)])
    if args.orthographic:
        cmd.append("--orthographic")


    cmd.extend(["--shape_ratio", str(getattr(args, 'shape_ratio', 0.6))])

    if getattr(args, 'decouple_training', True) is False:
        cmd.append("--no_decouple_training")

    if getattr(args, 'freeze_density_mapping', True) is False:
        cmd.append("--no_freeze_density_mapping")

    if getattr(args, 'disable_pruning_after_boundary', True) is False:
        cmd.append("--enable_pruning_after_boundary")
    
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Training finished.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}.")
        return False

def convert_to_mesh(args):
    """Convert a trained volume into a mesh (quiet log)."""

    model_path = args.model_path or get_latest_model_path(args.output_dir)
    if not model_path or not os.path.exists(model_path):
        print("Error: model checkpoint not found.")
        return False


    model_name = os.path.splitext(os.path.basename(model_path))[0]
    mesh_dir = os.path.join("meshes", model_name)
    os.makedirs(mesh_dir, exist_ok=True)
    output_path = os.path.join(mesh_dir, f"{model_name}.obj")


    cmd = [
        "python", "volume_to_mesh.py",
        "--volume_path", model_path,
        "--output_path", output_path,
        "--threshold", str(args.threshold),
        "--voxel_size", str(args.voxel_size)
    ]

    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])

    if getattr(args, 'no_smoothing', False):
        cmd.append("--no_smoothing")
    if hasattr(args, 'smooth_iterations') and args.smooth_iterations != 15:
        cmd.extend(["--smooth_iterations", str(args.smooth_iterations)])
    if hasattr(args, 'smooth_lambda') and args.smooth_lambda != 0.5:
        cmd.extend(["--smooth_lambda", str(args.smooth_lambda)])
    if hasattr(args, 'smooth_nu') and args.smooth_nu != 0.53:
        cmd.extend(["--smooth_nu", str(args.smooth_nu)])
    if getattr(args, 'no_enhanced_smoothing', False):
        cmd.append("--no_enhanced_smoothing")
    if getattr(args, 'no_volume_preprocessing', False):
        cmd.append("--no_volume_preprocessing")

    # Primary conversion pass
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Mesh saved to: {output_path}")
        return True
    except subprocess.CalledProcessError:
        print("Mesh conversion failed; attempting inspection + auto-fix...")
        try:
            from post_inspect import inspect_and_fix
            fixed_model_path, report_path = inspect_and_fix(
                model_path,
                output_dir=os.path.dirname(model_path),
                threshold=getattr(args, 'threshold', 0.5),
                supervision_image1=getattr(args, 'supervision_image1', None),
                supervision_image2=getattr(args, 'supervision_image2', None),
                device_str='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"Inspection completed; report at {report_path}. Retrying with the fixed volume...")
            cmd2 = [
                "python", "volume_to_mesh.py",
                "--volume_path", fixed_model_path,
                "--output_path", output_path,
                "--threshold", str(getattr(args, 'threshold', 0.5)),
                "--voxel_size", str(getattr(args, 'voxel_size', 0.1))
            ]
            if args.gpu is not None:
                cmd2.extend(["--gpu", str(args.gpu)])
            if getattr(args, 'no_smoothing', False):
                cmd2.append("--no_smoothing")
            if hasattr(args, 'smooth_iterations') and args.smooth_iterations != 15:
                cmd2.extend(["--smooth_iterations", str(args.smooth_iterations)])
            if hasattr(args, 'smooth_lambda') and args.smooth_lambda != 0.5:
                cmd2.extend(["--smooth_lambda", str(args.smooth_lambda)])
            if hasattr(args, 'smooth_nu') and args.smooth_nu != 0.53:
                cmd2.extend(["--smooth_nu", str(args.smooth_nu)])
            if getattr(args, 'no_enhanced_smoothing', False):
                cmd2.append("--no_enhanced_smoothing")
            if getattr(args, 'no_volume_preprocessing', False):
                cmd2.append("--no_volume_preprocessing")

            subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Mesh saved after repair: {output_path}")
            return True
        except Exception as e:
            print(f"Inspection/repair step failed: {e}")
            print("Mesh conversion failed.")
            return False

def run_demo(args):
    """Run a lightweight demonstration training loop."""
    print("Running the binary voxel demo...")
    
    # Demo arguments
    demo_args = argparse.Namespace()
    demo_args.supervision_image1 = None
    demo_args.supervision_image2 = None
    demo_args.volume_size = 64  
    demo_args.n_iter = 200      
    demo_args.lr = 0.05
    demo_args.output_dir = "demo_results"
    demo_args.gpu = args.gpu
    
    success = train_model(demo_args)
    if success:
        print("Demo finished.")
    return success

def main():
    parser = argparse.ArgumentParser(description="Binary voxel training and mesh conversion helper.")

    parser.add_argument('--train', action='store_true', help='Train a model.')
    parser.add_argument('--convert', action='store_true', help='Convert a checkpoint into a mesh.')
    parser.add_argument('--demo', action='store_true', help='Run the demo preset.')
    parser.add_argument('--list_gpus', action='store_true', help='Print visible CUDA devices.')

    parser.add_argument('--config_json', type=str, help='JSON file whose keys override CLI arguments.')

    parser.add_argument('--supervision_image1', type=str, help='Supervision image for view 1 (0 deg).')
    parser.add_argument('--supervision_image2', type=str, help='Supervision image for view 2 (180 deg).')
    parser.add_argument('--volume_size', type=int, default=128, help='Grid resolution, must be 128 or 256.')
    parser.add_argument('--n_iter', type=int, default=1000, help='Training iterations.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')

    parser.add_argument('--render_scale', type=float, default=1.0,
                        help='Render resolution multiplier (render_size = volume_size * render_scale).')
    parser.add_argument('--pts_per_ray', type=int, default=150,
                        help='Sample count per ray.')

    parser.add_argument('--render_width', type=int, default=200, help='Render width override in pixels.')
    parser.add_argument('--render_height', type=int, default=200, help='Render height override in pixels.')
    parser.add_argument('--azim1', type=float, default=0.0, help='Azimuth for view 1 (deg).')
    parser.add_argument('--azim2', type=float, default=180.0, help='Azimuth for view 2 (deg).')

    parser.add_argument('--elev1', type=float, default=0.0, help='Elevation for view 1 (deg).')
    parser.add_argument('--elev2', type=float, default=0.0, help='Elevation for view 2 (deg).')

    parser.add_argument('--shape_ratio', type=float, default=0.7,
                        help='Shape phase ratio (0-1).')
    parser.add_argument('--no_orthographic', action='store_false', dest='orthographic',
                        help='Use a perspective camera (default: orthographic).')

    parser.set_defaults(orthographic=True)
    parser.add_argument('--output_dir', type=str, default='results', help='Root directory for artifacts.')

    parser.add_argument('--model_path', type=str, help='Explicit checkpoint path for conversion.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Marching-cubes density threshold.')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Voxel spacing in world units.')

    parser.add_argument('--no_smoothing', action='store_true', help='Disable all surface smoothing.')
    parser.add_argument('--smooth_iterations', type=int, default=15, help='Taubin smoothing iterations.')
    parser.add_argument('--smooth_lambda', type=float, default=0.5, help='Taubin shrinkage parameter lambda.')
    parser.add_argument('--smooth_nu', type=float, default=0.53, help='Taubin expansion parameter nu.')

    parser.add_argument('--no_enhanced_smoothing', action='store_true', help='Skip enhanced multi-stage smoothing.')
    parser.add_argument('--no_volume_preprocessing', action='store_true', help='Skip voxel pre-filtering.')

    parser.add_argument('--gpu', type=int, help='CUDA device id.')
    
    args = parser.parse_args()

    if getattr(args, 'config_json', None):
        cfg_path = args.config_json
        if not os.path.exists(cfg_path):
            print(f"Error: configuration file not found: {cfg_path}")
            sys.exit(1)
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    print(f"Warning: ignoring unknown config key '{k}'.")
        except Exception as e:
            print(f"Failed to parse config file: {e}")
            sys.exit(1)
    

    if args.list_gpus:
        list_gpus()
        return

    if args.gpu is not None:
        if not torch.cuda.is_available():
            print("CUDA is unavailable; ignoring --gpu.")
            args.gpu = None
        elif args.gpu >= torch.cuda.device_count():
            print(f"GPU {args.gpu} does not exist; defaulting to GPU 0.")
            args.gpu = 0

    if args.volume_size not in (128, 256):
        print("Error: only volume_size=128 or 256 is supported.")
        sys.exit(1)

    if not any([args.train, args.convert, args.demo]):
        args.train = True
        args.convert = True
    

    success = True
    
    if args.demo:
        success &= run_demo(args)
    
    if args.train:
        success &= train_model(args)
    
    if args.convert:
        success &= convert_to_mesh(args)
    
    if success:
        print("All requested steps finished.")
    else:
        print("Some steps failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 