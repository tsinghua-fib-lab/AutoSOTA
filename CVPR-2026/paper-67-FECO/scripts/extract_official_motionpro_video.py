import os
import subprocess

input_root = "data/MotionPro/data"

for root, dirs, files in os.walk(input_root, topdown=True):
    dirs.sort()
    files.sort()

    for file in files:
        if 'data/MotionPro/data/0807' not in root:
            continue

        if file.lower().endswith(".mp4"):
            rel_path = os.path.relpath(root, input_root)
            parts = rel_path.split(os.sep)

            if len(parts) >= 3:
                video_path = os.path.join(root, file)
                filename_wo_ext = os.path.splitext(file)[0]
                outdir = os.path.join(root, filename_wo_ext)

                # Skip if already extracted
                if os.path.isdir(outdir) and any(f.lower().endswith(".jpg") for f in os.listdir(outdir)):
                    print(f"Skipping {video_path}, frames already extracted.")
                    continue

                os.makedirs(outdir, exist_ok=True)
                print(f"Extracting frames from {video_path} to {outdir}")

                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    os.path.join(outdir, "%06d.jpg")
                ]
                subprocess.run(cmd, check=True)
