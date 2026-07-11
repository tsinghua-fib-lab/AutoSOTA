import tarfile
import os
import tempfile
import shutil

def retag_tar_files(tar_dir, old_ext=".npy", new_ext=".input.npy"):
    for fname in os.listdir(tar_dir):
        if not fname.endswith(".tar.gz"):
            continue
        full_path = os.path.join(tar_dir, fname)
        print(f"Retagging {fname}")

        # Create a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(full_path, "r:gz") as tar:
                tar.extractall(path=tmpdir)

            # Rename files
            for item in os.listdir(tmpdir):
                if item.endswith(old_ext):
                    old_path = os.path.join(tmpdir, item)
                    new_path = os.path.join(tmpdir, item.replace(old_ext, new_ext))
                    os.rename(old_path, new_path)

            # Repack tar
            tmp_tar = full_path + ".tmp"
            with tarfile.open(tmp_tar, "w:gz") as new_tar:
                for f in os.listdir(tmpdir):
                    new_tar.add(os.path.join(tmpdir, f), arcname=f)

            shutil.move(tmp_tar, full_path)

# Example usage:
retag_tar_files("/media/data2/gamba_embed/enhancer_annotation/gamba/")
