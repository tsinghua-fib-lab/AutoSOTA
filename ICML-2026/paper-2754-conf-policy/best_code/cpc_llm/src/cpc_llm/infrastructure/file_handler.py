import shutil
import s3fs
import os


class LocalOrS3Client:
    def __init__(self, init_s3: bool = False, **s3fs_kwargs):
        self.init_s3 = init_s3
        if init_s3:
            self.fs = s3fs.S3FileSystem(**s3fs_kwargs)

    def exists(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.exists(path, **kwargs)
        else:
            return os.path.exists(path)

    def ls(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.ls(path, **kwargs)
        else:
            return os.listdir(path)

    def get(self, rpath, lpath, **kwargs):
        """Get file(s) from remote to local. Supports both S3 and local paths."""
        if rpath.startswith("s3://"):
            assert lpath and not lpath.startswith("s3://"), "lpath cannot be a S3 path."
            return self.fs.get(rpath, lpath, **kwargs)
        else:
            # Local to local copy
            if os.path.isdir(rpath):
                os.makedirs(lpath, exist_ok=True)
                for item in os.listdir(rpath):
                    src = os.path.join(rpath, item)
                    dst = os.path.join(lpath, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            else:
                os.makedirs(os.path.dirname(lpath), exist_ok=True)
                shutil.copy2(rpath, lpath)

    def put(self, lpath, rpath, **kwargs):
        """Put file(s) from local to remote. Supports both S3 and local paths."""
        if rpath.startswith("s3://"):
            assert lpath and not lpath.startswith("s3://"), "lpath cannot be a S3 path."
            return self.fs.put(lpath, rpath, **kwargs)
        else:
            # Local to local copy
            os.makedirs(
                os.path.dirname(rpath) if os.path.dirname(rpath) else ".", exist_ok=True
            )
            if os.path.isdir(lpath):
                os.makedirs(rpath, exist_ok=True)
                for item in os.listdir(lpath):
                    src = os.path.join(lpath, item)
                    dst = os.path.join(rpath, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            else:
                shutil.copy2(lpath, rpath)

    def open(self, fp, mode="rb", **kwargs):
        if fp.startswith("s3://"):
            return self.fs.open(fp, mode=mode, **kwargs)
        else:
            return open(fp, mode=mode, **kwargs)
