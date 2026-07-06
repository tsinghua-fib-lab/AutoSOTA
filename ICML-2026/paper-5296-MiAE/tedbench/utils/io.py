import json
import os
import gzip
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


def download_url(
    url: str, path: str, save_file: str | None = None, md5: str | None = None
):
    """Download a file from the specified URL.

    Skips downloading if a file with matching MD5 already exists.

    Args:
        url: URL to download.
        path: Directory to store the downloaded file.
        save_file: Local filename.  Inferred from the URL when not given.
        md5: Expected MD5 checksum.  If ``None``, the file is always downloaded.
    """

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[: save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        print("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file


def compute_md5(file_name: str, chunk_size: int = 65536) -> str:
    """
    Compute MD5 of the file.

    Parameters:
        file_name (str): file name
        chunk_size (int, optional): chunk size for reading large files
    """
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def extract(zip_file: str, member: str | None = None) -> str:
    """
    Extract files from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``, ``tar`` file types are supported.

    Parameters:
        zip_file (str): file name
        member (str, optional): extract specific member from the zip file.
            If not specified, extract all members.
    """
    import gzip
    import shutil
    import zipfile
    import tarfile

    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]
    save_path = os.path.dirname(zip_file)

    if extension == ".gz":
        member = os.path.basename(zip_name)
        members = [member]
        save_files = [os.path.join(save_path, member)]
        for _member, save_file in zip(members, save_files):
            with gzip.open(zip_file, "rb") as fin:
                if not os.path.exists(save_file):
                    print("Extracting %s to %s" % (zip_file, save_file))
                    with open(save_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        tar = tarfile.open(zip_file, "r")
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            print("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = tar.getnames()
            save_files = [os.path.join(save_path, _member) for _member in members]
            print("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if tar.getmember(_member).isdir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or tar.getmember(
                _member
            ).size != os.path.getsize(save_file):
                with tar.extractfile(_member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        zipped = zipfile.ZipFile(zip_file)
        if member is not None:
            members = [member]
            save_files = [os.path.join(save_path, os.path.basename(member))]
            print("Extracting %s from %s to %s" % (member, zip_file, save_files[0]))
        else:
            members = zipped.namelist()
            save_files = [os.path.join(save_path, _member) for _member in members]
            print("Extracting %s to %s" % (zip_file, save_path))
        for _member, save_file in zip(members, save_files):
            if zipped.getinfo(_member).is_dir():
                os.makedirs(save_file, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            if not os.path.exists(save_file) or zipped.getinfo(
                _member
            ).file_size != os.path.getsize(save_file):
                with zipped.open(_member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("Unknown file extension `%s`" % extension)

    if len(save_files) == 1:
        return save_files[0]
    else:
        return save_path


def extract_tar(
    tar_path: "str | Path",
    out_path: "str | Path",
    extract_members: bool = False,
    strip: int = 0,
) -> None:
    """Extracts a tar file.

    Parameters
    ----------
    tar_path:
        The path to the tar file.
    out_path:
        The directory to extract to.
    extract_members: bool, default False
        If `True`, the tar file member will be directly extracted to `out_path`, instead of creating a subdirectory.
    strip: int, default 0
        Remove `strip` folder hierarchies from the path of the extracted file.
    """

    def get_members(file):
        for member in file.getmembers():
            parts = Path(member.path).parts
            member.path = Path(*parts[min(strip, len(parts) - 1) :])
            yield member

    out_path = Path(out_path)
    with tarfile.open(tar_path, "r") as file:
        members = get_members(file)
        if extract_members:
            for member in members:
                file.extract(member, out_path)
        else:
            file.extractall(out_path, members=file)


def download_and_extract(
    url: str,
    root: "str | Path",
    archive_name: "str | None" = None,
    strip: int = 1,
):
    """Download a ``.tar.gz`` archive from *url* and extract it into *root*.

    The downloaded archive is deleted after successful extraction.

    Args:
        url: Direct download URL for the ``.tar.gz`` file.
        root: Destination directory.  Created if it does not exist.
        archive_name: Local filename for the downloaded archive.  Inferred from
            the URL when not given.
        strip: Number of leading path components to strip from archive members
            before extracting (default ``1`` removes the top-level directory so
            all files land directly in *root*).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if archive_name is None:
        name = url.split("/")[-1].split("?")[0]
        archive_name = name if name else "archive.tar.gz"
        if not any(archive_name.endswith(s) for s in (".tar.gz", ".tgz", ".tar")):
            archive_name += ".tar.gz"

    archive_path = root / archive_name
    download_url(url, str(root), save_file=archive_name)
    print(f"Extracting {archive_path} → {root}")
    extract_tar(archive_path, root, extract_members=True, strip=strip)
    archive_path.unlink()


def load_from_hf(model_path: str | Path):
    """Load a MiAE or MiAEClassifier from a local HF directory or a remote HF repo.

    The source must contain ``config.json`` (with a ``_model_class`` key) and
    ``pytorch_model.bin``, as produced by ``convert_checkpoint.py --hf-dir``.

    Args:
        model_path: Either a local directory path or a HuggingFace repo ID
            (e.g. ``"username/miae-b-tedbench"``).  A local path is used when
            ``Path(model_path).is_dir()``; otherwise the repo is downloaded via
            :func:`huggingface_hub.snapshot_download`.

    Returns:
        The loaded model in eval mode (``MiAE`` or ``MiAEClassifier``).
    """
    import torch
    from omegaconf import OmegaConf

    local_dir = Path(model_path)
    if not local_dir.is_dir():
        from huggingface_hub import snapshot_download
        local_dir = Path(snapshot_download(repo_id=str(model_path)))

    with open(local_dir / "config.json") as f:
        cfg_dict = json.load(f)

    model_class = cfg_dict.pop("_model_class", None)
    if model_class not in ("miae", "miae_classifier"):
        raise ValueError(
            f"config.json has _model_class={model_class!r}; "
            "expected 'miae' or 'miae_classifier'."
        )

    cfg = OmegaConf.create(cfg_dict)

    if model_class == "miae_classifier":
        from tedbench.model import MiAEClassifier
        model = MiAEClassifier(cfg)
    else:
        from tedbench.model import MiAE
        model = MiAE(cfg)

    state_dict = torch.load(
        local_dir / "pytorch_model.bin", map_location="cpu", weights_only=False
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def unzip_file(path: Path, remove: bool = True) -> Path:
    """Unzips a .gz file.

    Parameters
    ----------
    path:
        The path to the .gz file.

    """
    with gzip.open(path, "rb") as f_in:
        with open(path.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    if remove:
        os.remove(path)
    return path.with_suffix("")
