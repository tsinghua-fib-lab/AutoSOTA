import subprocess
import json
import shutil
import sys
from pathlib import Path
import datetime as dt


class Neo4jDockerSnapshotManager:
    """
    Snapshot-only backup & restore for Neo4j running in Docker.
    - Works on Windows 11 + Docker Desktop (Community).
    - Supports both bind mount and named volume for /data.
    - Backup:
        * bind mount  -> ZIP (shutil.make_archive)
        * named volume-> TGZ via a temporary 'alpine' container
    - Restore mirrors backup format and requires container stop.
    """

    def __init__(self, container_name: str):
        self.container = container_name

    def backup_snapshot(self, out_dir: str) -> Path:
        """
        Create a consistent snapshot backup of /data.
        Returns the path to the created archive (zip or tgz).
        """
        mounts = self._inspect_mounts()
        data_mount = self._get_data_mount(mounts)

        self._docker("stop", self.container)
        try:
            out_path = Path(out_dir).resolve()
            out_path.mkdir(parents=True, exist_ok=True)
            ts = self._now()
            if self._is_bind_mount(data_mount):
                # Directly ZIP the host directory
                src_dir = Path(data_mount["Source"])
                archive = out_path / f"{self.container}-data-snapshot-{ts}.zip"
                self._zip_dir(src_dir, archive)
                print(f"✅ Snapshot (bind) done: {archive}")
                return archive
            else:
                # Named volume -> use alpine to tar to TGZ on host
                vol_name = data_mount["Name"]
                archive = out_path / f"{self.container}-data-snapshot-{ts}.tgz"
                self._tar_volume_to_host(vol_name, out_path, archive.name)
                print(f"✅ Snapshot (volume) done: {archive}")
                return archive
        finally:
            self._docker("start", self.container)

    def restore_snapshot(self, archive_path: str):
        """
        Restore /data content from a given snapshot archive created by backup_snapshot().
        - .zip  -> bind mount restore
        - .tgz  -> named volume restore
        WARNING: This will erase current /data content.
        """
        mounts = self._inspect_mounts()
        data_mount = self._get_data_mount(mounts)
        archive = Path(archive_path).resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")

        # detect format
        is_zip = archive.suffix.lower() == ".zip"
        is_tgz = archive.suffix.lower() in (".tgz", ".tar.gz")

        if self._is_bind_mount(data_mount) and not is_zip:
            raise ValueError("Bind-mount restore expects a .zip archive.")
        if (not self._is_bind_mount(data_mount)) and not is_tgz:
            raise ValueError("Named-volume restore expects a .tgz (.tar.gz) archive.")

        self._docker("stop", self.container)
        try:
            if self._is_bind_mount(data_mount):
                dst = Path(data_mount["Source"])
                self._wipe_directory(dst)
                self._unzip_to_dir(archive, dst)
                print(f"✅ Restore (bind) done -> {dst}")
            else:
                vol_name = data_mount["Name"]
                self._wipe_volume(vol_name)
                self._untar_host_to_volume(vol_name, archive)
                print(f"✅ Restore (volume) done -> volume:{vol_name}")
        finally:
            self._docker("start", self.container)

    def _run(self, cmd: str, check=True):
        print(f"[RUN] {cmd}")
        return subprocess.run(cmd, shell=True, check=check,
                              capture_output=True, text=True)

    def _docker(self, subcmd: str, *args: str):
        return self._run(" ".join(["docker", subcmd, *args]))

    def _inspect_mounts(self):
        r = self._run(f'docker inspect {self.container} --format "{{{{json .Mounts}}}}"')
        try:
            mounts = json.loads(r.stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse docker inspect output: {e}\n{r.stdout}")
        return mounts

    @staticmethod
    def _get_data_mount(mounts):
        for m in mounts:
            if m.get("Destination") == "/data":
                return m
        raise RuntimeError("No /data mount found in container. Is Neo4j data persisted?")

    @staticmethod
    def _is_bind_mount(mount) -> bool:
        return mount.get("Type") == "bind"

    @staticmethod
    def _now():
        return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- bind mount ops

    @staticmethod
    def _zip_dir(src_dir: Path, zip_path: Path):
        base = zip_path.with_suffix("")  # shutil.make_archive auto-append .zip
        # Ensure temp base not occupied
        if base.exists():
            if base.is_dir():
                shutil.rmtree(base)
            else:
                base.unlink()
        # Create zip
        shutil.make_archive(str(base), "zip", root_dir=str(src_dir))

    @staticmethod
    def _unzip_to_dir(zip_path: Path, dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), extract_dir=str(dst_dir), format="zip")

    @staticmethod
    def _wipe_directory(dst_dir: Path):
        if dst_dir.exists():
            for child in dst_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)

    # ---- named volume ops (use temporary alpine)

    def _tar_volume_to_host(self, volume_name: str, host_out_dir: Path, out_name: str):
        cmd = (
            'docker run --rm '
            f'-v {volume_name}:/source '
            f'-v "{host_out_dir}":/backup '
            'alpine sh -lc '
            f'"cd /source && tar czf /backup/{out_name} ."'
        )
        self._run(cmd)

    def _wipe_volume(self, volume_name: str):
        cmd = (
            'docker run --rm '
            f'-v {volume_name}:/target '
            'alpine sh -lc "rm -rf /target/*"'
        )
        self._run(cmd)

    def _untar_host_to_volume(self, volume_name: str, archive_path: Path):
        # Map archive parent as /backup, then untar into /target
        cmd = (
            'docker run --rm '
            f'-v {volume_name}:/target '
            f'-v "{archive_path.parent}":/backup '
            f'alpine sh -lc "cd /target && tar xzf /backup/{archive_path.name}"'
        )
        self._run(cmd)


def backup_or_restore(
        archive_file_name: str = "",
        action: str = "backup"
):
    """
    backup / restore Neo4j graph database
    """

    CONTAINER_NAME = "neo4j-apoc"
    ACTION = action
    OUT_DIR = r"..\graphdb_backups"
    # ARCHIVE_TO_RESTORE = r"..\graphdb_backups\neo4j-data-snapshot-YYYYMMDD-HHMMSS.zip"
    ARCHIVE_TO_RESTORE = archive_file_name

    if shutil.which("docker") is None:
        print("The docker executable file was not found. Please confirm that Docker Desktop is installed and the docker command is available.")
        sys.exit(1)

    mgr = Neo4jDockerSnapshotManager(container_name=CONTAINER_NAME)

    if ACTION.lower() == "backup":
        archive = mgr.backup_snapshot(OUT_DIR)
        print(f"\n[OK] Backup created: {archive}")
    elif ACTION.lower() == "restore":
        mgr.restore_snapshot(ARCHIVE_TO_RESTORE)
        print("\n[OK] Restore finished.")
    else:
        print("ACTION is 'backup' or 'restore'。")


if __name__ == "__main__":
    backup_or_restore("", "backup")
    # backup_or_restore(r"..\graphdb_backups\neo4j-apoc-data-snapshot-???.zip", "restore")
