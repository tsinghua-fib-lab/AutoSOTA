import os
import csv
import stat
import sys
import time
import boto3
import requests
import warnings
import os.path as osp
from typing import Optional
from botocore.exceptions import ClientError
from appfl.misc.utils import (
    dump_data_to_file,
    load_data_from_file,
    id_generator,
    secure_appfl_dir,
)


def _open_credentials_file_securely(path: str):
    """Open an AWS credentials file, refusing to read it if the file's
    permissions or ownership would let another local user view it.

    The check is performed on the same file descriptor that is returned to
    the caller (TOCTOU-safe). On Windows the POSIX permission model does
    not apply, so the check is skipped with a warning.

    Returns a text-mode file object positioned at the start of the file.
    Raises PermissionError if the file is unsafe.
    """
    if os.name != "posix":
        warnings.warn(
            f"Cannot enforce AWS credentials file permissions on {os.name}; "
            "ensure the file is readable only by the current user.",
            stacklevel=2,
        )
        return open(path, encoding="utf-8")  # noqa: SIM115

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
    except Exception:
        os.close(fd)
        raise

    mode_bits = stat.S_IMODE(st.st_mode)
    bad_bits = mode_bits & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    if bad_bits:
        os.close(fd)
        raise PermissionError(
            f"AWS credentials file {path!r} has insecure permissions "
            f"{oct(mode_bits)}; refusing to read. Run "
            f"`chmod 600 {path}` (and ensure the file is owned by the current user)."
        )
    if st.st_uid != os.getuid():
        os.close(fd)
        raise PermissionError(
            f"AWS credentials file {path!r} is not owned by the current user "
            f"(uid={os.getuid()}, file uid={st.st_uid}); refusing to read."
        )

    return os.fdopen(fd, "r", encoding="utf-8")


class LargeObjectWrapper:
    # 3 MB maximum size limit for direct upload
    MAX_SIZE_LIMIT = 3 * 1024 * 1024

    def __init__(self, data, name: str):
        self.data = data
        self.name = name

    @property
    def size(self):
        return sys.getsizeof(self.data)

    @property
    def can_send_directly(self):
        return self.size < LargeObjectWrapper.MAX_SIZE_LIMIT


class CloudStorage:
    instc = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def get_instance(cls):
        if cls.instc is not None:
            return cls.instc
        else:
            raise RuntimeError("Please call CloudStorage.init(cfg) first")

    @classmethod
    def init(
        cls,
        s3_bucket: Optional[str] = None,
        s3_creds_file: Optional[str] = None,
        s3_tmp_dir: str = None,
        logger=None,
    ):
        if cls.instc is None:
            new_inst = cls.__new__(cls)
            new_inst.bucket = s3_bucket

            # Set s3_tmp_dir with fallback for read-only containers.
            # secure_appfl_dir handles the home/`/tmp` fallback and enforces
            # 0o700 / current-user ownership on every component.
            if s3_tmp_dir is None:
                s3_tmp_dir = secure_appfl_dir("s3_tmp_dir")
            s3_kwargs = {}
            if s3_creds_file is not None:
                with _open_credentials_file_securely(s3_creds_file) as file:
                    reader = csv.reader(file)
                    keys = next(reader)
                    s3_kwargs = {
                        "region_name": keys[0],
                        "aws_access_key_id": keys[1],
                        "aws_secret_access_key": keys[2],
                    }
            new_inst.client = boto3.client(service_name="s3", **s3_kwargs)
            new_inst.temp_dir = s3_tmp_dir
            new_inst.logger = logger
            new_inst.session_id = id_generator() + str(time.time())
            new_inst.registered_obj = set()
            new_inst.uploaded_obj = {}
            cls.instc = new_inst
        return cls.instc

    @staticmethod
    def is_cloud_storage_object(obj):
        """Check if the object corresponds to an object stored on the cloud (S3 bucket)"""
        if type(obj) is not dict:
            return False
        if "s3" in obj:
            return True
        else:
            return False

    @staticmethod
    def get_cloud_object_info(obj):
        """Obtain the storage information for an object stored on S3 bucket"""
        file_name = obj["s3"]["file_name"]
        object_name = obj["s3"]["object_name"] if "object_name" in obj["s3"] else None
        object_url = obj["s3"]["object_url"] if "object_url" in obj["s3"] else None
        return file_name, object_name, object_url

    def upload_file(
        self,
        file_path: str,
        object_url: Optional[str] = None,
        object_name: Optional[str] = None,
        expiration: int = 3600,
        delete_local: bool = True,
    ) -> dict:
        """
        Upload a local file to S3 directly via object_name or using the presigned url
        Inputs:
            file_path: path to local file
            object_url: presigned url for uploading the file to S3 bucket
            object_name: object key for the file on S3 bucket
            expiration: expiration second for the returned presigned url
            delete_local: whether to delete the local file
        Output:
            s3_obj: an object containing the following information
                - if uploaded directly: file name, presigned download url
                - if uploaded via presigned url: file name, object name
        """
        # Upload the object using presigned url
        if object_url is not None:
            try:
                with open(file_path, "rb") as f:
                    response = requests.put(object_url, data=f)
                if delete_local:
                    os.remove(file_path)
            except:  # noqa E722
                raise Exception("Error in uploading file using presigned url")
            s3_obj = {
                "s3": {"file_name": osp.basename(file_path), "object_name": object_name}
            }
            self.uploaded_obj[object_name] = s3_obj
            return s3_obj
        # Upload the object directly
        try:
            self.client.upload_file(file_path, self.bucket, object_name)
            if delete_local:
                os.remove(file_path)
        except ClientError as e:
            if self.logger is not None:
                self.logger.info(f"Error occurs in uploading file {e}")
            else:
                print(f"Error occurs in uploading file {e}")
            raise Exception(f"Error in uploading file {file_path} to S3")
        try:
            response = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_name},
                ExpiresIn=expiration,
            )
            s3_obj = {
                "s3": {"file_name": osp.basename(file_path), "object_url": response}
            }
            self.uploaded_obj[object_name] = s3_obj
            return s3_obj
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"Error occurs in generating presigned url {e}")
            else:
                print(f"Error occurs in generating presigned url {e}")
            raise

    def download_file(
        self, file_path, object_name=None, object_url=None, delete_cloud=False
    ):
        """
        Download an object from S3 bucket either using the object name (only possible with credentials)
        or using a presigned object url.
        Inputs:
            file_path: path to store the downloaded file
            object_name: key of the object on S3 to download
            object_url: presigned url for downloading the file
            delete_cloud: delete the cloud object after downloading using object name (only possible with credentials)
        """
        REPEAT_TIMES = 5
        if object_url is not None:
            for i in range(REPEAT_TIMES):
                try:
                    response = requests.get(object_url)
                    if response.status_code == 200:
                        with open(file_path, "wb") as file:
                            file.write(response.content)
                        if self.logger is not None:
                            self.logger.info(
                                "Successfully download object using presigned url"
                            )
                        else:
                            print("Successfully download object using presigned url")
                    return
                except:  # noqa E722
                    time.sleep(i + 1)
            if self.logger is not None:
                self.logger.info("Error in downloading object using presigned url")
            else:
                print("Error in downloading object using presigned url")
            raise
        if object_name is not None:
            download_success = False
            for i in range(REPEAT_TIMES):
                try:
                    self.client.download_file(self.bucket, object_name, file_path)
                    download_success = True
                    break
                except:  # noqa E722
                    time.sleep(i + 1)
            if not download_success:
                if self.logger is not None:
                    self.logger.info(
                        f"Error in downloading object {object_name} from S3"
                    )
                else:
                    print(f"Error in downloading object {object_name} from S3")
                raise

            delete_success = False
            for i in range(REPEAT_TIMES):
                try:
                    if delete_cloud:
                        self.client.delete_object(Bucket=self.bucket, Key=object_name)
                    delete_success = True
                    break
                except:  # noqa E722
                    time.sleep(i + 1)
            if not delete_success:
                if self.logger is not None:
                    self.logger.info(f"Error in deleting object {object_name} from S3")
                else:
                    print(f"Error in deleting object {object_name} from S3")
                raise

    @classmethod
    def upload_object(
        cls,
        data,
        object_name=None,
        object_url=None,
        ext="pt",
        temp_dir=None,
        register_for_clean=False,
    ):
        """
        Upload a python object to S3 bucket by first saving the object into a file in the temp directory,
        and then uploading to S3 directly via object name (only with credentials) or via presigned url.
        Inputs:
            cls: the cloud storage class instance
            data: the python object to be uploaded to S3 bucket
            object_name: key of the object to be uploaded to S3 bucket
            object_url: presigned url for uploading the object to S3 bucket
            ext: extension of the file for saving the object
            temp_dir: temporary directory for storing the saved object
            register_for_clean: if True, record the name of the object which will be cleaned in the cleanup function
        Outputs:
            s3_obj: an object containing the following information
                - if uploaded directly: file name, presigned download url
                - if uploaded via presigned url: file name, object name
        """
        if object_name is None:
            assert type(data) is LargeObjectWrapper
            object_name = data.name
            data = data.data
        cs = cls.get_instance()

        if object_name in cs.uploaded_obj:
            return cs.uploaded_obj[object_name]

        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, f"{object_name}.{ext}")

        dump_data_to_file(data, file_path)

        file_size = osp.getsize(file_path)
        size_metric = ["KB", "MB", "GB"]
        for metric in size_metric:
            file_size *= 1e-3
            if file_size < 1000 or metric == "GB":
                if cs.logger is not None:
                    cs.logger.info(f"Uploading model ({file_size:.2f} {metric}) to S3")
                else:
                    print(f"Uploading model ({file_size:.2f} {metric}) to S3")
                break

        if object_name is not None and register_for_clean:
            cs.registered_obj.add(object_name)

        return cs.upload_file(file_path, object_url, object_name)

    @classmethod
    def download_object(
        cls,
        data_info: dict,
        temp_dir: str = None,
        to_device: str = "cpu",
        delete_cloud: bool = False,
        delete_local: bool = True,
    ):
        """
        Download a python object from S3 either using the object_name (only possible with credentials)
        or using the presigned url, save the object into the temporary directory, and load into python object.
        Inputs:
            cls: the cloud storage class instance
            data_info: information about the cloud storage object
            temp_dir: temporary directory for storing the downloaded object
            to_device: the device of the downloaded object (only applicable to PyTorch model)
            delete_cloud: whether to delete the object stored on S3 after downloading the object
            delete_local: whether to delete the local file used for temporary storage of the object
        Output:
            object: the downloaded object
        """
        cs = cls.get_instance()
        file_name, object_name, object_url = cls.get_cloud_object_info(data_info)

        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, file_name)

        cs.download_file(file_path, object_name, object_url, delete_cloud)

        file_size = osp.getsize(file_path)
        size_metric = ["KB", "MB", "GB"]
        for metric in size_metric:
            file_size *= 1e-3
            if file_size < 1000 or metric == "GB":
                if cs.logger is not None:
                    cs.logger.info(
                        f"Downloaded model ({file_size:.2f} {metric}) from S3"
                    )
                else:
                    print(f"Downloaded model ({file_size:.2f} {metric}) from S3")
                break

        data = load_data_from_file(file_path, to_device)

        if delete_local:
            try:
                os.remove(file_path)
            except:  # noqa E722
                pass

        return data

    @classmethod
    def presign_upload_object(
        cls, object_name: str, expiration: int = 3600, register_for_clean=False
    ):
        """Presign a url for uploading an object to S3 bucket"""
        cs = cls.get_instance()
        try:
            response = cs.client.generate_presigned_url(
                "put_object",
                Params={"Bucket": cs.bucket, "Key": object_name},
                ExpiresIn=expiration,
            )
            if object_name is not None and register_for_clean:
                cs.registered_obj.add(object_name)
        except Exception as e:
            if cs.logger is not None:
                cs.logger.info(
                    f"Error in generating presigned url for uploading object {e}"
                )
            else:
                print(f"Error in generating presigned url for uploading object {e}")
            raise
        return response

    @classmethod
    def clean_up(cls):
        """Clean up registered objects on S3 bucket"""
        cs = cls.get_instance()
        for obj in cs.registered_obj:
            try:
                cs.client.delete_object(Bucket=cs.bucket, Key=obj)
                if cs.logger is not None:
                    cs.logger.info(f"Successfully cleaned object {obj} on S3")
                else:
                    print(f"Successfully cleaned object {obj} on S3")
            except Exception as e:
                if cs.logger is not None:
                    cs.logger.info(f"Error in cleaning object {obj} on S3 {e}")
                else:
                    print(f"Error in cleaning object {obj} on S3 {e}")
