import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Union
from appfl.comm.utils.s3_storage import CloudStorage, LargeObjectWrapper
from appfl.misc.utils import secure_appfl_dir


def send_model_by_pre_signed_s3(
    client_id: str,
    experiment_id: str,
    comm_type: str,
    model: Union[Dict, OrderedDict, bytes],
    model_key: Optional[str],
    model_url: Optional[str],
    logger: Optional[Any] = None,
):
    s3_tmp_dir = secure_appfl_dir(comm_type, client_id, experiment_id)
    model_wrapper = LargeObjectWrapper(model, model_key)
    if (
        comm_type == "globus_compute" and not model_wrapper.can_send_directly
    ) or comm_type != "globus_compute":
        CloudStorage.init(s3_tmp_dir=s3_tmp_dir, logger=logger)
        model = CloudStorage.upload_object(
            model_wrapper,
            object_url=model_url,
            ext="pt" if not isinstance(model, bytes) else "pkl",
            register_for_clean=True,
        )
    return model


def send_model_by_s3(experiment_id, comm_type, model, sender_id):
    model_wrapper = LargeObjectWrapper(
        data=model,
        name=experiment_id + str(uuid.uuid4()) + f"_{sender_id}_state",
    )
    if (
        comm_type == "globus_compute" and not model_wrapper.can_send_directly
    ) or comm_type != "globus_compute":
        model = CloudStorage.upload_object(model_wrapper, register_for_clean=True)
    return model


def extract_model_from_s3(
    client_id: str, experiment_id: str, comm_type: str, model: Any
):
    s3_tmp_dir = secure_appfl_dir(comm_type, client_id, experiment_id)
    if CloudStorage.is_cloud_storage_object(model):
        CloudStorage.init(s3_tmp_dir=s3_tmp_dir)
        model = CloudStorage.download_object(model)
    return model
