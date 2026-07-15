from globus_compute_sdk import Executor


def double(x):
    return x * 2


endpoint_id = ""  # YOUR-ENDPOINT-ID
with Executor(endpoint_id=endpoint_id) as gce:
    fut = gce.submit(double, 7)
    print(fut.result())
