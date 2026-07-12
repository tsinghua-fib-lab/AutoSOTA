import sys
sys.path.extend(['./', '../', '../../'])
import seml
from sacred import Experiment


# from https://github.com/n-gao/seml_logger
def add_default_observer_config(
    experiment: Experiment,
    notify_on_started=False,
    notify_on_completed=True,
    notify_on_failed=True,
    notify_on_interrupted=True,
    **kwargs,
):
    # We must use a global variable here due to the way sacred handles the configuration function.
    # It is only evaluated with the current global variables.
    kwargs = {**locals(), **kwargs}
    del kwargs["experiment"]
    del kwargs["kwargs"]
    global _kwargs, _ex
    _kwargs, _ex = kwargs, experiment

    def observer_config():
        global _ex, _kwargs
        name = "`{experiment[name]} ({config[db_collection]}:{_id})`"
        _ex.observers.append(
            seml.create_mattermost_observer(
                started_text=(
                    f":hourglass_flowing_sand: {name} "
                    "started on host `{host_info[hostname]}`."
                ),
                completed_text=(
                    f":white_check_mark: {name} "
                    "completed after _{elapsed_time}_ with result:\n"
                    "```json\n{result}\n````\n"
                ),
                interrupted_text=(
                    f":warning: {name} " "interrupted after _{elapsed_time}_."
                ),
                failed_text=(
                    f":x: {name} "
                    "failed after _{elapsed_time}_ with `{error}`.\n"
                    "```python\n{fail_trace}\n```\n"
                ),
                **_kwargs,
            )
        )
        # Clean global variables
        del _ex
        del _kwargs

    return _ex.config(observer_config)
