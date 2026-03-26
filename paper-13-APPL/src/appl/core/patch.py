import contextvars
import threading

APPL_PATCHED_NAME = "__APPL_PATCHED__"


def _get_new_target(target):
    # copy the current context and propagate it to the function in the new thread
    ctx = contextvars.copy_context()

    def new_target(*args, **kwargs):
        return ctx.run(target, *args, **kwargs)

    return new_target


def patch_threading() -> None:
    """Patch threading.Thread to automatically wrap the target with context."""
    if not hasattr(threading.Thread, APPL_PATCHED_NAME):
        # print("patching threading.Thread")
        class ThreadWithContext(threading.Thread):
            def __init__(self, *args, **kwargs):
                if "target" in kwargs and kwargs["target"] is not None:
                    kwargs["target"] = _get_new_target(kwargs["target"])
                elif len(args) > 0 and args[0] is not None:
                    args = (_get_new_target(args[0]),) + args[1:]
                super().__init__(*args, **kwargs)

        setattr(ThreadWithContext, APPL_PATCHED_NAME, True)
        threading.Thread = ThreadWithContext  # type: ignore
