from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from collections import defaultdict

from io_utils import write_json


def summarize_results(rows):
    """Summarize end-to-end results by task and method."""
    best_by_task = defaultdict(dict)
    for row in rows:
        task = row.get("task")
        method = row.get("method")
        residual = row.get("residual")
        if residual is None:
            continue
        current = best_by_task[task].get(method)
        if current is None or residual < current:
            best_by_task[task][method] = residual

    return {"best_residual_by_task": best_by_task}


def write_summary(path, summary):
    """Write a summary JSON file to disk."""
    write_json(path, summary)


def maybe_send_results(summary_text, files, send_slack):
    """Optionally send a summary and artifacts to Slack."""
    _ = (summary_text, files, send_slack)
    return
