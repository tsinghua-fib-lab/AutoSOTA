# Using Caching and Tracing

APPL supports tracing APPL functions and LM calls to facilitate users to understand and debug the program executions. Both the trace and the persistent LLM caching are useful for reproducing (potentially partial) execution results by loading cached responses of the LM calls, which enables failure recovery and avoids the extra costs of resending these calls. This also unlocks the possibility of conveniently debugging one specific LM call out of the whole program.

## Enabling APPL Caching
The persistent LLM caching (default path: `~/.appl/caches/cache.db`) is automatically enabled since v0.1.5.
LLM calls with temperature 0 will look up the cache first, use the cached responses if found, generate and cache the responses otherwise.

```yaml title="appl.yaml"
settings:
  caching:
    enabled: true # default to enable the caching
    folder: "~/.appl/caches" # The folder to store the cache files
    max_size: 100000  # Maximum number of entries in cache
    time_to_live: 43200 # Time-to-live in minutes (30 days)
    cleanup_interval: 1440 # Cleanup interval in minutes (1 day)
    allow_temp_greater_than_0: false # Whether to cache the generation results with temperature to be greater than 0
```

## Enabling APPL Tracing

To enable tracing in APPL, you can set the `tracing` configuration to `true` in `appl.yaml`:

```yaml title="appl.yaml"
settings:
  tracing:
    enabled: true
    path_format: <Your custom path for trace files>
    # The default path format is "./dumps/traces/{basename}_{time:YYYY_MM_DD__HH_mm_ss}"
    patch_threading: true # whether to patch `threading.Thread`
    strict_match: true # For loading the trace files, explain later.
```

### Obtaining the Trace File
Then we run the [QA example](./2_qa_example.md#answer-follow-up-questions) with tracing enabled:

```python linenums="1" title="answer_questions.py"
--8<-- "examples/basic/answer_questions.py"
```

```bash
$ python answer_questions.py
```

You can find the result trace file in the specified path. The default location is `./dumps/traces/answer_questions_<the timestamp>.pkl`.
You can [visualize](#visualizing-the-trace) the trace file using the method you want.

### Resuming from a Previous Trace

You can reproduce the execution results from a previous trace by specifying the `APPL_RESUME_TRACE` environment variable with the path to the trace file:

```bash
$ APPL_RESUME_TRACE=<path to the trace file> python answer_questions.py
```

Then each LM call will be loaded from the trace file if it exists (loading from the trace is of higher priority than the persistent cache). Such loading can be useful for:

- Debugging a specific LM call: the LM calls before that can be loaded from the trace file, therefore no need to resend them with extra costs.
- Reproducible results: the trace file can be shared with others to reproduce the same results.
- Recovery from failures: if the program fails, you can resume from the trace file to avoid resending the LM calls.

!!! info "`strict_match` for calls with same prompts"
    When `strict_match` is `False`, the LM calls with the same prompt will load the same response from the trace file. To load the response for each LM call correspondingly, you can set `strict_match` to `True` (which is the default setting), then the `gen_id` of the LM call will also be used for matching.

## Visualizing the Trace

### Langfuse (Recommended)

Langfuse is an open-source web-based tool for visualizing traces and LLM calls.

You can host Langfuse [locally](https://langfuse.com/self-hosting) or use [public version](https://langfuse.com/).

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```

Then you can set the environment variables for the Langfuse server by:

```bash title=".env"
LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>
LANGFUSE_SECRET_KEY=<your-langfuse-secret-key>
LANGFUSE_HOST=<your-langfuse-host>
# Set to http://localhost:3000 if you are hosting Langfuse locally
```
You can find your Langfuse public and private API keys in the project settings page (Project Dashboard -> Configure Tracing).

Then you can visualize the traces by:

```bash
$ appltrace <path to the trace file>
```

Then you will see conversation like:

![Langfuse Conversation](../_assets/tracing/langfuse_convo.png)

and the timeline like:

![Langfuse Timeline](../_assets/tracing/langfuse_timeline.png)

??? question "Troubleshooting: Incomplete traces on Langfuse"
    You may see incomplete traces (function calls tree) in Langfuse when you click from the `Traces` page. This might because langfuse apply a filter based on the timestamp. Try to remove the `?timestamp=<timestamp>` in the url and refresh the page.

### Lunary 

Lunary is another open-source web-based tool for visualizing traces and LLM calls.

You can host Lunary [locally](https://github.com/lunary-ai/lunary?tab=readme-ov-file#running-locally) or use [their hosted version](https://lunary.ai/).

You can follow [the steps](https://github.com/lunary-ai/lunary?tab=readme-ov-file#running-locally) to start a local Lunary server.
After installing the Postgres, you may create a database for Lunary:
```bash
createuser postgres --createdb
createdb lunary -U postgres
# Your postgres url is: "postgresql://postgres:@localhost:5432/lunary"
# you can verify the database by:
psql postgresql://postgres:@localhost:5432/lunary
```

Then you can use this url to set the DATABASE_URL in "packages/backend/.env". You may also change other environment variables in the ".env" file according to your needs.
Then you can set the environment variables for the Lunary server by:

```bash
# `1c1975c5-13b9-4977-8003-89fff5c71c27` is the project ID of the default project, you can get the project ID from the website.
export LUNARY_API_KEY=<your project ID>
# `http://localhost:3333` is the default url
export LUNARY_API_URL=<the url of the Lunary server>
```

Then you can visualize the traces by:

```bash
$ appltrace <path to the trace file> --platform lunary
```

Then you will see:

![Lunary](../_assets/tracing/lunary.png)

### Simple HTML and Chrome Tracing

You can then visualize the traces using the script:

```bash
$ appltrace <path to the trace file> -o <output file>
```

The default output file is a HTML file, which can be viewed in a browser. We provide a sample trace file [here](../_assets/tracing/example_trace.html).

If you specify the output file to be a `.json` file, the script will generate a JSON file that is loadable by Chrome's tracing viewer (with address: chrome://tracing/). The loaded trace will look like this:

![Chrome Trace Viewer](../_assets/tracing/chrome_viewer.png)

This way is going to be deprecated where Langfuse provides much better visualization.

### LangSmith

Optionally, you can use LangSmith to inspect the LM calls and responses in the trace files. You need to [obtain your API key](https://smith.langchain.com/settings) from LangSmith and add the following environment variables to your `.env` file:

```bash title=".env"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your api key>
# [Optional] specify the project name
# LANGCHAIN_PROJECT=<your project name>
```

??? warning "LangSmith may contain inaccurate statistics for asynchronous LM calls"
    ![Langsmith](../_assets/tracing/langsmith.png)
    
    When running the [example](./2_qa_example.md#answer-follow-up-questions), the time statistics for the `get_answer` function calls are not consistent.

    Nonetheless, it is sometimes useful to record and inspect the LM calls and responses using LangSmith.
