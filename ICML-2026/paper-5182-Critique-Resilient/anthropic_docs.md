[Anthropic home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/dark.svg)](https://docs.anthropic.com/)

English

Search...

Ctrl K

Search...

Navigation

Capabilities

Batch processing

[Welcome](https://docs.anthropic.com/en/home) [Developer Platform](https://docs.anthropic.com/en/docs/intro) [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) [Model Context Protocol (MCP)](https://docs.anthropic.com/en/docs/mcp) [API Reference](https://docs.anthropic.com/en/api/messages) [Resources](https://docs.anthropic.com/en/resources/overview) [Release Notes](https://docs.anthropic.com/en/release-notes/overview)

Batch processing is a powerful approach for handling large volumes of requests efficiently. Instead of processing requests one at a time with immediate responses, batch processing allows you to submit multiple requests together for asynchronous processing. This pattern is particularly useful when:

- You need to process large volumes of data
- Immediate responses are not required
- You want to optimize for cost efficiency
- You’re running large-scale evaluations or analyses

The Message Batches API is our first implementation of this pattern.

* * *

# [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#message-batches-api)  Message Batches API

The Message Batches API is a powerful, cost-effective way to asynchronously process large volumes of [Messages](https://docs.anthropic.com/en/api/messages) requests. This approach is well-suited to tasks that do not require immediate responses, with most batches finishing in less than 1 hour while reducing costs by 50% and increasing throughput.

You can [explore the API reference directly](https://docs.anthropic.com/en/api/creating-message-batches), in addition to this guide.

## [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#how-the-message-batches-api-works)  How the Message Batches API works

When you send a request to the Message Batches API:

1. The system creates a new Message Batch with the provided Messages requests.
2. The batch is then processed asynchronously, with each request handled independently.
3. You can poll for the status of the batch and retrieve results when processing has ended for all requests.

This is especially useful for bulk operations that don’t require immediate results, such as:

- Large-scale evaluations: Process thousands of test cases efficiently.
- Content moderation: Analyze large volumes of user-generated content asynchronously.
- Data analysis: Generate insights or summaries for large datasets.
- Bulk content generation: Create large amounts of text for various purposes (e.g., product descriptions, article summaries).

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#batch-limitations)  Batch limitations

- A Message Batch is limited to either 100,000 Message requests or 256 MB in size, whichever is reached first.
- We process each batch as fast as possible, with most batches completing within 1 hour. You will be able to access batch results when all messages have completed or after 24 hours, whichever comes first. Batches will expire if processing does not complete within 24 hours.
- Batch results are available for 29 days after creation. After that, you may still view the Batch, but its results will no longer be available for download.
- Batches are scoped to a [Workspace](https://console.anthropic.com/settings/workspaces). You may view all batches—and their results—that were created within the Workspace that your API key belongs to.
- Rate limits apply to both Batches API HTTP requests and the number of requests within a batch waiting to be processed. See [Message Batches API rate limits](https://docs.anthropic.com/en/api/rate-limits#message-batches-api). Additionally, we may slow down processing based on current demand and your request volume. In that case, you may see more requests expiring after 24 hours.
- Due to high throughput and concurrent processing, batches may go slightly over your Workspace’s configured [spend limit](https://console.anthropic.com/settings/limits).

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#supported-models)  Supported models

The Message Batches API currently supports:

- Claude Opus 4.1 ( `claude-opus-4-1-20250805`)
- Claude Opus 4 ( `claude-opus-4-20250514`)
- Claude Sonnet 4 ( `claude-sonnet-4-20250514`)
- Claude Sonnet 3.7 ( `claude-3-7-sonnet-20250219`)
- Claude Sonnet 3.5 ( [deprecated](https://docs.anthropic.com/en/docs/about-claude/model-deprecations)) ( `claude-3-5-sonnet-20240620` and `claude-3-5-sonnet-20241022`)
- Claude Haiku 3.5 ( `claude-3-5-haiku-20241022`)
- Claude Haiku 3 ( `claude-3-haiku-20240307`)
- Claude Opus 3 ( [deprecated](https://docs.anthropic.com/en/docs/about-claude/model-deprecations)) ( `claude-3-opus-20240229`)

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#what-can-be-batched)  What can be batched

Any request that you can make to the Messages API can be included in a batch. This includes:

- Vision
- Tool use
- System messages
- Multi-turn conversations
- Any beta features

Since each request in the batch is processed independently, you can mix different types of requests within a single batch.

Since batches can take longer than 5 minutes to process, consider using the [1-hour cache duration](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration) with prompt caching for better cache hit rates when processing batches with shared context.

* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#pricing)  Pricing

The Batches API offers significant cost savings. All usage is charged at 50% of the standard API prices.

| Model | Batch input | Batch output |
| --- | --- | --- |
| Claude Opus 4.1 | $7.50 / MTok | $37.50 / MTok |
| Claude Opus 4 | $7.50 / MTok | $37.50 / MTok |
| Claude Sonnet 4 | $1.50 / MTok | $7.50 / MTok |
| Claude Sonnet 3.7 | $1.50 / MTok | $7.50 / MTok |
| Claude Sonnet 3.5 ( [deprecated](https://docs.anthropic.com/en/docs/about-claude/model-deprecations)) | $1.50 / MTok | $7.50 / MTok |
| Claude Haiku 3.5 | $0.40 / MTok | $2 / MTok |
| Claude Opus 3 ( [deprecated](https://docs.anthropic.com/en/docs/about-claude/model-deprecations)) | $7.50 / MTok | $37.50 / MTok |
| Claude Haiku 3 | $0.125 / MTok | $0.625 / MTok |

* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#how-to-use-the-message-batches-api)  How to use the Message Batches API

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#prepare-and-create-your-batch)  Prepare and create your batch

A Message Batch is composed of a list of requests to create a Message. The shape of an individual request is comprised of:

- A unique `custom_id` for identifying the Messages request
- A `params` object with the standard [Messages API](https://docs.anthropic.com/en/api/messages) parameters

You can [create a batch](https://docs.anthropic.com/en/api/creating-message-batches) by passing this list into the `requests` parameter:

Shell

Python

TypeScript

Java

Copy

```bash
curl https://api.anthropic.com/v1/messages/batches \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "requests": [\
        {\
            "custom_id": "my-first-request",\
            "params": {\
                "model": "claude-opus-4-1-20250805",\
                "max_tokens": 1024,\
                "messages": [\
                    {"role": "user", "content": "Hello, world"}\
                ]\
            }\
        },\
        {\
            "custom_id": "my-second-request",\
            "params": {\
                "model": "claude-opus-4-1-20250805",\
                "max_tokens": 1024,\
                "messages": [\
                    {"role": "user", "content": "Hi again, friend"}\
                ]\
            }\
        }\
    ]
}'

```

In this example, two separate requests are batched together for asynchronous processing. Each request has a unique `custom_id` and contains the standard parameters you’d use for a Messages API call.

**Test your batch requests with the Messages API**

Validation of the `params` object for each message request is performed asynchronously, and validation errors are returned when processing of the entire batch has ended. You can ensure that you are building your input correctly by verifying your request shape with the [Messages API](https://docs.anthropic.com/en/api/messages) first.

When a batch is first created, the response will have a processing status of `in_progress`.

JSON

Copy

```JSON
{
  "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 2,
    "succeeded": 0,
    "errored": 0,
    "canceled": 0,
    "expired": 0
  },
  "ended_at": null,
  "created_at": "2024-09-24T18:37:24.100435Z",
  "expires_at": "2024-09-25T18:37:24.100435Z",
  "cancel_initiated_at": null,
  "results_url": null
}

```

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#tracking-your-batch)  Tracking your batch

The Message Batch’s `processing_status` field indicates the stage of processing the batch is in. It starts as `in_progress`, then updates to `ended` once all the requests in the batch have finished processing, and results are ready. You can monitor the state of your batch by visiting the [Console](https://console.anthropic.com/settings/workspaces/default/batches), or using the [retrieval endpoint](https://docs.anthropic.com/en/api/retrieving-message-batches):

Shell

Python

TypeScript

Java

Copy

```bash
curl https://api.anthropic.com/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d \
 --header "x-api-key: $ANTHROPIC_API_KEY" \
 --header "anthropic-version: 2023-06-01" \
 | sed -E 's/.*"id":"([^"]+)".*"processing_status":"([^"]+)".*/Batch \1 processing status is \2/'

```

You can [poll](https://docs.anthropic.com/en/api/messages-batch-examples#polling-for-message-batch-completion) this endpoint to know when processing has ended.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#retrieving-batch-results)  Retrieving batch results

Once batch processing has ended, each Messages request in the batch will have a result. There are 4 result types:

| Result Type | Description |
| --- | --- |
| `succeeded` | Request was successful. Includes the message result. |
| `errored` | Request encountered an error and a message was not created. Possible errors include invalid requests and internal server errors. You will not be billed for these requests. |
| `canceled` | User canceled the batch before this request could be sent to the model. You will not be billed for these requests. |
| `expired` | Batch reached its 24 hour expiration before this request could be sent to the model. You will not be billed for these requests. |

You will see an overview of your results with the batch’s `request_counts`, which shows how many requests reached each of these four states.

Results of the batch are available for download at the `results_url` property on the Message Batch, and if the organization permission allows, in the Console. Because of the potentially large size of the results, it’s recommended to [stream results](https://docs.anthropic.com/en/api/retrieving-message-batch-results) back rather than download them all at once.

Shell

Python

TypeScript

Java

Copy

```bash
#!/bin/sh
curl "https://api.anthropic.com/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d" \
  --header "anthropic-version: 2023-06-01" \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  | grep -o '"results_url":[[:space:]]*"[^"]*"' \
  | cut -d'"' -f4 \
  | while read -r url; do
    curl -s "$url" \
      --header "anthropic-version: 2023-06-01" \
      --header "x-api-key: $ANTHROPIC_API_KEY" \
      | sed 's/}{/}\n{/g' \
      | while IFS= read -r line
    do
      result_type=$(echo "$line" | sed -n 's/.*"result":[[:space:]]*{[[:space:]]*"type":[[:space:]]*"\([^"]*\)".*/\1/p')
      custom_id=$(echo "$line" | sed -n 's/.*"custom_id":[[:space:]]*"\([^"]*\)".*/\1/p')
      error_type=$(echo "$line" | sed -n 's/.*"error":[[:space:]]*{[[:space:]]*"type":[[:space:]]*"\([^"]*\)".*/\1/p')

      case "$result_type" in
        "succeeded")
          echo "Success! $custom_id"
          ;;
        "errored")
          if [ "$error_type" = "invalid_request" ]; then
            # Request body must be fixed before re-sending request
            echo "Validation error: $custom_id"
          else
            # Request can be retried directly
            echo "Server error: $custom_id"
          fi
          ;;
        "expired")
          echo "Expired: $line"
          ;;
      esac
    done
  done

```

The results will be in `.jsonl` format, where each line is a valid JSON object representing the result of a single request in the Message Batch. For each streamed result, you can do something different depending on its `custom_id` and result type. Here is an example set of results:

.jsonl file

Copy

```JSON
{"custom_id":"my-second-request","result":{"type":"succeeded","message":{"id":"msg_014VwiXbi91y3JMjcpyGBHX5","type":"message","role":"assistant","model":"claude-opus-4-1-20250805","content":[{"type":"text","text":"Hello again! It's nice to see you. How can I assist you today? Is there anything specific you'd like to chat about or any questions you have?"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":36}}}}
{"custom_id":"my-first-request","result":{"type":"succeeded","message":{"id":"msg_01FqfsLoHwgeFbguDgpz48m7","type":"message","role":"assistant","model":"claude-opus-4-1-20250805","content":[{"type":"text","text":"Hello! How can I assist you today? Feel free to ask me any questions or let me know if there's anything you'd like to chat about."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":34}}}}

```

If your result has an error, its `result.error` will be set to our standard [error shape](https://docs.anthropic.com/en/api/errors#error-shapes).

**Batch results may not match input order**

Batch results can be returned in any order, and may not match the ordering of requests when the batch was created. In the above example, the result for the second batch request is returned before the first. To correctly match results with their corresponding requests, always use the `custom_id` field.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#using-prompt-caching-with-message-batches)  Using prompt caching with Message Batches

The Message Batches API supports prompt caching, allowing you to potentially reduce costs and processing time for batch requests. The pricing discounts from prompt caching and Message Batches can stack, providing even greater cost savings when both features are used together. However, since batch requests are processed asynchronously and concurrently, cache hits are provided on a best-effort basis. Users typically experience cache hit rates ranging from 30% to 98%, depending on their traffic patterns.

To maximize the likelihood of cache hits in your batch requests:

1. Include identical `cache_control` blocks in every Message request within your batch
2. Maintain a steady stream of requests to prevent cache entries from expiring after their 5-minute lifetime
3. Structure your requests to share as much cached content as possible

Example of implementing prompt caching in a batch:

Shell

Python

TypeScript

Java

Copy

```bash
curl https://api.anthropic.com/v1/messages/batches \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "requests": [\
        {\
            "custom_id": "my-first-request",\
            "params": {\
                "model": "claude-opus-4-1-20250805",\
                "max_tokens": 1024,\
                "system": [\
                    {\
                        "type": "text",\
                        "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n"\
                    },\
                    {\
                        "type": "text",\
                        "text": "<the entire contents of Pride and Prejudice>",\
                        "cache_control": {"type": "ephemeral"}\
                    }\
                ],\
                "messages": [\
                    {"role": "user", "content": "Analyze the major themes in Pride and Prejudice."}\
                ]\
            }\
        },\
        {\
            "custom_id": "my-second-request",\
            "params": {\
                "model": "claude-opus-4-1-20250805",\
                "max_tokens": 1024,\
                "system": [\
                    {\
                        "type": "text",\
                        "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.\n"\
                    },\
                    {\
                        "type": "text",\
                        "text": "<the entire contents of Pride and Prejudice>",\
                        "cache_control": {"type": "ephemeral"}\
                    }\
                ],\
                "messages": [\
                    {"role": "user", "content": "Write a summary of Pride and Prejudice."}\
                ]\
            }\
        }\
    ]
}'

```

In this example, both requests in the batch include identical system messages and the full text of Pride and Prejudice marked with `cache_control` to increase the likelihood of cache hits.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#best-practices-for-effective-batching)  Best practices for effective batching

To get the most out of the Batches API:

- Monitor batch processing status regularly and implement appropriate retry logic for failed requests.
- Use meaningful `custom_id` values to easily match results with requests, since order is not guaranteed.
- Consider breaking very large datasets into multiple batches for better manageability.
- Dry run a single request shape with the Messages API to avoid validation errors.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#troubleshooting-common-issues)  Troubleshooting common issues

If experiencing unexpected behavior:

- Verify that the total batch request size doesn’t exceed 256 MB. If the request size is too large, you may get a 413 `request_too_large` error.
- Check that you’re using [supported models](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#supported-models) for all requests in the batch.
- Ensure each request in the batch has a unique `custom_id`.
- Ensure that it has been less than 29 days since batch `created_at` (not processing `ended_at`) time. If over 29 days have passed, results will no longer be viewable.
- Confirm that the batch has not been canceled.

Note that the failure of one request in a batch does not affect the processing of other requests.

* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#batch-storage-and-privacy)  Batch storage and privacy

- **Workspace isolation**: Batches are isolated within the Workspace they are created in. They can only be accessed by API keys associated with that Workspace, or users with permission to view Workspace batches in the Console.

- **Result availability**: Batch results are available for 29 days after the batch is created, allowing ample time for retrieval and processing.


* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing\#faq)  FAQ

How long does it take for a batch to process?

Batches may take up to 24 hours for processing, but many will finish sooner. Actual processing time depends on the size of the batch, current demand, and your request volume. It is possible for a batch to expire and not complete within 24 hours.

Is the Batches API available for all models?

See [above](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#supported-models) for the list of supported models.

Can I use the Message Batches API with other API features?

Yes, the Message Batches API supports all features available in the Messages API, including beta features. However, streaming is not supported for batch requests.

How does the Message Batches API affect pricing?

The Message Batches API offers a 50% discount on all usage compared to standard API prices. This applies to input tokens, output tokens, and any special tokens. For more on pricing, visit our [pricing page](https://www.anthropic.com/pricing#anthropic-api).

Can I update a batch after it's been submitted?

No, once a batch has been submitted, it cannot be modified. If you need to make changes, you should cancel the current batch and submit a new one. Note that cancellation may not take immediate effect.

Are there Message Batches API rate limits and do they interact with the Messages API rate limits?

The Message Batches API has HTTP requests-based rate limits in addition to limits on the number of requests in need of processing. See [Message Batches API rate limits](https://docs.anthropic.com/en/api/rate-limits#message-batches-api). Usage of the Batches API does not affect rate limits in the Messages API.

How do I handle errors in my batch requests?

When you retrieve the results, each request will have a `result` field indicating whether it `succeeded`, `errored`, was `canceled`, or `expired`. For `errored` results, additional error information will be provided. View the error response object in the [API reference](https://docs.anthropic.com/en/api/creating-message-batches).

How does the Message Batches API handle privacy and data separation?

The Message Batches API is designed with strong privacy and data separation measures:

1. Batches and their results are isolated within the Workspace in which they were created. This means they can only be accessed by API keys from that same Workspace.
2. Each request within a batch is processed independently, with no data leakage between requests.
3. Results are only available for a limited time (29 days), and follow our [data retention policy](https://support.anthropic.com/en/articles/7996866-how-long-do-you-store-personal-data).
4. Downloading batch results in the Console can be disabled on the organization-level or on a per-workspace basis.

Can I use prompt caching in the Message Batches API?

Yes, it is possible to use prompt caching with Message Batches API. However, because asynchronous batch requests can be processed concurrently and in any order, cache hits are provided on a best-effort basis.

Was this page helpful?

YesNo

[Streaming Messages](https://docs.anthropic.com/en/docs/build-with-claude/streaming) [Citations](https://docs.anthropic.com/en/docs/build-with-claude/citations)

On this page

- [Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#message-batches-api)
- [How the Message Batches API works](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#how-the-message-batches-api-works)
- [Batch limitations](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#batch-limitations)
- [Supported models](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#supported-models)
- [What can be batched](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#what-can-be-batched)
- [Pricing](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#pricing)
- [How to use the Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#how-to-use-the-message-batches-api)
- [Prepare and create your batch](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#prepare-and-create-your-batch)
- [Tracking your batch](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#tracking-your-batch)
- [Retrieving batch results](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#retrieving-batch-results)
- [Using prompt caching with Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#using-prompt-caching-with-message-batches)
- [Best practices for effective batching](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#best-practices-for-effective-batching)
- [Troubleshooting common issues](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#troubleshooting-common-issues)
- [Batch storage and privacy](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#batch-storage-and-privacy)
- [FAQ](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#faq)