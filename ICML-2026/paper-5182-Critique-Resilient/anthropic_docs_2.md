[Anthropic home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/anthropic/logo/dark.svg)](https://docs.anthropic.com/)

English

Search...

Ctrl K

Search...

Navigation

Messages

Messages

[Welcome](https://docs.anthropic.com/en/home) [Developer Platform](https://docs.anthropic.com/en/docs/intro) [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) [Model Context Protocol (MCP)](https://docs.anthropic.com/en/docs/mcp) [API Reference](https://docs.anthropic.com/en/api/messages) [Resources](https://docs.anthropic.com/en/resources/overview) [Release Notes](https://docs.anthropic.com/en/release-notes/overview)

POST

/

v1

/

messages

cURL

Python

JavaScript

PHP

Go

Java

Copy

```
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [\
        {"role": "user", "content": "Hello, world"}\
    ]
}'
```

200

4XX

Copy

```
{
  "content": [\
    {\
      "text": "Hi! My name is Claude.",\
      "type": "text"\
    }\
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-sonnet-4-20250514",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503
  }
}
```

#### Headers

[​](https://docs.anthropic.com/en/api/messages#parameter-anthropic-beta)

anthropic-beta

string\[\]

Optional header to specify the beta version(s) you want to use.

To use multiple betas, use a comma separated list like `beta1,beta2` or specify the header multiple times for each beta.

[​](https://docs.anthropic.com/en/api/messages#parameter-anthropic-version)

anthropic-version

string

required

The version of the Anthropic API you want to use.

Read more about versioning and our version history [here](https://docs.anthropic.com/en/api/versioning).

[​](https://docs.anthropic.com/en/api/messages#parameter-x-api-key)

x-api-key

string

required

Your unique API key for authentication.

This key is required in the header of all API requests, to authenticate your account and access Anthropic's services. Get your API key through the [Console](https://console.anthropic.com/settings/keys). Each key is scoped to a Workspace.

#### Body

application/json

[​](https://docs.anthropic.com/en/api/messages#body-model)

model

string

required

The model that will complete your prompt.

See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

Required string length: `1 - 256`

Examples:

`"claude-sonnet-4-20250514"`

[​](https://docs.anthropic.com/en/api/messages#body-messages)

messages

object\[\]

required

Input messages.

Our models are trained to operate on alternating `user` and `assistant` conversational turns. When creating a new `Message`, you specify the prior conversational turns with the `messages` parameter, and the model then generates the next `Message` in the conversation. Consecutive `user` or `assistant` turns in your request will be combined into a single turn.

Each input message must be an object with a `role` and `content`. You can specify a single `user`-role message, or you can include multiple `user` and `assistant` messages.

If the final message uses the `assistant` role, the response content will continue immediately from the content in that message. This can be used to constrain part of the model's response.

Example with a single `user` message:

```json
[{"role": "user", "content": "Hello, Claude"}]

```

Example with multiple conversational turns:

```json
[\
  {"role": "user", "content": "Hello there."},\
  {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},\
  {"role": "user", "content": "Can you explain LLMs in plain English?"},\
]

```

Example with a partially-filled response from Claude:

```json
[\
  {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},\
  {"role": "assistant", "content": "The best answer is ("},\
]

```

Each input message `content` may be either a single `string` or an array of content blocks, where each block has a specific `type`. Using a `string` for `content` is shorthand for an array of one content block of type `"text"`. The following input messages are equivalent:

```json
{"role": "user", "content": "Hello, Claude"}

```

```json
{"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}

```

See [examples](https://docs.anthropic.com/en/api/messages-examples) for more input examples.

Note that if you want to include a [system prompt](https://docs.anthropic.com/en/docs/system-prompts), you can use the top-level `system` parameter — there is no `"system"` role for input messages in the Messages API.

There is a limit of 100,000 messages in a single request.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-messages-content)

messages.content

stringobject\[\]

required

[​](https://docs.anthropic.com/en/api/messages#body-messages-role)

messages.role

enum<string>

required

Available options:

`user`,

`assistant`

[​](https://docs.anthropic.com/en/api/messages#body-max-tokens)

max\_tokens

integer

required

The maximum number of tokens to generate before stopping.

Note that our models may stop _before_ reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate.

Different models have different maximum values for this parameter. See [models](https://docs.anthropic.com/en/docs/models-overview) for details.

Required range: `x >= 1`

Examples:

`1024`

[​](https://docs.anthropic.com/en/api/messages#body-container)

container

string \| null

Container identifier for reuse across requests.

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers)

mcp\_servers

object\[\]

MCP servers to be utilized in this request

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-name)

mcp\_servers.name

string

required

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-type)

mcp\_servers.type

enum<string>

required

Available options:

`url`

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-url)

mcp\_servers.url

string

required

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-authorization-token)

mcp\_servers.authorization\_token

string \| null

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-tool-configuration)

mcp\_servers.tool\_configuration

object \| null

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-tool-configuration-allowed-tools)

mcp\_servers.tool\_configuration.allowed\_tools

string\[\] \| null

[​](https://docs.anthropic.com/en/api/messages#body-mcp-servers-tool-configuration-enabled)

mcp\_servers.tool\_configuration.enabled

boolean \| null

[​](https://docs.anthropic.com/en/api/messages#body-metadata)

metadata

object

An object describing metadata about the request.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-metadata-user-id)

metadata.user\_id

string \| null

An external identifier for the user who is associated with the request.

This should be a uuid, hash value, or other opaque identifier. Anthropic may use this id to help detect abuse. Do not include any identifying information such as name, email address, or phone number.

Maximum length: `256`

Examples:

`"13803d75-b4b5-4c3e-b2a2-6f21399b021b"`

[​](https://docs.anthropic.com/en/api/messages#body-service-tier)

service\_tier

enum<string>

Determines whether to use priority capacity (if available) or standard capacity for this request.

Anthropic offers different levels of service for your API requests. See [service-tiers](https://docs.anthropic.com/en/api/service-tiers) for details.

Available options:

`auto`,

`standard_only`

[​](https://docs.anthropic.com/en/api/messages#body-stop-sequences)

stop\_sequences

string\[\]

Custom text sequences that will cause the model to stop generating.

Our models will normally stop when they have naturally completed their turn, which will result in a response `stop_reason` of `"end_turn"`.

If you want the model to stop generating when it encounters custom strings of text, you can use the `stop_sequences` parameter. If the model encounters one of the custom sequences, the response `stop_reason` value will be `"stop_sequence"` and the response `stop_sequence` value will contain the matched stop sequence.

[​](https://docs.anthropic.com/en/api/messages#body-stream)

stream

boolean

Whether to incrementally stream the response using server-sent events.

See [streaming](https://docs.anthropic.com/en/api/messages-streaming) for details.

[​](https://docs.anthropic.com/en/api/messages#body-system)

system

stringobject\[\]

System prompt.

A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role. See our [guide to system prompts](https://docs.anthropic.com/en/docs/system-prompts).

Examples:

```json
[\
  {\
    "text": "Today's date is 2024-06-01.",\
    "type": "text"\
  }\
]

```

`"Today's date is 2023-01-01."`

[​](https://docs.anthropic.com/en/api/messages#body-temperature)

temperature

number

Amount of randomness injected into the response.

Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to `1.0` for creative and generative tasks.

Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

Required range: `0 <= x <= 1`

Examples:

`1`

[​](https://docs.anthropic.com/en/api/messages#body-thinking)

thinking

object

Configuration for enabling Claude's extended thinking.

When enabled, responses include `thinking` content blocks showing Claude's thinking process before the final answer. Requires a minimum budget of 1,024 tokens and counts towards your `max_tokens` limit.

See [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for details.

- Enabled
- Disabled

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-thinking-budget-tokens)

thinking.budget\_tokens

integer

required

Determines how many tokens Claude can use for its internal reasoning process. Larger budgets can enable more thorough analysis for complex problems, improving response quality.

Must be ≥1024 and less than `max_tokens`.

See [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for details.

Required range: `x >= 1024`

[​](https://docs.anthropic.com/en/api/messages#body-thinking-type)

thinking.type

enum<string>

required

Available options:

`enabled`

[​](https://docs.anthropic.com/en/api/messages#body-tool-choice)

tool\_choice

object

How the model should use the provided tools. The model can use a specific tool, any available tool, decide by itself, or not use tools at all.

- Auto
- Any
- Tool
- None

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-tool-choice-type)

tool\_choice.type

enum<string>

required

Available options:

`auto`

[​](https://docs.anthropic.com/en/api/messages#body-tool-choice-disable-parallel-tool-use)

tool\_choice.disable\_parallel\_tool\_use

boolean

Whether to disable parallel tool use.

Defaults to `false`. If set to `true`, the model will output at most one tool use.

[​](https://docs.anthropic.com/en/api/messages#body-tools)

tools

object\[\]

Definitions of tools that the model may use.

If you include `tools` in your API request, the model may return `tool_use` content blocks that represent the model's use of those tools. You can then run those tools using the tool input generated by the model and then optionally return results back to the model using `tool_result` content blocks.

There are two types of tools: **client tools** and **server tools**. The behavior described below applies to client tools. For [server tools](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#server-tools), see their individual documentation as each has its own behavior (e.g., the [web search tool](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool)).

Each tool definition includes:

- `name`: Name of the tool.
- `description`: Optional, but strongly-recommended description of the tool.
- `input_schema`: [JSON schema](https://json-schema.org/draft/2020-12) for the tool `input` shape that the model will produce in `tool_use` output content blocks.

For example, if you defined `tools` as:

```json
[\
  {\
    "name": "get_stock_price",\
    "description": "Get the current stock price for a given ticker symbol.",\
    "input_schema": {\
      "type": "object",\
      "properties": {\
        "ticker": {\
          "type": "string",\
          "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."\
        }\
      },\
      "required": ["ticker"]\
    }\
  }\
]

```

And then asked the model "What's the S&P 500 at today?", the model might produce `tool_use` content blocks in the response like this:

```json
[\
  {\
    "type": "tool_use",\
    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",\
    "name": "get_stock_price",\
    "input": { "ticker": "^GSPC" }\
  }\
]

```

You might then run your `get_stock_price` tool with `{"ticker": "^GSPC"}` as an input, and return the following back to the model in a subsequent `user` message:

```json
[\
  {\
    "type": "tool_result",\
    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",\
    "content": "259.75 USD"\
  }\
]

```

Tools can be used for workflows that include running client-side tools and functions, or more generally whenever you want the model to produce a particular JSON structure of output.

See our [guide](https://docs.anthropic.com/en/docs/tool-use) for more details.

- Custom tool
- Bash tool (2024-10-22)
- Bash tool (2025-01-24)
- Code execution tool (2025-05-22)
- Computer use tool (2024-01-22)
- Computer use tool (2025-01-24)
- Text editor tool (2024-10-22)
- Text editor tool (2025-01-24)
- Text editor tool (2025-04-29)
- TextEditor\_20250728
- Web search tool (2025-03-05)

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-tools-name)

tools.name

string

required

Name of the tool.

This is how the tool will be called by the model and in `tool_use` blocks.

Required string length: `1 - 128`

[​](https://docs.anthropic.com/en/api/messages#body-tools-input-schema)

tools.input\_schema

object

required

[JSON schema](https://json-schema.org/draft/2020-12) for this tool's input.

This defines the shape of the `input` that your tool accepts and that the model will produce.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-tools-input-schema-type)

tools.input\_schema.type

enum<string>

required

Available options:

`object`

[​](https://docs.anthropic.com/en/api/messages#body-tools-input-schema-properties)

tools.input\_schema.properties

object \| null

[​](https://docs.anthropic.com/en/api/messages#body-tools-input-schema-required)

tools.input\_schema.required

string\[\] \| null

[​](https://docs.anthropic.com/en/api/messages#body-tools-type)

tools.type

enum<string> \| null

Available options:

`custom`

[​](https://docs.anthropic.com/en/api/messages#body-tools-description)

tools.description

string

Description of what this tool does.

Tool descriptions should be as detailed as possible. The more information that the model has about what the tool is and how to use it, the better it will perform. You can use natural language descriptions to reinforce important aspects of the tool input JSON schema.

Examples:

`"Get the current weather in a given location"`

[​](https://docs.anthropic.com/en/api/messages#body-tools-cache-control)

tools.cache\_control

object \| null

Create a cache control breakpoint at this content block.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#body-tools-cache-control-type)

tools.cache\_control.type

enum<string>

required

Available options:

`ephemeral`

[​](https://docs.anthropic.com/en/api/messages#body-tools-cache-control-ttl)

tools.cache\_control.ttl

enum<string>

The time-to-live for the cache control breakpoint.

This may be one the following values:

- `5m`: 5 minutes
- `1h`: 1 hour

Defaults to `5m`.

Available options:

`5m`,

`1h`

Examples:

```json
{
  "description": "Get the current weather in a given location",
  "input_schema": {
    "properties": {
      "location": {
        "description": "The city and state, e.g. San Francisco, CA",
        "type": "string"
      },
      "unit": {
        "description": "Unit for the output - one of (celsius, fahrenheit)",
        "type": "string"
      }
    },
    "required": ["location"],
    "type": "object"
  },
  "name": "get_weather"
}

```

[​](https://docs.anthropic.com/en/api/messages#body-top-k)

top\_k

integer

Only sample from the top K options for each subsequent token.

Used to remove "long tail" low probability responses. [Learn more technical details here](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277).

Recommended for advanced use cases only. You usually only need to use `temperature`.

Required range: `x >= 0`

Examples:

`5`

[​](https://docs.anthropic.com/en/api/messages#body-top-p)

top\_p

number

Use nucleus sampling.

In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by `top_p`. You should either alter `temperature` or `top_p`, but not both.

Recommended for advanced use cases only. You usually only need to use `temperature`.

Required range: `0 <= x <= 1`

Examples:

`0.7`

#### Response

200

2004XX

application/json

Message object.

[​](https://docs.anthropic.com/en/api/messages#response-id)

id

string

required

Unique object identifier.

The format and length of IDs may change over time.

Examples:

`"msg_013Zva2CMHLNnXjNJJKqJ2EF"`

[​](https://docs.anthropic.com/en/api/messages#response-type)

type

enum<string>

default:message

required

Object type.

For Messages, this is always `"message"`.

Available options:

`message`

[​](https://docs.anthropic.com/en/api/messages#response-role)

role

enum<string>

default:assistant

required

Conversational role of the generated message.

This will always be `"assistant"`.

Available options:

`assistant`

[​](https://docs.anthropic.com/en/api/messages#response-content)

content

object\[\]

required

Content generated by the model.

This is an array of content blocks, each of which has a `type` that determines its shape.

Example:

```json
[{"type": "text", "text": "Hi, I'm Claude."}]

```

If the request input `messages` ended with an `assistant` turn, then the response `content` will continue directly from that last turn. You can use this to constrain the model's output.

For example, if the input `messages` were:

```json
[\
  {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},\
  {"role": "assistant", "content": "The best answer is ("}\
]

```

Then the response `content` might be:

```json
[{"type": "text", "text": "B)"}]

```

- Thinking
- Redacted thinking
- Tool use
- Server tool use
- Web search tool result
- Code execution tool result
- MCP tool use
- MCP tool result
- Container upload

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#response-content-signature)

content.signature

string

required

[​](https://docs.anthropic.com/en/api/messages#response-content-thinking)

content.thinking

string

required

[​](https://docs.anthropic.com/en/api/messages#response-content-type)

content.type

enum<string>

default:thinking

required

Available options:

`thinking`

Examples:

```json
[\
  {\
    "text": "Hi! My name is Claude.",\
    "type": "text"\
  }\
]

```

[​](https://docs.anthropic.com/en/api/messages#response-model)

model

string

required

The model that handled the request.

Required string length: `1 - 256`

Examples:

`"claude-sonnet-4-20250514"`

[​](https://docs.anthropic.com/en/api/messages#response-stop-reason)

stop\_reason

enum<string> \| null

required

The reason that we stopped.

This may be one the following values:

- `"end_turn"`: the model reached a natural stopping point
- `"max_tokens"`: we exceeded the requested `max_tokens` or the model's maximum
- `"stop_sequence"`: one of your provided custom `stop_sequences` was generated
- `"tool_use"`: the model invoked one or more tools
- `"pause_turn"`: we paused a long-running turn. You may provide the response back as-is in a subsequent request to let the model continue.
- `"refusal"`: when streaming classifiers intervene to handle potential policy violations

In non-streaming mode this value is always non-null. In streaming mode, it is null in the `message_start` event and non-null otherwise.

Available options:

`end_turn`,

`max_tokens`,

`stop_sequence`,

`tool_use`,

`pause_turn`,

`refusal`

[​](https://docs.anthropic.com/en/api/messages#response-stop-sequence)

stop\_sequence

string \| null

required

Which custom stop sequence was generated, if any.

This value will be a non-null string if one of your custom stop sequences was generated.

[​](https://docs.anthropic.com/en/api/messages#response-usage)

usage

object

required

Billing and rate-limit usage.

Anthropic's API bills and rate-limits by token counts, as tokens represent the underlying cost to our systems.

Under the hood, the API transforms requests into a format suitable for the model. The model's output then goes through a parsing stage before becoming an API response. As a result, the token counts in `usage` will not match one-to-one with the exact visible content of an API request or response.

For example, `output_tokens` will be non-zero, even for an empty string response from Claude.

Total input tokens in a request is the summation of `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#response-usage-cache-creation)

usage.cache\_creation

object \| null

required

Breakdown of cached tokens by TTL

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#response-usage-cache-creation-ephemeral-1h-input-tokens)

usage.cache\_creation.ephemeral\_1h\_input\_tokens

integer

default:0

required

The number of input tokens used to create the 1 hour cache entry.

Required range: `x >= 0`

[​](https://docs.anthropic.com/en/api/messages#response-usage-cache-creation-ephemeral-5m-input-tokens)

usage.cache\_creation.ephemeral\_5m\_input\_tokens

integer

default:0

required

The number of input tokens used to create the 5 minute cache entry.

Required range: `x >= 0`

[​](https://docs.anthropic.com/en/api/messages#response-usage-cache-creation-input-tokens)

usage.cache\_creation\_input\_tokens

integer \| null

required

The number of input tokens used to create the cache entry.

Required range: `x >= 0`

Examples:

`2051`

[​](https://docs.anthropic.com/en/api/messages#response-usage-cache-read-input-tokens)

usage.cache\_read\_input\_tokens

integer \| null

required

The number of input tokens read from the cache.

Required range: `x >= 0`

Examples:

`2051`

[​](https://docs.anthropic.com/en/api/messages#response-usage-input-tokens)

usage.input\_tokens

integer

required

The number of input tokens which were used.

Required range: `x >= 0`

Examples:

`2095`

[​](https://docs.anthropic.com/en/api/messages#response-usage-output-tokens)

usage.output\_tokens

integer

required

The number of output tokens which were used.

Required range: `x >= 0`

Examples:

`503`

[​](https://docs.anthropic.com/en/api/messages#response-usage-server-tool-use)

usage.server\_tool\_use

object \| null

required

The number of server tool requests.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#response-usage-server-tool-use-web-search-requests)

usage.server\_tool\_use.web\_search\_requests

integer

default:0

required

The number of web search tool requests.

Required range: `x >= 0`

Examples:

`0`

[​](https://docs.anthropic.com/en/api/messages#response-usage-service-tier)

usage.service\_tier

enum<string> \| null

required

If the request used the priority, standard, or batch tier.

Available options:

`standard`,

`priority`,

`batch`

[​](https://docs.anthropic.com/en/api/messages#response-container)

container

object \| null

required

Information about the container used in this request.

This will be non-null if a container tool (e.g. code execution) was used.

Showchild attributes

[​](https://docs.anthropic.com/en/api/messages#response-container-expires-at)

container.expires\_at

string

required

The time at which the container will expire.

[​](https://docs.anthropic.com/en/api/messages#response-container-id)

container.id

string

required

Identifier for the container used in this request

Was this page helpful?

YesNo

[Beta headers](https://docs.anthropic.com/en/api/beta-headers) [Count Message tokens](https://docs.anthropic.com/en/api/messages-count-tokens)

cURL

Python

JavaScript

PHP

Go

Java

Copy

```
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [\
        {"role": "user", "content": "Hello, world"}\
    ]
}'
```

200

4XX

Copy

```
{
  "content": [\
    {\
      "text": "Hi! My name is Claude.",\
      "type": "text"\
    }\
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-sonnet-4-20250514",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503
  }
}
```