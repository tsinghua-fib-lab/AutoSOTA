# SimpleTool Skill — Real-Time AI Application Development

> **This is a skill file.** Feed it to any AI coding assistant (Claude, Gemini, GPT, Cursor, etc.) as context, then describe the app you want. The AI will generate a working SimpleTool-powered application.
>
> Example prompt: *"Read the attached SimpleTool skill, then build me a Pong game where AI controls one paddle in real-time."*

---

## 1. What is SimpleTool?

SimpleTool is a **multi-head parallel decoding** server for real-time LLM function calling. It runs on vLLM and decodes function name + arguments simultaneously instead of sequentially.

```
Traditional:  function → arg1 → arg2 → ...  (sequential, ~200-500ms)
SimpleTool:   [function, arg1, arg2, ...]    (parallel,   ~25-60ms)
```

**Application domains**: game AI, robotic arm control, digital human animation, IoT automation — anything that needs < 100ms LLM decision-making.

## 2. Server API

Server default: `http://localhost:8899`

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check, returns `{status, version, model}` |
| POST | `/v1/function_call` | Multi-head parallel function call |

### Request Format (v2)
```javascript
{
  messages: [{role: 'user', content: 'your query'}],
  tools: [...],                // OpenAI-format tool definitions
  system: "domain prompt",     // Domain-specific system prompt (v2)
  environment: [...],          // Current state info (string array, optional)
  history: [...],              // Action history (string array, max 6)
  include_content_head: false  // Whether to generate <content> head
}
```

The `system` field lets you inject a domain-specific system prompt (e.g., "You are a robotic arm controller"). If omitted, the server uses a generic default. The `environment` field is optional context folded into the user message.

### Response Format
```javascript
{
  success: true,
  function: "move",
  args: {direction: "up", speed: "fast"},   // Named args (param names from tool def)
  heads: {                                   // Raw per-head output
    function: "move",
    arg1: "up",
    arg2: "fast",
    arg3: "<|null|>"
  },
  content: null,       // Only if include_content_head was true
  latency_ms: 35.2
}
```

## 3. Dynamic Head Count (Critical for Latency!)

**The server automatically prunes unused heads.** If your tools have at most 2 parameters, only 3 heads are spawned (`<function>`, `<arg1>`, `<arg2>`), not 8. This saves ~40% latency.

```
Active heads = [<function>] + [<arg1>...<argN>]
where N = max parameter count across all tool definitions
```

**Design tip**: Keep your tools to 1–3 parameters when possible. Fewer params = fewer heads = lower latency.

## 4. Tool Definition

### Constraints
- Maximum **6 arguments** per function (arg1–arg6)
- Arguments map to `arg1, arg2, ...` in the order defined in `properties`
- Server auto-converts types: numeric strings → int/float, otherwise lowercase string
- Use `enum` to constrain options — this dramatically improves accuracy

### Template
```javascript
const TOOLS = [{
  type: "function",
  function: {
    name: "action_name",
    description: "Clear, concise — what this action does and when to use it",
    parameters: {
      type: "object",
      properties: {
        param1: {
          type: "string",
          enum: ["opt_a", "opt_b", "opt_c"],  // Constrain! Improves accuracy
          description: "What this param controls"
        },
        param2: {
          type: "number",
          description: "Numeric value with unit, e.g. 'Force in Newtons'"
        }
      },
      required: ["param1"]
    }
  }
}];
```

### Multi-Tool Example (Game)
```javascript
const TOOLS = [
  {type:"function", function:{name:"move",    description:"Move unit to position", parameters:{type:"object", properties:{unit:{type:"string"}, target:{type:"string", enum:["north","south","east","west"]}}}}},
  {type:"function", function:{name:"attack",  description:"Attack enemy",          parameters:{type:"object", properties:{unit:{type:"string"}, target:{type:"string"}}}}},
  {type:"function", function:{name:"retreat",  description:"Pull back unit",        parameters:{type:"object", properties:{unit:{type:"string"}}}}},
  {type:"function", function:{name:"pass",     description:"Do nothing this turn",  parameters:{type:"object", properties:{}}}}
];
// Max params = 2 → only 3 heads spawned
```

## 5. Query Design

### Principles
1. **Be imperative** — tell the model what to decide, not just describe state
2. **Include decision context** — "Ball is BELOW paddle, intercept it" not "Ball y=250"
3. **List valid options** — "Choose: up/down/stay"
4. **Keep it short** — shorter query = faster prefill

### Good vs Bad
```
✅ "Ball 50px BELOW paddle, approaching fast. Move DOWN to intercept. Choose: up/down/stay"
❌ "Ball position: 250, Paddle position: 200. What should I do?"

✅ "Red gear at (300,150,50). Move arm there slowly for pickup."
❌ "There is a gear somewhere on the table. The arm needs to go to it."

✅ "Stream starting, viewers saying hello. Greet them warmly."
❌ "Viewers are in the chat. Do something appropriate."
```

### Environment & History
```javascript
// Environment: current state as key=value strings
const env = [
  `ball_y=${ballY}`,
  `paddle_y=${paddleY}`,
  `gap=${gap}`,
  `approaching=true`
];

// History: recent actions (max 6, server trims automatically)
const history = [
  "move(up)", "move(up)", "stay()"
];
```

### Domain System Prompts (v2)
For v2 server, set a domain-specific system prompt:
```javascript
// Game AI
const SYSTEM = "You are the AI controller for a Pong game. Move the paddle to intercept the ball. React quickly.";

// Robotic arm
const SYSTEM = "You are the voice controller for a 6-axis robotic arm. Convert commands to precise function calls. Coordinates in mm.";

// Digital human
const SYSTEM = "You are the animation controller for a virtual streamer. Convert director instructions to expression and speech calls.";
```

## 6. Frontend Code Standards

### Required: Type-Safe Value Extraction
```javascript
// Values in args may be int, not string — always coerce
function safeStr(v) {
  if (v === null || v === undefined) return '';
  return String(v).trim().toLowerCase();
}

// Extract with args (named) first, heads (positional) as fallback
let direction = safeStr(d.args?.direction) || safeStr(d.heads?.arg1);
```

### Required: Validate Return Values
```javascript
const VALID = ['up', 'down', 'stay'];
if (!VALID.includes(direction)) {
  console.warn(`Invalid: "${direction}", fallback to stay`);
  direction = 'stay';
}
```

### Required: Error Handling with Fallback
```javascript
async function callAI() {
  try {
    const r = await fetch(SERVER_URL + '/v1/function_call', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(request)
    });
    const data = await r.json();
    if (!data.success) throw new Error(data.error);
    applyAction(data);
  } catch (e) {
    console.error('[AI] Failed:', e);
    applyFallbackAI();  // MUST have fallback — never freeze the app
  }
}
```

### Required: Logging
```javascript
console.log(`[Game] Query: ${query}`);
console.log(`[Game] → ${data.function}(${JSON.stringify(data.args)}) ${data.latency_ms.toFixed(0)}ms`);
```

### Recommended: Debug UI Overlay
Show in a corner of your app: current query, raw response, latency (current + rolling average).

## 7. Game Loop Pattern

**Decouple AI from rendering.** The AI loop runs at 10–16 Hz; the render loop runs at 60 fps.

```javascript
const AI_INTERVAL = 100;  // 100ms = 10 Hz
let aiPending = false;

// Render loop (60fps) — never blocks on AI
function gameLoop() {
  update();
  render();
  requestAnimationFrame(gameLoop);
}

// AI loop (async, non-blocking)
async function aiLoop() {
  if (aiPending) return;
  aiPending = true;
  await callAI();
  aiPending = false;
}

setInterval(aiLoop, AI_INTERVAL);
gameLoop();
```

## 8. FCClient Template

Drop-in client class for any HTML/JS application:

```javascript
class FCClient {
  constructor(url = 'http://localhost:8899') {
    this.url = url.replace(/\/$/, '');
  }

  async health() {
    try {
      const r = await fetch(`${this.url}/health`, {signal: AbortSignal.timeout(3000)});
      const d = await r.json();
      return {ok: d.loaded === true || d.status === 'ok', version: d.version};
    } catch (e) {
      return {ok: false};
    }
  }

  async call({query, tools, system, env, history, includeContent = false}) {
    const t0 = performance.now();
    try {
      const r = await fetch(`${this.url}/v1/function_call`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          messages: [{role: 'user', content: query}],
          tools,
          system,                              // v2: domain system prompt
          environment: env,
          history,
          include_content_head: includeContent
        })
      });
      const d = await r.json();
      return {...d, wall_ms: performance.now() - t0};
    } catch (e) {
      return {success: false, error: e.message, wall_ms: performance.now() - t0};
    }
  }
}
```

Usage:
```javascript
const ai = new FCClient('http://localhost:8899');

const result = await ai.call({
  query: "Ball is BELOW. Move down. Choose: up/down/stay",
  tools: TOOLS,
  system: "You are a Pong AI. Move paddle to intercept ball.",
  env: ["ball_y=300", "paddle_y=200", "gap=100"],
  history: ["move(down)", "move(down)"]
});

if (result.success) {
  console.log(`${result.function}(${JSON.stringify(result.args)}) in ${result.latency_ms}ms`);
}
```

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| AI stuck / no movement | Query too vague | Add decision hints: "Move DOWN to intercept" |
| `.trim is not a function` | `args` values may be int | Use `String(v)` before `.trim()` |
| High latency (>100ms) | Too many heads / long query | Reduce tool params, shorten query/env |
| Wrong function called | Ambiguous tool descriptions | Add `enum`, improve `description` fields |
| `<|null|>` in all args | Model confused | Check tool param order matches expectations |

---

**Skill Version**: 2.0 — Supports v1/v2 server, multi-domain (game, robotics, avatar)  
**Last Updated**: 2026-03
