#!/usr/bin/env python3
"""
test_server.py — Hit running rt_server /v1/function_call with 4 scenarios
Usage:  python test_server.py [--url http://localhost:8899]
"""

import argparse, json, time, sys, requests

# ==================== Test Scenarios ====================
SCENARIOS = [
    # ── 1. Game: Tower Defense (from benchmark) ──
    {
        "name": "Game — Tower Defense",
        "desc": "use_skill(Amiya)",
        "expected_fn": "use_skill",
        "request": {
            "messages": [{"role": "user", "content":
                "Wave 5, BOSS appeared, 8 enemies remaining\n"
                "Operators: Blaze(north,HP50%,skill ready) Amiya(center,HP90%,skill ready)\n"
                "Enemy direction: concentrated north\n\n"
                "Amiya use skill now"
            }],
            "tools": [
                {"type":"function","function":{"name":"move","description":"Move a deployed operator to a new position on the battlefield. Use this when the player wants to reposition a unit to a different lane or strategic point.","parameters":{"type":"object","properties":{
                    "unit_id":{"type":"string","description":"The name of the operator to move. Must match one of the currently deployed operators shown in the battlefield state. Supports fuzzy matching for ASR input, e.g. 'blaze', 'Blaze', 'BLAZE' all refer to the same operator."},
                    "target":{"type":"string","description":"The destination position on the battlefield grid. Must be one of: 'north' (top lane), 'south' (bottom lane), 'east' (right/enemy side), 'west' (left/base side), 'center' (middle area). Choose based on the player's spoken direction."}},"required":["unit_id","target"]}}},
                {"type":"function","function":{"name":"use_skill","description":"Activate the special skill of a deployed operator. Each operator has a unique skill that can be triggered when the skill gauge is ready. The skill effect depends on the operator type (e.g. AoE damage, healing, buff).","parameters":{"type":"object","properties":{
                    "unit_id":{"type":"string","description":"The name of the operator whose skill should be activated. The operator must be currently deployed on the battlefield and have their skill ready (skill gauge full). Supports fuzzy name matching for ASR input."},
                    "skill_id":{"type":"string","description":"Optional skill identifier when an operator has multiple skills. If the operator only has one skill or the player did not specify which skill, this can be omitted. Format: 's1', 's2', 's3' for skill slot 1/2/3."}},"required":["unit_id"]}}},
                {"type":"function","function":{"name":"retreat","description":"Withdraw a single operator from the battlefield back to the reserve bench. The operator's redeployment timer starts after retreat. Use when the player wants to pull back a specific unit to save them or free up a deployment slot.","parameters":{"type":"object","properties":{
                    "unit_id":{"type":"string","description":"The name of the operator to retreat. Must be currently deployed on the battlefield. After retreat, this operator enters cooldown before they can be redeployed. Supports fuzzy name matching for ASR input."}},"required":["unit_id"]}}},
                {"type":"function","function":{"name":"set_stance","description":"Change the combat behavior mode of a deployed operator. This affects how the operator selects targets and whether they prioritize attacking or surviving.","parameters":{"type":"object","properties":{
                    "unit_id":{"type":"string","description":"The name of the operator whose stance should be changed. Must be currently deployed on the battlefield. Supports fuzzy name matching for ASR input."},
                    "stance":{"type":"string","description":"The behavior mode to set. Must be one of: 'aggressive' (prioritize attacking nearest enemy, maximize DPS), 'defensive' (prioritize blocking and damage reduction, focus on survival), 'hold' (stay in position and only attack enemies in range, do not chase)."}},"required":["unit_id","stance"]}}},
                {"type":"function","function":{"name":"retreat_all","description":"Emergency retreat of all currently deployed operators from the battlefield at once. Use only when the player explicitly requests a full withdrawal, typically in dire situations. All operators enter redeployment cooldown simultaneously.","parameters":{"type":"object","properties":{}}}},
                {"type":"function","function":{"name":"pass","description":"Take no action this turn. Use when the player's command has already been fulfilled in history, or when the player explicitly says to wait, skip, or do nothing. Also use when the voice input is ambiguous and no clear command can be extracted.","parameters":{"type":"object","properties":{}}}}
            ],
            "system": "You are the voice command interpreter for a real-time tower defense game. The player issues orders by voice. You convert ASR-transcribed commands into function calls.\n\nRules:\n- One function call per command\n- Fuzzy match operator names\n- Positions: north, south, east, west, center\n- If all tasks in history are done, call pass",
            "history": []
        }
    },
    # ── 2. Robotic Arm — Assembly (from benchmark) ──
    {
        "name": "Robotic Arm — Assembly",
        "desc": "move_to(300,150,50,slow)",
        "expected_fn": "move_to",
        "request": {
            "messages": [{"role": "user", "content":
                "Arm at home (0,0,500), gripper open\n"
                "Workpiece: red gear at (300,150,50), target tray at (600,0,80)\n\n"
                "Move to the red gear position slowly"
            }],
            "tools": [
                {"type":"function","function":{"name":"move_to","description":"Move the robotic arm end-effector (tool center point) to a specified 3D coordinate in the workspace. The arm plans a collision-free path from its current position to the target. Optionally control movement speed for precision tasks.","parameters":{"type":"object","properties":{
                    "x":{"type":"number","description":"Target X coordinate in millimeters, relative to the robot base frame origin. Positive X points forward (away from the robot base). Valid range depends on arm reach, typically -800 to 800 mm."},
                    "y":{"type":"number","description":"Target Y coordinate in millimeters, relative to the robot base frame origin. Positive Y points to the left when facing the robot. Valid range depends on arm reach, typically -800 to 800 mm."},
                    "z":{"type":"number","description":"Target Z coordinate in millimeters, relative to the robot base frame origin (table surface = 0). Positive Z points upward. Must be >= 0 to avoid collision with the work surface. Typical range: 0 to 500 mm."},
                    "speed":{"type":"string","description":"Movement speed profile for the path. 'slow' (25% max velocity) for precision placement and delicate parts, 'normal' (50% max velocity) for standard pick-and-place, 'fast' (100% max velocity) for rapid repositioning when precision is not critical. Default: 'normal'."}},"required":["x","y","z"]}}},
                {"type":"function","function":{"name":"grip","description":"Close the gripper jaws to grasp an object at the current end-effector position. The gripper applies the specified force and holds it. Must be called after positioning the arm above/around the target object.","parameters":{"type":"object","properties":{
                    "force":{"type":"number","description":"Gripping force in Newtons applied by the gripper jaws. Choose based on object fragility: 10N for light/fragile items (electronics, thin plastic), 50N for medium items (standard gears, metal parts), 100N for heavy/robust items (large castings, steel blocks). Excessive force may damage delicate workpieces."}},"required":["force"]}}},
                {"type":"function","function":{"name":"release","description":"Open the gripper jaws to release the currently held object. The gripper fully opens to its maximum width. Should be called after positioning the arm at the target placement location. Ensure the object is at a safe height above the surface before releasing.","parameters":{"type":"object","properties":{}}}},
                {"type":"function","function":{"name":"rotate","description":"Rotate the end-effector around a specified axis without changing its position. Used to orient the gripper or tool for proper approach angle before grasping, or to rotate a held workpiece for assembly alignment.","parameters":{"type":"object","properties":{
                    "axis":{"type":"string","description":"The rotation axis in the end-effector frame. 'roll' rotates around the approach direction (Z-axis of tool frame, like turning a screwdriver), 'pitch' tilts the end-effector up/down (like nodding), 'yaw' swings the end-effector left/right (like shaking head). Choose based on the desired orientation change."},
                    "angle":{"type":"number","description":"Rotation angle in degrees. Positive values follow the right-hand rule around the specified axis. Typical range: -180 to 180 degrees. Small angles (< 15°) for fine adjustment, larger angles for major reorientation."}},"required":["axis","angle"]}}},
                {"type":"function","function":{"name":"home","description":"Return the robotic arm to its predefined home position (0, 0, 500) with the gripper pointing straight down and jaws open. Use as a safe starting/ending position for task sequences, or to clear the workspace. The arm takes a collision-free path at normal speed.","parameters":{"type":"object","properties":{}}}}
            ],
            "system": "You are the voice controller for an industrial 6-axis robotic arm. You convert spoken commands into function calls.\n\nRules:\n- One function call per command\n- Coordinates in mm, angles in degrees\n- Gripper force: light=10N, medium=50N, heavy=100N\n- Speed: slow/normal/fast",
            "history": []
        }
    },
    # ── 3. Digital Human — Streamer (from benchmark) ──
    {
        "name": "Digital Human — Streamer",
        "desc": "speak(welcome,cheerful)",
        "expected_fn": "speak",
        "request": {
            "messages": [{"role": "user", "content":
                "Stream just started, viewers flooding in\n"
                "Chat: \"Hello streamer!\" \"Good evening!\"\n"
                "Director: greet the audience warmly, say welcome and look at camera"
            }],
            "tools": [
                {"type":"function","function":{"name":"set_expression","description":"Set the facial expression of the digital human avatar. Controls the blend shapes for eyes, eyebrows, and mouth to display the target emotion. The expression persists until changed by another set_expression call or overridden by a speak animation.","parameters":{"type":"object","properties":{
                    "emotion":{"type":"string","description":"The target facial expression to display. Must be one of: 'happy' (smile, raised cheeks), 'sad' (downturned mouth, drooping eyebrows), 'surprised' (wide eyes, raised eyebrows, open mouth), 'angry' (furrowed brows, tight lips), 'neutral' (relaxed default face), 'thinking' (slightly furrowed brow, eyes looking up/away, subtle lip purse)."},
                    "intensity":{"type":"number","description":"The strength of the facial expression blend, from 0.0 (barely visible, subtle hint) to 1.0 (maximum exaggeration, full expression). Recommended: 0.3-0.5 for natural conversation, 0.6-0.8 for reactive moments, 0.9-1.0 for comedic or dramatic emphasis."}},"required":["emotion","intensity"]}}},
                {"type":"function","function":{"name":"speak","description":"Make the digital human speak the given text with lip-sync animation and appropriate facial expressions. The TTS engine converts text to audio while the avatar performs real-time viseme-based lip synchronization. The tone parameter affects both voice prosody and accompanying facial micro-expressions.","parameters":{"type":"object","properties":{
                    "text":{"type":"string","description":"The speech content for the digital human to say aloud. Should be natural conversational language appropriate for a live stream context. Keep sentences concise (under 50 characters preferred for real-time responsiveness). May include casual expressions, emoji descriptions, or audience interaction phrases."},
                    "tone":{"type":"string","description":"The vocal tone and emotional coloring of the speech delivery. Must be one of: 'cheerful' (upbeat, warm, higher pitch, for greetings and positive moments), 'calm' (steady, soothing, moderate pace, for explanations and transitions), 'serious' (lower pitch, measured pace, for important announcements), 'excited' (high energy, faster pace, emphasis peaks, for reactions and hype moments)."}},"required":["text","tone"]}}},
                {"type":"function","function":{"name":"gesture","description":"Trigger a pre-defined body gesture animation on the digital human avatar. The gesture plays once and blends back to the idle pose. Can be combined with speak or set_expression for more natural multi-channel communication.","parameters":{"type":"object","properties":{
                    "type":{"type":"string","description":"The gesture animation to play. Must be one of: 'wave' (friendly hand wave, for greetings and farewells), 'nod' (head nod, to show agreement or acknowledgment), 'shake_head' (head shake, to express disagreement or disbelief), 'bow' (respectful bow, for gratitude or formal greeting), 'point' (index finger pointing forward, to direct attention), 'thumbs_up' (approval gesture, for positive feedback), 'clap' (both hands clapping, for celebration or applause)."}},"required":["type"]}}},
                {"type":"function","function":{"name":"look_at","description":"Direct the digital human's eye gaze and subtle head orientation toward a specified target. Creates natural eye contact or directional attention. The gaze shift is smoothly interpolated over ~200ms for realistic movement.","parameters":{"type":"object","properties":{
                    "target":{"type":"string","description":"The gaze target direction. Must be one of: 'camera' (look directly at the audience through the camera lens, creates eye contact with viewers), 'left' (glance to the left side of the screen, e.g. toward a chat panel or co-host), 'right' (glance to the right, e.g. toward a game screen or secondary content), 'up' (look upward, conveys thinking or reacting to something above), 'down' (look downward, conveys reading chat, shyness, or sadness)."}},"required":["target"]}}},
                {"type":"function","function":{"name":"idle","description":"Return the digital human to its default idle animation loop. Resets any active expression to neutral, stops ongoing gestures, and returns gaze to a soft forward direction with natural idle micro-movements (subtle breathing, occasional blinks, slight sway). Use during pauses or transitions between active segments.","parameters":{"type":"object","properties":{}}}}
            ],
            "system": "You are the expression controller for a virtual digital human streamer. You convert director instructions into animation function calls.\n\nRules:\n- One function call per instruction\n- Emotion intensity: 0.0-1.0\n- Speech text should be natural\n- Tone: cheerful/calm/serious/excited",
            "history": []
        }
    },
    # ── 4. Neon Arena (what HTML actually sends, legacy env style) ──
    {
        "name": "Neon Arena — Legacy HTML",
        "desc": "fire(left) or move(left)",
        "expected_fn": "move",
        "request": {
            "messages": [{"role": "user", "content":
                "Arena 900x600. FIRE or Call move(dir) or fire(dir). dir:up/down/left/right"
            }],
            "tools": [
                {"type":"function","function":{"name":"move","description":"Move the player's spaceship in the specified direction by one step on the 900x600 arena grid. Use to reposition for better firing angle, dodge incoming bullets, or approach/retreat from enemies.","parameters":{"type":"object","properties":{
                    "direction":{"type":"string","enum":["up","down","left","right"],"description":"The movement direction on the arena. 'up' decreases Y (toward top edge), 'down' increases Y (toward bottom edge), 'left' decreases X (toward left edge), 'right' increases X (toward right edge). Choose based on tactical positioning relative to the player and arena walls."}},"required":["direction"]}}},
                {"type":"function","function":{"name":"fire","description":"Fire a bullet from the spaceship in the specified direction. The bullet travels in a straight line until it hits a target or exits the arena boundary. Use when aligned with the player's position on the horizontal or vertical axis for best hit probability.","parameters":{"type":"object","properties":{
                    "direction":{"type":"string","enum":["up","down","left","right"],"description":"The firing direction of the bullet. 'up' fires toward top edge, 'down' fires toward bottom edge, 'left' fires toward left edge (toward player's side), 'right' fires toward right edge. Choose based on current alignment with the player: fire horizontally when align_h=true, vertically when align_v=true."}},"required":["direction"]}}}
            ],
            "environment": ["pos=700,300","player=100,310","dist=600","align_h=true","align_v=false","cd=0","wall=no"],
            "history": ["fire(left)","move(up)","fire(left)"]
        }
    },
]


def check_health(url: str) -> dict:
    r = requests.get(f"{url}/health", timeout=5)
    return r.json()


def call_fc(url: str, req: dict) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/function_call", json=req, timeout=30)
    wall_ms = (time.perf_counter() - t0) * 1000
    d = r.json()
    d["_wall_ms"] = wall_ms
    return d


def fmt_heads(heads: dict) -> str:
    lines = []
    for k in ["function","arg1","arg2","arg3","arg4","arg5","arg6","content"]:
        if k in heads:
            v = heads[k]
            tag = "NULL" if (not v or v == "<|null|>") else v
            lines.append(f"    {k:<10} = {tag}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Test SimpleTool server")
    ap.add_argument("--url", default="http://localhost:8899")
    ap.add_argument("--rounds", type=int, default=3, help="hot rounds per scenario")
    args = ap.parse_args()

    url = args.url.rstrip("/")

    # ── Health ──
    print(f"\n{'='*65}")
    print(f"  SimpleTool Server Test")
    print(f"  Target: {url}")
    print(f"{'='*65}\n")

    try:
        h = check_health(url)
        print(f"  /health → {json.dumps(h)}")
        if not h.get("loaded") and h.get("status") != "ok":
            print("  ⚠ Model not loaded!"); sys.exit(1)
    except Exception as e:
        print(f"  ✗ Cannot connect: {e}"); sys.exit(1)

    version = h.get("version", "unknown")
    print(f"  Server version: {version}\n")

    # ── Cold start (first call warms KV cache) ──
    print(f"{'='*65}")
    print(f"  COLD START")
    print(f"{'='*65}")
    cold_ms = []
    for sc in SCENARIOS:
        r = call_fc(url, sc["request"])
        ms = r.get("latency_ms", r.get("_wall_ms", 0))
        cold_ms.append(ms)
        ok = "✓" if r.get("function", "") == sc["expected_fn"] else "✗"
        print(f"  {ok} {sc['name']:<35} {ms:7.1f}ms  → {r.get('function','?')}({r.get('args',{})})")
    print()

    # ── Hot rounds ──
    print(f"{'='*65}")
    print(f"  HOT ROUNDS (×{args.rounds})")
    print(f"{'='*65}")
    hot_ms = [[] for _ in SCENARIOS]
    for rd in range(args.rounds):
        parts = []
        for i, sc in enumerate(SCENARIOS):
            r = call_fc(url, sc["request"])
            ms = r.get("latency_ms", r.get("_wall_ms", 0))
            hot_ms[i].append(ms)
            parts.append(f"{ms:6.1f}ms")
        print(f"  Round {rd+1}: {'  '.join(parts)}")
    print()

    # ── Detailed test ──
    print(f"{'='*65}")
    print(f"  DETAILED RESULTS")
    print(f"{'='*65}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        r = call_fc(url, sc["request"])
        fn = r.get("function", "")
        ok = fn == sc["expected_fn"]
        results.append((sc, r, ok))

        status = "PASS ✓" if ok else "FAIL ✗"
        ms_server = r.get("latency_ms", 0)
        ms_wall = r.get("_wall_ms", 0)

        print(f"─── {sc['name']} ───")
        print(f"  {status}  expected={sc['expected_fn']}  got={fn}")
        print(f"  args: {json.dumps(r.get('args', {}), ensure_ascii=False)}")
        print(f"  server={ms_server:.1f}ms  wall={ms_wall:.1f}ms  overhead={ms_wall-ms_server:.1f}ms")
        if r.get("heads"):
            print(f"  heads:")
            print(fmt_heads(r["heads"]))
        if r.get("error"):
            print(f"  error: {r['error']}")
        print()

    # ── Summary ──
    n = len(results)
    passed = sum(1 for _, _, ok in results if ok)
    avg_cold = sum(cold_ms) / n
    avg_hot = sum(sum(h) for h in hot_ms) / sum(len(h) for h in hot_ms) if hot_ms else 0
    avg_detail = sum(r.get("latency_ms", 0) for _, r, _ in results) / n

    print(f"{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Server version  : {version}")
    print(f"  Accuracy        : {passed}/{n}")
    print(f"  Cold start avg  : {avg_cold:.1f}ms")
    print(f"  Hot avg         : {avg_hot:.1f}ms")
    print(f"  Detail avg      : {avg_detail:.1f}ms")
    print()
    print(f"  {'Scenario':<35} {'Cold':>7} {'Hot':>7} {'Detail':>7} {'Status':>6}")
    print(f"  {'─'*65}")
    for i, (sc, r, ok) in enumerate(results):
        havg = sum(hot_ms[i]) / len(hot_ms[i]) if hot_ms[i] else 0
        print(f"  {sc['name']:<35} {cold_ms[i]:6.1f}  {havg:6.1f}  {r.get('latency_ms',0):6.1f}  {'✓' if ok else '✗':>5}")
    print()


if __name__ == "__main__":
    main()