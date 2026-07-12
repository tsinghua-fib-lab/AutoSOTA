"""Parse eval result JSON and record score."""
import json, sys, subprocess, os

result_path = sys.argv[1]
iteration = sys.argv[2]
idea_id = sys.argv[3]
title = sys.argv[4]
status = sys.argv[5] if len(sys.argv) > 5 else "success"
notes = sys.argv[6] if len(sys.argv) > 6 else ""
is_best = sys.argv[7] if len(sys.argv) > 7 else ""

with open(result_path) as f:
    result = json.load(f)

pm = result["posterior_mean"]
psnr = pm["PSNR"]
ssim = pm["SSIM"]
lpips = pm["LPIPS"]

metrics = json.dumps({
    "PSNR": round(psnr, 2),
    "SSIM": round(ssim, 2),
    "LPIPS": round(lpips, 2),
})

cmd = [
    "/tools/record_score.sh",
    "--scores", "/autosota_artifacts/paper-4513/sota/scores.jsonl",
    "--iter", iteration,
    "--idea-id", idea_id,
    "--title", title,
    "--status", status,
    "--primary", str(round(psnr, 4)),
    "--metrics", metrics,
    "--notes", notes,
]
if is_best:
    cmd.extend(["--is-best", is_best])

os.chdir("/repo")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Recorded: PSNR={psnr:.2f} SSIM={ssim:.2f} LPIPS={lpips:.2f}")
