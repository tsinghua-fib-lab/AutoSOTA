"""
Generate DAVIS-Tracking dataset for RP-GSSM reproduction.

Uses RatInABox to simulate an agent in a rectangular arena and
DAVIS 2017 videos as background distractors.

Output: tracking_data.npz with keys: train_obs, train_states, test_obs, test_states
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# Setup
CACHE_DIR = os.environ.get("RP_GSSM_CACHE_DIR", "/autosota_cache")
OUTPUT_DIR = Path(os.environ.get("RP_GSSM_DATA_DIR", "/repo/rp_ssm/data"))
SEED = 42
FRAME_SIZE = 64
N_TRAIN = 500
N_TEST = 125
T = 100
DT = 0.05  # 50ms per step = ~20fps = ~5s per 100-frame sequence


def download_davis(cache_dir: str) -> Path:
    """Download DAVIS 2017 480p and extract to cache."""
    davis_zip = Path(cache_dir) / "DAVIS-2017-trainval-480p.zip"
    davis_dir = Path(cache_dir) / "DAVIS-2017-trainval-480p"

    if davis_dir.exists() and any(davis_dir.iterdir()):
        print(f"DAVIS already extracted at {davis_dir}")
        return davis_dir

    if not davis_zip.exists():
        url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
        print(f"Downloading DAVIS from {url} ... (~794 MB)")
        tmp_dir = Path(cache_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / "davis_download.zip"
        urllib.request.urlretrieve(url, tmp_file)
        tmp_file.rename(davis_zip)
        print(f"Downloaded to {davis_zip}")

    print(f"Extracting DAVIS to {davis_dir} ...")
    davis_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(davis_zip, "r") as zf:
        zf.extractall(davis_dir)
    print("DAVIS extraction complete")
    return davis_dir


def load_davis_frames(davis_dir: Path) -> dict:
    """Load all DAVIS video frame paths organized by video."""
    for candidate in [
        davis_dir / "DAVIS" / "JPEGImages" / "480p",
        davis_dir / "JPEGImages" / "480p",
    ]:
        if candidate.exists():
            jpeg_dir = candidate
            break
    else:
        matches = list(davis_dir.rglob("JPEGImages"))
        if matches:
            jpeg_dir = matches[0] / "480p" if (matches[0] / "480p").exists() else matches[0]
        else:
            raise FileNotFoundError(f"Could not find JPEGImages in {davis_dir}")

    videos = {}
    for video_dir in sorted(jpeg_dir.iterdir()):
        if video_dir.is_dir():
            frames = sorted(video_dir.glob("*.jpg"))
            if frames:
                videos[video_dir.name] = frames

    total_frames = sum(len(v) for v in videos.values())
    print(f"Found {len(videos)} DAVIS videos, {total_frames} total frames")
    return videos


def render_frame(env_bounds, pos, head_dir, vel, size: int) -> np.ndarray:
    """
    Render a single frame: gray arena with red dot (agent) and blue arrow (head direction).
    Returns uint8 RGB (size, size, 3).
    """
    frame = np.ones((size, size, 3), dtype=np.uint8) * 240

    # Draw arena walls
    pts = []
    for x, y in env_bounds:
        px = int(x * (size - 1))
        py = int((1 - y) * (size - 1))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(100, 100, 100), thickness=1)

    # Agent position
    ax = int(np.clip(pos[0], 0, 1) * (size - 1))
    ay = int((1 - np.clip(pos[1], 0, 1)) * (size - 1))

    # Red dot for agent
    cv2.circle(frame, (ax, ay), radius=3, color=(220, 30, 30), thickness=-1)

    # Blue arrow for head direction
    arrow_len = 8
    ex = int(ax + head_dir[0] * arrow_len)
    ey = int(ay - head_dir[1] * arrow_len)
    cv2.arrowedLine(frame, (ax, ay), (ex, ey), color=(30, 30, 220), thickness=1, tipLength=0.4)

    return frame


def generate_agent_data_ratinabox(
    num_sequences: int,
    num_timesteps: int,
    frame_size: int,
    dt: float,
    seed: int,
):
    """
    Generate agent tracking data using RatInABox.
    Creates a fresh Environment and Agent per sequence to avoid state accumulation.
    Returns: obs (N,T,H,W,3) uint8, states (N,T,6) float32
    """
    from ratinabox.Environment import Environment
    from ratinabox.Agent import Agent

    np.random.seed(seed)

    all_obs = []
    all_states = []

    env_bounds = [[0, 0], [1, 0], [1, 1], [0, 1]]

    for seq_idx in tqdm(range(num_sequences), desc="Generating agent trajectories"):
        # Create fresh Environment and Agent for each sequence
        Env = Environment(params={"boundary": env_bounds})
        Ag = Agent(
            Env,
            params={
                "dt": dt,
                "speed_mean": 0.08,
                "speed_std": 0.02,
                "rotational_velocity_std": 120 * (np.pi / 180),
                "thigmotaxis": 0.5,
                "wall_repel_distance": 0.05,
            },
        )

        frames = []
        states = []

        for t_idx in range(num_timesteps):
            # Capture current state
            pos = Ag.pos.copy()
            vel = Ag.velocity.copy()
            hd = Ag.head_direction.copy()

            state = np.concatenate([pos, vel, hd])
            states.append(state)

            frame = render_frame(env_bounds, pos, hd, vel, frame_size)
            frames.append(frame)

            # Update for next timestep
            if t_idx < num_timesteps - 1:
                Ag.update()

        all_obs.append(np.stack(frames))
        all_states.append(np.stack(states))

    obs = np.stack(all_obs).astype(np.uint8)
    states = np.stack(all_states).astype(np.float32)
    return obs, states


def overlay_davis_background(
    agent_frames: np.ndarray,
    davis_videos: dict,
    seed: int = 123,
) -> np.ndarray:
    """
    Overlay DAVIS video backgrounds onto agent frames.
    agent_frames: (N, T, H, W, 3) uint8
    Returns: (N, T, H, W, 3) uint8
    """
    rng = np.random.RandomState(seed)
    N, T_frames, H, W, C = agent_frames.shape
    video_names = list(davis_videos.keys())
    result = np.zeros_like(agent_frames)

    for seq_idx in tqdm(range(N), desc="Overlaying DAVIS backgrounds"):
        video_name = video_names[rng.randint(0, len(video_names))]
        davis_frame_paths = davis_videos[video_name]

        n_frames = len(davis_frame_paths)
        if n_frames > T_frames:
            start = rng.randint(0, n_frames - T_frames)
        else:
            start = rng.randint(0, max(1, n_frames))
        # Use modulo indexing so we always get exactly T_frames paths
        selected_paths = [davis_frame_paths[(start + i) % n_frames] for i in range(T_frames)]

        for t_idx in range(T_frames):
            davis_img = cv2.imread(str(selected_paths[t_idx]))
            if davis_img is None:
                davis_img = np.ones((H, W, 3), dtype=np.uint8) * 128
            else:
                davis_img = cv2.resize(davis_img, (W, H))
                davis_img = cv2.cvtColor(davis_img, cv2.COLOR_BGR2RGB)

            agent_img = agent_frames[seq_idx, t_idx]

            # Only agent (non-background) pixels go on top of DAVIS
            bg_mask = np.all(agent_img == 240, axis=-1)
            composite = davis_img.copy()
            composite[~bg_mask] = agent_img[~bg_mask]

            result[seq_idx, t_idx] = composite

    return result


def main():
    output_dir = OUTPUT_DIR
    datasets_dir = output_dir / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load DAVIS
    print("=" * 60)
    print("STEP 1: Loading DAVIS dataset")
    print("=" * 60)
    davis_dir = download_davis(CACHE_DIR)
    davis_videos = load_davis_frames(davis_dir)

    # Step 2: Generate agent data
    print("\n" + "=" * 60)
    print("STEP 2: Generating agent tracking data")
    print("=" * 60)
    total_seqs = N_TRAIN + N_TEST
    obs, states = generate_agent_data_ratinabox(
        num_sequences=total_seqs,
        num_timesteps=T,
        frame_size=FRAME_SIZE,
        dt=DT,
        seed=SEED,
    )
    print(f"Agent data: obs={obs.shape}, states={states.shape}")

    # Step 3: Overlay DAVIS backgrounds
    print("\n" + "=" * 60)
    print("STEP 3: Overlaying DAVIS backgrounds")
    print("=" * 60)
    obs_with_davis = overlay_davis_background(obs, davis_videos, seed=SEED + 1)

    # Step 4: Split and save
    print("\n" + "=" * 60)
    print("STEP 4: Splitting and saving")
    print("=" * 60)

    # Last N_TEST sequences are test data
    train_obs = obs_with_davis[N_TEST:]
    train_states = states[N_TEST:]
    test_obs = obs_with_davis[:N_TEST]
    test_states = states[:N_TEST]

    output_path = datasets_dir / "tracking_data.npz"
    print(f"Saving to {output_path}")
    np.savez_compressed(
        output_path,
        train_obs=train_obs,
        train_states=train_states,
        test_obs=test_obs,
        test_states=test_states,
    )

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    loaded = np.load(output_path)
    for key in ["train_obs", "train_states", "test_obs", "test_states"]:
        arr = loaded[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.1f}, {arr.max():.1f}]")

    file_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nFile size: {file_mb:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
