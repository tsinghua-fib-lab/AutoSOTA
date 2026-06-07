import sys, os, json, numpy as np
from scipy import spatial
sys.path.insert(0, '/repo')
os.environ["DISPLAY"] = ""

from kasal.utils.io_ply import load_ply_model

with open("/repo/DSRSTO/DSRSTO dataset/shape_meshes/obj_000001_ours_sym_type.json") as f:
    gt = json.load(f)

info = gt["current_obj_info"]
diameter = info["diameter"]

model = load_ply_model("/repo/DSRSTO/DSRSTO dataset/shape_meshes/obj_000001_ours.ply", color_op=False)
verts = model["vertices"]
print("Diameter:", diameter)

# CORRECT ADI: mean(min distance from each point in original to rotated)
def compute_adi_correct(pts_original, pts_rotated):
    # For each point in original, find closest point in rotated
    dists = spatial.distance.cdist(pts_original, pts_rotated, metric='euclidean')
    min_dists = np.min(dists, axis=1)
    return np.mean(min_dists)

# GT discrete symmetry
sm = info["symmetries_discrete"][0]
mat = np.array(sm).reshape(4,4)
R_gt = mat[:3,:3]
t_gt = mat[:3,3]
rotated_gt = np.dot(verts, R_gt.T) + t_gt
adi_correct = compute_adi_correct(verts, rotated_gt)
print("ADI (correct) using GT matrix:", adi_correct, "ADI/d:", adi_correct/diameter)

# Continuous symmetry
sc = info["symmetries_continuous"][0]
axis = np.array(sc["axis"])
offset = np.array(sc["offset"])
a = axis / np.linalg.norm(axis)

for ang_deg in [30, 60, 90, 120, 180]:
    theta = np.radians(ang_deg)
    c = np.cos(theta); s = np.sin(theta)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + s*K + (1-c)*np.dot(K,K)
    t = offset - np.dot(R, offset)
    rotated = np.dot(verts, R.T) + t
    adi = compute_adi_correct(verts, rotated)
    print(f"  {ang_deg} deg: ADI={adi:.6f}, ADI/d={adi/diameter:.6f}")

# Also test z-axis rotation
a_z = np.array([0, 0, 1.0])
for ang_deg in [30, 90, 180]:
    theta = np.radians(ang_deg); c = np.cos(theta); s = np.sin(theta)
    K = np.array([[0, -a_z[2], a_z[1]], [a_z[2], 0, -a_z[0]], [-a_z[1], a_z[0], 0]])
    R = np.eye(3) + s*K + (1-c)*np.dot(K,K)
    rotated = np.dot(verts, R.T)
    adi = compute_adi_correct(verts, rotated)
    print(f"  z-axis {ang_deg} deg: ADI={adi:.6f}, ADI/d={adi/diameter:.6f}")
