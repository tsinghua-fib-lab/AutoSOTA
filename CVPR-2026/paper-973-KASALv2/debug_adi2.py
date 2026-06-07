import sys, os, json, numpy as np
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
print("Vertices:", verts.shape)

# Ground truth symmetries
sm = info["symmetries_discrete"][0]
mat = np.array(sm).reshape(4,4)
R_gt = mat[:3,:3]
t_gt = mat[:3,3]
print("\nGT matrix det(R):", np.linalg.det(R_gt))
print("GT R * R.T:")
print(np.dot(R_gt, R_gt.T))

# Apply ground truth
rotated_gt = np.dot(verts, R_gt.T) + t_gt
dists_gt = np.linalg.norm(verts - rotated_gt, axis=1)
print("\nADI using GT matrix:", np.mean(dists_gt), "ADI/d:", np.mean(dists_gt)/diameter)

# Continuous symmetry
sc = info["symmetries_continuous"][0]
axis = np.array(sc["axis"])
offset = np.array(sc["offset"])
print("\nAxis:", axis)
print("Offset:", offset)

a = axis / np.linalg.norm(axis)
for ang_deg in [30, 60, 90, 120, 180]:
    theta = np.radians(ang_deg)
    c = np.cos(theta); s = np.sin(theta)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + s*K + (1-c)*np.dot(K,K)
    t = offset - np.dot(R, offset)
    rotated = np.dot(verts, R.T) + t
    dists = np.linalg.norm(verts - rotated, axis=1)
    adi = np.mean(dists)
    print(f"  {ang_deg} deg: ADI={adi:.4f}, ADI/d={adi/diameter:.6f}")

# Key question: is the cylinder centered at origin?
center = np.mean(verts, axis=0)
print("\nVertices center:", center)
print("Vertices min:", np.min(verts, axis=0))
print("Vertices max:", np.max(verts, axis=0))

# The cylinder should be aligned with z-axis
# Let me check if rotating around z gives small ADI for a cylinder
a_z = np.array([0, 0, 1.0])
for ang_deg in [30, 90, 180]:
    theta = np.radians(ang_deg)
    c = np.cos(theta); s = np.sin(theta)
    K = np.array([[0, -a_z[2], a_z[1]], [a_z[2], 0, -a_z[0]], [-a_z[1], a_z[0], 0]])
    R = np.eye(3) + s*K + (1-c)*np.dot(K,K)
    rotated = np.dot(verts, R.T)
    dists = np.linalg.norm(verts - rotated, axis=1)
    adi = np.mean(dists)
    print(f"  z-axis {ang_deg} deg: ADI={adi:.4f}, ADI/d={adi/diameter:.6f}")
