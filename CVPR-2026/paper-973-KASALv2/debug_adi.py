import sys, os, json, numpy as np
sys.path.insert(0, '/repo')
os.environ["DISPLAY"] = ""

from kasal.utils.io_ply import load_ply_model

# Load ground truth
with open("/repo/DSRSTO/DSRSTO dataset/shape_meshes/obj_000001_ours_sym_type.json") as f:
    gt = json.load(f)

info = gt["current_obj_info"]
diameter = info["diameter"]
print("Diameter:", diameter)

# Load the model
model = load_ply_model("/repo/DSRSTO/DSRSTO dataset/shape_meshes/obj_000001_ours.ply", color_op=False)
vertices = model["vertices"]
print("Vertices shape:", vertices.shape)
print("Vertices range:", np.min(vertices, axis=0), np.max(vertices, axis=0))

# Check discrete symmetries from ground truth
if "symmetries_discrete" in info:
    sym_mats = info["symmetries_discrete"]
    print("\nDiscrete symmetries:", len(sym_mats))
    for i, sm in enumerate(sym_mats):
        mat = np.array(sm).reshape(4,4)
        R = mat[:3,:3]
        t = mat[:3,3]
        print(f"  Matrix {i}: det(R)={np.linalg.det(R):.6f}, t={t}")
        rotated = np.dot(vertices, R.T) + t
        dists = np.linalg.norm(vertices - rotated, axis=1)
        adi = np.mean(dists)
        print(f"    ADI={adi:.6f}, ADI/d={adi/diameter:.6f}, max_dist={np.max(dists):.4f}")

# Check continuous symmetries from ground truth
if "symmetries_continuous" in info:
    sym_conts = info["symmetries_continuous"]
    print("\nContinuous symmetries:", len(sym_conts))
    for i, sc in enumerate(sym_conts):
        axis = np.array(sc["axis"])
        offset = np.array(sc["offset"])
        print(f"  Axis: {axis}, offset: {offset}")
        a_norm = axis / np.linalg.norm(axis)
        for ang_deg in [30, 90, 180]:
            theta = np.radians(ang_deg)
            c = np.cos(theta); s = np.sin(theta)
            K = np.array([[0, -a_norm[2], a_norm[1]], [a_norm[2], 0, -a_norm[0]], [-a_norm[1], a_norm[0], 0]])
            R = np.eye(3) + s*K + (1-c)*np.dot(K,K)
            t = offset - np.dot(R, offset)
            rotated = np.dot(vertices, R.T) + t
            dists = np.linalg.norm(vertices - rotated, axis=1)
            adi = np.mean(dists)
            print(f"    Angle {ang_deg}: ADI={adi:.6f}, ADI/d={adi/diameter:.6f}")

# Now load the sym PLY (pre-computed result) and check its transform
print("\n\n=== Ground truth SYM file ===")
sym_ply = "/repo/DSRSTO/DSRSTO dataset/shape_meshes/obj_000001_ours_sym.ply"
model_sym = load_ply_model(sym_ply, color_op=False)
v_sym = model_sym["vertices"]
print("Sym vertices shape:", v_sym.shape)
print("Sym vertices range:", np.min(v_sym, axis=0), np.max(v_sym, axis=0))
# The sym file might have been transformed; check difference
print("Mean vertex diff:", np.mean(np.linalg.norm(vertices - v_sym, axis=1)))
