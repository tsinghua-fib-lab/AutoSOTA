from pathlib import Path
from build.othello import game, inference, RMCTS

onnx_path = Path(__file__).parent / 'othello' / 'models' / 'ResNet_8blocks_48channels.onnx'
engine = inference.Engine(onnx_path)
R = RMCTS.RMCTS_Tree(game.rootState(), engine)
T,L = R.explore(1024, temperature=1)
R.export_tree_json("R_tree.json")

