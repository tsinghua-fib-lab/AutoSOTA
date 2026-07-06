# python scripts
__author__ = "Du Jiawei NUS/IHPC"
__email__ = "dujiawei@u.nus.edu"
# Descrption:

import os

# Read API key from environment to avoid leaking secrets in code.
openai_api_key = os.getenv("OPENAI_API_KEY", "")
# Put your name
key_owner = "key"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True
