echo "=== Acquiring Pre-trained Models ==="
echo "---"

echo "Installing gdown package (for downloading from Google Drive)"
pip install gdown

echo "Start downloading model [A-M]"
gdown --id 1eU07wj0avbNO-R2ij9w15zskmJofeXO6 -O final_submission/models/model_A/model.ckpt  # graph_transformer_650x10_final200_A.ckpt
gdown --id 1TJfd4k7TzJ8UsnEh8N_5MoxhQ_bck1Fg -O final_submission/models/model_B/model.ckpt  # graph_transformer_650x10_final200_B.ckpt
gdown --id 1JaxLPm4hxVoPws3WqFHPSr0iSIKItW3d -O final_submission/models/model_C/model.ckpt  # graph_transformer_650x10_final200_C.ckpt
gdown --id 1B70uVBGDRYxso3q-i-n_SO4Dt0Bc64wB -O final_submission/models/model_D/model.ckpt  # graph_transformer_800x11_final220_D.ckpt
gdown --id 1sDjwDcyyTRP5A_V0qFCSigJSvLOIBEwX -O final_submission/models/model_E/model.ckpt  # graph_transformer_600x10_final280_E.ckpt
gdown --id 1I0icRGNTWn3g91XldrmRh6ezh9k1WpnA -O final_submission/models/model_F/model.ckpt  # graph_transformer_800x10_final200_F.ckpt
gdown --id 1ZzAoIkMjZX8rtZSRwEAVzSqNemQp2cty -O final_submission/models/model_G/model.ckpt  # graph_transformer_600x11_final270_G.ckpt
gdown --id 1gh1gKFYJSadkHgqC3U2LvwdrRCADAzAw -O final_submission/models/model_H/model.ckpt  # graph_transformer_700x15_final280_1res_H.ckpt
gdown --id 1Piob2pof-DjRe-H4oz29k-tOzI9GxoPE -O final_submission/models/model_I/model.ckpt  # graph_transformer_700x16_final280_I.ckpt
gdown --id 1PEeTBDJVuyiEzxaaSBu7-37afFOj6zCZ -O final_submission/models/model_J/model.ckpt  # graph_transformer_650x13_final270_1res_J.ckpt
gdown --id 1TYoXuyx-ELsB91trPgeoqdMmoZl-jL0r -O final_submission/models/model_K/model.ckpt  # graph_transformer_600x14_final250_1res_K.ckpt
gdown --id 1Cbw7oEETAapl1oqO36Y4p7uwOIGvFQkG -O final_submission/models/model_L/model.ckpt  # graph_transformer_750x18_final300_quad_L.ckpt
gdown --id 1p0GLutPVuWqxsR7i95pzyKOOC4MLYF5K -O final_submission/models/model_M/model.ckpt  # graph_transformer_660x11_final220_M.ckpt
