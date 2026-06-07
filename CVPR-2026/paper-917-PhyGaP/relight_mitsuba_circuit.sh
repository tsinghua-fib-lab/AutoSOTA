CKPT_PATH=output/mitsuba_david_museum/mitsuba_david_museum-0412_1631-st-4.0-ind-obj_env/chkpnt15000.pth
item=mitsuba_david_museum
envmap=christmas_studio

python scripts/render_relight_orbit_rotation.py\
     --ckpt ${CKPT_PATH} \
     --env_map assets/envmap/${envmap}.hdr \
     --idx_list "0,8,16,24" \
     --final_itr 15000  \
     --angle_step_deg 2  \
     --output_dir relighting_results/${item}_${envmap}_180 \
     --save_envview   --mode direct   --env_view_hfov_deg 90 --reverse_rotation