"""
Visualization utils code for SAVVY pipeline - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_trajectories(object_traj, my_traj, final_obj, obj_loc, outfig_path="vis_traj.jpg"):
    """
    Visualize my trajectory and object possible locations and final locations
    """
    plt.figure(figsize=(15, 10))
    
    # Plot my trajectory as a connected line
    traj_times = sorted(my_traj.keys())
    traj_x = [my_traj[t]["loc"][0] for t in traj_times]
    traj_y = [my_traj[t]["loc"][1] for t in traj_times]
    plt.plot(traj_x, traj_y, 'b-', linewidth=2, label='My Trajectory')
    
    for t in traj_times[::1]:  # Plot every 5th point to avoid clutter
        x, y = my_traj[t]["loc"]
        fx, fy = my_traj[t]["forward_vec"]
        # Scale the forward vector for visibility
        arrow_scale = 0.5
        plt.arrow(x, y, fx * arrow_scale, fy * arrow_scale, 
                    head_width=0.2, head_length=0.3, fc='blue', ec='blue')

    object_traj_x = [object_traj[t][0] for t in traj_times]
    object_traj_y = [object_traj[t][1] for t in traj_times]
    plt.plot(object_traj_x, object_traj_y, 'r-', linewidth=2, label='Object Trajectory')
    
    
    # Plot object possible locations with different colors for each object
    colors = plt.cm.tab10(np.linspace(0, 1, len(obj_loc)))
    
    for i, (obj_name, positions_list) in enumerate(obj_loc.items()):
        color = colors[i]
        
        # Plot all possible positions as small dots
        for positions_set in positions_list:
            x_vals = [pos[0] for pos in positions_set]
            y_vals = [pos[1] for pos in positions_set]
            # plt.scatter(x_vals, y_vals, s=5, alpha=0.3, color=color)
        
        # If we have determined a final position for this object, mark it
        if obj_name in final_obj and isinstance(final_obj[obj_name], tuple):
            final_x, final_y = final_obj[obj_name]
            plt.scatter(final_x, final_y, s=100, color=color, edgecolor='black',
                    label=f'{obj_name} (Final)')
            plt.annotate(obj_name, (final_x, final_y), 
                        xytext=(5, 5), textcoords='offset points')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('X coordinate (m)')
    plt.ylabel('Y coordinate (m)')
    plt.title('Trajectory and Object Locations')
    plt.legend()
    
    # Ensure equal aspect ratio
    plt.axis('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(outfig_path, dpi=300)