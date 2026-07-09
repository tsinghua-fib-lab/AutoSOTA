import numpy as np
import scipy as sp

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib as mpl
from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import polyscope as ps

import os

def visualize_optimization(mapping, opt_data):
    n_output = len(opt_data)

    t = np.arange(1, n_output + 1)
    l_dual = np.zeros(np.shape(t))
    l_reg = np.zeros(np.shape(t))
    time_forward_dXPP = np.zeros(np.shape(t))
    time_backward_dXPP = np.zeros(np.shape(t))
    time_forward = np.zeros(np.shape(t))
    nActive = np.zeros(np.shape(t))
    for ii in np.arange(0, n_output):
        l_dual[ii] = opt_data[str(ii)]["loss_dual"]
        l_reg[ii] = opt_data[str(ii)]["loss_reg"]
        time_forward_dXPP[ii] = opt_data[str(ii)]["dXPP_forward_time"]
        time_backward_dXPP[ii] = opt_data[str(ii)]["dXPP_backward_time"]
        time_forward[ii] = opt_data[str(ii)]["total_forward_time"]
        nActive[ii] = opt_data[str(ii)]["nActive"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.0))

    active_change = np.concatenate((np.diff(nActive), [0])) != 0
    active_change[np.isnan(active_change)] = 0
    nChange = t[active_change]
    active_set_line = ax1.vlines(nChange,1e-4,1e1)
    loss_dual_line = ax1.plot(t, l_dual, c="k", label="|\mu|_\inf")
    loss_reg_line = ax1.plot(t, l_reg, c="k", linestyle="--", label="|L - L_c|_\inf")
    loss_total_line = ax1.plot(t, l_reg+l_dual, c="r", label="Loss")

    dXPP_forward_line = ax2.plot(t, time_forward_dXPP, c="b", linewidth=2, label='dXPP Forward')
    dXPP_backward_line = ax2.plot(t, time_backward_dXPP, c="b", linewidth=2, linestyle='--', label='dXPP Backward')
    forward_line = ax2.plot(t, time_forward, c="orange", linestyle=':', label='Total Forward')

    ax1.set(xlim=[0, n_output], ylim=[1e-4, 1e1], xlabel='Iteration', ylabel='Loss', yscale='log')
    if mapping.option == "ant" or mapping.option == "parameterization":
        ax2.set(xlim=[0, n_output], ylim=[0, 1], xlabel='Iteration', ylabel='QP Forward Time (s)',
                yscale='linear', title="#v: " + str(mapping.nv) + " dim: " + str(mapping.dim * mapping.nv) + " #Ineq: " + str(
                mapping.dim * mapping.nc) + " #Eq: " + str(mapping.dim * mapping.nb))
    else:
        ax2.set(xlim=[0, n_output], ylim=[1e-6, 1e1], xlabel='Iteration', ylabel='QP Forward Time (s) / 1s',
                yscale='log', title="#v: " + str(mapping.nv) + " dim: " + str(mapping.dim * mapping.nv) + " #Ineq: " + str(
                mapping.dim * mapping.nc) + " #Eq: " + str(mapping.dim * mapping.nb))
    
    ax2.legend()
    plt.savefig('results/' + mapping.option + '/optimization_' + "_nv=" + str(mapping.nv)  + "_lreg=" + str(mapping.lambda_reg) + '.pdf')
    plt.close(fig)

    # save into mapping_timings.dat
    if mapping.option == "corner":
        # columns: nv, nc, dXPP_forward, dXPP_backward
        with open("results/corner/mapping_timings.dat", "a+") as f:
            print(mapping.nv, mapping.nc,
                np.nanmean(time_forward_dXPP), np.nanmean(time_backward_dXPP), file=f)


    if mapping.video:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.0, 6.0))

        n_frames = min(n_output, 600)  # maximum 5s video at 60fps

        ax2.set_aspect('equal')
        ax2.set_yscale('linear')

        loss_dual_line = ax1.plot(1, opt_data["1"]["loss_dual"], c="k", label="|\mu|_2")
        loss_dual_line = loss_dual_line[0]

        loss_reg_line = ax1.plot(1, opt_data["1"]["loss_reg"], c="k", linestyle="--", label="|L - L_c|_\inf")
        loss_reg_line = loss_reg_line[0]

        dXPP_forward_line = ax3.plot(1, opt_data["1"]["dXPP_forward_time"], c="b", linewidth=2, label='dXPP Forward')
        dXPP_forward_line = dXPP_forward_line[0]
        dXPP_backward_line = ax3.plot(1, opt_data["1"]["dXPP_backward_time"], c="b", linewidth=2, linestyle='--',
                                       label='dXPP Backward')
        dXPP_backward_line = dXPP_backward_line[0]
        forward_line = ax3.plot(1, opt_data["1"]["total_forward_time"], c="orange", linestyle=':', label='Total Forward')
        forward_line = forward_line[0]

        ax1.set(xlim=[0, n_output], ylim=[1e-4, 1e3], xlabel='Iteration', ylabel='Loss', yscale='log')
        ax2.set(xlim=[np.min(mapping.v[:, 0]), np.max(mapping.v[:, 0])], ylim=[np.min(mapping.v[:, 1]), np.max(mapping.v[:, 1])])
        if mapping.option == "ant" or mapping.option == "parameterization":
            ax3.set(xlim=[0, n_output], ylim=[0, 1], xlabel='Iteration', ylabel='QP Forward Time (s)',
                    yscale='linear', title="#v: " + str(mapping.nv) + " dim: " + str(mapping.dim * mapping.nv) + " #Ineq: " + str(
                    mapping.dim * mapping.nc) + " #Eq: " + str(mapping.dim * mapping.nb))
        else:
            ax3.set(xlim=[0, n_output], ylim=[1e-6, 1e1], xlabel='Iteration', ylabel='QP Forward Time (s) / 1s',
                    yscale='log', title="#v: " + str(mapping.nv) + " dim: " + str(mapping.dim * mapping.nv) + " #Ineq: " + str(
                    mapping.dim * mapping.nc) + " #Eq: " + str(mapping.dim * mapping.nb))
        ax1.legend()
        ax3.legend()

        def update(frame):
            print((frame, np.floor(n_output / n_frames)))
            frame = int(np.floor(n_output / n_frames) * frame)
            print((frame, n_output))

            # update the loss plot:
            loss_dual_line.set_xdata(t[:frame])
            loss_dual_line.set_ydata(l_dual[:frame])

            loss_reg_line.set_xdata(t[:frame])
            loss_reg_line.set_ydata(l_reg[:frame])

            dXPP_forward_line.set_xdata(t[:frame])
            dXPP_forward_line.set_ydata(time_forward_dXPP[:frame])

            dXPP_backward_line.set_xdata(t[:frame])
            dXPP_backward_line.set_ydata(time_backward_dXPP[:frame])

            forward_line.set_xdata(t[:frame])
            forward_line.set_ydata(time_forward[:frame])

            for line in ax2.get_lines():
                line.remove()

            for collection in ax2.collections:
                collection.remove()

            for patch in ax2.patches:
                patch.remove()

            b_rep = np.vstack((mapping.b, mapping.b[0, :]))
            b_pad = np.vstack((mapping.b[-1, :], mapping.b, mapping.b[0, :]))
            theta_start = np.zeros(mapping.nb)
            for ii in np.arange(1, mapping.nb + 1):
                v_curr = b_pad[ii, :]
                dl_2 = b_pad[ii + 1, :] - v_curr
                theta_start[ii - 1] = np.arctan2(dl_2[1], dl_2[0])

            x_mapping = opt_data[str(frame)]["x_star"]
            inverted = mapping.compute_inversion(x_mapping)
            dxdn = opt_data[str(frame)]["L"] @ x_mapping

            ax2.plot(b_rep[:, 0], b_rep[:, 1], 'k', linewidth=2)
            title = "interior |Lv|=" + str(
                np.round(np.linalg.norm(mapping.P_interior @ opt_data[str(frame)]["L"] @ x_mapping),
                         2)) + " reflex |Lv|=" + str(
                np.round(np.linalg.norm(mapping.P_cone @ opt_data[str(frame)]["L"] @ x_mapping), 2))
            ax2.set_title(title)

            x_mapping = np.reshape(x_mapping, (mapping.nv, mapping.dim), order='F')
            dxdn = np.reshape(dxdn, (mapping.nv, mapping.dim), order='F')

            if inverted is None:
                ax2.triplot(Triangulation(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f), color='k')
            else:
                ax2.tripcolor(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f, facecolors=inverted,
                              cmap=mpl.colors.ListedColormap(['white', 'red']), edgecolor="k", linewidth=0.5,
                              alpha=0.5)

            try:
                # reverse sign convention only visually to match paper
                for ii in np.arange(1, mapping.nb + 1):
                    theta_cone = mapping.theta[ii - 1]
                    if theta_cone > np.pi:  # use to only plot constrained cones
                        if theta_cone > np.pi:
                            theta_cone = 2 * np.pi - theta_cone
                            theta_displace = np.pi - theta_cone
                        else:
                            theta_displace = 0

                        ax2.add_patch(
                            patches.Wedge(
                                (b_pad[ii, 0], b_pad[ii, 1]),  # (x,y)
                                0.1,  # radius
                                (theta_start[ii - 1] + theta_displace) * 180 / np.pi,  # theta1 (in degrees)
                                (theta_start[ii - 1] + theta_displace) * 180 / np.pi + theta_cone * 180 / np.pi,
                                # theta2
                                color="b", alpha=0.2
                            )
                        )

                # reverse sign convention only visually to match paper
                for ii in np.arange(0, mapping.nv):
                    dxdn_curr = -dxdn[ii, :]
                    dxdn_norm = np.linalg.norm(dxdn_curr)
                    if dxdn_norm > 1e-12 and ii in mapping.l_boundary[mapping.l_cone]:
                        ax2.quiver(np.squeeze(x_mapping[ii, 0]), np.squeeze(x_mapping[ii, 1]), dxdn_curr[0],
                                   dxdn_curr[1],
                                   color='r')
            except:
                print("Failed to plot vectors")

            # plt.gca().set_aspect('equal')

            return (ax1, ax2)

        fps = np.floor(np.min([60, n_frames / 5])).astype(
            int)  # 60 fps maximum, if < 60 fps then make fps such that video is 5 seconds
        anim = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=30)
        FFwriter = animation.FFMpegWriter(fps=fps)
        anim.save('results/' + mapping.option + '/optimization_'  + '_nv=' + str(mapping.nv)  + '_lreg=' + str(mapping.lambda_reg) + '.mp4', writer=FFwriter)
        plt.close()



def visualize_mapping(mapping,map_type):
    if map_type == "free":
        title = "Lv_int=" + str(
            np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping_free), 2)) + "_Lv_ref=" + str(
            np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping_free), 2)) + "_Ninv=" + str(
            np.sum(mapping.inverted_free == 1)) + "_nv=" + str(mapping.nv)
        dxdn = mapping.dxdn_free.reshape((mapping.nv, 2), order='F')
        x_mapping = mapping.x_mapping_free.reshape((mapping.nv, 2), order='F')
    elif map_type == "constrained":
        title = "Lv_int=" + str(
            np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping), 2)) + "_Lv_ref=" + str(
            np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping), 2)) + "_Ninv=" + str(
            np.sum(mapping.inverted == 1))  + "_nv=" + str(mapping.nv) + "_lreg=" + str(mapping.lambda_reg)
        dxdn = mapping.dxdn.reshape((mapping.nv, 2), order='F')
        x_mapping = mapping.x_mapping.reshape((mapping.nv, 2), order='F')

    b_pad = np.vstack((mapping.b[-1, :], mapping.b, mapping.b[0, :]))
    b_rep = np.vstack((mapping.b, mapping.b[0, :]))
    # internal angles of constraints
    theta_start = np.zeros(mapping.nb)
    for ii in np.arange(1, mapping.nb + 1):
        v_curr = b_pad[ii, :]
        dl_2 = b_pad[ii + 1, :] - v_curr
        theta_start[ii - 1] = np.arctan2(dl_2[1], dl_2[0])

    fig, ax = plt.subplots()
    plt.plot(b_rep[:, 0], b_rep[:, 1], 'k', linewidth=2)

    inverted = mapping.compute_inversion(np.reshape(x_mapping, (mapping.nv * mapping.dim, 1), order='F'))
    inverted_bool = inverted == -1
    if inverted is None:
        ax.triplot(Triangulation(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f), color='k')
    else:
        ax.tripcolor(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f, facecolors=inverted,
                     cmap=mpl.colors.ListedColormap(['white', 'red']), edgecolor="k", linewidth=1, alpha=(1-inverted_bool)) # trick to get only red faces colored
        ax.triplot(Triangulation(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f), color='k',linewidth=0.5)

    try:
        # reverse sign convention only visually to match paper
        for ii in np.arange(1, mapping.nb + 1):

            theta_cone = mapping.theta[ii - 1]
            if theta_cone > np.pi:  # use to only plot constrained cones
                if theta_cone > np.pi:
                    theta_cone = 2 * np.pi - theta_cone
                    theta_displace = np.pi - theta_cone
                else:
                    theta_displace = 0

                ax.add_patch(
                    patches.Wedge(
                        (b_pad[ii, 0]-0*mapping.d_bisect[ii-1,0], b_pad[ii, 1]-0*mapping.d_bisect[ii-1,1]),  # (x,y)
                        0.1,  # radius
                        (theta_start[ii - 1] + theta_displace) * 180 / np.pi,  # theta1 (in degrees)
                        (theta_start[ii - 1] + theta_displace) * 180 / np.pi + theta_cone * 180 / np.pi,  # theta2
                        color="b", alpha=0.2
                    )
                )

        # reverse sign convention only visually to match paper
        for ii in np.arange(0, mapping.nv):
            dxdn_curr = -dxdn[ii, :]
            dxdn_norm = np.linalg.norm(dxdn_curr)
            if dxdn_norm > 1e-8 and ii in mapping.l_boundary[mapping.l_cone]:
                plt.quiver(np.squeeze(x_mapping[ii, 0]), np.squeeze(x_mapping[ii, 1]), dxdn_curr[0]/dxdn_norm, dxdn_curr[1]/dxdn_norm, # TODO : note normalize then scale
                           color='b',scale=10)
    except:
        print("Failed to plot vectors")

    plt.gca().set_aspect('equal')
    plt.axis('off')

    if map_type == "free":
        plt.savefig('results/' + mapping.option + '/final_free_' + title + '.pdf')
        plt.savefig('results/' + mapping.option + '/final_free_' + title + '.png',dpi=300,transparent=True,bbox_inches='tight')
    elif map_type == "constrained":
        plt.savefig('results/' + mapping.option + '/final_constrained_' + title + '.pdf')
        plt.savefig('results/' + mapping.option + '/final_constrained_' + title + '.png',dpi=300,transparent=True,bbox_inches='tight')
    plt.close(fig)

    plt.close()

def inversion_zoom(mapping):
    for map_type in ["free","constrained"]:

        if map_type == "free":
            title = "Lv_int=" + str(
                np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping_free),
                         2)) + "_Lv_ref=" + str(
                np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping_free), 2)) + "_Ninv=" + str(
                np.sum(mapping.inverted_free == 1)) + "_nv=" + str(mapping.nv)
            x_mapping = mapping.x_mapping_free.reshape((mapping.nv, 2), order='F')
        elif map_type == "constrained":
            title = "Lv_int=" + str(
                np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping), 2)) + "_Lv_ref=" + str(
                np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping), 2)) + "_Ninv=" + str(
                np.sum(mapping.inverted == 1)) + "_nv=" + str(mapping.nv) + "_lreg=" + str(mapping.lambda_reg)
            x_mapping = mapping.x_mapping.reshape((mapping.nv, 2), order='F')


        fig, ax = plt.subplots()
        b_rep = np.vstack((mapping.b, mapping.b[0, :]))
        plt.plot(b_rep[:, 0], b_rep[:, 1], 'k', linewidth=0.5)

        inverted = mapping.compute_inversion(np.reshape(x_mapping, (mapping.nv * mapping.dim, 1), order='F'))
        assert(inverted is not None)
        ax.tripcolor(x_mapping[:, 0], x_mapping[:, 1], triangles=mapping.f, facecolors=inverted,
                     cmap=mpl.colors.ListedColormap(['white', 'red']), edgecolor="k", linewidth=0.5, alpha=0.75)

        if map_type == "free": # save for constrained, to see the inversion removed
            v_inverted = mapping.f[inverted==1,:]
            v_inverted = x_mapping[v_inverted[0,:],:] # get first triangle with inversions
            dl = np.ptp(v_inverted)
            center = np.mean(v_inverted,0)

        plt.gca().set_aspect('equal')

        plt.axis('off')

        tri_marker = plt.scatter(center[0],center[1],color='k',s=20)

        if map_type == "free":
            plt.savefig('results/' + mapping.option + '/final_free_inversion_zoomout_' + title + '.pdf')
        elif map_type == "constrained":
            plt.savefig(
                'results/' + mapping.option + '/final_constrained_inversion_zoomout_' + title + '.pdf')

        tri_marker.remove()

        plt.xlim(center[0] - 5 * dl, center[0] + 5 * dl)
        plt.ylim(center[1] - 5 * dl, center[1] + 5 * dl)

        if map_type == "free":
            plt.savefig('results/' + mapping.option + '/final_free_inversion_zoomin_' + title + '.pdf')
        elif map_type == "constrained":
            plt.savefig(
                'results/' + mapping.option + '/final_constrained_inversion_zoomin_' + title + '.pdf')

        plt.close(fig)


def manual_polyscope(mapping):
    ps.init()

    for map_type in ["free","constrained"]:
        if map_type == "free":
            x_mapping = mapping.x_mapping_free.reshape((mapping.nv, 2), order='F')
        elif map_type == "constrained":
            x_mapping = mapping.x_mapping.reshape((mapping.nv, 2), order='F')

        ps.register_surface_mesh(map_type, x_mapping,mapping.f)

    ps.show()

def export_to_MATLAB(mapping):
    for map_type in ["free","constrained"]:
        if map_type == "free":
            title = "Lv_int=" + str(
                np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping_free),
                         2)) + "_Lv_ref=" + str(
                np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping_free), 2)) + "_Ninv=" + str(
                np.sum(mapping.inverted_free == 1)) + "_nv=" + str(mapping.nv)
            x_mapping = mapping.x_mapping_free.reshape((mapping.nv, 2), order='F')
        elif map_type == "constrained":
            title = "Lv_int=" + str(
                np.round(np.linalg.norm(mapping.P_interior @ mapping.L @ mapping.x_mapping), 2)) + "_Lv_ref=" + str(
                np.round(np.linalg.norm(mapping.P_cone @ mapping.L @ mapping.x_mapping), 2)) + "_Ninv=" + str(
                np.sum(mapping.inverted == 1)) + "_nv=" + str(mapping.nv) + "_lreg=" + str(mapping.lambda_reg)
            x_mapping = mapping.x_mapping.reshape((mapping.nv, 2), order='F')

        b_loop = np.vstack((mapping.b, mapping.b[0, :]))
        inverted = mapping.compute_inversion(np.reshape(x_mapping, (mapping.nv * mapping.dim, 1), order='F'))

        dict = {"v":x_mapping,"f":mapping.f,"flip":inverted==1,"b_loop":b_loop}
        if map_type == "free":
            sp.io.savemat(
                'results/' + mapping.option + '/final_free_' + title + '.mat',
                dict)
        elif map_type == "constrained":
            sp.io.savemat(
                'results/' + mapping.option + '/final_constrained_' + title + '.mat',
                dict)
