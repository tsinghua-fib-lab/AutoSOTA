import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import imageio

def calc_aolp_dop(stokes,mask=None):
    if mask is not None:
        mask = (mask[...,:] > 1e-5)
        stokes_masked = torch.stack([stokes[...,0]*mask,stokes[...,1]*mask,stokes[...,2]*mask],-1)
    cues = cues_from_stokes(stokes)
    aolp = cues['aolp']
    dop = cues['dop']
    return aolp, dop


def cues_from_stokes(stokes):
    import torch
    if torch.is_tensor(stokes):
        sqrt, atan2 = torch.sqrt, torch.atan2
    else:
        sqrt, atan2 = np.sqrt, np.arctan2
    # Assumes last dimension is 4
    dop = sqrt((stokes[...,1:]**2).sum(-1))/stokes[...,0] # sqrt(S1^2+S2^2+S3^2)/S0 S3=0
    dop[stokes[...,0]<1e-6] = 0.
    aolp = 0.5*atan2(stokes[...,2],stokes[...,1])
    aolp = (aolp%np.pi)/np.pi*180  #转化为弧度
    s0 = stokes[...,0]
    return {'dop':dop,
            'aolp':aolp,
            's0':s0}
def quat_to_rot(q):
    prefix, _ = q.shape[:-1]
    q = F.normalize(q, dim=-1)
    R = torch.ones([*prefix, 3, 3]).to(q.device)
    qr = q[... ,0]
    qi = q[..., 1]
    qj = q[..., 2]
    qk = q[..., 3]
    R[..., 0, 0]=1-2 * (qj**2 + qk**2)
    R[..., 0, 1] = 2 * (qj *qi -qk*qr)
    R[..., 0, 2] = 2 * (qi * qk + qr * qj)
    R[..., 1, 0] = 2 * (qj * qi + qk * qr)
    R[..., 1, 1] = 1-2 * (qi**2 + qk**2)
    R[..., 1, 2] = 2*(qj*qk - qi*qr)
    R[..., 2, 0] = 2 * (qk * qi-qj * qr)
    R[..., 2, 1] = 2 * (qj*qk + qi*qr)
    R[..., 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def lift(x, y, z, intrinsics):
    device = x.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = intrinsics[..., 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(device)), dim=-1)

def normalize(v):
    import torch
    return torch.nn.functional.normalize(v, dim=-1)

def normalize_np(v):
    return  v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),
                     1e-12)



def viz_gt_stokes(stokes):
    stokes_out = stokes
    print(stokes_out)
    print(stokes_out.shape)
    stokes_out_copy = stokes_out.clone()
    cues_out = cues_from_stokes(stokes_out_copy)

    imageio.imwrite('debug/gt_s0.exr',cues_out['s0'].cpu().numpy().astype('float32'))

    plt.imshow(cues_out['aolp'][...,0].cpu().numpy(), 
            vmin=0, vmax=180,
            cmap='twilight')
    plt.colorbar()
    plt.savefig('debug/gt_aolp.png')
    plt.close()

    plt.imshow(cues_out['dop'][...,0].cpu().numpy())
    plt.colorbar()
    plt.savefig('debug/gt_dop.png')
    plt.close()


def stokes_fac_from_normal(rays_o, rays_d, normal, 
                           clip_spec=False, eta = None):
    # This function is modified from PANDORA
    # The input direction vector of PANDORA is in Blender's coordinate system, which is different from GS.
    # Args:
    #   rays, normal :(H,W,3)
    #   diff_rads, spec_rads: (H,W,3)

    # Add singleton dimension for Num_lights
    rays_o = rays_o[...,None,:]
    rays_d = rays_d[...,None,:]
    normal = normal[...,None,:]
    if eta is not None:
        eta = eta[...,None,:]  
    # import pdb;pdb.set_trace()
    # Define helper functions
    # flag to check if torch or np
    pt = torch.is_tensor(rays_o)
    cos = torch.cos if pt else np.cos
    sin = torch.sin if pt else np.sin
    acos = torch.acos if pt else np.arccos
    atan2 = torch.atan2 if pt else np.arctan2
    sqrt = torch.sqrt if pt else np.sqrt
    normize = normalize if pt else normalize_np
    stack = torch.stack if pt else np.stack
    acos = lambda x: torch.acos(torch.clamp(x, min=-1.+1e-7,
                                               max=1.-1e-7))
    dot = lambda x, y: (x*y).sum(-1, keepdim=True)
    clamp = lambda x,y: torch.clamp(x,min=y)
    mask_fn = lambda x,mask: torch.where(mask, x, torch.zeros_like(x))
    cross = lambda x,y: torch.cross(x,y.broadcast_to(x.shape),dim=-1) 
    clip_max = lambda x,y:torch.clamp(x, max=y)

    # Helper variables    
    eta = eta if eta is not None else 1.5
    n = normize(normal) 
    o = normize(-rays_d)
    h = n #if train_mode else normize(i+o)
    # n_o = normize(cross(n,o))
    # h_o = normize(cross(h,o))
    n_o = normize(n - dot(n,o)*o)
    h_o = normize(h - dot(h,o)*o)

    # Using Dirs_up in global coordinates
    # From https://ksimek.github.io/2012/08/22/extrinsic/
    x_o = normize(stack([-o[...,1],o[...,0],0*o[...,2]],-1))
    # Cross product with local up vector of [0,0,-1]
    y_o = cross(x_o,o)
    phi_o = atan2(-dot(n_o,x_o), dot(n_o,y_o))
    psi_o = atan2(-dot(h_o, x_o), dot(h_o, y_o))
    # Using Dirs_up in local coordinates
    # phi_o = atan2(n_o[...,[0]],n_o[...,[1]])
    # psi_o = atan2(h_o[...,[0]],h_o[...,[1]]) 
    # Variables for Fresnel
    # incidence
    eta_i_1, eta_i_2 = 1.0, eta
    theta_i_1  = acos(dot(n,o))
    theta_i_2 = acos(sqrt(clamp(1-(sin(theta_i_1)/eta)**2,
                                1e-7))) 

    # exitance
    eta_o_1, eta_o_2 = eta, 1.0
    theta_o_2  = acos(dot(n,o))
    theta_o_1 = acos(sqrt(clamp(1-(sin(theta_o_2)/eta)**2,
                                1e-7))) 
    # reflectance
    theta_d = acos(dot(h,o))
    eta_r_1, eta_r_2 = 1.0, eta
    theta_r_1 = theta_d
    theta_r_2 = acos(sqrt(clamp(1-(sin(theta_r_1)/eta)**2,
                                1e-7))) 
    
    # Transmission components
    T_i__perp = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_1)+eta_i_2*cos(theta_i_2))**2,
                1e-7)
    T_i__perp = T_i__perp*(cos(theta_i_1)>1e-7)
    T_i__para = (2*eta_i_1*cos(theta_i_1))**2\
    /clamp((eta_i_1*cos(theta_i_2)+eta_i_2*cos(theta_i_1))**2,
                1e-7)
    T_i__para = T_i__para*(cos(theta_i_1)>1e-7)
    T_i__plus, T_i__min = 0.5*(T_i__perp+T_i__para), 0.5*(T_i__perp-T_i__para)
    # exitance
    T_o__perp = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_1)+eta_o_2*cos(theta_o_2))**2,
                1e-7)
    T_o__para = (2*eta_o_1*cos(theta_o_1))**2\
    /clamp((eta_o_1*cos(theta_o_2)+eta_o_2*cos(theta_o_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)

    # Reflection components
    R__perp = (eta_r_1*cos(theta_r_1)-eta_r_2*cos(theta_r_2))**2\
    /clamp((eta_r_1*cos(theta_r_1)+eta_r_2*cos(theta_r_2))**2,
                1e-7)
    R__para = (eta_r_1*cos(theta_r_2)-eta_r_2*cos(theta_r_1))**2\
    /clamp((eta_r_1*cos(theta_r_2)+eta_r_2*cos(theta_r_1))**2,
                1e-7)
    T_o__plus, T_o__min = 0.5*(T_o__perp+T_o__para), 0.5*(T_o__perp-T_o__para)
    R__plus, R__min = 0.5*(R__perp+R__para), 0.5*(R__perp-R__para)

    # R__plus = clip_max(clamp(R__plus, 0.04),1.)
    # R__min = clip_max(clamp(R__min, 0.),0.16)

    # Exitant  stokes  Unpolarized illumination
    stokes_diff_fac = stack([ 1+0.*T_o__min,
                        T_o__min/T_o__plus*cos(2*phi_o),
                        -T_o__min/T_o__plus*sin(2*phi_o)],
                        -1) 
                        # (H, W, Num_lights, 1, 3)
    # if train_mode:
    #     stokes_diff_fac = stack([ 1+0.*T_o__min,
    #                           T_o__min/T_o__plus*cos(2*phi_o),
    #                          -T_o__min/T_o__plus*sin(2*phi_o)],
    #                          -1) 
    #                         # (H, W, Num_lights, 1, 3)
    # else:
    #     stokes_diff_fac = stack([ T_o__plus,
    #                               T_o__min*cos(2*phi_o),
    #                              -T_o__min*sin(2*phi_o)],
    #                              -1) 
    #                             # (H, W, Num_lights, 1, 3)
    # if not train_mode:
    #     stokes_diff_fac = stokes_diff_fac*T_i__plus[...,None]

    # Mask the regions where angles at interface are larger than 90
    # diff_mask = dot(n,o) > 1e-7
    # diff_mask = diff_mask*(dot(n,o)>1e-7)
    # stokes_diff_fac = mask_fn(stokes_diff_fac, diff_mask[...,None])

    stokes_spec_fac = stack([  1+0.*R__plus,
                                R__min/R__plus*cos(2*psi_o),
                                -R__min/R__plus*sin(2*psi_o)],-1)
                            # (H,W,Num_lights,1,3)
    
    if clip_spec:
        spec_mask = dot(h,o) > 1e-7
        spec_mask = spec_mask*(dot(h,o)>1e-7)
        stokes_spec_fac = mask_fn(stokes_spec_fac, spec_mask[..., None])
        R__plus = mask_fn(R__plus, spec_mask)

    # stokes_diff_fac = stokes_diff_fac*T_o__plus[...,None]

    return stokes_diff_fac, stokes_spec_fac

