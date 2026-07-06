import torch
import einops as ein
from src_pr.grad_utils import *
from torchvision import transforms
import solidspy.uelutil as ue
import torch.nn.functional as F
from einops import rearrange
import cv2

def resize_image(tensor, target_size):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    assert len(tensor.shape) > 3, f"Expected image, got {tensor.shape}"
    original_shape = tensor.shape
    batch_size = original_shape[0]
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b' + ' (' + ' '.join([f'c{i}' for i in range(num_dims)]) + ') ' + 'x y'
    tensor = ein.rearrange(tensor, pattern)
    tensor = transforms.Resize((target_size, target_size), antialias=False)(tensor).view(batch_size, *original_shape[1:-2], target_size, target_size)
    return tensor

class StiffnessMatrix:
    def __init__(self, no_BC_folder, nels_per_side = 64, ndof = 8, device = 'cpu', dtype = torch.float32):

        self.ndof = ndof
        self.nels = nels_per_side**2
        nodes, mats, elements, loads = self.readin(folder = no_BC_folder)
        tot_local_stiffness = np.zeros((self.nels, 8, 8))
        default_mats = np.ones_like(mats)
        default_mats[:, 1] = 0.3
        for ele in range(self.nels):
            tot_local_stiffness[ele] = self.retriever(elements, default_mats, nodes[:, :3], ele)[0]
        assembler, neq = self.DME(nodes[:, -2:], elements, ndof_node=2, ndof_el_max=18, ndof_el=8)
        self.neq = neq
        eles = torch.arange(self.nels)
        glob_assembler = assembler[eles][:,:ndof].to(device)

        self.tot_local_stiffness = torch.tensor(tot_local_stiffness, dtype=dtype).to(device)
        self.indices_ext = torch.cartesian_prod(torch.arange(ndof), torch.arange(ndof)).to(device)
        self.glob_assembler_idcs = glob_assembler[:, self.indices_ext].to(device)

    def readin(self, folder=""):
        """Read the input files"""
        nodes = np.loadtxt(folder + 'nodes.txt', ndmin=2)
        mats = np.loadtxt(folder + 'mater.txt', ndmin=2)
        elements = np.loadtxt(folder + 'eles.txt', ndmin=2, dtype=int)
        loads = np.loadtxt(folder + 'loads.txt', ndmin=2)
        return nodes, mats, elements, loads

    def eqcounter(self, cons, ndof_node=2):
        nnodes = cons.shape[0]
        bc_array = cons.copy().astype(int)
        neq = 0
        for i in range(nnodes):
            for j in range(ndof_node):
                if bc_array[i, j] == 0:
                    bc_array[i, j] = neq
                    neq += 1
        return neq, bc_array

    def DME(self, cons, elements, ndof_node=2, ndof_el_max=18, ndof_el=8):
        nels = elements.shape[0]
        assem_op = np.zeros([nels, ndof_el_max], dtype=int)
        neq, bc_array = self.eqcounter(cons, ndof_node=ndof_node)
        ndof = ndof_el
        for ele in range(nels):
            assem_op[ele, :ndof] = bc_array[elements[ele, 3:]].flatten() # only last 4 elements define the nodes
        return torch.tensor(assem_op), neq

    def loadasem(self, loads, bc_array, neq, ndof_node=2):
        nloads = loads.shape[0]
        rhs_vec = np.zeros([neq])
        for cont in range(nloads):
            node = int(loads[cont, 0])
            for dof in range(ndof_node):
                dof_id = bc_array[node, dof]
                if dof_id != -1:
                    rhs_vec[dof_id] = loads[cont, dof + 1]
        return rhs_vec

    def image_to_stiffness_coord(self, image_coord, dof, tot_dofs=2):
        batch_size, height, width = image_coord.shape
        stiffness_coord = torch.zeros((batch_size, height * width, tot_dofs), dtype=image_coord.dtype, device=image_coord.device)
        image_coord = rearrange(image_coord, 'b x y -> b (x y)')
        stiffness_coord[:, :, dof] = image_coord
        return rearrange(stiffness_coord, 'b (x y) d -> b (x y d)', x = height, y = width, d=tot_dofs)

    def stiffness_to_image_coord(self, stiffness_flat, dof, tot_dofs=2):
        # add batch dim if nonexisting
        if len(stiffness_flat.shape) == 1:
            stiffness_flat = stiffness_flat.unsqueeze(0)
        nodes = stiffness_flat.shape[1] // tot_dofs
        sqrt_nodes = int(nodes ** 0.5)
        assert sqrt_nodes ** 2 == nodes, "The number of nodes is not a perfect square."
        stiffness_reshaped = rearrange(stiffness_flat, 'b (x y d) -> b x y d', x = sqrt_nodes, y = sqrt_nodes, d = tot_dofs)
        return stiffness_reshaped[:, :, :, dof]

    def retriever(self, elements, mats, nodes, ele, uel=ue.elast_quad4):
        params = mats[elements[ele, 2], :]
        elcoor = nodes[elements[ele, 3:], 1:]
        kloc, mloc = uel(elcoor, params)
        return kloc, mloc

class ResidualsMechanics:
    def __init__(self, model, pixels_per_dim, pixels_at_boundary, no_BC_folder, device = 'cpu', bcs = 'none', E=1.0, nu=0.3, topopt_eval = False, use_ddim_x0 = False, ddim_steps = 0):
        """
        Initialize the residual evaluation.

        :param model: The neural network model to compute the residuals for.
        :param n_steps: Number of steps for time discretization.
        :param E: Young's Modulus.
        :param nu: elastic material parameter.
        """
        self.gov_eqs = 'mechanics'
        self.model = model # do not change this since we use model in main class (not as a direct instance of ResidualsMechanics)
        self.stiffs = StiffnessMatrix(no_BC_folder = no_BC_folder, device=device, dtype=torch.float32)
        self.deriv_mode = None
        self.pixels_at_boundary = pixels_at_boundary
        self.E = E  # Young's Modulus # NOTE: not required here as we directly use the stiffness matrix from Solidspy (which is based on E=1 and nu=0.3)
        self.nu = nu  # same as above
        self.periodic = False
        self.device = device
        if bcs == 'periodic':
            self.periodic = True

        self.relu = torch.nn.ReLU()
        self.pixels_per_dim = pixels_per_dim
        
        if self.pixels_at_boundary:
            # adjust for boundary pixels via trapezoidal rule
            self.use_trapezoid = True
        else:
            # no need for boundary adjustments, simply take mean
            self.use_trapezoid = False

        if self.use_trapezoid:
            self.trapezoidal_weights = self.create_trapezoidal_weights()

        self.topopt_eval = topopt_eval

        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

    def create_trapezoidal_weights(self):        
        # identify corner nodes
        trapezoidal_weights = torch.zeros((1, self.pixels_per_dim, self.pixels_per_dim))
        trapezoidal_weights = trapezoidal_weights.to(self.device)
        trapezoidal_weights[..., 0,0] = 1.
        trapezoidal_weights[..., 0,-1] = 1.
        trapezoidal_weights[..., -1,0] = 1.
        trapezoidal_weights[..., -1,-1] = 1.
        # identify boundary nodes
        trapezoidal_weights[..., 1:-1,0] = 2.
        trapezoidal_weights[..., 1:-1,-1] = 2.
        trapezoidal_weights[..., 0,1:-1] = 2.
        trapezoidal_weights[..., -1,1:-1] = 2.
        # identify interior nodes
        trapezoidal_weights[..., 1:-1,1:-1] = 4.
        # assert that no node is 0
        assert torch.all(trapezoidal_weights != 0)
        trapezoidal_weights *= (1./self.pixels_per_dim)**2 / 4.
        trapezoidal_weights = generalized_image_to_b_xy_c(trapezoidal_weights)
        return trapezoidal_weights

    def compute_residual(self, input_tuple, reduce='none', return_model_out = False, return_optimizer = False, return_inequality = False, sample = False, ddim_func = None, pass_through = False, return_projections = False, skip_model_call = False, given_model_output = None):
        self.deriv_mode = 'stiffness'

        # assemble stiffness matrix
        input = input_tuple[0]
        bcs = input_tuple[1]
        vf = input_tuple[2]
        bc_x, bc_y, load_x, load_y = torch.chunk(bcs, 4, dim=1)
        bcs_red = resize_image(bcs, 64)

        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
        else:
            assert len(input) == 2 and isinstance(input, tuple), 'Input must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(input)
            noisy_in = generalized_b_xy_c_to_image(noisy_in)
            noisy_in_red = resize_image(noisy_in, 64)

        # Projection-output bookkeeping.
        projections = None
        
        if skip_model_call and given_model_output is not None:
            # Use the provided output for projection-head residuals.
            x0_pred = generalized_b_xy_c_to_image(given_model_output) if len(given_model_output.shape) == 3 else given_model_output
            model_out = x0_pred
        elif pass_through:
            x0_pred = input
            model_out = x0_pred
        else:
            # model requires 10 channels (fields, vf, strain_energy_dens, von-mises, 4 boundary conditions)
            noisy_in_red = torch.cat((noisy_in_red, bcs_red), dim=1)
            
            # Prepare model-call arguments.
            call_kwargs = {}
            if return_projections and hasattr(self.model, 'use_projection_heads') and getattr(self.model, 'use_projection_heads'):
                call_kwargs['return_projections'] = True
            
            if self.use_ddim_x0:
                x0_pred, model_out = ddim_func(noisy_in_red, time, self.model, noisy_in.shape, self.ddim_steps, 0., gov_eqs = 'mechanics')
            else:
                result = self.model(noisy_in_red, time, **call_kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    x0_pred, projections = result
                else:
                    x0_pred = result
                model_out = x0_pred
        assert len(x0_pred.shape) == 4, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape
                
        displacements = x0_pred[:,:-1]
        rho = x0_pred[:,-1]
        rho_flatten = rearrange(rho, 'b x y -> b (x y)')

        # upscale displacements
        displacements = resize_image(displacements, 65)
        displacements_x_stiff = self.stiffs.image_to_stiffness_coord(displacements[:,0], 0)
        displacements_y_stiff = self.stiffs.image_to_stiffness_coord(displacements[:,1], 1)
        displacements_stiff = displacements_x_stiff+displacements_y_stiff
        # extend indices to batch
        global_batch_idcs = torch.arange(batch_size).repeat_interleave(self.stiffs.nels*(self.stiffs.ndof**2)).to(displacements.device)
        glob_assembler_idcs_ext = self.stiffs.glob_assembler_idcs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # initialize the global stiffness matrix
        k_closed_glob_vec_batched_temp = torch.zeros((batch_size, self.stiffs.neq, self.stiffs.neq), dtype = displacements.dtype, device = displacements.device)
        # scale the local stiffness matrices (which is constant) over the batch
        scaled_kloc = self.stiffs.tot_local_stiffness.unsqueeze(0) * rho_flatten[:, :, None, None]
        # extract the values according to dofs
        scaled_kloc_val = scaled_kloc[:, :, self.stiffs.indices_ext[:,0], self.stiffs.indices_ext[:,1]]
        # use advanced indexing to sum the contributions
        # would be nice to use sparse matrices here, but not yet supported by index_put_
        k_closed_glob_vec_batched = k_closed_glob_vec_batched_temp.index_put((global_batch_idcs.flatten(), glob_assembler_idcs_ext[:,:,:,0].flatten(), glob_assembler_idcs_ext[:,:,:,1].flatten()), scaled_kloc_val.flatten(), accumulate=True)
        load_x_stiff = self.stiffs.image_to_stiffness_coord(load_x.squeeze(1), 0)
        load_y_stiff = self.stiffs.image_to_stiffness_coord(load_y.squeeze(1), 1)
        f_glob = load_x_stiff+load_y_stiff

        BC_node_x_stiff = self.stiffs.image_to_stiffness_coord(bc_x.squeeze(1), 0)
        BC_node_y_stiff = self.stiffs.image_to_stiffness_coord(bc_y.squeeze(1), 1)

        # replace rows in stiffness matrix that are defined with BCs with 1 at diagonal and 0 elsewhere
        # Identify the indices where BC_node_x_stiff and BC_node_y_stiff are not zero
        mask = BC_node_x_stiff+BC_node_y_stiff != 0
        # Zero out the rows where the mask is True
        mask_ext = mask.unsqueeze(-1).expand_as(k_closed_glob_vec_batched)
        k_closed_glob_vec_batched[mask_ext] = 0
        # create mask that is True where mask is False and if on diagonal
        identity = torch.eye(self.stiffs.neq, device=k_closed_glob_vec_batched.device, dtype=k_closed_glob_vec_batched.dtype).expand(batch_size, -1, -1)
        identity_masked = identity * mask_ext
        # Set the diagonal elements to 1 where the mask is True
        k_closed_glob_vec_batched += identity_masked
        # Set the corresponding elements in f_glob to 0
        f_glob[mask] = 0

        residual = ein.einsum(k_closed_glob_vec_batched, displacements_stiff, 'b i j, b j -> b i') - f_glob
        
        output = {}
        output['residual'] = residual

        if return_model_out:            
            displacements_model_out = model_out[:,:-1]
            rho_model_out = model_out[:,-1]
            # upscale displacements
            displacements_model_out = resize_image(displacements_model_out, 65)
            # pad E similar to dataset
            pad_width = (0, 1, 0, 1)
            # pad the array with 0s at the end
            rho_padded_model_out = F.pad(rho_model_out, pad=pad_width, mode='constant', value = 0)
            model_out = torch.cat((displacements_model_out, rho_padded_model_out.unsqueeze(1)), dim=1)
            output['model_out'] = model_out

        if return_optimizer:
            # compliance = ein.einsum(displacements_stiff, f_glob, 'b i, b i -> b')
            compliance = ein.einsum(displacements_stiff, k_closed_glob_vec_batched, displacements_stiff, 'b i, b i j, b j -> b') # NOTE this works much better since this is not sparse
            # compliance = -ein.einsum(displacements_stiff.detach(), k_closed_glob_vec_batched, displacements_stiff.detach(), 'b i, b i j, b j -> b') # HACK see NTopo
            output['optimizer'] = compliance

        if return_inequality:
            # NOTE: ToPy just uses binary solid/void ratios here, not considering the small values for the voids. We here use the actual E values. Difference should be negligible though.
            threshold = vf
            mode = 'leq'

            shift = rho_flatten.mean(1) - threshold
            if mode == 'leq':
                ineq_loss = self.relu(shift)
            elif mode == 'geq':
                ineq_loss = self.relu(-shift)
            # output['inequality'] = ineq_loss
            output['inequality'] = shift # NOTE: we here consider the volume missmatch as an equality constraint

        with torch.no_grad():
            if self.topopt_eval and sample:
                solution = input_tuple[3]
                opt_disp = solution[:, :2]
                rho_simp = solution[:, 2,:-1, :-1] # remove padding here
                rho_simp_flatten = rearrange(rho_simp, 'b x y -> b (x y)')
                # compliance of data (opt_disp)
                disp_data_x_stiff = self.stiffs.image_to_stiffness_coord(opt_disp[:,0], 0)
                disp_data_y_stiff = self.stiffs.image_to_stiffness_coord(opt_disp[:,1], 1)
                disp_data_stiff = disp_data_x_stiff+disp_data_y_stiff

                # initialize the global stiffness matrix
                k_closed_glob_vec_batched = torch.zeros((batch_size, self.stiffs.neq, self.stiffs.neq), dtype = displacements.dtype, device = displacements.device)
                scaled_kloc = self.stiffs.tot_local_stiffness.unsqueeze(0) * rho_simp_flatten[:, :, None, None]
                scaled_kloc_val = scaled_kloc[:, :, self.stiffs.indices_ext[:,0], self.stiffs.indices_ext[:,1]]
                k_closed_glob_vec_batched.index_put_((global_batch_idcs.flatten(), glob_assembler_idcs_ext[:,:,:,0].flatten(), glob_assembler_idcs_ext[:,:,:,1].flatten()), scaled_kloc_val.flatten(), accumulate=True)
                
                BC_node_x_stiff = self.stiffs.image_to_stiffness_coord(bc_x.squeeze(1), 0)
                BC_node_y_stiff = self.stiffs.image_to_stiffness_coord(bc_y.squeeze(1), 1)

                mask = BC_node_x_stiff+BC_node_y_stiff != 0
                mask_ext = mask.unsqueeze(-1).expand_as(k_closed_glob_vec_batched)
                k_closed_glob_vec_batched[mask_ext] = 0
                identity = torch.eye(self.stiffs.neq, device=k_closed_glob_vec_batched.device, dtype=k_closed_glob_vec_batched.dtype).expand(batch_size, -1, -1)
                identity_masked = identity * mask_ext
                k_closed_glob_vec_batched += identity_masked

                # sanity check - residual of opt_disp should be zero
                residual_data = ein.einsum(k_closed_glob_vec_batched, disp_data_stiff, 'b i j, b j -> b i') - f_glob
                assert torch.isclose(torch.abs(residual_data).mean(), torch.tensor(0., device=residual_data.device), atol = 1.e-5), 'Residual of opt_disp is not zero.'
                compliance_data = ein.einsum(disp_data_stiff, f_glob, 'b i, b i -> b')

                # actual compliance based on predicted binarized field and true displacements (solved via FEM)
                rho_flatten_binarized = rho_flatten.clone()
                rho_flatten_binarized[rho_flatten_binarized > 0.5] = 1
                rho_flatten_binarized[rho_flatten_binarized <= 0.5] = 1.e-3

                # initialize the global stiffness matrix
                k_closed_glob_vec_batched = torch.zeros((batch_size, self.stiffs.neq, self.stiffs.neq), dtype = displacements.dtype, device=displacements.device)

                scaled_kloc = self.stiffs.tot_local_stiffness.unsqueeze(0) * rho_flatten_binarized[:, :, None, None]
                scaled_kloc_val = scaled_kloc[:, :, self.stiffs.indices_ext[:,0], self.stiffs.indices_ext[:,1]]
                k_closed_glob_vec_batched.index_put_((global_batch_idcs.flatten(), glob_assembler_idcs_ext[:,:,:,0].flatten(), glob_assembler_idcs_ext[:,:,:,1].flatten()), scaled_kloc_val.flatten(), accumulate=True)
                BC_node_x_stiff = self.stiffs.image_to_stiffness_coord(bc_x.squeeze(1), 0)
                BC_node_y_stiff = self.stiffs.image_to_stiffness_coord(bc_y.squeeze(1), 1)

                mask = BC_node_x_stiff+BC_node_y_stiff != 0
                mask_ext = mask.unsqueeze(-1).expand_as(k_closed_glob_vec_batched)
                k_closed_glob_vec_batched[mask_ext] = 0
                identity = torch.eye(self.stiffs.neq, device=k_closed_glob_vec_batched.device, dtype=k_closed_glob_vec_batched.dtype).expand(batch_size, -1, -1)
                identity_masked = identity * mask_ext
                k_closed_glob_vec_batched += identity_masked

                compliance_true_binarized = torch.zeros(batch_size, dtype=displacements.dtype, device=displacements.device)
                for i in range(batch_size):
                    u_sol = torch.linalg.solve(k_closed_glob_vec_batched[i], f_glob[i])
                    compliance_true_binarized[i] = ein.einsum(u_sol, f_glob[i], 'i, i ->')

                rel_CE_error = (compliance_true_binarized - compliance_data) / compliance_data
                output['rel_CE_error_full_batch'] = rel_CE_error

                # volume fraction error
                # NOTE small inconsistency since model evaluates vf via rho_flatten_binarized and not rho_flatten_binarized_zeros, but should be negligible
                rho_flatten_binarized_zeros = rho_flatten_binarized.clone()
                rho_flatten_binarized_zeros[rho_flatten_binarized <= 0.5] = 0
                vf_error = torch.abs(rho_flatten_binarized.mean(1) - vf) / vf
                output['vf_error_full_batch'] = vf_error

                # floating material presence
                batch_size = rho_flatten_binarized.shape[0]
                fm = compute_fm(rearrange(rho_flatten_binarized, 'b (x y) -> b x y', b = batch_size, x = self.pixels_per_dim, y = self.pixels_per_dim))
                output['fm_error_full_batch'] = fm

                # # OPTIONAL: actual compliance based on as-is field in case this is of interest
                # compliance_true = torch.zeros(batch_size, dtype=displacements.dtype, device=displacements.device)
                # for i in range(batch_size):
                #     u_sol = torch.linalg.solve(k_closed_glob_vec_batched[i], f_glob[i])
                #     compliance_true[i] = ein.einsum(u_sol, f_glob[i], 'i, i ->')
                # output['optimizer_true'] = compliance_true

        # Attach projection outputs when requested.
        if return_projections and (projections is not None):
            output['projections'] = projections

        if reduce == 'full':
            # mean over all items in dict
            return {k: v.mean() for k, v in output.items()}
        elif reduce == 'per-batch':
            # mean over all but first dimension (batch dimension) 
            # only if tensor has more than one dimension and key is not in excluded list
            excluded_keys = ['model_out', 'residual', 'projections']
            return {k: v.mean(dim=tuple(range(1, v.ndim))) if hasattr(v, 'ndim') and v.ndim > 1 and k not in excluded_keys else v for k, v in output.items()}
        elif reduce == 'none':
            # return as-is
            return output
        else:
            raise ValueError('Unknown reduction method.')
        
def compute_fm(gen):
    gen = gen.detach().cpu().numpy()
    res = torch.empty(len(gen), dtype = int)
    for i in range(len(gen)):
        res[i] = check_floating_material(gen[i])
    return res

def check_floating_material(image):
    _, im = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    im = im.astype(np.uint8)
    sa = cv2.connectedComponents(im)
    return sa[0] != 2