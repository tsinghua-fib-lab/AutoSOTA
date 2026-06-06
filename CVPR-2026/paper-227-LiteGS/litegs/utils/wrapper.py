import torch
import typing
import numpy as np
import math
from torch.cuda import nvtx

from .platform import add_cmake_output_path
from . import spherical_harmonics
from ..utils.statistic_helper import StatisticsHelperInst


try:
    import litegs_fused
except:
    add_cmake_output_path()
    import litegs_fused


class BaseWrapper:
    '''
    Base class for comparing the forward and backward results of two functions.
    '''

    _fused: typing.Callable = None  # Optimized function to be tested
    _script: typing.Callable = None  # Reference implementation
    _absolute_error_threshold = 1e-5  # Threshold for absolute error comparison
    _relative_error_threshold = 1e-3  # Threshold for relative error comparison

    test_inputs: list[tuple[list[int], typing.Any]] = []
    '''Input Parameters for testing. 

    (list[int],dtypes,requires_grad) for random input generate by torch.randn

    (typing.Any,None,None) for constant parameter for testing'''

    @staticmethod
    def compute_forward_and_backward(func: typing.Callable, input_tensors: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute forward and backward passes for a given function.

        Args:
            func (typing.Callable): The function to evaluate.
            input_tensors (list[torch.Tensor]): List of input tensors with gradients enabled.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: 
                - Forward outputs as a list of tensors.
                - Gradients of the input tensors.
        """
        forward_outputs_list = []
        gradients_list = []

        # Forward pass
        forward_outputs: list[torch.Tensor] = func(*input_tensors)
        if isinstance(forward_outputs, torch.Tensor):
            forward_outputs = [forward_outputs]

        # Compute sum of outputs for backward pass
        total_output_sum = 0
        for output_data in forward_outputs:
            if isinstance(output_data, torch.Tensor):
                forward_outputs_list.append(output_data.detach())
                total_output_sum += output_data.sum()
            else:
                forward_outputs_list.append(output_data)
        
        # Backward pass
        if total_output_sum.requires_grad:
            total_output_sum.backward()
            for input_tensor in input_tensors:
                if isinstance(input_tensor, torch.Tensor) and input_tensor.requires_grad:
                    gradients_list.append(input_tensor.grad)
                    input_tensor.grad = None
                else:
                    gradients_list.append(None)

        return forward_outputs_list, gradients_list

    @classmethod
    def compare_tensors(cls, outputs_1: list[torch.Tensor], outputs_2: list[torch.Tensor], phase: str) -> bool:
        """
        Compare two lists of tensors for similarity within error thresholds.

        Args:
            outputs_1 (list[torch.Tensor]): First list of tensors (e.g., from fused function).
            outputs_2 (list[torch.Tensor]): Second list of tensors (e.g., from script function).
            phase (str): A label indicating the phase of comparison (e.g., 'forward', 'backward').

        Returns:
            bool: True if all tensors are within the defined error thresholds; False otherwise.
        """
        tensors_match = True

        if len(outputs_1) != len(outputs_2):
            print(f"[{cls.__name__}-{phase}]: Mismatch in the number of tensors.")
            return False

        for i, (tensor_1, tensor_2) in enumerate(zip(outputs_1, outputs_2)):
            
            if tensor_1.__class__ != tensor_2.__class__:
                print(f"[{cls.__name__}-{phase}]: Ojbect #{i} does not match.")
                tensors_match = False
                continue

            if isinstance(tensor_1, torch.Tensor):
                absolute_error = (tensor_1 - tensor_2).abs()
                relative_error = absolute_error / tensor_2.abs()

                within_threshold = ((absolute_error < cls._absolute_error_threshold) | (relative_error < cls._relative_error_threshold)).all()

                if not within_threshold:
                    print(f"[{cls.__name__}-{phase}]: Tensor #{i} does not match.")
                    tensors_match = False

        return tensors_match
    
    @classmethod
    def gen_inputs(cls):
        inputs = []
        for obj, obj_type,requires_grad in cls.test_inputs:
            if obj_type is not None:
                obj = torch.randn(obj, dtype=obj_type, device='cuda', requires_grad=requires_grad)
            inputs.append(obj)
        return inputs

    @classmethod
    def validate(cls):
        """
        Validate the consistency between the `fused` and `script` functions for forward and backward passes.

        Returns:
            bool: True if both forward and backward results match; False otherwise.
        """
        inputs=cls.gen_inputs()

        fused_forward, fused_grads = cls.compute_forward_and_backward(cls.call_fused, inputs)
        script_forward, script_grads = cls.compute_forward_and_backward(cls.call_script, inputs)

        forward_match = cls.compare_tensors(fused_forward, script_forward, 'forward')
        backward_match = cls.compare_tensors(fused_grads, script_grads, 'backward')

        if forward_match and backward_match:
            print(f"[{cls.__name__}]: Validation successful.")
            return True
        return False
    
    @classmethod
    def call_fused(cls, *args, **kwargs):
        return cls._fused(*args, **kwargs)

    @classmethod
    def call_script(cls, *args, **kwargs):
        return cls._script(*args, **kwargs)
    
    @classmethod
    def call(cls, *args, **kwargs):
        return cls._fused(*args, **kwargs)

def check():
    for wrapper_class in BaseWrapper.__subclasses__():
        wrapper_class.validate()
    return

class CreateTransformMatrix(BaseWrapper):
    """
    A wrapped class for creating 3D transformation matrices.

    This class provides implementations for generating 3D transformation matrices based on scaling vectors and quaternion-based rotation vectors.
    Users can invoke the computations through `call_fused`, `call_script`, or `call` methods.

    Args:
        scaling_vec (torch.Tensor): A tensor of shape [3, num_points] representing scaling factors for the transformation along x, y, and z axes.
        rotator_vec (torch.Tensor): A tensor of shape [4, num_points] containing quaternion components (r, x, y, z) for rotation.

    Returns:
        torch.Tensor: A 3D transformation matrix of shape [3, 3, num_points], where each slice corresponds to the transformation for one point.
    """
    def __create_transform_matrix_fused(scaling_vec:torch.Tensor,rotator_vec:torch.Tensor)->torch.Tensor:

        class CreateTransformMatrixFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx,quaternion:torch.Tensor,scale:torch.Tensor):
                ctx.save_for_backward(quaternion,scale)
                transform_matrix=litegs_fused.createTransformMatrix_forward(quaternion,scale)
                return transform_matrix
            
            @staticmethod
            def backward(ctx,grad_transform_matrix:torch.Tensor):
                (quaternion,scale)=ctx.saved_tensors
                grad_quaternion,grad_scale=litegs_fused.createTransformMatrix_backward(grad_transform_matrix,quaternion,scale)
                return grad_quaternion,grad_scale
            
        transform_matrix=CreateTransformMatrixFunc.apply(rotator_vec,scaling_vec)
        return transform_matrix

    def __create_transform_matrix_script(scaling_vec:torch.Tensor,rotator_vec:torch.Tensor)->torch.Tensor:
        rotation_matrix=torch.zeros((3,3,rotator_vec.shape[-1]),device='cuda')

        r=rotator_vec[0]
        x=rotator_vec[1]
        y=rotator_vec[2]
        z=rotator_vec[3]


        rotation_matrix[0,0]=1 - 2 * (y * y + z * z)
        rotation_matrix[0,1]=2 * (x * y + r * z)
        rotation_matrix[0,2]=2 * (x * z - r * y)

        rotation_matrix[1,0]=2 * (x * y - r * z)
        rotation_matrix[1,1]=1 - 2 * (x * x + z * z)
        rotation_matrix[1,2]=2 * (y * z + r * x)

        rotation_matrix[2,0]=2 * (x * z + r * y)
        rotation_matrix[2,1]=2 * (y * z - r * x)
        rotation_matrix[2,2]=1 - 2 * (x * x + y * y)

        transform_matrix=rotation_matrix*scaling_vec.unsqueeze(1)
        return transform_matrix

    _fused=__create_transform_matrix_fused
    _script=__create_transform_matrix_script
    test_inputs=[([3,1024*512],torch.float32,True),
                  ([4,1024*512],torch.float32,True)]

class CreateRaySpaceTransformMatrix(BaseWrapper):
    """
    A wrapped class for creating ray-space transformation matrices.

    This class provides methods to compute the transformation matrices in ray space using both a fused implementation and a script-based implementation.
    The transformations are calculated based on the positions of points in 3D space, a view matrix, and camera focal lengths.

    Args:
        point_positions (torch.Tensor): A tensor representing the 3D positions of points with shape [4, num_points].
        view_matrix (torch.Tensor): A tensor representing the camera view matrix with shape [num_views, 4, 4].
        proj_matrix (torch.Tensor): A tensor representing the focal lengths of the camera with shape [num_views, 4, 4].
        output_shape (tuple[int,int]): ...
        bTranspose (bool, optional): A flag indicating whether to transpose certain matrix components during the computation. Default is True.

    Returns:
        torch.Tensor: A ray-space transformation matrix with shape [num_views, 3, 3, num_points].
    """
    @torch.no_grad()
    def __create_rayspace_transform_script(point_positions:torch.Tensor,view_matrix:torch.Tensor,proj_matrix:torch.Tensor,output_shape:tuple[int,int],bTranspose:bool=True)->torch.Tensor:
        t=torch.matmul(view_matrix.transpose(-1,-2),point_positions)
        t[:,2].clamp_(1e-2)#near plane 0.01
        J=torch.zeros((t.shape[0],3,3,t.shape[-1]),device=t.device)#view point mat3x3
        tz_square=t[:,2]*t[:,2]
        focal_length_x=output_shape[1]*proj_matrix[:,0,0]*0.5
        focal_length_y=output_shape[0]*proj_matrix[:,1,1]*0.5
        J[:,0,0]=focal_length_x/t[:,2]#focal x
        J[:,1,1]=focal_length_y/t[:,2]#focal y
        if bTranspose:
            J[:,0,2]=-(focal_length_x*t[:,0])/tz_square
            J[:,1,2]=-(focal_length_y*t[:,1])/tz_square
        else:
            J[:,2,0]=-(focal_length_x*t[:,0])/tz_square
            J[:,2,1]=-(focal_length_y*t[:,1])/tz_square
        return J

    @torch.no_grad()
    def __create_rayspace_transform_fused(point_positions:torch.Tensor,view_matrix:torch.Tensor,proj_matrix:torch.Tensor,output_shape:tuple[int,int],bTranspose:bool=True)->torch.Tensor:
        t=torch.matmul(view_matrix.transpose(-1,-2),point_positions)
        J=litegs_fused.jacobianRayspace(t,proj_matrix,output_shape[0],output_shape[1],bTranspose)
        return J
    
    _fused=__create_rayspace_transform_fused
    _script=__create_rayspace_transform_script
    test_inputs=[([4,1024*512],torch.float32,True),
                 ([1,4,4],torch.float32,False),
                 ([1,4,4],torch.float32,False),
                 ((1080,1920),None,None),
                 (True,None,None)]

class World2NdcFunc(torch.autograd.Function):
    '''
    A custom autograd function for transforming world coordinates to normalized device coordinates (NDC).

    This implementation overrides the backward computation to address potential floating-point precision issues 
    that may arise in the standard autograd process for `world2ndc` transformations.

    Args:
        position (torch.Tensor): Input tensor representing world coordinates with shape [4, num_points].
        view_project_matrix (torch.Tensor): View-projection matrix with shape [num_views, 4, 4].
    Returns:
        torch.Tensor: Normalized device coordinates (NDC) with shape [num_views, 4, num_points].
    '''
    @staticmethod
    def forward(ctx,position:torch.Tensor,view_project_matrix:torch.Tensor):
        hom_pos=torch.matmul(view_project_matrix.transpose(-1,-2),position)
        repc_hom_w=1/(hom_pos[:,3:4]+1e-7)
        ndc_pos=hom_pos*repc_hom_w
        ctx.save_for_backward(view_project_matrix,ndc_pos,repc_hom_w)
        return ndc_pos
    
    @staticmethod
    def backward(ctx,grad_ndc_pos:torch.Tensor):
        (view_project_matrix,ndc_pos,repc_hom_w)=ctx.saved_tensors
        position_grad=litegs_fused.world2ndc_backword(view_project_matrix,ndc_pos,repc_hom_w,grad_ndc_pos)
        return (position_grad,None)

class CreateCovarianceMatrixFunc(torch.autograd.Function):
    '''
    A custom autograd function for efficiently computing the forward and backward passes of gaussian 3D covariance matrix.

    This function assumes the input `transforms` is a symmetric matrix and optimizes the computation of the backward pass.

    Args:
        transforms (torch.Tensor): Input transform matrix of shape [num_views, num_points, 3, 3]. Assumed to be symmetric.
    Returns:
        torch.Tensor: 3D covariance matrix of shape [num_views, num_points, 3, 3].
    '''
    @staticmethod
    def forward(ctx,transforms:torch.Tensor):
        ctx.save_for_backward(transforms)
        cov=transforms.transpose(-1,-2).contiguous()@transforms
        return cov
    
    @staticmethod
    def backward(ctx,CovarianceMatrixGradient:torch.Tensor):
        (transforms,)=ctx.saved_tensors
        return (2*transforms@CovarianceMatrixGradient)

class ProjCov3dTo2dFunc(torch.autograd.Function):
    """
    A custom autograd function for projecting a 3D covariance matrix to a 2D covariance matrix.

    This function assumes the input `cov3d` and `transforms_t` is a symmetric matrix and optimizes the computation of the backward pass.

    Args:
        cov3d (torch.Tensor): Input 3D covariance matrix of shape [num_views, num_points, 3, 3]. Assumed to be symmetric.
        transforms_t (torch.Tensor): Translated transformation matrices of shape [num_views, num_points, 2, 3]. Assumed to be symmetric.

    Returns:
        torch.Tensor: Projected 2D covariance matrix of shape [num_views, num_points, 2, 2].
    """


    @staticmethod
    def forward(ctx,cov3d:torch.Tensor,transforms_t:torch.Tensor):
        ctx.save_for_backward(transforms_t)
        cov2d=transforms_t@cov3d@(transforms_t.transpose(-1,-2).contiguous())
        return cov2d
    
    @staticmethod
    def backward(ctx,cov2d_gradient:torch.Tensor):
        (transforms_t,)=ctx.saved_tensors
        N,P=transforms_t.shape[0:2]
        # cov3d_gradient=torch.zeros((N,P,3,3),device=transforms_t.device)
        # for i in range(0,3):
        #     for j in range(0,3):
        #         cov3d_gradient[:,:,i,j]=\
        #             (transforms_t[:,:,0,i]*transforms_t[:,:,0,j])*cov2d_gradient[:,:,0,0]\
        #             + (transforms_t[:,:,0,i]*transforms_t[:,:,1,j])*cov2d_gradient[:,:,0,1]\
        #             + (transforms_t[:,:,1,i]*transforms_t[:,:,0,j])*cov2d_gradient[:,:,1,0]\
        #             + (transforms_t[:,:,1,i]*transforms_t[:,:,1,j])*cov2d_gradient[:,:,1,1]
        temp_matrix_A=transforms_t[:,:,(0,0,1,1),:].transpose(-1,-2).contiguous()
        temp_matrix_B=(transforms_t[:,:,(0,1,0,1),:]*cov2d_gradient.reshape(N,P,-1,1)).contiguous()
        cov3d_gradient=temp_matrix_A@temp_matrix_B

        return cov3d_gradient,None

class CreateCov2dDirectly(BaseWrapper):
    """
    A wrapped class for creating 2D covariance matrices.

    This class provides implementations for efficiently computing 2D covariance matrices by minimizing intermediate matrix operations.

    Users can invoke the computations through `call_fused`, `call_script`, or `call` methods.
    """
    def create_2dcov_fused(J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor)->torch.Tensor:
        '''
        An optimized function to calculate cov2d

        The usual method contains several matrix multiplications with a large batch number and a small K. Loading and writing these intermediate variables takes a lot of time.

        Args:
            J (torch.Tensor): Input tensor representing transformations with shape [num_views, 3, 3, num_points].
            view_matrix (torch.Tensor): View matrix with shape [num_views, 4, 4].
            transform_matrix (torch.Tensor): Transformation matrix with shape [num_views, 3, 3, num_points].

        Returns:
            torch.Tensor: Computed 2D covariance matrix with shape [num_views, 2, 2, num_points].

        '''
        class Cov2dCreateV2Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx,J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor)->torch.Tensor:
                ctx.save_for_backward(J,view_matrix,transform_matrix)
                cov2d=litegs_fused.createCov2dDirectly_forward(J,view_matrix,transform_matrix)
                return cov2d
            
            @staticmethod
            def backward(ctx,grad_cov2d:torch.Tensor):
                (J,view_matrix,transform_matrix)=ctx.saved_tensors
                transform_matrix_grad=litegs_fused.createCov2dDirectly_backward(grad_cov2d,J,view_matrix,transform_matrix)
                return (None,None,transform_matrix_grad)

        cov2d=Cov2dCreateV2Func.apply(J,view_matrix,transform_matrix)
        return cov2d
    
    _fused=create_2dcov_fused
    _script=None
    test_inputs=[([1,3,3,1024*512],torch.float32,False),
                 ([1,4,4],torch.float32,False),
                 ([3,3,1024*512],torch.float32,True)]
    _relative_error_threshold=5e-2#ProjCov3dTo2dFunc 引入浮点误差，适度放大relative error
    
    @classmethod
    def call_script(cls, J:torch.Tensor,view_matrix:torch.Tensor,transform_matrix:torch.Tensor):
        """
        Script-based implementation for creating 2D covariance matrices.

        This method uses a step-by-step approach involving intermediate computations such as 3D covariance matrix 
        generation and matrix transformations to compute the final 2D covariance matrix.

        Args:
            J (torch.Tensor): Input tensor representing transformations with shape [num_views, 3, 3, num_points].
            view_matrix (torch.Tensor): View matrix with shape [num_views, 4, 4].
            transform_matrix (torch.Tensor): Transformation matrix with shape [num_views, 3, 3, num_points].

        Returns:
            torch.Tensor: Computed 2D covariance matrix with shape [num_views, 2, 2, num_points].
        """
        cov3d=CreateCovarianceMatrixFunc.apply(transform_matrix.permute((2,0,1)))
        trans_J=J[:,:,:2].permute(0,3,2,1)
        trans_M=view_matrix[:,0:3,0:3].unsqueeze(0).transpose(-1,-2)
        trans_T=(trans_J@trans_M).contiguous()
        cov2d=ProjCov3dTo2dFunc.apply(cov3d,trans_T)
        cov2d[:,:,0,0]+=0.3
        cov2d[:,:,1,1]+=0.3
        return cov2d.permute((0,2,3,1))

class GaussiansRasterFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sorted_pointId:torch.Tensor,
        tile_start_index:torch.Tensor,
        ndc:torch.Tensor,
        cov2d_inv:torch.Tensor,
        color:torch.Tensor,
        opacities:torch.Tensor,
        tiles:torch.Tensor,
        tile_size:int,
        img_h:int,
        img_w:int,
        enable_transmitance:bool=False,
        enable_depth:bool=False,
        enable_entropy:bool=False
    ):
        transmitance=None
        depth=None
        entropy=None
        normal=None

        # The backward kernel reads final_transmitance unconditionally, so it must
        # be stored whenever backward will run (not just when explicitly requested).
        need_backward_aux = any(ctx.needs_input_grad[i] for i in (2, 3, 4, 5))
        create_transmitance = enable_transmitance or need_backward_aux
        img,transmitance,depth,entropy,lst_contributor=litegs_fused.rasterize_forward(
            sorted_pointId,tile_start_index,ndc,cov2d_inv,color,opacities,tiles,
            tile_size,img_h,img_w,create_transmitance,enable_depth,enable_entropy
        )

        ctx.save_for_backward(sorted_pointId,tile_start_index,transmitance,lst_contributor,ndc,cov2d_inv,color,opacities,tiles)
        ctx.arg_tile_size=tile_size
        ctx.img_hw=(img_h,img_w)
        if enable_depth==False:
            depth=None
        if enable_transmitance==False:
            transmitance=None
        if enable_entropy==False:
            entropy=None
        return img,transmitance,depth,entropy,normal

    @staticmethod
    def backward(ctx, grad_rgb_image:torch.Tensor, grad_transmitance_image:torch.Tensor,grad_depth_image:torch.Tensor,grad_entropy_image:torch.Tensor,grad_normal_image:torch.Tensor):
        sorted_pointId,tile_start_index,transmitance,lst_contributor,ndc,cov2d_inv,color,opacities,tiles=ctx.saved_tensors
        (img_h,img_w)=ctx.img_hw
        tile_size=ctx.arg_tile_size

        grad_ndc,grad_cov2d_inv,grad_color,grad_opacities=litegs_fused.rasterize_backward(sorted_pointId,tile_start_index,ndc,cov2d_inv,color,opacities,tiles,
                                                                                                        transmitance,lst_contributor,
                                                                                                        grad_rgb_image,grad_transmitance_image,grad_depth_image,grad_entropy_image,
                                                                                                        tile_size,img_h,img_w)

        grads = (
            None,
            None,
            grad_ndc,
            grad_cov2d_inv,
            grad_color,
            grad_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

        return grads

class SphericalHarmonicToRGB(BaseWrapper):
    """
    A derived class for converting spherical harmonics to RGB color values.

    This class provides both a fused implementation and a script-based fallback for evaluating spherical harmonics (SH) and converting them to RGB values for a given set of directions.

    Args:
            deg (int): Degree of the spherical harmonics.
            sh_base (torch.Tensor): Base spherical harmonic coefficients with shape [1, num_channels, num_points].
            sh_rest (torch.Tensor): Remaining SH coefficients with shape [(deg+1)**2-1, num_channels, num_points].
            dirs (torch.Tensor): Directions tensor with shape [num_views,3, num_points].

    Returns:
        torch.Tensor: RGB values computed from the spherical harmonics with shape [num_views, num_channels, num_points].
    """
    def __sh2rgb_fused(deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
        class SphericalHarmonicFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx,deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
                ctx.save_for_backward(dirs,sh_base,sh_rest)
                ctx.degree=deg
                ctx.sh_rest_dim=sh_rest.shape[0]
                rgb=litegs_fused.sh2rgb_forward(deg,sh_base,sh_rest,dirs)
                return rgb
            
            @staticmethod
            def backward(ctx, grad_rgb):
                (dirs,sh_base,sh_rest)=ctx.saved_tensors
                degree=ctx.degree
                sh_rest_dim=ctx.sh_rest_dim
                sh_base_grad,sh_reset_grad,dir_grad=litegs_fused.sh2rgb_backward(degree,grad_rgb,sh_rest_dim,dirs,sh_base,sh_rest)
                return None,sh_base_grad,sh_reset_grad,dir_grad
        return SphericalHarmonicFunc.apply(deg,sh_base,sh_rest,dirs).clamp_min(0)
    def __sh2rgb_script(deg:int, sh_base:torch.Tensor,sh_rest:torch.Tensor, dirs:torch.Tensor):
        return spherical_harmonics.sh_to_rgb(deg,torch.cat((sh_base,sh_rest),dim=0),dirs).clamp_min(0)
    _fused=__sh2rgb_fused
    _script=__sh2rgb_script
    test_inputs=[(3,None,None),
        ([1,3,1024*512],torch.float32,True),
        ([(3+1)**2-1,3,1024*512],torch.float32,True),
        ([1,3,1024*512],torch.float32,True)]

class EighAndInverse2x2Matrix(BaseWrapper):
    def __eight_inverse_2x2matrix_script(cov2d:torch.Tensor):
        with torch.no_grad():
            eigen_val,eigen_vec=torch.linalg.eigh(cov2d.permute(0,3,1,2).reshape(-1,2,2))
            eigen_val=eigen_val.reshape(cov2d.shape[0],cov2d.shape[3],2).permute(0,2,1)
            eigen_vec=eigen_vec.reshape(cov2d.shape[0],cov2d.shape[3],2,2).permute(0,2,3,1)

        cov2d_inv=torch.linalg.inv(cov2d.permute(0,3,1,2).reshape(-1,2,2))
        cov2d_inv=cov2d_inv.reshape(cov2d.shape[0],cov2d.shape[3],2,2).permute(0,2,3,1)
        return eigen_val,eigen_vec,cov2d_inv

    def __eight_inverse_2x2matrix_fused(cov2d:torch.Tensor):
        class EighAndInverse2x2Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx,input_matrix:torch.Tensor):
                val,vec,inverse_matrix=litegs_fused.eigh_and_inv_2x2matrix_forward(input_matrix)
                ctx.save_for_backward(inverse_matrix)
                return val,vec,inverse_matrix
            
            @staticmethod
            def backward(ctx,val_grad,vec_grad,inverse_matrix_grad):
                (inverse_matrix,)=ctx.saved_tensors
                matrix_grad:torch.Tensor=litegs_fused.inv_2x2matrix_backward(inverse_matrix,inverse_matrix_grad)
                matrix_grad.nan_to_num_(0)
                return matrix_grad
        return EighAndInverse2x2Func.apply(cov2d)

    @classmethod
    def gen_inputs(cls):
        cov2d=torch.randn([1,2,2,512*1024], dtype=torch.float32, device='cuda', requires_grad=False)
        cov2d[:,0,1,:]=cov2d[:,1,0,:]
        cov2d[:,0,0,:]*=10
        cov2d[:,1,1,:]*=10
        cov2d.requires_grad_(True)
        return [cov2d,]
    
    _fused=__eight_inverse_2x2matrix_fused
    _script=__eight_inverse_2x2matrix_script
    test_inputs=None
    _relative_error_threshold=1e-2


class Binning(BaseWrapper):
    @torch.no_grad()
    def __binning_script(ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,
            img_pixel_shape:tuple[int,int],tile_size:int):
        def craete_2d_AABB(ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,tile_size:int,img_pixel_shape:tuple[int,int],img_tile_shape:tuple[int,int]):
            # Major and minor axes -> AABB extensions
            opacity_clamped=opacity.unsqueeze(0).clamp_min(1/255)
            coefficient=2*((255*opacity_clamped).log())
            axis_length=(coefficient*eigen_val.abs()).sqrt()
            extension=(axis_length.unsqueeze(-2)*eigen_vec).abs().sum(dim=-3)

            screen_uv=(ndc[:,:2]+1.0)*0.5
            screen_uv[:,0]*=img_pixel_shape[1]#x
            screen_uv[:,1]*=img_pixel_shape[0]#y
            screen_coord=screen_uv-0.5
            b_visible=~((ndc[:,0]<-1.3)|(ndc[:,0]>1.3)|(ndc[:,1]<-1.3)|(ndc[:,1]>1.3)|(ndc[:,2]>1)|(ndc[:,2]<0))
            left_up=((screen_coord-extension)/tile_size).int()*b_visible
            right_down=((screen_coord+extension)/tile_size).ceil().int()*b_visible
            left_up[:,0].clamp_(0,img_tile_shape[1])#x
            left_up[:,1].clamp_(0,img_tile_shape[0])#y
            right_down[:,0].clamp_(0,img_tile_shape[1])
            right_down[:,1].clamp_(0,img_tile_shape[0])

            return left_up,right_down
        
        nvtx.range_push("binning_allocate")
        img_tile_shape=(int(math.ceil(img_pixel_shape[0]/float(tile_size))),int(math.ceil(img_pixel_shape[1]/float(tile_size))))
        tiles_num=img_tile_shape[0]*img_tile_shape[1]

        left_up,right_down=craete_2d_AABB(ndc,eigen_val,eigen_vec,opacity,tile_size,img_pixel_shape,img_tile_shape)

        #splatting area of each points
        rect_length=right_down-left_up
        tiles_touched=rect_length[:,0]*rect_length[:,1]
        b_visible=(tiles_touched!=0)

        #sort by depth
        values,point_ids=ndc[:,2].sort(dim=-1,descending=True)
        for i in range(ndc.shape[0]):
            tiles_touched[i]=tiles_touched[i,point_ids[i]]

        #calc the item num of table and the start index in table of each point
        prefix_sum=tiles_touched.cumsum(1,dtype=torch.int32)#start index of points
        total_tiles_num_batch=prefix_sum[:,-1]
        allocate_size=total_tiles_num_batch.max().cpu()
        nvtx.range_pop()
        
        # allocate table and fill it (Table: tile_id-uint16,point_id-uint16)
        large_points_index=(tiles_touched>=32).nonzero()
        my_table=litegs_fused.duplicateWithKeys(left_up,right_down,prefix_sum,point_ids,large_points_index,int(allocate_size),img_tile_shape[1])
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]

        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)

        # range
        tile_start_index=litegs_fused.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)
            
        return tile_start_index,sorted_pointId,b_visible
    
    @torch.no_grad()
    def __binning_fused(ndc:torch.Tensor,eigen_val:torch.Tensor,eigen_vec:torch.Tensor,opacity:torch.Tensor,
            img_pixel_shape:tuple[int,int],tile_size:int):

        img_tile_shape=(int(math.ceil(img_pixel_shape[0]/float(tile_size))),int(math.ceil(img_pixel_shape[1]/float(tile_size))))
        tiles_num=img_tile_shape[0]*img_tile_shape[1]

        left_up,right_down=litegs_fused.create_ROI_AABB(ndc,eigen_val,eigen_vec,opacity,img_pixel_shape[0],img_pixel_shape[1],tile_size)
        
        rect_length=right_down-left_up
        tiles_touched=rect_length[:,0]*rect_length[:,1]
        b_visible=(tiles_touched!=0)
        if StatisticsHelperInst.bStart:
            StatisticsHelperInst.update_max_min_compact('radii',rect_length.max(dim=1).values.float())

        # Inverse-z convention: ndc.z is near->1, far->0.
        # Use descending ndc.z so rasterization iterates near-to-far (front-to-back alpha compositing).
        values,point_ids=ndc[:,2].sort(dim=-1,descending=True)
        for i in range(ndc.shape[0]):
            tiles_touched[i]=tiles_touched[i,point_ids[i]]

        #calc the item num of table and the start index in table of each point
        prefix_sum=tiles_touched.cumsum(1,dtype=torch.int32)#start index of points
        total_tiles_num_batch=prefix_sum[:,-1]
        allocate_size=total_tiles_num_batch.max().cpu()
        
        # allocate table and fill it (Table: tile_id-uint16,point_id-uint16)
        large_points_index=(tiles_touched>=32).nonzero()
        my_table=litegs_fused.duplicateWithKeys(left_up,right_down,prefix_sum,point_ids,large_points_index,int(allocate_size),img_tile_shape[1])

        '''
        Example - Before sorting by tile ID:
        tileId_table: tensor([1014, 1013, 1012, 1011, 1010, 1009, 1008, 1007...], device='cuda:0', dtype=torch.int32)
        pointId_table: tensor([1489, 1489, 1489, 1489, 1489, 1489, 1489, 1489...], device='cuda:0', dtype=torch.int32)
        '''
        tileId_table:torch.Tensor=my_table[0]
        pointId_table:torch.Tensor=my_table[1]

        # sort tile_id with torch.sort
        sorted_tileId,indices=torch.sort(tileId_table,dim=1,stable=True)
        sorted_pointId=pointId_table.gather(dim=1,index=indices)
        '''
        After sorting by tile ID:
        sorted_tileId: tensor([1, 1, 1, 1, 1, 1, 1, 1...], device='cuda:0', dtype=torch.int32)
        sorted_pointId: tensor([12529, 12530,  9960, 11940, 11975, 11636, 30885, 30886...], device='cuda:0', dtype=torch.int32)
        '''

        # range
        tile_start_index=litegs_fused.tileRange(sorted_tileId,int(allocate_size),int(tiles_num-1+1))#max_tile_id:tilesnum-1, +1 for offset(tileId 0 is invalid)

        return tile_start_index,sorted_pointId,b_visible
    
    
    _fused=__binning_fused
    _script=__binning_script
###
### compact params
###

class CompactVisibleWithSparseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,visible_id:torch.Tensor,*args:list[torch.Tensor])->list[torch.Tensor]:
        compacted_tensors=[]
        for tensor in args:
            compacted_tensors.append(tensor[...,visible_id,:].contiguous())
        ctx.chunk_num=args[0].shape[-2]
        ctx.chunk_size=args[0].shape[-1]
        return *compacted_tensors,
    
    @staticmethod
    def backward(ctx,*args):
        chunk_num=ctx.chunk_num
        chunk_size=ctx.chunk_size
        grads=[]#the index of sprase tensor is invalid!! backward compact with Our Optimizer
        for grad in args:
            sparse_value=grad.reshape(-1,chunk_size)
            placeholder_grad=torch.sparse_coo_tensor(torch.empty(grad.dim()-1,sparse_value.shape[0],device='cuda'),sparse_value,(*grad.shape[:-2],chunk_num,chunk_size))
            grads.append(placeholder_grad)
        return None,*grads

def sparse_adam_update(param:torch.Tensor, grad:torch.Tensor, exp_avg:torch.Tensor, exp_avg_sq:torch.Tensor, visible_chunk:torch.Tensor, 
                       lr:float, b1:float, b2:float, eps:float):
    litegs_fused.adamUpdate(param,grad,exp_avg,exp_avg_sq,visible_chunk,lr,b1,b2,eps)
    return
