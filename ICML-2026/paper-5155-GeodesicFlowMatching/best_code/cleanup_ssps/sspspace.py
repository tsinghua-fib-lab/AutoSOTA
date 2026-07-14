import numpy as np
from scipy.stats import qmc
from scipy.stats import special_ortho_group
from scipy.optimize import minimize
from nengo.dists import UniformHypersphere

import warnings

class SPSpace:
    r"""  Class for Semantic Pointer (SP) representation mapping
    
    This is similar to nengo_spa vocabs but structured differently.

    Parameters
    ----------
        domain_size : int
            The number of discrete symbols that will be encoded in this space.
            
        dim : int
            The dimensionality of the SPs, should be >= domain_size.
            
        seed : int
            The seed for generating the SPs. Default is None.
            
    Attributes
    ----------
        domain_size, dim : int
        
        vectors : np.ndarray
           A (domain_size x dim) array of all SPs
           
        inverse_vectors : Node
            Inverse (under binding) SPs
            
    Examples
    --------
       from sspslam import SPSpace
       sp_space = SPSpace(5, 100)

    """
    def __init__(self, domain_size: int, dim: int, seed=None,**kwargs):
        self.domain_size = int(domain_size)
        self.dim = int(dim)
        if np.issubdtype(type(seed), np.integer):
            rng = np.random.RandomState(seed)
        elif seed is None:
            rng = np.random.RandomState()
        self.rng = rng
        if self.domain_size==1: # only one is special case, vectors only contains identity
            self.vectors = np.zeros((self.domain_size,self.dim))
            self.vectors[:,0] = 1
        else:
            self.vectors = self.make_unitary(UniformHypersphere(surface=True).sample(self.domain_size,self.dim, rng=rng) )

            for j in range(self.domain_size):
                q = self.vectors[j,:]/np.linalg.norm(self.vectors[j,:])
                for k in range(j+1,self.domain_size):
                    self.vectors[k,:] = self.vectors[k,:] - (q.T @ self.vectors[k,:])*q
        self.inverse_vectors = self.invert(self.vectors)
        #self.make_unitary(self.rng.randn(self.domain_size,self.dim))
            
    def encode(self, i):
        """
        Maps index to SP
        
        Parameters
        ----------
        i : np.array
            An array of ints, each in [0, domain_size)

        Returns
        -------
        np.array
            Semantic Pointers.

        """
        return self.vectors[i.reshape(-1).astype(int)]
    
    def decode(self, v, **kwargs):
        """
        Maps dim-D vector to index 
        
        Parameters
        ----------
        v : np.array
            A (n_samples x ssp_dim) vector

        Returns
        -------
        np.array
            A n_samples length vector of indexes 

        """
        sims = self.vectors @ v.T
        return np.argmax(sims, axis=0)
    
    def clean_up(self, v, **kwargs):
        """
        Maps dim-D vector to SP
        
        Parameters
        ----------
        v : np.array
            A (n_samples x ssp_dim) vector

        Returns
        -------
        np.array
            A (n_samples x ssp_dim) vector, each row a Semantic Pointer.

        """
        sims = self.vectors @ v.T
        return self.vectors[np.argmax(sims, axis=0)]
    
    def normalize(self,v):
        """
        Normalizes input
        """
        return v/np.sqrt(np.sum(v**2))
    
    def make_unitary(self,v):
        """
        Makes input unitary (Fourier components have magnitude of 1)
        """
        fv = np.fft.fft(v, axis=1)
        fv = fv/np.sqrt(fv.real**2 + fv.imag**2)
        return np.fft.ifft(fv, axis=1).real  
    
    def identity(self):
        """
        Returns
        -------
        np.array
            dim-D identity vector under binding

        """
        s = np.zeros(self.dim)
        s[0] = 1
        return s
    
    def bind(self,a,b):
        """
        Binds togther input
        
        Parameters
        ----------
        a : np.array
            A vector with shape (n_samples x ssp_dim) 
            
        b : np.array
            A vector with shape (n_samples x ssp_dim) 

        Returns
        -------
        np.array
            A vector with shape (n_samples x ssp_dim). Row i is a[i,:] binded with b[i,:]

        """
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real
    
    def invert(self,a):
        """
        Inverts input under binding
        """
        a = np.atleast_2d(a)
        return a[:,-np.arange(self.dim)]
    
    def get_binding_matrix(self,v):
        """
        Maps input vector to a matrix that, when multiplied with another vecotr, will bind vectors
        """
        C = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                C[i,j] = v[:,(i - j) % self.dim] 
        return C


class SSPSpace:
    def __init__(self, domain_dim: int, ssp_dim: int, axis_matrix=None, phase_matrix=None,
                 domain_bounds=None, length_scale=1):
        '''
        Represents a domain using spatial semantic pointers.

        Parameters:
        -----------

        domain_dim : int 
            The dimensionality of the domain being represented.

        ssp_dim : int
            The dimensionality of the spatial semantic pointer vector.

        axis_matrix : np.ndarray
            A ssp_dim X domain_dim ndarray representing the axis vectors for
            the domain.

        phase_matrix : np.ndarray
            A ssp_dim x domain_dim ndarray representing the frequency 
            components of the SSP representation.

        domain_bounds : np.ndarray
            A domain_dim X 2 ndarray giving the lower and upper bounds of the 
            domain, used in decoding from an ssp to the point it represents.

        length_scale : float or np.ndarray
            Scales values before encoding.
            
        '''
        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim
        self.length_scale = length_scale * np.ones((self.domain_dim,1))
        
        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim
        
        self.domain_bounds = domain_bounds
        self.decoder_model = None
        
        if (axis_matrix is None) & (phase_matrix is None):
            raise RuntimeError("SSP spaces must be defined by either a axis matrix or phase matrix. Use subclasses to construct spaces with predefined axes.")
        elif (phase_matrix is None):
            assert axis_matrix.shape[0] == ssp_dim, f'Expected ssp_dim {axis_matrix.shape[0]}, got {ssp_dim}.'
            assert axis_matrix.shape[1] == domain_dim
            self.axis_matrix = axis_matrix
            self.phase_matrix = (-1.j*np.log(np.fft.fft(axis_matrix,axis=0))).real
            
        elif (axis_matrix is None):
            assert phase_matrix.shape[0] == ssp_dim
            assert phase_matrix.shape[1] == domain_dim
            self.phase_matrix = phase_matrix
            self.axis_matrix = np.fft.ifft(np.exp(1.j*phase_matrix), axis=0).real
            
    def update_lengthscale(self, scale):
        '''
        Changes the lengthscale being used in the encoding.
        '''
        if not isinstance(scale, np.ndarray) or scale.size == 1:
            self.length_scale = scale * np.ones((self.domain_dim,))
        else:
            assert scale.size == self.domain_dim
            self.length_scale = scale
        assert self.length_scale.size == self.domain_dim
        ### end if
        
    def optimize_lengthscale(self, init_xs, init_ys):
        ls_0 = self.length_scale
        self.length_scale = np.ones((self.domain_dim,1))
        
        def min_func(length_scale):
            init_phis = self.encode(init_xs/ length_scale)
            W = np.linalg.pinv(init_phis.T) @ init_ys
            mu = np.dot(init_phis.T,W)
            diff = init_ys - mu.T
            err = np.sum(np.power(diff, 2))
            return err

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B', bounds = self.domain_dim*[(1e-8,1e5)])
        self.length_scale = retval.x.reshape(-1,1)
    
    def encode(self,x):
        '''
        Transforms input data into an SSP representation.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the data
            
        '''

        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T), axis=0 ).real
        return data.T
    
    def encode_and_deriv(self,x):
        '''
        Returns the ssp representation of the data and the derivative of
        the encoding.

        Parameters:
        -----------
        x : np.ndarray
            A (num_samples, domain_dim) array representing data to be encoded.

        Returns:
        --------
        data : np.ndarray
            A (num_samples, ssp_dim) array of the ssp representation of the 
            data

        grad : np.ndarray
            A (num_samples, ssp_dim, domain_dim) array of the ssp representation of the data

        '''
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1 / self.length_scale))
        scaled_x = x @ ls_mat
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        ddata = np.fft.ifft( 1.j * (self.phase_matrix @ ls_mat) @ np.exp( 1.j * self.phase_matrix @ scaled_x.T ), axis=0 ).real
        return data.T, ddata.T
    
    def encode_fourier(self,x):
        x = np.atleast_2d(x)
        ls_mat = np.atleast_2d(np.diag(1/self.length_scale.flatten()))
        assert ls_mat.shape == (self.domain_dim, self.domain_dim), f'Expected Len Scale mat with dimensions {(self.domain_dim, self.domain_dim)}, got {ls_mat.shape}'
        scaled_x = x @ ls_mat
        data = np.exp( 1.j * self.phase_matrix @ scaled_x.T)

        return data.T
    
    # def encode_as_SSP(self,x):
    #     assert x.shape[0] == self.domain_dim
    #     data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ x / self.length_scale ), axis=0 ).real
    #     return SSP(data,self)
    
 
    def decode(self,ssp,method='from-set',sampling_method='grid',
               num_samples =128, samples=None): # other args for specfic methods
        '''
        Transforms ssp representation back into domain representation.

        Parameters:
        -----------
        ssp : np.ndarray
            SSP representation of a data point.

        method : {'from-set', 'direct-optim'}
            The technique for decoding the ssp.  from-set samples the domain
            and finds the closest match under the dot product. direct-optim
            does an initial coarse sampling and then optimizes the decoded
            value starting from the initial best match in the coarse sampling.

        sampling_method : {'grid'|'length-scale'|'sobol'}
            Evenly distributes samples along the domain axes

        num_samples : int
            The number of samples along each axis.

        Returns:
        --------
        x : np.ndarray
            The decoded point
        '''
        if (method=='direct-optim') | (method=='from-set'):
            if samples is None:
                sample_ssps, sample_points = self.get_sample_pts_and_ssps(method=sampling_method, 
                        num_points_per_dim=num_samples)
            else:
                sample_ssps, sample_points = samples
                assert sample_ssps.shape[1] == ssp.shape[1], f'Expected {sample_ssps.shape} dim, got {ssp.shape}'
            

#         unit_ssp = ssp / np.linalg.norm(ssp, axis=1)
        unit_ssp = np.zeros(ssp.shape)
        for s_idx, s in enumerate(ssp):
            if np.linalg.norm(s) < 1e-6:
                unit_ssp[s_idx,:] = s
            else:
                unit_ssp[s_idx,:] = s / np.linalg.norm(s)
        
        if method=='from-set': 
            sims = sample_ssps @ unit_ssp.T
            return sample_points[np.argmax(sims),:]

        elif method == 'direct-optim':
            retvals = np.zeros((ssp.shape[0], self.domain_dim))
            for i, u_ssp in enumerate(unit_ssp):
                x0 = self.decode(np.atleast_2d(u_ssp), method='from-set',
                                sampling_method='length-scale',
                                num_samples=num_samples, samples=samples)
                soln = minimize(
                    lambda x, target=u_ssp: -np.inner(self.encode(np.atleast_2d(x)), target).flatten(),
                    x0,
                    method='L-BFGS-B',
                    bounds=self.domain_bounds
                )
                retvals[i] = soln.x
            return retvals
        elif method=='network':
            if self.decoder_model is None:
                raise Exception('Network not trained for decoding. You must first call train_decoder_net')
            return self.decoder_model.predict(ssp)
        elif method=='network-optim':
            if self.decoder_model is None:
                raise Exception('Network not trained for decoding. You must first call train_decoder_net')
            x0 = self.decoder_model.predict(ssp)
            

            solns = np.zeros(x0.shape)
            for i in range(x0.shape[0]):
                def min_func(x,target=ssp[i,:]):
                    x_ssp = self.encode(np.atleast_2d(x))
                    return -np.inner(x_ssp, target).flatten()
                soln = minimize(min_func, x0[i,:], 
                            method='L-BFGS-B',
                            bounds=self.domain_bounds)
                solns[i,:] = soln.x
            return solns
        else:
            raise NotImplementedError(f'Unrecognized decoding method: {method}')
        

    def clean_up(self, ssp, method='from-set', grid_resolution=128):
        ssp = np.atleast_2d(ssp)

        if method == 'from-set':
            sample_ssps, _ = self.get_sample_pts_and_ssps(
                num_points_per_dim=grid_resolution,
                method='grid'
            )
            sims = sample_ssps @ ssp.T   # (N, D) @ (D, 1)
            return sample_ssps[np.argmax(sims)]
        else:  # direct-optim
            x = self.decode(ssp, method, num_samples=grid_resolution)
            return self.encode(x)



    def get_sample_points(self, samples_per_dim=100, method='length-scale'):
        '''
        Identifies points in the domain of the SSP encoding that 
        will be used to determine optimal decoding.

        Parameters
        ----------

        method: {'grid'|'length-scale'|'sobol'}
            The way to select samples from the domain. 
            'grid' uniformly spaces samples_per_dim points on the domain
            'sobol' decodes using samples_per_dim**data_dim points generated 
                using a sobol sampling
            'length-scale' uses the selected lengthscale to determine the number
                of sample points generated per dimension.

        Returns
        -------

        sample_pts : np.ndarray 
            A (num_samples, domain_dim) array of candiate decoding points.
        '''

        if self.domain_bounds is None:
            bounds = np.vstack([-10*np.ones(self.domain_dim), 10*np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds

        if method == 'grid':
            num_pts_per_dim = [samples_per_dim for _ in range(bounds.shape[0])]
        elif method == 'length-scale':
            num_pts_per_dim = [2*int(np.ceil((b[1]-b[0])/self.length_scale[b_idx])) for b_idx, b in enumerate(bounds)]
        else:
            num_pts_per_dim = samples_per_dim 

        if method=='grid' or method=='length-scale':
            xxs = np.meshgrid(*[np.linspace(bounds[i,0], 
                                            bounds[i,1],
                                            num_pts_per_dim[i]
                                        ) for i in range(self.domain_dim)])
            retval = np.array([x.reshape(-1) for x in xxs]).T
            assert retval.shape[1] == self.domain_dim, f'Expected {self.domain_dim}d data, got {retval.shape[1]}d data'
            return retval

        elif method=='sobol':
            num_points = np.prod(num_pts_per_dim)

            sampler = qmc.Sobol(d=self.domain_dim) 
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T
        elif method=='Rd':
            num_points = np.prod(samples_per_dim)
            u_sample_points = _Rd_sampling(num_points, self.domain_dim)
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds).T
        else:
            raise NotImplementedError(f'Sampling method {method} is not implemented')
        return sample_points.T 
        
    
    def get_sample_ssps(self,num_points): 
        sample_points = self.get_sample_points(num_points)
        sample_ssps = self.encode(sample_points)
        return sample_ssps
    
    def get_sample_pts_and_ssps(self,num_points_per_dim=100, method='grid'): 
        sample_points = self.get_sample_points(
                                        method=method, 
                                        samples_per_dim=num_points_per_dim
                                    )
        if method == 'grid':
            expected_points = int(num_points_per_dim**(self.domain_dim))
            assert sample_points.shape[0] == expected_points, f'Expected {expected_points} samples, got {sample_points.shape[0]}.'

        sample_ssps = self.encode(sample_points)

        if method == 'grid':
            assert sample_ssps.shape[0] == expected_points

        return sample_ssps, sample_points
    
    def normalize(self,ssp):
        return ssp/np.maximum(np.sqrt(np.sum(ssp**2)), 1e-8)
    
    def make_unitary(self,ssp):
        fssp = np.fft.fft(ssp)
        fssp = fssp/np.maximum(np.sqrt(fssp.real**2 + fssp.imag**2), 1e-8)
        return np.fft.ifft(fssp).real  
    
    def make_unitary_fourier(self,fssp):
        fssp = fssp/np.maximum(np.sqrt(fssp.real**2 + fssp.imag**2), 1e-8)
        return fssp
    
    def identity(self):
        s = np.zeros(self.ssp_dim)
        s[0] = 1
        return s
    
    def bind(self,a,b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        return np.fft.ifft(np.fft.fft(a, axis=1) * np.fft.fft(b,axis=1), axis=1).real
    
    def invert(self,a):
        a = np.atleast_2d(a)
        return a[:,-np.arange(self.ssp_dim)]
    
    def similarity_plot(self,ssp,n_grid=100,plot_type='heatmap',ax=None,**kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        if self.domain_dim == 1:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            sims = ssp @ self.encode(np.atleast_2d(xs).T).T
            im = ax.plot(xs, sims.reshape(-1) )
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
        elif self.domain_dim == 2:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            ys = np.linspace(self.domain_bounds[1,0],self.domain_bounds[1,1], n_grid)
            X,Y = np.meshgrid(xs,ys)
            sims = ssp @ self.encode(np.vstack([X.reshape(-1),Y.reshape(-1)]).T).T 
            if plot_type=='heatmap':
                im = ax.pcolormesh(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contour':
                im = ax.contour(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contourf':
                im = ax.contourf(X,Y,sims.reshape(X.shape),**kwargs)
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
            ax.set_ylim(self.domain_bounds[1,0],self.domain_bounds[1,1])
        else:
            raise NotImplementedError()
        return im
    
    def train_decoder_net(self,n_training_pts=200000,n_hidden_units = 8,
                          learning_rate=1e-3,n_epochs = 20, load_file=True, save_file=True):
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')

        import sklearn
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        
        if (type(self).__name__ == 'HexagonalSSPSpace'):
            path_name = './saved_decoder_nets/domaindim' + str(self.domain_dim) + '_lenscale' + str(self.length_scale[0]) + '_nscales' + str(self.n_scales) + '_nrotates' + str(self.n_rotates) + '_scale_min' + str(self.scale_min) + '_scalemax' + str(self.scale_max) +'.h5'
        else:
            #warnings.warn("Cannot load decoder net for non HexagonalSSPSpace class")
            load_file = False
            save_file=False
        
        
        if load_file:
            try:
                self.decoder_model = keras.models.load_model(path_name)
                return
            except BaseException as be:
                print('Error loading decoder:')
                print(be)
                pass

        model = keras.Sequential([
             layers.Dense(self.ssp_dim, activation="relu", name="layer1"),# layers.Dropout(.1),
             layers.Dense(n_hidden_units, activation="relu", name="layer2"), # kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
             layers.Dense(self.domain_dim, name="output"),
            ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error')
        
        sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_points_per_dim=n_training_pts,
                                                                           method='Rd')
        shuffled_ssps, shuffled_pts = sklearn.utils.shuffle(sample_ssps, sample_points)
        history = model.fit(shuffled_ssps, shuffled_pts,
            epochs=n_epochs,verbose=True, validation_split = 0.1)
        
        if save_file:
           model.save(path_name)

            
        self.decoder_model = model
        return history
            
class RandomSSPSpace(SSPSpace):
    '''
    Creates an SSP space using randomly generated frequency components.
    '''
    def __init__(self, domain_dim: int, ssp_dim: int,  domain_bounds=None, length_scale=1, rng=np.random.default_rng()):
#         partial_phases = rng.random.rand(ssp_dim//2,domain_dim)*2*np.pi - np.pi
        
        
        #partial_phases = rng.random((ssp_dim // 2, domain_dim)) * 2 * np.pi - np.pi
        #axis_matrix = _constructaxisfromphases(partial_phases)
        def make_good_unitary(dim, eps=1e-3, rng=np.random):
            a = rng.rand((dim - 1) // 2)
            sign = rng.choice((-1, +1), len(a))
            phi = sign * np.pi * (eps + a * (1 - 2 * eps))
            assert np.all(np.abs(phi) >= np.pi * eps)
            assert np.all(np.abs(phi) <= np.pi * (1 - eps))
        
            fv = np.zeros(dim, dtype='complex64')
            fv[0] = 1
            fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
            fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
            if dim % 2 == 0:
                fv[dim // 2] = 1
        
            assert np.allclose(np.abs(fv), 1)
            v = np.fft.ifft(fv)
            
            v = v.real
            assert np.allclose(np.fft.fft(v), fv)
            assert np.allclose(np.linalg.norm(v), 1)
            return v

        axis_matrix = np.zeros((ssp_dim,domain_dim))
        for i in range(domain_dim):
            axis_matrix[:,i] = make_good_unitary(ssp_dim)

        super().__init__(domain_dim, 
                         axis_matrix.shape[0],
                         axis_matrix=axis_matrix,
                         domain_bounds=domain_bounds,
                         length_scale=length_scale,
                         )


        
class HexagonalSSPSpace(SSPSpace):
    '''
    Creates an SSP space using the Hexagonal Tiling developed by NS Dumont 
    (2020)
    '''
    def __init__(self,  domain_dim:int,ssp_dim: int=151, n_rotates:int=5, n_scales:int=5, 
                 scale_min=0.1, scale_max=3,
                 domain_bounds=None, length_scale=1):
        if (n_rotates==5) & (n_scales==5) & (ssp_dim!=151): # user wants to define ssp with total dim, not number of simplex rotates and scales
            n_rotates = int(np.sqrt((ssp_dim-1)/(2*(domain_dim+1))))
            n_scales = n_rotates
            ssp_dim = n_rotates*n_scales*(domain_dim+1)*2 + 1
            
        phases_hex = np.hstack([np.sqrt(1+ 1/domain_dim)*np.identity(domain_dim) - (domain_dim**(-3/2))*(np.sqrt(domain_dim+1) + 1),
                         (domain_dim**(-1/2))*np.ones((domain_dim,1))]).T
        
        self.grid_basis_dim = domain_dim + 1
        self.num_grids = n_rotates*n_scales
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n_scales = n_scales
        self.n_rotates = n_rotates
        
        #scales = scale_max*(np.linspace((scale_min/scale_max)**2,1,n_scales))**(1/domain_dim)
        scales = np.linspace(scale_min,scale_max,n_scales)
        phases_scaled = np.vstack([phases_hex*i for i in scales])
        
        if (n_rotates==1):
            phases_scaled_rotated = phases_scaled
        elif (domain_dim==1):
            scales = np.linspace(scale_min,scale_max,n_scales+n_rotates)
            phases_scaled_rotated = np.vstack([phases_hex*i for i in scales])
        elif (domain_dim == 2):
            angles = np.linspace(0,2*np.pi/3,n_rotates,endpoint=False)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        else:
            R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates, random_state=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        
        axis_matrix = _constructaxisfromphases(phases_scaled_rotated)
        ssp_dim = axis_matrix.shape[0]
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
  
    def sample_grid_encoders(self, n_neurons):
        d = self.ssp_dim
        n = self.domain_dim
        A = self.phase_matrix
        sample_pts = self.get_sample_points(n, method='sobol')
        if d % 2 == 0:
            N = ((d-2)//2)//(n+1)
        else:
            N = ((d-1)//2)//(n+1)

        sample_pts = self.get_sample_points(n_neurons, method='sobol')   
        sorts =  np.random.randint(0, N, size = n_neurons)      
        
        encoders = np.zeros((n_neurons,d))
        for i in range(n_neurons):
            res = np.zeros(d, dtype=complex)
            res[(1 + sorts[i]*(n+1)):(n + 2 + sorts[i]*(n+1)) ] = np.exp( 1.j * A[(1 + sorts[i]*(n+1)):(n + 2 + sorts[i]*(n+1)) ] @ sample_pts[i,:])
            res[-(n + 1 + sorts[i]*(n+1)):-(sorts[i]*(n+1)+ (sorts[i]==0))] = np.exp( 1.j * A[-(n + 1 + sorts[i]*(n+1)):-(sorts[i]*(n+1) + (sorts[i]==0))] @ sample_pts[i,:])
            encoders[i,:] = np.fft.ifft(res).real
        res[0] = 1
        if d%2==0:
            res[d//2] = 1
            
        return encoders
        
    
def _constructaxisfromphases(K):
    d = K.shape[0]
    F = np.ones((d*2 + 1,K.shape[1]), dtype="complex")
    F[0:d,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[0:d,:]),axis=0)
    axes =  np.fft.ifft(np.fft.ifftshift(F,axes=0),axis=0).real
    return axes

def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real

def _Rd_sampling(n,d,seed=0.5):
    def phi(d): 
        x=2.0000 
        for i in range(10): 
          x = pow(1+x,1/(d+1)) 
        return x
    g = phi(d) 
    alpha = np.zeros(d) 
    for j in range(d): 
      alpha[j] = pow(1/g,j+1) %1 
    z = np.zeros((n, d)) 
    for i in range(n):
        z[i] = seed + alpha*(i+1)
    z = z %1
    return z
