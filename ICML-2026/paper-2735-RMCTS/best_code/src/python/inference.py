from asyncio.log import logger
import pathlib
from pathlib import Path
import numpy as np
import torch # helps onnxruntime find torch libraries

# next we try to load tensorrt if available
# allows onnxruntime to find TensorRT libraries
try:
	import tensorrt as trt
except ImportError:
	pass

import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

from . import game

class OrtModel:
	def __init__(self, onnx_path : pathlib.Path | str):
		'''
		onnx_path is a path to a model in .onnx format
		'''
		self.onnx_path = Path(onnx_path)
		assert self.onnx_path.exists()
		assert self.onnx_path.suffix == '.onnx'
		self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
		#self.providers = ort.get_available_providers()
		self.session = ort.InferenceSession(self.onnx_path, providers=self.providers)
		self.inputs = self.session.get_inputs()
		self.outputs = self.session.get_outputs()
		assert len(self.inputs) == 1
		assert len(self.outputs) == 2
		assert self.inputs[0].name == 'input'
		assert self.outputs[0].name == 'log_policy'
		assert self.outputs[1].name == 'value'
		self.input_shape = tuple(self.inputs[0].shape[1:]) # excludes input batchsize (dim 0)
		assert self.input_shape == game.metaparm.input_shape
		assert self.inputs[0].shape[0] is None or isinstance(self.inputs[0].shape[0], str)  # dynamic batch size
		#print(f"first dimension of input is {self.inputs[0].shape[0]}")
		self.num_actions = self.outputs[0].shape[1]
		assert self.num_actions == game.numActions()

	def pred(self, G):
		'''
		input G is a list (or array) of gamestates of dtype = np.float32
		returns policy = exp(log_policy) and value
		'''
		if type(G) != np.ndarray:
			G = np.array(G, dtype=np.float32)

		X = game.inputNetworkMany(G)
		assert X.shape[1:] == self.input_shape
		num_games = X.shape[0]

		if num_games == 0:
			# nothing to do
			return None, None
		
		log_policy, value = self.session.run(['log_policy', 'value'], {'input': X})
		policy = np.exp(log_policy)

		return policy, value

	def __call__(self, G):
		return self.pred(G)

class Engine:
	def __init__(self, path : pathlib.Path | str, max_batchsize = 1024, opt_batchsize = 256):
		'''
		path is a path to a model in .onnx or .engine format
		'''
		self.path = Path(path)
		assert self.path.exists()
		assert self.path.suffix in ['.onnx', '.engine']
		assert 1 <= opt_batchsize <= max_batchsize
		self.min_batchsize = 1
		self.opt_batchsize = opt_batchsize
		self.max_batchsize = max_batchsize
		logger = trt.Logger(trt.Logger.WARNING)
		if self.path.suffix == '.onnx':
			ort_model = OrtModel(self.path)
			input_shape = ort_model.input_shape
			builder = trt.Builder(logger)
			network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
			parser = trt.OnnxParser(network, logger)
			success = parser.parse_from_file(str(self.path.resolve()))
			if not success:
				for idx in range(parser.num_errors):
					print(parser.get_error(idx))
			assert success
			# set optimization profile
			profile = builder.create_optimization_profile()
			profile.set_shape(
				"input",           # input tensor name
				(1, *input_shape), # min shape
				(opt_batchsize, *input_shape),# opt shape (typical batch size)
				(max_batchsize, *input_shape) # max shape
			)
			config = builder.create_builder_config()
			config.add_optimization_profile(profile)
			config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 26) # TODO
			self.serialized_engine = builder.build_serialized_network(network, config)
		else:
			with open(self.path, "rb") as f:
				self.serialized_engine = f.read()
		runtime = trt.Runtime(logger)

		self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)
		self.context = self.engine.create_execution_context()
		assert self.engine.get_tensor_shape('input')[0] == -1  # dynamic batch size
		self.input_shape = self.engine.get_tensor_shape('input')[1:]
		assert self.input_shape == game.metaparm.input_shape
		self.num_actions = self.engine.get_tensor_shape('log_policy')[1]
		assert self.num_actions == game.numActions()

		self.d_X = cuda.mem_alloc(int(self.max_batchsize * np.prod(self.input_shape) * np.dtype(np.float32).itemsize))
		self.d_P = cuda.mem_alloc(int(self.max_batchsize * self.num_actions * np.dtype(np.float32).itemsize))
		self.d_V = cuda.mem_alloc(int(self.max_batchsize * 1 * np.dtype(np.float32).itemsize))
		self.buffers = [int(self.d_X), int(self.d_P), int(self.d_V)]

	def save_serialized_engine(self, engine_path: pathlib.Path | str):
		'''
		save the serialized engine to engine_path
		'''
		engine_path = Path(engine_path)
		with open(engine_path, "wb") as f:
			f.write(self.serialized_engine)	

	def pred(self, G):
		'''
		input G is a list (or array) of gamestates of dtype = np.float32
		returns policy = exp(log_policy) and value
		'''
		if type(G) != np.ndarray:
			G = np.array(G, dtype=np.float32)

		X = game.inputNetworkMany(G)
		assert X.shape[1:] == self.input_shape
		num_games = X.shape[0]

		if num_games == 0:
			# nothing to do
			return None, None

		batchsize = self.max_batchsize		
		log_policy = np.empty((num_games, self.num_actions), dtype=np.float32)
		value =  np.empty((num_games, 1), dtype=np.float32)

		for i in range(0, num_games, batchsize):
			num_items = min(batchsize, num_games - i)
			batch_X = X[i:i+batchsize]
			batch_size = batch_X.shape[0]
			cuda.memcpy_htod(self.d_X, batch_X)
			self.context.set_input_shape('input', (num_items, *self.input_shape))
			self.context.execute_v2(bindings=self.buffers)
			cuda.memcpy_dtoh(log_policy[i:i+batch_size], int(self.d_P))
			cuda.memcpy_dtoh(value[i:i+batch_size], int(self.d_V))
		policy = np.exp(log_policy)
		return policy, value

	def __call__(self, G):
		return self.pred(G)
	