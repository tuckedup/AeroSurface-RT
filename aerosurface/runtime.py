"""Portable ONNX and native TensorRT execution using PyTorch-owned CUDA buffers."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def ort_session(path: Path, threads: int = 2):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


class TensorRTRuntime:
    """TensorRT 10/11 name-based execution; explicit ownership of all buffers."""

    def __init__(self, path: Path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        self.input_name = self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(s <= 0 for s in shape):
                raise ValueError("Only static batch-one engines are supported")
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16}.get(
                self.engine.get_tensor_dtype(name)
            )
            if dtype is None:
                raise ValueError("Unsupported tensor dtype")
            self.buffers[name] = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        if self.engine.num_io_tensors != 2 or not self.input_name or not self.output_name:
            raise ValueError("Expected one input and one output")
        torch.cuda.synchronize()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if tuple(x.shape) != tuple(self.buffers[self.input_name].shape):
            raise ValueError("TensorRT input shape mismatch")
        with torch.cuda.stream(self.stream):
            self.buffers[self.input_name].copy_(torch.from_numpy(x))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return self.buffers[self.output_name].float().cpu().numpy().copy()


def build_engine(onnx_path: Path, engine_path: Path) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Explicit precision comes from the ONNX tensor types (TensorRT 11 strongly typed).
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(plan))
    return {"tensorrt": trt.__version__, "engine_bytes": engine_path.stat().st_size}
