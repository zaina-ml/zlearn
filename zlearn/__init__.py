import numpy as np
import inspect

def matmul(x1, x2):
    if x1.dtype.istype(x2.dtype):
        return Tensor(np.dot(x1._numpy(), x2._numpy()), dtype=x1.dtype)
    
    raise TypeError(f"Target tensors do not have the same dtype {x1.dtype, x2.dtype}")

def add(x1, x2):
    if x1.dtype.istype(x2.dtype):
        return Tensor(np.add(x1._numpy(), x2._numpy()), dtype=x1.dtype)
    
    raise TypeError(f"Target tensors do not have the same dtype {x1.dtype, x2.dtype}")

def subtract(x1, x2):
    if x1.dtype.istype(x2.dtype):
        return Tensor(np.subtract(x1._numpy(), x2._numpy()), dtype=x1.dtype)
    
    raise TypeError(f"Target tensors do not have the same dtype {x1.dtype, x2.dtype}")

def multiply(x1, x2):
    if x1.dtype.istype(x2.dtype):
        return Tensor(np.multiply(x1._numpy(), x2._numpy()), dtype=x1.dtype)
    
    raise TypeError(f"Target tensors do not have the same dtype {x1.dtype, x2.dtype}")

def divide(x1, x2):
    if x1.dtype.istype(x2.dtype):
        return Tensor(np.divide(x1._numpy(), x2._numpy()), dtype=x1.dtype)
    
    raise TypeError(f"Target tensors do not have the same dtype {x1.dtype, x2.dtype}")

def relu(x):
    return Tensor(x._numpy() * (x._numpy() > 0), dtype=x.dtype)

def sigmoid(x):
    return Tensor(1/(1 + np.exp(-x._numpy())), dtype=x.dtype)  

def softmax(x):
    e_x = np.exp(x._numpy() - np.max(x._numpy()))

    return Tensor(e_x / e_x.sum(axis=0), dtype=Float32())

def arange(start: int, step: int, stop: int, dtype):
    return Tensor(np.arange(start=start, step=step, stop=stop), dtype=dtype)

class Float64:
    def istype(self, dtype) -> bool:
        if str(dtype) == str(Float64()):
            return True

    def _port_to_type(self, tensor) -> np.ndarray:
        return tensor.astype(np.float64)

    def __repr__(self) -> str:
        return 'zlearn.Float64'

class Float32:
    def istype(self, dtype) -> bool:
        if str(dtype) == str(Float32()):
            return True

    def _port_to_type(self, tensor) -> np.ndarray:
        return tensor.astype(np.float32)

    def __repr__(self) -> str:
        return 'zlearn.Float32'

class Int64:
    def is_type(self, dtype) -> bool:
        if str(dtype) == str(Int64()):
            return True
    
    def _port_to_type(self, tensor) -> np.ndarray:
        return tensor.astype(np.int64)

    def __repr__(self) -> str:
        return 'zlearn.Int64'

class Int32:
    def istype(self, dtype) -> bool:
        if str(dtype) == str(Int32()):
            return True
    
    def _port_to_type(self, tensor) -> np.ndarray:
        return tensor.astype(np.int32)

    def __repr__(self) -> str:
        return 'zlearn.Int32'


class Tensor:
    def __init__(self, tensor: list, dtype) -> None:
        if inspect.isclass(dtype):
            raise TypeError(f"{dtype} is a class, not an object. Perhaps you forgot parenthesis? (ex. zlearn.Float64())")
        
        self.tensor = dtype._port_to_type(self._numpy(tensor))
        self.dtype = dtype
    
    def to_type(self, dtype):
        return Tensor(tensor=dtype._port_to_type(self._numpy(self.tensor)), dtype=dtype)

    def to_scalar(self) -> int:
        if self.tensor.shape:
            raise ValueError(f"Target Tensor: {self.tensor} is not scalar")

        return self.tensor.item()

    def max(self):
        return Tensor(np.max(self.tensor), dtype=self.dtype)

    def min(self):
        return Tensor(np.min(self.tensor), dtype=self.dtype)

    def mean(self):
        return Tensor(np.mean(self.tensor), dtype=self.dtype)

    def sum(self):
        return Tensor(np.sum(self.tensor), dtype=self.dtype)
    
    def transpose(self):
        return Tensor(np.transpose(self.tensor).tolist(), dtype=self.dtype)

    def numel(self):
        return Tensor(len(self.tensor), dtype=Int32())
    
    def _numpy(self, tensor=None) -> np.ndarray:
        if tensor is None:
            return np.array(self.tensor)
        
        return np.array(tensor) 
    
    def __repr__(self) -> str:
        return f"zlearn.Tensor({self.tensor}, {self.dtype})"
    
    def __getitem__(self, index):
        return Tensor(self.tensor[index], dtype=self.dtype)