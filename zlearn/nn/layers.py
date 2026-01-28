import random

from zlearn import Tensor
from zlearn import Int32, Int64, Float32, Float64
from zlearn import multiply, matmul, add, subtract, relu, sigmoid, softmax

class Linear:
    def __init__(self, in_features, out_features, bias=True, has_params=True) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.bias = bias
        self.has_params = has_params

        self.params = None
    
    def __call__(self, input_: Tensor) -> Tensor:
        return add(matmul(add(input_, self.params['bias']['in']), self.params['weight'].transpose()), self.params['bias']['out'])
    
    def set_params(self) -> None:
        params = {
            'bias':{},
            'weight': self.set_weight(in_features=self.in_features, out_features=self.out_features)
        }

        if self.bias:
            params['bias']['in'] = self.set_bias(features=self.in_features)
        else:
            params['bias']['in'] = Tensor([0], dtype=Float32())
        
        params['bias']['out'] = self.set_bias(features=self.out_features)

        self.params = params

    def set_bias(self, features) -> Tensor:
        bias = []

        for i in range(features):
            bias.append(self.bias_param())
        
        return Tensor(bias, dtype=Float32())

    def set_weight(self, in_features, out_features) -> Tensor:
        weight = []

        for i in range(out_features):
            weight_stack = []

            for x in range(in_features):
                weight_stack.append(self.weight_param())
            weight.append(weight_stack)

        return Tensor(weight, dtype=Float32())

    def weight_param(self) -> float:
        return round(random.uniform(-0.02, 0.02), 3)

    def bias_param(self) -> float:
        return round(random.uniform(-0.02, 0.02), 3)
    
    def __str__(self) -> str:
        return f'(Linear) (in_features: {self.in_features}) (out_features: {self.out_features})'


class ReLU:
    def __init__(self, has_params=False):
        self.has_params = has_params

    def __call__(self, input_: Tensor):
        return relu(input_)
    
    def __str__(self) -> str:
        return f'(ReLU)'
    
class Sigmoid:
    def __init__(self, has_params=False):
        self.has_params = has_params

    def __call__(self, input_: Tensor):
        return sigmoid(input_)
    
    def __str__(self) -> str:
        return f'(Sigmoid)'

class Softmax:
    def __init__(self, has_params=False):
        self.has_params = has_params

    def __call__(self, input_: Tensor):
        return softmax(input_)
    
    def __str__(self) -> str:
        return f'(Softmax)'