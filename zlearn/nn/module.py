class Sequential:
    def __init__(self, *layers):
        self.layers = self.init_layers(layers)
    
    def init_layers(self, layers):
        for idx, layer in enumerate(layers):
            if idx == 0:
                layer.bias = False

            if layer.has_params:   
                layer.set_params()
        
        return layers
        

    def __str__(self):
        cout = 'module.Sequential(\n'
        
        for idx, layer in enumerate(self.layers):
            cout += f'  ({idx}):{layer}\n'       
        
        cout += ')'
        
        return cout

    def __call__(self, input_):
        output = input_

        for layer in self.layers:
            output = layer(output)
        
        return output
