from torch.autograd import grad

def nth_derivative(function, variable, derivatives_degree):
    for i in range(derivatives_degree):
        grads = grad(function, variable, create_graph=True)[0]
        function = grads.sum()
    return grads
