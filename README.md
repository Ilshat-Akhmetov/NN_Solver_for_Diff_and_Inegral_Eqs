## An application of a neural network for solving ODEs and integral equations. 

The main idea is to minimize sum of residual's squares for each boundary or initial condition
and for main equation on its domain. 

For example, if we have equation 

$$ u' +2xu = 5, x \in [0,1] $$

with initial condition

$$ u'(0) = 3 $$

then the proposed algorithm will try to find a neural network minimizing the following expression:

$$ argmin_{NN}((NN(x)' +2xNN(x) - 5)^2 + (NN(0)' - 3)^2) $$

on equation's domain. So we approximate unknown function **u(x)** with a neural network (**NN(x)**). 
For inegral equations idea essentially is the same. 
As you might guess this method tries to minimize sum of squares of residuals for main equation and 
boundary conditions simultaneously at each epoch, 
so eventually NN-approximator becomes quite accurate at representing unknown function **u**.

In **SourceCode** you can take a look at code and find out how classes and methods are implemented.

In **JupyterPresentations\Solving_ODE_with_NN.ipynb** you can find various examples of NN's
application for solving ordinary differential equations. Sometimes it does not converge. 
I cannot say now why it not always work, this question requires further investigation.

In **JupyterPresentations\Solving_Integral_Equation_with_NN.ipynb** you can find various examples of NN's
application for solving integral equations.

This version is rather a prototype than a finished project. In future i hope to expand this program on 2 and 3
dimensional cases and an arbitrary system of equations.
