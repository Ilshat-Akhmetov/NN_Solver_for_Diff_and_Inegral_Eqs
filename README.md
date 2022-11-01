## An application of a neural network for solving ordinary differential equations. 

It  tries to minimise sum of residual's squares for each boundary or initial condition
and for main equation on its domain. 

For example, if we have equation 

$$ u' +2xu = 5, x \in [0,1] $$

with initial condition

$$ u'(0) = 3 $$

then the  neural network will try to minimize following expression:

$$ argmin_{NN}((NN(x)' +2xNN(x) - 5)^2 + (NN(0)' - 3)^2) $$

on equation's domain. So we approximate unknown function **u(x)** with a neural network (**NN(x)**). 
As you might guess, this method tries to minimize square residuals for main equation and 
boundary conditions simultaneously at each epoch, 
so eventually NN-approximator becomes quite accurate at representing unknown function **u**.

In **SourceCode** you can take a look at code and find out how classes and methods are implemented.

In **JupyterPresentations\NN_Solve_for_ODE_Presentation.ipynb** you may find various examples of NN's
application for solving ordinary differential equations. Sometimes it does not converge. 
I cannot say now why it not always work, this question requires further investigation.

In future im going to expand it on 2 and 3 dimensional cases and separate plot and report classes. 
This version is rather a prototype than a finished project.
