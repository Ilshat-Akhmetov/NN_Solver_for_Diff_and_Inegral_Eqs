## A neural network for solving ordinary differential equations. 

It  tries to minimise sum of residual's squares for each boundary or initial condition
and for main equation on its domain. 

For example, if we have equation 

<img src="https://latex.codecogs.com/gif.latex? u' +2xu = 5, x \in [0,1]" /> 

with initial condition

<img src="https://latex.codecogs.com/gif.latex? u'(0) = 3 " /> 

then the  neural network will try to minimize following expression:

<img src="https://latex.codecogs.com/gif.latex? argmin_{NN}((NN' +2xNN - 5)^2 + (NN'(0) - 3)^2) " />

on each point of domain x. So we approximate unknown function u with a neural network (NN). 
As you might guess, this method tries to minimise residuals for main equation and 
initial conditions simultaneously, altogether.

In **NN_Solve_for_ODE_Presentation.ipynb** you may found various examples of NN's
application for solving ordinary differential equations. Sometimes it does not converge. 
I cannot say now why it not always work, this question requires further investigation.

In future im going to expand it on 2 and 3 dimensional cases and separate plot and report classes. 
This version is rather a prototype than a finished project.
