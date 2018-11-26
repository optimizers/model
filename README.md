model
=====

An NonLinear Program model class for Matlab.

Installation
------------
This is a Matlab *package*, and therefore the local directory should be prefixed with the "+" symbol, i.e.,

  ./+model

nlpmodel.m
----------
`nlpmodel.m` is a general interface for nonlinear programming models that all other model classes inherit from.

Available models
----------------
- `nlpproblem.m`: supports custom user function implementations (for objective, gradient, etc.).
- `amplmodel.m`: interface for AMPL problems (requires [AmplMexInterface](https://github.com/optimizers/AmplMexInterface)).
- `cutestproblm.m`: interface for CUTEST problems (requires [CUTEST](https://github.com/optimizers/cutest-mirror)).
- `slackmodel.m`: converts an instance of nlpmodel into slack formulation (equality constraints and bounds only).
- `slackmodelnn.m`: converts an instance of nlpmodel into nonnegative slack formulation (equality constraints and nonnegative bounds only).

Helper Functions
----------------
`model_classify.m`: write metadata and classification (unconstrained, equality, or bnd constrained) for AMPL or CUTEST problems in a directory.
