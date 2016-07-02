% A model with limited-memory BFGS Hessian in forward mode.
%

classdef lbfgsmodel < model.nlpmodel

  properties
    nlp    % original model because inheritance sucks in Matlab
    lbfgs  % L-BFGS Hessian approximation
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  methods

    function self = lbfgsmodel(nlp, mem)

      % Instantiate the base class.
      self = self@model.nlpmodel(nlp.name, nlp.x0, nlp.cL, nlp.cU, nlp.bL, nlp.bU);
      self.nlp = nlp;
      self.lbfgs = opLBFGS(nlp.n, mem);
      self.lbfgs.update_forward = true;  % Use L-BFGS in forward mode.
      self.lbfgs.scaling = true;
    end

    function f = fobj_local(self, x)
      f = self.nlp.fobj(x);
    end

    function g = gobj_local(self, x)
      g = self.nlp.gobj(x);
    end

    function H = hobj_local(self, ~)
      H = self.lbfgs;
    end

    function H = hlag_local(self, ~, ~)
      H = self.lbfgs;
    end

    function w = hlagprod_local(self, ~, ~, v)
      w = self.lbfgs * v;
    end

  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  methods (Sealed = true)

    function self = quasi_newton_update(self, s, y)
      self.lbfgs = update(self.lbfgs, s, y);
    end

  end

end
