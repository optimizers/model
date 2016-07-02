% A model with limited-memory SR1 Hessian in forward mode.
%

classdef lsr1model < model.nlpmodel

  properties
    nlp    % original model because inheritance sucks in Matlab
    lsr1  % L-SR1 Hessian approximation
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  methods

    function self = lsr1model(nlp, mem)

      % Instantiate the base class.
      self = self@model.nlpmodel(nlp.name, nlp.x0, nlp.cL, nlp.cU, nlp.bL, nlp.bU);
      self.nlp = nlp;
      self.lsr1 = opLSR1(nlp.n, mem);
      self.lsr1.update_forward = true;
      self.lsr1.scaling = true;
    end

    function f = fobj_local(self, x)
      f = self.nlp.fobj(x);
    end

    function g = gobj_local(self, x)
      g = self.nlp.gobj(x);
    end

    function H = hobj_local(self, ~)
      H = self.lsr1;
    end

    function H = hlag_local(self, ~, ~)
      H = self.lsr1;
    end

    function w = hlagprod_local(self, ~, ~, v)
      w = self.lsr1 * v;
    end

  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  methods (Sealed = true)

    function self = quasi_newton_update(self, s, y)
      self.lsr1 = update(self.lsr1, s, y);
    end

  end

end
