classdef nlpproblem < model.nlpmodel
   
  properties
    fobj_loc
    gobj_loc
    hobj_loc
    fcon_loc
    gcon_loc
    gconprod_loc
    hcon_loc
    hlag_loc
    hconprod_loc
    hlagprod_loc
    ghivprod_loc
   end

   methods

      function self = nlpproblem(name, x0, cL, cU, bL, bU)
         % Instantiate the base class.
         self = self@model.nlpmodel(name, x0, cL, cU, bL, bU);
      end
      
      function f = fobj_local(self, x)
         %FOBJ  Objective function.
         f = self.fobj_loc(x);
      end

      function g = gobj_local(self, x)
         %GOBJ  Gradient bjective function.
         g = self.gobj_loc(x);
      end
      
      function H = hobj_local(self, x)
         %HOBJ  Hessian of objective function.
         H = self.hobj_loc(x);
      end
      
      function c = fcon_local(self, x)
         %FCON  Constraint functions, nonlinear followed by linear.
         c = self.fcon_loc(x);
      end
      
      function J = gcon_local(self, x)
         %GCON  Constraint functions Jacobian, nonlinear followed by linear.
         J = self.gcon_loc(x);
      end
      
      function [Jprod, Jtprod] = gconprod_local(self, x)
         %GCON  Constraint functions Jacobian, nonlinear followed by linear.
         [Jprod, Jtprod] = self.gconprod_loc(x);
      end
      
      function HC = hcon_local(self, x, y)
         HC = self.hcon_loc(x, y);
      end

      function Hv = hconprod_local(self, x, y, v)
         Hv = self.hconprod_loc(x, y, v);
      end

      function Hv = hlagprod_local(self, x, y, v)
         Hv = self.hlagprod_loc(x, y, v);
      end

      function HL = hlag_local(self, x, y)
         HL = self.ah.hlag_loc(x, y);
      end
      
      function gHiv = ghivprod_local(self, x, g, v)
         gHiv = self.ghivprod_loc(x, g, v);
      end
      
   end

end % classdef
