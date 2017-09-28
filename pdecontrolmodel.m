classdef pdecontrolmodel < model.nlpmodel

   properties
      n_cells;  % Number of cells
      alpha;    % Penalty parameter
      
      pdeobj;   % Objective function struct
      pdecon;   % Constraint function struct
   end % properties
   
   methods
       function self = pdecontrolmodel(name, n_cells, alpha, pdeobj, pdecon, x0)
          n = length(x0);
          c0 = pdecon.value(x0);
          m = length(c0);
          bL = -Inf(n,1);
          bU = Inf(n,1);
          cL = zeros(m,1);
          cU = zeros(m,1);
           
          self = self@model.nlpmodel(name, x0, cL, cU, bL, bU);
          
          % TODO: Remove when this becomes matrix-free
          J = pdecon.Jacobian(x0);
          self.Jpattern = spones(J');
          
          self.n_cells = n_cells;
          self.alpha = alpha;
          self.pdeobj = pdeobj;
          self.pdecon = pdecon;
       end
       
       function f = fobj_local(self, x)
          f = self.pdeobj.value(x); 
       end
       
       function g = gobj_local(self, x)
          g = self.pdeobj.gradient(x);
       end
       
       function H = hobj_local(self, x)
          H = self.pdeobj.hessian(x);
       end
       
       function c = fcon_local(self, x)
          c = self.pdecon.value(x);
       end
       
       function J = gcon_local(self, x)
          J = self.pdecon.Jacobian(x)'; 
       end
       
       function HL = hlag_local(self, x, y)
          H = self.pdeobj.hessian(x);
          Hc = self.pdecon.constraintHessian(x, y);
          HL = H - Hc;
       end
       
       function w = hlagprod_local(self, x, y, v)
          w = self.pdeobj.hessVec(v, x);
          w = w - self.pdecon.applyConstraintHessian(x, y, v);
       end

       function w = hconprod_local(self, x, y, v)
          w = self.pdecon.applyConstraintHessian(x, y, v);
       end

       function w = ghivprod_local(self, x, y, v)
          w = self.pdecon.ghivprod(x, y, v);
       end

   end % methods
    
    
end