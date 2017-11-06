classdef pdecontrolmodel < model.nlpmodel

   properties
      n_cells;      % Number of cells
      alpha;        % Penalty parameter
      
      lin_explicit   % Separate linear and nonlinear parts of constraints 
      
      pdeobj;       % Objective function struct
      pdecon;       % Constraint function struct
   end % properties
   
   methods
       function self = pdecontrolmodel(name, n_cells, alpha, pdeobj, pdecon, x0, varargin)
           
          p = inputParser;
          addParameter(p,'lin_explicit', false);
          parse(p, varargin{:});
          
          n = length(x0);
          
          if p.Results.lin_explicit
             % We separate the linear part of the constraint from the
             % nonlinear part using a slack formulation. We set
             % A*x + c(x) = b => c(x) - s = 0, A*x + s = b
             
             s0 = pdecon.evaluateNonlinearityValue(x0);
             x0 = [x0;s0];
             n = n + length(s0);
             m = 2*length(s0);
          else
             c0 = pdecon.value(x0);
             m = length(c0);             
          end

          bL = -Inf(n,1);
          bU = Inf(n,1);
          cL = zeros(m,1);
          cU = zeros(m,1); 
          
          self = self@model.nlpmodel(name, x0, cL, cU, bL, bU);

          self.lin_explicit = p.Results.lin_explicit;
          
          % TODO: Remove when this becomes matrix-free
          if self.lin_explicit
             nn = n_cells;
             JL = pdecon.evaluateLinearityJacobian(x0(1:2*nn));
             JN = pdecon.evaluateNonlinearityJacobian(x0(1:2*nn));
             J = [JN -speye(nn-1);JL speye(nn-1)];
             self.Jpattern = spones(J);
             self.linear(nn:end) = true(nn-1,1);
          else
             J = pdecon.Jacobian(x0);  
             self.Jpattern = spones(J');
          end
          
          self.n_cells = n_cells;
          self.alpha = alpha;
          self.pdeobj = pdeobj;
          self.pdecon = pdecon;
       end
       
       function f = fobj_local(self, x)
          n = self.n_cells;
          f = self.pdeobj.value(x(1:2*n)); 
       end
       
       function g = gobj_local(self, x)
          n = self.n_cells;
          g = self.pdeobj.gradient(x(1:2*n));
          if self.lin_explicit
              g = [g; zeros(n-1,1)];
          end
       end
       
       function H = hobj_local(self, x)
          n = self.n_cells;
          H = self.pdeobj.hessian(x(1:2*n));
          if self.lin_explicit
             H = [H sparse(2*n,n-1); sparse(n-1,2*n) sparse(n-1,n-1)];
          end
       end
       
       function c = fcon_local(self, x)
          n = self.n_cells;
          if self.lin_explicit
             s = x(2*n+1:end);
             cl = self.pdecon.evaluateLinearityValue(x(1:2*n));
             cn = self.pdecon.evaluateNonlinearityValue(x(1:2*n));
             c = [cn - s; cl + s];
          else
             c = self.pdecon.value(x);
          end
       end
       
       function J = gcon_local(self, x)
          if self.lin_explicit
             nn = self.n_cells;
             JL = self.pdecon.evaluateLinearityJacobian(x(1:2*nn));
             JN = self.pdecon.evaluateNonlinearityJacobian(x(1:2*nn));
             J = [JN -speye(nn-1); JL speye(nn-1)];             
          else
             J = self.pdecon.Jacobian(x)';
          end
       end

       function [Jprod, Jtprod] = gconprod_local(self, x)
          J = self.gcon(x);
          Jprod = @(v) J*v;
          Jtprod = @(v) J'*v;       
       end
       
       function HL = hlag_local(self, x, y)
          n = self.n_cells;
          H = self.pdeobj.hessian(x(1:2*n));
          Hc = self.pdecon.constraintHessian(x(1:2*n), y(1:n-1));
          HL = H - Hc;
          if self.lin_explicit
             HL = [HL sparse(2*n,n-1);sparse(n-1,2*n) sparse(n-1,n-1)]; 
          end
       end
       
       function w = hlagprod_local(self, x, y, v)
          n = self.n_cells;
          w = self.pdeobj.hessVec(v(1:2*n), x(1:2*n));
          w = w - self.pdecon.applyConstraintHessian(x(1:2*n), y(1:n-1), v(1:2*n));
          if self.lin_explicit
             w = [w; zeros(n-1,1)];
          end
       end

       function w = hconprod_local(self, x, y, v)
          n = self.n_cells;
          w = self.pdecon.applyConstraintHessian(x(1:2*n), y(1:n-1), v(1:2*n));
          if self.lin_explicit
             w = [w; zeros(n-1,1)];
          end
       end

       function w = ghivprod_local(self, x, y, v)
          n = self.n_cells;
          w = self.pdecon.ghivprod(x(1:2*n), y(1:n-1), v(1:2*n));
          if self.lin_explicit
             w = [w; zeros(n-1,1)];
          end
       end

       function P = preconditioner(self, x)
           nn = self.n_cells;
           A = self.pdecon.evaluateLinearityJacobian(x(1:2*nn));
           % P = @(v) [v(1:self.n); (A*A')\v(self.n+1:end)];
           P = @(v) (A*A')\v;
       end
        
   end % methods
    
    
end