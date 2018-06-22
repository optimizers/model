classdef RegModel < model.NlpModel
   % REGMODEL  Dual-Regularized nonlinear program
   %
   % Adds dual regularization to problem
   %
   % minimize  f(x) + 0.5*|r|^2
   % subj to   cL <= c(x) + d*r <= cU
   %           bL <= x <= bU
   %
   % from the inequality-based formulation:
   %
   % minimize  f(x)
   % subj to   cL <= c(x) <= cU
   %           bL <=   x  <= bU

   
   properties
      nlp    % original inequality-based object
      ireg   % indictor of regularization variables
      delta  % Regularization parameter
   end
   
   methods
      
      function self = RegModel(nlp, delta)

         % Upper and lower bounds for the variables and constraints.
         bL = [nlp.bL; -Inf*ones(nlp.m,1)];
         bU = [nlp.bU; Inf*ones(nlp.m,1)];
         cL = nlp.cL;
         cU = nlp.cU;

         % Initial point. Set r to 0
         x0 = [ nlp.x0; zeros(nlp.m,1) ];

         % Instantiate from the base class.
         self = self@model.NlpModel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;
         
         % Create an identifier for regularization variables.
         self.ireg = [ false(nlp.n,1); true(nlp.m,1) ];   

         % Jacobian sparsity pattern of the slack model.
         J = nlp.gcon(nlp.x0);
         self.Jpattern = [spones(J)  speye(nlp.m)];

         % Hessian sparsity pattern.
         y = ones(nlp.m,1);
         HL = nlp.hlag(nlp.x0, y);
         nR = self.n - nlp.n;
         self.Hpattern = [ spones(HL)         sparse(nlp.n, nR)
                           sparse(nR, nlp.n)  speye(nR)         ];
         
         % Store the original NLP model.
         self.nlp = nlp;
         
         % Store the regularization parameter
         self.delta = delta;
         
      end

      function f = fobj_local(self, xr)
         %FOBJ  Objective function.
         x = xr(~self.ireg,:);
         f = self.nlp.fobj(x);
         f = f + 0.5*(xr(self.ireg,:)'*xr(self.ireg,:));
      end
      
      function g = gobj_local(self, xr)
         %HOBJ  Gradient of objective function.
         x = xr(~self.ireg,:);
         gx = self.nlp.gobj(x);
         g = [gx; xr(self.ireg,:)];
      end
      
      function H = hobj_local(self, xr)
         %HOBJ  Hessian of objective function.
         x = xr(~self.ireg,:);
         Hx = self.nlp.hobj(x);
         nmZ = sparse(self.nlp.n, self.nlp.m);
         mmI = speye(self.nlp.m, self.nlp.m);
         H = [ Hx     nmZ
               nmZ'   mmI ];
      end
      
      function c = fcon_local(self, xr)
         %FCON  Constraint function.
         x = xr(~self.ireg,:);
         r = xr( self.ireg,:);
         c = self.nlp.fcon(x) + self.delta*r;
      end
      
      function J = gcon_local(self, xr)
         %GCON  Constraint Jacobian.
         x = xr(~self.ireg,:);
         Jx = self.nlp.gcon(x);
         J = [Jx self.delta*speye(self.nlp.m)];
      end
      
      function [Jprod, Jtprod] = gconprod_local(self, xr)
         [Jxprod, Jxtprod] = self.nlp.gconprod(xr(~self.ireg,:));
         Jprod = @(v) Jxprod(v(~self.ireg)) + self.delta*v(self.ireg,:);
         Jtprod = @(v) [Jxtprod(v); self.delta*v];
      end
      
      function HL = hlag_local(self, xr, y)
         %HLAG  Hessian of Lagrangian (sparse matrix).
         x = xr(~self.ireg,:);
         H = self.nlp.hlag(x, y);
         nmZ = zeros(self.nlp.n, self.nlp.m);
         mmI = speye(self.nlp.m);
         HL = [ H     nmZ
                nmZ'  mmI ];
      end
      
      function Hv = hlagprod_local(self, xr, y, vv)
         %HLAGPROD  Hessian-vector product with Hessian of Lagrangian.
         x = xr(~self.ireg);
         v = vv(~self.ireg);
         Hv = zeros(self.n, 1);
         Hv(~self.ireg) = self.nlp.hlagprod(x, y, v);
         Hv(self.ireg) = vv(self.ireg);
      end
      
      function HC = hcon_local(self, xr, yy)
         %HCON  Hessian of Lagrangian (without objective; sparse matrix).
         x = xr(~self.ireg,:);
         y = yy(1:self.nlp.m,:);
         H = self.nlp.hcon(x, y);
         n = self.nlp.n;
         nS = self.n - n;
         HC = [ H              sparse(n, nS)
                sparse(nS, n)  sparse(nS, nS) ];
      end

      function Hv = hconprod_local(self, xr, yy, vv)
         %HCONPROD  Hessian-vector product with HCON.
         x = xr(~self.ireg,:);
         v = vv(~self.ireg,:);
         Hv = zeros(self.n, 1);
         Hv(~self.ireg) = self.nlp.hconprod(x, yy, v);
      end
      
      function z = ghivprod_local(self, xr, gxr, vxr)
         x = xr (~self.ireg,:);
         g = gxr(~self.ireg,:);
         v = vxr(~self.ireg,:);
         z = zeros(self.m,1);
         z(~self.linear) = self.nlp.ghivprod(x, g, v);
      end
      
   end % methods
      
end % classdef
