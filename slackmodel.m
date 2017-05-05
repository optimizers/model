classdef slackmodel < model.nlpmodel
   % SLACKMODEL  Equality constraints with bounds.
   %
   % Derives the following slack formulation
   %
   % minimize  f(x,s)
   % subj to   ceq(x) = ceq
   %           c(x) - s = 0
   %           bL <= x <= bU
   %           cL <= s <= cU
   %
   % from the inequality-based formulation:
   %
   % minimize  f(x)
   % subj to   ceq(x) = ceq
   %           cL <= c(x) <= cU
   %           bL <=   x  <= bU
   %
   % Thus, the problem appears as such:
   %
   % minimize  f(x)  subj to  c(x) = 0, bL <= x <= bU.
   
   properties
      nlp    % original inequality-based object
      islack % indictor of slack variables
      nslack % number of slack variables
   end

   properties (SetAccess = private, Hidden = false)
      Aeq    % Jacobian of linear constraints
      beq    % RHS of linear constraints
   end
   
   methods
      
      function self = slackmodel(nlp)

         % Upper and lower bounds for the variables and slacks.
         bL = [ nlp.bL; nlp.cL(~nlp.iFix) ];
         bU = [ nlp.bU; nlp.cU(~nlp.iFix) ];

         % The linear and nonlinear constraints are equalities, ie,
         % 0 <= c(x) - s <= 0.
         cL = nlp.cL;
         cL(~nlp.iFix) = zeros(sum(~nlp.iFix), 1);
         cU = nlp.cU;
         cU(~nlp.iFix) = zeros(sum(~nlp.iFix), 1);

         % Initial point. Set slacks to be feasible.
         c = nlp.fcon(nlp.x0);
         x0 = [ nlp.x0; c(~nlp.iFix) ];

         % Instantiate from the base class.
         self = self@model.nlpmodel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;
         
         % Create an indetifier for slack variables.
         nS = sum(~nlp.iFix);
         self.nslack = nS;
         self.islack = [ false(nlp.n,1); true(nS,1) ];   

         % Jacobian sparsity pattern of the slack model.
         J = nlp.gcon(nlp.x0);
         self.Jpattern = [spones(J)  speye(nS)];

         % Hessian sparsity pattern.
         y = ones(size(c));
         HL = nlp.hlag(nlp.x0, y);
         self.Hpattern = [ spones(HL)         sparse(nlp.n, nS)
                           sparse(nS, nlp.n)  sparse(nS   , nS) ];
         
         % Store linear Jacobian and RHS.
         self.Aeq =  J(self.linear, :);
         self.beq = cL(self.linear, :);
         
         % Store the original NLP model.
         self.nlp = nlp;
         
      end

      function f = fobj_local(self, xs)
         x = xs(~self.islack,:);
         f = self.nlp.fobj(x);
      end
      
      function g = gobj_local(self, xs)
         x = xs(~self.islack,:);
         gx = self.nlp.gobj(x);
         g = [gx; zeros(self.nslack, 1)];
      end
      
      function H = hobj_local(self, xs)
         x = xs(~self.islack,:);
         Hx = self.nlp.hobj(x);
         nmZ = sparse(self.nlp.n, self.nslack);
         mmZ = sparse(self.nlp.m, self.nslack);
         H = [ Hx     nmZ
               nmZ'   mmZ ];
      end
      
      function c = fcon_local(self, xs)
         x = xs(~self.islack,:);
         s = xs( self.islack,:);
         c = self.nlp.fcon(x);
         c(~self.nlp.iFix) = c(~self.nlp.iFix) - s;
      end
      
      function J = gcon_local(self, xs)
         x = xs(~self.islack,:);
         Jx = self.nlp.gcon(x);
         J = [Jx -speye(self.nslack)];
      end
      
      function HL = hlag_local(self, xs, y)
         x = xs(~self.islack,:);
         H = self.nlp.hlag(x, y);
         nmZ = zeros(self.nlp.n, self.nslack);
         mmZ = zeros(self.nslack);
         HL = [ H     nmZ
                nmZ'  mmZ ];
      end

      function Hv = hconprod_local(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = zeros(self.n, 1);
         Hv(~self.islack) = self.nlp.hconprod(x, y, v);
      end
      
      function Hv = hlagprod_local(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = zeros(self.n, 1);
         Hv(~self.islack) = self.nlp.hlagprod(x, y, v);
      end
      
      function z = ghivprod_local(self, xs, gxs, vxs)
         x = xs (~self.islack);
         g = gxs(~self.islack);
         v = vxs(~self.islack);
         z = zeros(self.m,1);
         z(~self.linear) = self.nlp.ghivprod(x, g, v);
      end
      
   end % methods
      
end % classdef
