classdef slackmodel < model.nlpmodel
   % SLACKMODEL  Equality constraints with bounds.
   %
   % Derives the following slack formulation
   %
   % minimize  f(x,s)
   % subj to   c(x) - s = 0
   %           bL <= x <= bU
   %           cL <= s <= cU
   %
   % from the inequality-based formulation:
   %
   % minimize  f(x)
   % subj to   cL <= c(x) <= cU
   %           bL <=   x  <= bU
   %
   % Thus, the problem appears as such:
   %
   % minimize  f(x)  subj to  c(x) = 0, bL <= x <= bU.
   
   properties
      nlp    % original inequality-based object
      islack % indictor of slack variables
   end

   properties (SetAccess = private, Hidden = false)
      Aeq    % Jacobian of linear constraints
      beq    % RHS of linear constraints
   end
   
   methods
      
      function self = slackmodel(nlp)

         % Upper and lower bounds for the variables and slacks.
         bL = [ nlp.bL; nlp.cL ];
         bU = [ nlp.bU; nlp.cU ];

         % The linear and nonlinear constraints are equalities, ie,
         % 0 <= c(x) - s <= 0.
         cL = zeros(nlp.m, 1);
         cU = zeros(nlp.m, 1);

         % Initial point. Set slacks to be feasible.
         c = nlp.fcon(nlp.x0);
         x0 = [ nlp.x0; c ];

         % Instantiate from the base class.
         self = self@model.nlpmodel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;
         
         % Create an indetifier for slack variables.
         self.islack = [ false(nlp.n,1); true(nlp.m,1) ];   

         % Jacobian sparsity pattern of the slack model.
         J = nlp.gcon(nlp.x0);
         self.Jpattern = [spones(J)  speye(nlp.m)];

         % Hessian sparsity pattern.
         y = ones(size(c));
         HL = nlp.hlag(nlp.x0, y);
         nS = self.n - nlp.n;
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
         g = [gx; zeros(self.nlp.m, 1)];
      end
      
      function H = hobj_local(self, xs)
         x = xs(~self.islack,:);
         Hx = self.nlp.hobj(x);
         nmZ = sparse(self.nlp.n, self.nlp.m);
         mmZ = sparse(self.nlp.m, self.nlp.m);
         H = [ Hx     nmZ
               nmZ'   mmZ ];
      end
      
      function c = fcon_local(self, xs)
         x = xs(~self.islack,:);
         s = xs( self.islack,:);
         c = self.nlp.fcon(x) - s;
      end
      
      function J = gcon_local(self, xs)
         x = xs(~self.islack,:);
         Jx = self.nlp.gcon(x);
         J = [Jx -speye(self.nlp.m)];
      end
      
      function HL = hlag_local(self, xs, y)
         x = xs(~self.islack,:);
         H = self.nlp.hlag(x, y);
         nmZ = zeros(self.nlp.n, self.nlp.m);
         mmZ = zeros(self.nlp.m);
         HL = [ H     nmZ
                nmZ'  mmZ ];
      end
      
      function Hv = hlagprod_local(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = zeros(self.n, 1);
         Hv(~self.islack) = self.nlp.hlagprod(x, y, v);
      end
      
   end % methods
      
end % classdef
