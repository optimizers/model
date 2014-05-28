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
         [c, J] = nlp.con(nlp.x0);
         x0 = [ nlp.x0; c ];

         % Instantiate from the base class.
         self = self@model.nlpmodel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;
         
         % Create an indetifier for slack variables.
         self.islack = [ false(nlp.n,1); true(nlp.m,1) ];   

         % Jacobian sparsity pattern of the slack model.
         self.Jpattern = [spones(J)  speye(nlp.m)];

         % Hessian sparsity pattern.
         y = ones(size(c));
         HL = nlp.hesslag(nlp.x0, y);
         nS = self.n - nlp.n;
         self.Hpattern = [ spones(HL)         sparse(nlp.n, nS)
                           sparse(nS, nlp.n)  sparse(nS   , nS) ];
         
         % Store linear Jacobian and RHS.
         self.Aeq =  J(self.linear, :);
         self.beq = cL(self.linear, :);
         
         % Store the original NLP model.
         self.nlp = nlp;
         
      end
      
      function varargout = obj(self, xs)
         %OBJ  Objective functions.
         
         % gxs = [ g ]  original gradient
         %       [ 0 ]  gradient wrt to slacks
         
         varargout = cell(nargout, 1);
         x = self.extract_original_x(xs);
         
         if nargout <= 1
            f = self.nlp.obj(x);
         elseif nargout <= 2
            [f, gx] = self.nlp.obj(x);
         else
            [f, gx, Hx] = self.nlp.obj(x);
         end
         
         % Objective
         varargout(1) = {f};
         
         % Gradient
         if nargout >= 2
            gxs = [gx; zeros(self.m, 1)];
            varargout(2) = {gxs};
         end
         
         % Hessian
         if nargout >= 3
            nmZ = sparse(self.nlp.n, self.nlp.m);
            mmZ = sparse(self.nlp.n, self.nlp.n);

            %       x(n)   s(m)
            Hxs = [ Hx     nmZ
                    nmZ'   mmZ ];
            varargout(3) = {Hxs};
         end
      end

      function [c, J] = con(self, xs)
         [x, s] = self.extract_original_x(xs);
         if nargout == 1
            c = self.nlp.con(x);
         else
            [c, Jx] = self.nlp.con(x);
            J = [Jx -speye(self.nlp.m)];         
         end
         c = c - s;
      end

      function HL = hesslag(self, xs, y)
         x = self.extract_original_x(xs);
         H = self.nlp.hesslag(x, y);

         nmZ = zeros(self.nlp.n, self.nlp.m);
         mmZ = zeros(self.nlp.m);
         HL = [ H     nmZ
                nmZ'  mmZ ];
      end
      
      function Hv = hesslagprod(self, xs, y, vv)
         x = xs(~self.islack);
         v = vv(~self.islack);
         Hv = zeros(self.n, 1);
         Hv(~self.islack) = self.nlp.hesslagprod(x, y, v);
      end
      
   end % methods
   
   methods (Access=private)
      function [x, s] = extract_original_x(self, xs)
         x = xs(~self.islack);
         if nargout > 1
            s = xs(self.islack);
         end
      end
   end
   
end % classdef
