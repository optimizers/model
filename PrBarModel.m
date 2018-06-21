classdef PrBarModel < model.NlpModel
   % PRBARMODEL  Convert bound constrained problems to primal barrier.
   %
   % Given the following slack formulation
   %
   % minimize  f(x)
   % subj to   c(x) = 0
   %           bL <= x <= bU
   %
   % change the problem to
   %
   % minimize f(x) - mu*log(x-bL) - mu*log(bU-x)
   % subj to  c(x) = 0
   
   properties
      nlp    % original slack based formulation
      mu     % Barrier parameter
   end
   
   methods
      
      function self = PrBarModel(nlp, mu)

         % Upper and lower bounds for the variables and slacks.
         bL = -Inf(nlp.n,1);
         bU =  Inf(nlp.n,1);

         % The linear and nonlinear constraints are equalities, ie,
         % 0 <= c(x) - s <= 0.
         cL = nlp.cL;
         cU = nlp.cU;

         % Initial point. Must be strictly inside of bounds
         if ~(sum(nlp.x0 > nlp.bL) == nlp.n) ...
                 || ~(sum(nlp.x0 < nlp.bU) == nlp.n)
                   ME = MException('PRBARMODEL:initial point',...
                      'initial point must be strictly inside of bounds');
                   throw(ME);             
         end
         x0 = nlp.x0;

         % Instantiate from the base class.
         self = self@model.NlpModel(nlp.name, x0, cL, cU, bL, bU);

         % Identify the linear constraints.
         self.linear = nlp.linear;

         % Jacobian sparsity pattern of the slack model.
         J = nlp.gcon(nlp.x0);
         self.Jpattern = spones(J);

         % Hessian sparsity pattern.
         y = ones(size(nlp.m));
         HL = nlp.hlag(nlp.x0, y);
         self.Hpattern = spones(HL);
         
         % Store the original NLP model.
         self.nlp = nlp;
         
         % Store the barrier parameter.
         self.mu = mu;
      end

      function f = fobj_local(self, x)
         bL = self.nlp.bL;
         jLow = self.nlp.jLow;
         bU = self.nlp.bU;
         jUpp = self.nlp.jUpp;
         f = self.nlp.fobj(x) - self.mu*sum(log(x(jLow) - bL(jLow))) ...
                              - self.mu*sum(log(bU(jUpp) - x(jUpp)));
      end
      
      function g = gobj_local(self, x)
         bL = self.nlp.bL;
         jLow = self.nlp.jLow;
         bU = self.nlp.bU;
         jUpp = self.nlp.jUpp;
         
         g = self.nlp.gobj(x);
         g(jLow) = g(jLow) - self.mu./(x(jLow) - bL(jLow));
         g(jUpp) = g(jUpp) + self.mu./(bU(jUpp) - x(jUpp));
      end
      
      function H = hobj_local(self, x)
         bL = self.nlp.bL;
         jLow = self.nlp.jLow;
         bU = self.nlp.bU;
         jUpp = self.nlp.jUpp;

         H = self.nlp.hobj(x);
         Hd = zeros(self.n,1);
         Hd(jLow) = self.mu./((x(jLow) - bL(jLow)).^2);
         Hd(jUpp) = Hd(jUpp) + self.mu./((x(jUpp) - bU(jUpp)).^2);
         H = H + spdiags(Hd,0,self.n,self.n);
      end
      
      function c = fcon_local(self, x)
         c = self.nlp.fcon(x);
      end
      
      function J = gcon_local(self, x)
         J = self.nlp.gcon(x);
      end
      
      function [Jprod, Jtprod] = gconprod_local(self, x)
         [Jprod, Jtprod] = self.nlp.gconprod(x);
      end
      
      function HL = hlag_local(self, x, y)
         bL = self.nlp.bL;
         jLow = self.nlp.jLow;
         bU = self.nlp.bU;
         jUpp = self.nlp.jUpp;
          
         HL = self.nlp.hlag(x, y);
         Hd = zeros(self.n,1);
         Hd(jLow) = self.mu./((x(jLow) - bL(jLow)).^2);
         Hd(jUpp) = Hd(jUpp) + self.mu./((x(jUpp) - bU(jUpp)).^2);
         HL = HL + spdiags(Hd,0,self.n,self.n);
      end

      function Hv = hconprod_local(self, x, y, v)
         Hv = self.nlp.hconprod(x, y, v);
      end
      
      function Hv = hlagprod_local(self, x, y, v)
         bL = self.nlp.bL;
         jLow = self.nlp.jLow;
         bU = self.nlp.bU;
         jUpp = self.nlp.jUpp;

         Hv = self.nlp.hlagprod(x, y, v);
         Hv(jLow) = Hv(jLow) + self.mu*v(jLow)./((x(jLow) - bL(jLow)).^2);
         Hv(jUpp) = Hv(jUpp) + self.mu*v(jUpp)./((bU(jUpp) - x(jUpp)).^2);
      end
      
      function z = ghivprod_local(self, x, g, v)
         z = self.nlp.ghivprod(x, g, v);
      end
      
   end % methods
      
end % classdef
