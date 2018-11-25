classdef intrelabmodel < model.nlpmodel

   properties
      pdeobj;       % Objective function struct
      pdecon;       % Constraint function struct
      
      Aeq;            % Jacobian of linear constraints
      beq;            % rhs of linear constraints
      m_nln;          % Number of nonlinear constraints
      m_lin;          % Number of linear constraints
      n_nln;          % Number variables in original pde
      s;              % Number of slack variables
   end % properties
   
   methods
       function self = intrelabmodel(name, pdeobj, pdecon, x0, varargin)
          
          n = length(x0);
           
          p = inputParser;
          addParameter(p,'bL', -Inf(n,1));
          addParameter(p,'bU',  Inf(n,1));
          addParameter(p,'Aeq', sparse(0,n));
          addParameter(p,'beq', []);
          addParameter(p,'n_nln', n);
          parse(p, varargin{:});

          bL = p.Results.bL;
          bU = p.Results.bU;
          Aeq = p.Results.Aeq;
          beq = p.Results.beq;
          n_nln = p.Results.n_nln;
          
          c0 = pdecon.value(x0);
          m_nln = length(c0);
          
          m_lin = size(Aeq,1);
          
          cL = [zeros(m_nln,1); beq];
          cU = [zeros(m_nln,1); beq]; 
          
          self = self@model.nlpmodel(name, x0, cL, cU, bL, bU);
          
          self.pdeobj = pdeobj;
          self.pdecon = pdecon;
          
          self.beq = beq;
          self.Aeq = Aeq;
          self.m_nln = m_nln;
          self.m_lin = m_lin;
          self.n_nln = n_nln;
          self.s = self.n - n_nln;
          
          self.linear(end-self.s+1:end) = true;
          
          % TODO: Remove when this becomes matrix-free
          J = self.gcon(x0);
          self.Jpattern = spones(J);
       end
       
       function f = fobj_local(self, x)
          f = self.pdeobj.value(x(1:self.n_nln)); 
       end
       
       function g = gobj_local(self, x)
          g = self.pdeobj.gradient(x(1:self.n_nln));
          g = [g; zeros(self.s,1)];
       end
       
       function H = hobj_local(self, x)
          H = self.pdeobj.hessian(x(1:self.n_nln));
          H = [H sparse(self.n_nln,self.s); sparse(self.s, self.n)];
       end
       
       function c = fcon_local(self, x)
          c = self.pdecon.value(x(1:self.n_nln));
          c = [c; self.Aeq*x];
       end
       
       function J = gcon_local(self, x)
          J = self.pdecon.Jacobian(x(1:self.n_nln));
          J = [J sparse(self.m_nln, self.s); self.Aeq];
       end

       function [Jprod, Jtprod] = gconprod_local(self, x)
          J = self.gcon(x);
          Jprod = @(v) J*v;
          Jtprod = @(v) J'*v;       
       end
       
%        function HL = hlag_local(~, ~, ~)
%           % Do nothing
%           HL = [];
%        end
       
       function w = hlagprod_local(self, x, y, v)
          w = self.pdeobj.hessVec(v(1:self.n_nln),x(1:self.n_nln)) - ...
              self.pdecon.applyAdjointHessian(y(1:self.m_nln),v(1:self.n_nln),x(1:self.n_nln));
          w = [w; zeros(self.s,1)];
       end

       function w = hconprod_local(self, x, y, v)
          w = self.pdecon.applyAdjointHessian(y(1:self.m_nln),v(1:self.n_nln),x(1:self.n_nln));
          w = [w; zeros(self.s,1)];
       end

       function w = ghivprod_local(self, x, y, v)
          % Do nothing for now
          w = [];
       end

       function P = gcon_prcnd(self, x)
           nu = self.pdecon.var.nu;
           A = self.pdecon.Jacobian(x(1:self.n_nln));
           A = A(1:self.m_nln,1:nu);
           %G = chol(A*A');
%            if self.n == self.n_nln
%                P = @(v) (G\(G'\v));
%            else
%                P = @(v) ([G\(G'\v(1:self.pdecon.var.nu)); v(self.pdecon.var.nu+1:end)]);
%            end
           G = chol(A);
           if self.n == self.n_nln
               P = @(v) (G\(G'\(G\(G'\v))));
           else
               P = @(v) ([(G\(G'\(G\(G'\v(1:self.pdecon.var.nu))))); v(self.pdecon.var.nu+1:end)]);
           end
       end

       function s = gcon_sval(~, ~)
           s = 1;
       end
       
   end % methods
    
    
end