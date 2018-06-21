classdef TopologyModel < model.NlpModel

   properties
      nelx  %
      nely  %
      nx    % Number of x variables
      nu    % Number of u variables
      p     % penalty exponent
      KE    % stiffness matrix
      
      fixeddofs %
      freedofs  %
   end % properties
   
   methods
       function self = TopologyModel(name, x0, nelx, nely, volfrac, penal)
           
          nx = nelx*nely;
          nu = 2*(nelx+1)*(nely+1);

          m = nu + 1;
          
          fixeddofs   = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);
          alldofs     = [1:2*(nely+1)*(nelx+1)];
          freedofs    = setdiff(alldofs,fixeddofs);
          
          % RHS for PDE constraint
          F = zeros(2*(nely+1)*(nelx+1),1);
          F(2) = -1;
          
          F = F(freedofs);
          
          bL = [1e-3*ones(nx,1); -Inf(nu,1); 0];
          bU = [ones(nx,1); Inf(nu,1); Inf];
          cL = [F; volfrac*nx];
          cU = [F; volfrac*nx];
          
          bL(nx + fixeddofs) = 0;
          bU(nx + fixeddofs) = 0;
          
          self = self@model.NlpModel(name, x0, cL, cU, bL, bU);
          
          self.fixeddofs = fixeddofs;
          self.freedofs = freedofs;
          
          self.linear(end) = true;
          
          self.nelx = nelx;
          self.nely = nely;
          self.nx = nx;
          self.nu = nu;
          self.p = penal;
          self.KE = self.stiffness_matrix();
          
          % TODO: Remove when this becomes matrix-free
          J = self.gcon(x0);
          self.Jpattern = spones(J);
       end
       
       function f = fobj_local(self, x)
          f = 0;
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));
          
          p = self.p;
          for ely = 1:self.nely
            for elx = 1:self.nelx
              n1 = (self.nely+1)*(elx-1)+ely; 
              n2 = (self.nely+1)* elx   +ely;
              edof = [2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2];
              Ue = uu(edof);
              f = f + (xx(ely,elx)^p)*Ue'*self.KE*Ue;
            end
          end
       end
       
       function g = gobj_local(self, x)
          g = zeros(self.n,1);
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));
         
          p = self.p;
          for ely = 1:self.nely
            for elx = 1:self.nelx
              n1 = (self.nely+1)*(elx-1)+ely; 
              n2 = (self.nely+1)* elx   +ely;
              edof = [2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2];
              Ue = uu(edof);
              ix = (elx-1)*self.nely + ely;
              g(ix) = p*xx(ely,elx)^(p-1)*Ue'*self.KE*Ue;
              g(self.nx+edof) = g(self.nx+edof) + 2*xx(ely,elx)^(p)*self.KE*Ue;
            end
          end
       end
       
       function H = hobj_local(self, x)
          H = sparse(self.n,self.n);
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));
          
          p = self.p;
          for ely = 1:self.nely
            for elx = 1:self.nelx
              n1 = (self.nely+1)*(elx-1)+ely; 
              n2 = (self.nely+1)* elx   +ely;
              edof = [2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2];
              Ue = uu(edof);
              ix = (elx-1)*self.nely + ely;
              H(ix,ix) = p*(p-1)*xx(ely,elx)^(p-2)*Ue'*self.KE*Ue;
              H(self.nx + edof, self.nx + edof) = H(self.nx + edof, self.nx + edof) + 2*xx(ely,elx)^(p)*self.KE;
              H(self.nx + edof, ix) = H(self.nx + edof, ix) + 2*p*xx(ely,elx)^(p-1)*self.KE*Ue;
            end
          end
          
          H(1:self.nx, (self.nx+1):(self.nx+self.nu)) = H((self.nx+1):(self.nx+self.nu), 1:self.nx)';
       end
       
       function c = fcon_local(self, x)
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));
          K = self.k_matrix(xx);
          K = K(self.freedofs, :);
          c = [K*uu; sum(sum(xx)) + x(end)];
       end
       
       function J = gcon_local(self, x)
          Jx = sparse(self.nu,self.nx);
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));
          
          for elx = 1:self.nelx
              for ely = 1:self.nely
                n1 = (self.nely+1)*(elx-1)+ely; 
                n2 = (self.nely+1)* elx   +ely;
                edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
                ix = (elx-1)*self.nely + ely;
                Jx(edof, ix) = Jx(edof, ix) + self.p*xx(ely, elx)^(self.p-1)*self.KE*uu(edof);
              end
          end
          
          K = self.k_matrix(xx);
          K = K(self.freedofs,:);
          %K(:, self.fixeddofs) = 0;
          Jx = Jx(self.freedofs,:);
          J = [Jx K zeros(self.m-1,1); ones(1,self.nx) sparse(1,self.nu) 1];
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
          w = self.hobj_local(x)*v - self.hconprod_local(x, y, v);
       end

       function HC = hcon_local(self, x, y)
          HC = sparse(self.n, self.n); 
          xx = reshape(x(1:self.nx), self.nely, self.nelx);
          uu = x((self.nx+1):(self.nx+self.nu));   
          yy = zeros(self.nu);
          yy(self.freedofs) = y(1:end-1);
          yy(end) = y(end);
          
          p = self.p;
          for elx = 1:self.nelx
            for ely = 1:self.nely
              n1 = (self.nely+1)*(elx-1)+ely; 
              n2 = (self.nely+1)* elx   +ely;
              edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
              ix = (elx-1)*self.nely + ely;
              
              HC(ix,ix) = p*(p-1)*xx(ely,elx)^(p-2)*(yy(edof)'*self.KE*uu(edof));
              HC(self.nx+edof, ix) = HC(self.nx+edof, ix) + p*xx(ely,elx)^(p-1)*self.KE*yy(edof);
              HC(ix, self.nx+edof) = HC(ix, self.nx+edof) + HC(self.nx+edof,ix)';
            end
          end
       end
       
       function w = hconprod_local(self, x, y, v)
           w = self.hcon(x,y)*v;
       end

       function w = ghivprod_local(self, x, y, v)
          % Do nothing for now
          w = [];
       end

       function P = preconditioner(self, x)
          P = @(x) x;
       end

       function s = gcon_min_singular_value(~, ~)
           s = 0;
       end
       
   end % methods
    
   methods(Access = private)
       function KE = stiffness_matrix(~)
            E = 1.; 
            mu = 0.3;
            k=[ 1/2-mu/6   1/8+mu/8 -1/4-mu/12 -1/8+3*mu/8 ... 
               -1/4+mu/12 -1/8-mu/8  mu/6       1/8-3*mu/8];
            KE = E/(1-mu^2)*[ k(1) k(2) k(3) k(4) k(5) k(6) k(7) k(8)
                              k(2) k(1) k(8) k(7) k(6) k(5) k(4) k(3)
                              k(3) k(8) k(1) k(6) k(7) k(4) k(5) k(2)
                              k(4) k(7) k(6) k(1) k(8) k(3) k(2) k(5)
                              k(5) k(6) k(7) k(8) k(1) k(2) k(3) k(4)
                              k(6) k(5) k(4) k(3) k(2) k(1) k(8) k(7)
                              k(7) k(4) k(5) k(2) k(3) k(8) k(1) k(6)
                              k(8) k(3) k(2) k(5) k(4) k(7) k(6) k(1)];
       end
       
       function K = k_matrix(self, xx)
           K = sparse(2*(self.nelx+1)*(self.nely+1), 2*(self.nelx+1)*(self.nely+1));
           
           p = self.p;
           for elx = 1:self.nelx
             for ely = 1:self.nely
               n1 = (self.nely+1)*(elx-1)+ely; 
               n2 = (self.nely+1)* elx   +ely;
               edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
               K(edof,edof) = K(edof,edof) + xx(ely,elx)^(p)*self.KE;
             end
           end           
       end
   end
end