classdef nlpmodel < handle

   properties
      n       % number of variables
      m       % number of constraints
      cL, cU  % lower and upper bounds on constraints
      bL, bU  % lower and upper bounds on variables
      x0      % initial point point
      name    % problem name

      % Variables; logical arrays
      jFix    % logical array of fixed variables
      jInf    % logical array of infeasible bounds
      jLow    % logical array of lower-bounded variables
      jUpp    % logical array of upper-bounded variables
      jFre    % logical array of free variables
      jTwo    % logical array of upper/lower-bounded variables

      % Constraints; logical arrays
      iFix    % fixed constraints
      iInf    % infeasible constraints
      iLow    % lower-bounded constraints
      iUpp    % upper-bounded constraints
      iFre    % free constraints
      iTwo    % upper/lower-bounded constraints
      linear  % logical array indicating linear constraints

      Jpattern% Jacobian sparsity pattern
      Hpattern% Jacobian sparsity pattern

      % Number of calls counter:
      ncalls_fobj = 0 % objective function
      ncalls_gobj = 0 % objective function
      ncalls_fcon = 0 % constraint function
      ncalls_gcon = 0 % constraint function
      ncalls_hvp  = 0 % Hessian Lagrangian vector-product function
      ncalls_hes  = 0 % Hessian Lagrangian function
      ncalls_ghiv = 0 % gHiv products

      % Time in calls:
      time_fobj = 0 % objective function
      time_gobj = 0 % objective function
      time_fcon = 0 % constraint function
      time_gcon = 0 % constraint function
      time_hvp  = 0 % Hessian Lagrangian vector-product function
      time_hes  = 0 % Hessian Lagrangian function
      time_ghiv = 0 % gHiv products

      obj_scale     % objective scaling
      
      dc            % Variable scaling (columns of Jacobian)
      dr            % Constraint scaling (rows of Jacobian)
                    % dr*J*dc = Jbar where Jbar is well-conditioned
   end % properties

   properties (Hidden=true, Constant)
      BMAX   =   1e32;  % Free upper bound limit
      BMIN   =  -1e32;  % Free lower bound limit
   end
   
   methods (Sealed = true)

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function o = nlpmodel(name, x0, cL, cU, bL, bU)

         % Ensure that the bounds are sensible.
         assert( all(bL <= bU) );
         assert( all(cL <= cU) );

         % Dimensions of the problem.
         o.n = length(bL);
         o.m = length(cL);

         % Catergorize the bounds.
         o.jFix = bU - bL <= eps;
         o.jInf = bL > bU;
         o.jLow = bL >  o.BMIN;
         o.jUpp = bU <  o.BMAX;
         o.jFre = ~o.jLow  &  ~o.jUpp;
         o.jTwo =  o.jLow  &   o.jUpp;

         % Categorize the linear/nonlinear constraints.
         o.iFix = cU - cL <= eps;
         o.iInf =  cL >  cU;
         o.iLow =  cL >  o.BMIN;
         o.iUpp =  cU <  o.BMAX;
         o.iFre =  ~o.iLow &  ~o.iUpp;
         o.iTwo =   o.iLow &   o.iUpp;

         % Store other things.
         o.name = name;
         o.x0 = x0;
         o.cL = cL;
         o.cU = cU;
         o.bL = bL;
         o.bU = bU;

         % By default, all constraints are categorized as nonlinear. The
         % subclass should override this if it's known which constraints
         % are linear.
         o.linear = false(o.m, 1);

         % No scaling by default.
         o.obj_scale = 1.0;
         o.dc = ones(o.n,1);
         o.dr = ones(o.m,1);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function set_scaling(self, dc, dr)
         % Apply new rhs scaling while undoing previous
         self.cL = (dr.*self.cL)./self.dr;
         self.cU = (dr.*self.cU)./self.dr;
         self.bL = (self.bL./dc).*self.dc;
         self.bU = (self.bU./dc).*self.dc;
         
         self.dc = dc;
         self.dr = dr;
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function f = fobj(self, x)
         self.ncalls_fobj = self.ncalls_fobj + 1;
         t = tic;
         f = self.fobj_local(self.dc.*x) * self.obj_scale;
         self.time_fobj = self.time_fobj + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function g = gobj(self, x)
         self.ncalls_gobj = self.ncalls_gobj + 1;
         t = tic;
         g = self.dc.*self.gobj_local(self.dc.*x) * self.obj_scale;
         self.time_gobj = self.time_gobj + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function H = hobj(self, x)
         self.ncalls_hes = self.ncalls_hes + 1;
         t = tic;
         dC = spdiags(self.dc,0,self.n,self.n);
         H = dC*self.hobj_local(self.dc.*x) * dC * self.obj_scale;
         self.time_hes = self.time_hes + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function c = fcon(self, x)
         self.ncalls_fcon = self.ncalls_fcon + 1;
         t = tic;
         c = self.dr.*self.fcon_local(self.dc.*x);
         self.time_fcon = self.time_fcon + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function J = gcon(self, x)
         self.ncalls_gcon = self.ncalls_gcon + 1;
         t = tic;
         dR = spdiags(self.dr,0,self.m,self.m);
         dC = spdiags(self.dc,0,self.n,self.n);
         J = dR*self.gcon_local(self.dc.*x)*dC;
         self.time_gcon = self.time_gcon + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function Hc = hcon(self, x, y)
         self.ncalls_hes = self.ncalls_hes + 1;
         t = tic;
         dC = spdiags(self.dc,0,self.n,self.n);
         Hc = dC*self.hcon_local(self.dc.*x, self.dr.*y)*dC;
         self.time_hes = self.time_hes + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function H = hlag(self, x, y)
         self.ncalls_hes = self.ncalls_hes + 1;
         t = tic;
         dC = spdiags(self.dc,0,self.n,self.n);
         H = dC*self.hlag_local(self.dc.*x, self.dr.*y)*dC;
         self.time_hes = self.time_hes + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function w = hlagprod(self, x, y, v)
         self.ncalls_hvp = self.ncalls_hvp + 1;
         t = tic;
         dC = spdiags(self.dc,0,self.n,self.n);
         w = dC*self.hlagprod_local(self.dc.*x, self.dr.*y, self.dc.*v);
         self.time_hvp = self.time_hvp + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function w = hconprod(self, x, y, v)
         self.ncalls_hvp = self.ncalls_hvp + 1;
         t = tic;
         dC = spdiags(self.dc,0,self.n,self.n);
         w = dC*self.hconprod_local(self.dc.*x, self.dr.*y, self.dc.*v);
         self.time_hvp = self.time_hvp + toc(t);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      function w = ghivprod(self, x, y, v)
         self.ncalls_ghiv = self.ncalls_ghiv + 1;
         t = tic;
         w = self.dr.*self.ghivprod_local(self.dc.*x, self.dc.*y, self.dc.*v);
         self.time_ghiv = self.time_ghiv + toc(t);
      end
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function c = fcon_lin(self, x)
         %FCON_LIN  Constraint functions, linear only.
         c = self.dr(self.linear).*self.fcon_select(self.dc.*x, self.linear);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function J = gcon_lin(self, x)
         %GCON_LIN  Constraint Jacobian, linear only.
         dR = spdiags(self.dr,0,self.m,self.m);
         dC = spdiags(self.dc,0,self.n,self.n);
         J = dR(self.linear,self.linear)*self.gcon_select(self.dc.*x, self.linear)*dC;
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function c = fcon_nln(self, x)
         %FCON_NLN  Constraint functions, non-linear only.
         dR = spdiags(self.dr,0,self.m,self.m);
         c = dR(~self.linear,~self.linear)*self.fcon_select(self.dc.*x, ~self.linear);
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function J = gcon_nln(self, x)
         %GCON_NLN  Constraint Jacobian, non-linear only.
         dR = spdiags(self.dr,0,self.m,self.m);
         dC = spdiags(self.dc,0,self.n,self.n);
         J = dR(~self.linear,~self.linear)*self.gcon_select(self.dc.*x, ~self.linear)*dC;
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function scale = scale_obj(self, x)
        if nargin == 1
          x = self.x0;
        end
        g_max = 1.0e+2;
        f = self.fobj(x);  % Ensure fobj has been called at the same x
        g = self.gobj(x);
        gNorm = norm(g, inf);
        scale = g_max / max(g_max, gNorm);  % <= 1 always
        self.obj_scale = scale;
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function [nln_feas, lin_feas, bnd_feas] = prResidual(self, x, c, scaled)
          
         bl = self.bL;
         bu = self.bU;
         cl = self.cL;
         cu = self.cU;
         
         if nargin < 4 || scaled == true
         else
            x = x ./ self.dc;
            c = c ./ self.dr;
            bl = bl .* self.dc;
            bu = bu .* self.dc;
            cl = cl ./ self.dr;
            cu = cu ./ self.dr;
         end

         % Constraint residuals.
         rcL = min(c - cl, 0);
         rcU = min(cu - c, 0);

         % Nonlinear infeasibility.
         nln_rcL = rcL(~self.linear);
         nln_rcU = rcU(~self.linear);
         nln_feas = max( norm(nln_rcL, inf), norm(nln_rcU, inf ));

         % Linear infeasibility.
         lin_rcL = rcL(self.linear);
         lin_rcU = rcU(self.linear);
         lin_feas = max( norm(lin_rcL, inf), norm(lin_rcU, inf ));

         % Bounds infeasibility.
         bnd_feas_low = norm(max(bl - x, 0), inf);
         bnd_feas_upp = norm(max(x - bu, 0), inf);
         bnd_feas = max(bnd_feas_low, bnd_feas_upp);

         % Bundle into an aggregate if only one output.
         if nargout == 1
            nln_feas = max( [nln_feas, lin_feas, bnd_feas] );
         end

      end % function prResidual

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function rNorm = duResidual(self, x, c, g, J, y, zL, zU, scaled)
         r = g - J'*y;
         if nargin < 7 || (isempty(zL) && isempty(zU))
             zL = zeros(self.n,1);
             zU = zeros(self.n,1);
             zL(r>0) =  r(r>0);
             zU(r<0) = -r(r<0);
             zL(self.jFre) = 0;
             zU(self.jFre) = 0;
         end

         if nargin < 8 || scaled == true
         else
            x  = x ./  self.dc;
            c  = c ./ self.dr;
            r  = r ./ self.dc;
            y  = y .* self.dr;
            zL = zL ./ self.dc;
            zU = zU ./ self.dc;
         end
         
         rD1 = norm(r - zL + zU, inf ) / max([1, norm(zL), norm(zU)]);

         jj = ~self.jFix & self.jLow;
         rC1 = norm( min(1,zL(jj)) .* (x(jj) - self.bL(jj)), inf );

         jj = ~self.jTwo & self.jUpp;
         rL  = norm( zL(jj), inf );
         
         jj = ~self.jFix & self.jUpp;
         rC2 = norm( min(1,zU(jj)) .* (self.bU(jj) - x(jj)), inf );

         jj = ~self.jTwo & self.jLow;
         rU  = norm( zU(jj), inf );
         
         ii = ~self.iFix & self.iLow;
         yp  = min(1, +max(y, 0));
         rC3 = norm( yp(ii) .* (c(ii) - self.cL(ii)), inf );

         ii = ~self.iFix & self.iUpp;
         ym  = min(1, -min(y, 0));
         rC4 = norm( ym(ii) .* (self.cU(ii) - c(ii)), inf);

         rNorm = max( [rD1, rC1, rC2, rC3, rC4, rL, rU ] );

      end % function duResidual

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function display(self)
         %DISPLAY Display details about the problem.
         fprintf(self.formatting());
      end % function display

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function s = formatting(o)
         s = [];
         s = [s  sprintf('     Problem name: %-22s\n',o.name)];
         s = [s  sprintf('  Total variables: %5i'    ,o.n)];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('Total constraints: %5i\n'  ,o.m)];
         s = [s  sprintf('             free: %5i'    ,sum(o.jFre))];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('           linear: %5i\n'  ,sum(o.linear))];
         s = [s  sprintf('            lower: %5i'    ,sum(o.jLow & ~o.jTwo))];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('        nonlinear: %5i\n'  ,o.m - sum(o.linear))];
         s = [s  sprintf('            upper: %5i'    ,sum(o.jUpp & ~o.jTwo))];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('         equality: %5i\n'  ,sum(o.iFix))];
         s = [s  sprintf('          low/upp: %5i'    ,sum(o.jTwo))];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('            lower: %5i\n'  ,sum(o.iLow & ~o.iTwo))];
         s = [s  sprintf('            fixed: %5i'    ,sum(o.jFix))];
         s = [s  sprintf('%10s','')];
         s = [s  sprintf('            upper: %5i\n'    ,sum(o.iUpp & ~o.iTwo))];
         s = [s  sprintf('%44slow/upp: %5i','',sum(o.iTwo & ~o.iFix))];
         s = [s  sprintf('\n')];
      end

   end % methods

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   methods

      function [f, g, H] = obj(self, x)
         f = self.fobj(x);
         if nargout > 1
          g = self.gobj(x);
        end
        if nargout > 2
          H = self.hobj(x);
        end
      end

   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   methods (Access = private, Hidden = true)

      function c = fcon_select(self, x, select)
         %FCON_SELECT
         c = self.fcon(x);
         c = c(select);
      end
      function J = gcon_select(self, x, select)
         %GCON_SELECT
         J = self.gcon(x);
         J = J(select,:);
      end

   end

end % classdef
