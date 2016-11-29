classdef CFlashModel < model.ProjRecModel
    %% CFlashModel - Sovling reconstruction problem Cflash algorithm
    %   This class was developped to solve the problem represented by
    %   RecModel using the Cflash solver under NLPlab's framework.
    %
    %   This model implements the projection functions required by the
    %   Cflash algorithm (generalization of the TRON algorithm to linear
    %   inequalities). The following methods have to be implemented:
    %
    %           * project
    %           * eqProject
    %
    %   The project method is carried out by ProjRecModel, whereas this
    %   model implements the method eqProject, that is specific to the
    %   Cflash solver.
    %
    %   A public attribute named 
    %           * normJac 
    %   must also be provided. It must represent the norm of the Jacobian 
    %   of the constraint set. It is used in the evaluation of relative 
    %   tolerances.
    %
    %   This model is only compatible with the structure of the tomographic
    %   reconstruction algorithm made by Poly-LION. The terms in the
    %   objective function should be provided by a Critere object, whereas
    %   the preconditionner should be a Precond object. This model also
    %   expects the sinogram to be a Sinogramme object and the geos a
    %   GeometrieSeries object.
    %
    %   * For additional information about the reconstruction problem, see
    %   RecModel.
    %   * For additional information about the projection sub-problem, see
    %   ProjModel.
    %   ---
    
    properties (SetAccess = private, Hidden = false)
        % -- eqProject parameters --
        C;
        CCt;
        eqProjMethod;
    end
    
    properties (SetAccess = public)
        % Norm of the jacobian (in our case preconditionner)
        normJac; % Used in relative tolerances
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods (Access = public)
        
        function self = CFlashModel(crit, prec, sino, geos, projModel, ...
                projSolver, projOptions, varargin)
            %% Constructor
            % Inputs:
            %   - crit: Critere object from the Poly-LION repository
            %   - prec: Precond object from the Poly-Lion repository
            %   - sino: Sinogramme object from the Poly-Lion repository
            %   - geos: Geometrie_Series object from the Poly-Lion repo
            %   - projModel: model representing the dual of the projection
            %   problem. It is used in the project function. NOTE: 
            %   generally speaking, it is way easier to solve the dual of 
            %   the projection problem than the primal.
            %   - projSolver: solver that can receive and solve projModel.
            %   - options: struct containing projSolver's parameters.
            %   - mu0: (optional) initial vector for the reconstruction
            %   problem
            %   - name: (optional) name of the nlp model
            %   - eqMethod (optional): string containing the name of the
            %   method that will be used to solve the sub-problem of
            %   projecting on the equality constraint B*C*x = 0
            
            % Gathering optinal arguments and setting default values
            p = inputParser;
            p.addParameter('eqMethod', 'pcg');
            p.addParameter('mu0', []);
            p.addParameter('name', '');
            
            % Parsing input arguments
            p.parse(varargin{:});
            
            % Setting values
            eqProjMeth = p.Results.eqMethod;
            mu0 = p.Results.mu0;
            name = p.Results.name;
            
            % Calling the RecModel superclass (inputs will be checked here)
            self = self@model.ProjRecModel(crit, prec, sino, geos, ...
                projModel, projSolver, projOptions, mu0, name);
            
            % Building Spot operators for the eqProject function.
            % Using wrappers to call object function.
            cWrap = @(x, mode) self.precMult(x, mode);
            cctWrap = @(X, mode) self.prec.AdjointDirect(X);
            
            self.C = opFunction(self.objSize, self.objSize, cWrap);
            self.CCt = opFunction(self.objSize, self.objSize, cctWrap);
            
            % Getting the norm of the preconditionner, helps to evaluate
            % "relative" decreases/zeros.
            self.normJac = self.prec.norm();
            
            self.eqProjMethod = eqProjMeth;
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %                   --- Cflash methods ---
        % The following methods are required in order for the Cflash
        % algorithm to be functional
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function wProj = eqProject(self, wbar, ind)
            %% EqProject - project vector wbar on equality constraints
            % Solves the problem
            % min   1/2 || w - \bar{w}||^2
            %   w   sc (C'*w)_i = 0, if (C*x)_i = l_i
            % where C is the jacobian of the linear constraints and i
            % denotes the indices of the fixed variables. This problem has
            % an analytical solution.
            %
            % From the first order KKT conditions, we can obtain the
            % following set of equations:
            %
            %   w               = wbar + B*C*z
            %   (B*C*C'*B') * z = -B*C*wbar
            %
            % where z is the lagrange mutliplier associated with the
            % equality constraint.
            %
            % Inputs:
            %   - wbar: vector to project on the equality constraint set
            %   - ind: indices of the active constraints (C*x)_i = 0
            % Ouput:
            %   - v: projected direction
            %
            % * EqProject is a special case since it doesn't use the
            % bcflash solver. A new projection model doesn't have to be
            % created. It is located inside the CFlashProjModel class
            % because useful methods and attributes are already defined.
            
            % If B is the mask matrix such that
            % B := {b_i' = ith col of I for all i \in ~indfree}
            switch self.eqProjMethod
                case 'pcg'
                    % Building reduced operators from object's attributes
                    subC = self.C(ind, :); % B * C
                    subCCt = self.CCt(ind, ind); % (B * C * C' * B')^-1
                    % Using CG to solve (B*C*C'*B') z = -B*C*wbar
                    [z, ~] = pcg(subCCt, -subC*wbar, 1e-12, 5e2);
                    % For the unconstrained case, the solution is trivial
                    wProj = wbar + (subC' * z);
                case 'lsqr'
                    % Building reduced operators from object's attributes
                    subC = self.C(ind, :); % B * C
                    lsqrOpts.atol = 1e-12;
                    lsqrOpts.btol = 1e-15;
                    [deltaW, flags, ~] = lsqr_spot(subC, subC*(-wbar), ...
                        lsqrOpts);
                    if ~flags.solved
                        error('LSQR didn''t converge');
                    end
                    wProj = deltaW + wbar;
                case 'lsmr'
                    % Building reduced operators from object's attributes
                    subC = self.C(ind, :); % B * C
                    lsmrOpts.atol = 1e-12;
                    lsmrOpts.btol = 1e-15;
                    [deltaW, flags, ~] = lsmr_spot(subC, subC*(-wbar), ...
                        lsmrOpts);
                    if ~flags.solved
                        error('LSMR didn''t converge');
                    end
                    wProj = deltaW + wbar;
                case 'facto'
                    nAct = sum(ind);
                    nObj = self.objSize;
                    if nAct == 0
                        % B is empty, the solution is trivial
                        wProj = wbar;
                    elseif nAct == nObj % This will never happen!
                        % B is the identity, B*C*C'*B' = C*C'
                        z = self.prec.DirAdjInv((self.C * (-wbar)));
                        wProj = wbar + (self.C' * z);
                    elseif nAct <= 0.5 * nObj
                        % # active constraints below or equal to 50%
                        subC = self.C(ind, :); % B * C
                        
                        %                 --- NOTES ---
                        % THE CODE BELOW won't form an explicit matrix
                        % subCCt = self.CCt(ind, ind);
                        % We must force B as a full matrix, fft (in C) 
                        % doesn't support sparse matrices...
                        % THE CODE BELOW USES LSQR...
                        % z = opInverse(subCCt) * (subC * (-wbar));
                        
                        B = sparse(1:nAct, find(ind), ones(1, nAct), ...
                            nAct, nObj);
                        
                        z = (B * self.prec.reducedDirAdj(ind)) ...
                            \ (subC * (-wbar));
                        
                        wProj = wbar + (subC' * z);
                    else % # active cstr above 50%
                        error('Not implemented yet');
                    end
%                 case 'test'
%                     subC = self.C(ind, :); % B * C
%                     wProj = wbar - subC' * subC * wbar;
                otherwise
                    error('Unrecognized solver for eqProject');
            end
        end
    end
    
    methods (Access = private)
        function z = precMult(self, z, mode)
            %% Evaluates C * z
            if mode == 1
                z = self.prec.Direct(z);
            elseif mode == 2
                z = self.prec.Adjoint(z);
            end
        end
        
    end
end
