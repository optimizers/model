function result = check_model(model, x, varargin)
% Checks finite differences of various parts of model for small models

logH = '%10s  %7s    %7s  %7s\n';
logB = '%10s  %1.3e  %1.3e  %4i\n';

p = inputParser;
p.addParameter('y', []);
p.addParameter('check_obj', true);
p.addParameter('check_con', true);
p.addParameter('second_deriv', true);
p.addParameter('atol', 1e-6);
p.addParameter('rtol', 1e-6);
p.addParameter('step', (eps / 3)^(1/3));
p.addParameter('check_all', false);
p.addParameter('k', 10);
p.parse(varargin{:});

y = p.Results.y;
check_obj = p.Results.check_obj;
check_con = p.Results.check_con;
second_deriv = p.Results.second_deriv;
atol = p.Results.atol;
rtol = p.Results.rtol;
e = p.Results.step;
check_all = p.Results.check_all;
k = p.Results.k;

n = model.n;
m = model.m;
h = zeros(n,1);

if isempty(y)
    y = 2*rand(m,1);
end

% Display header
fprintf(logH, 'func', 'eabs', 'erel', 'pass');
fprintf('      ---------------------------------\n');

if check_obj
% Check objective gradient
gobj = true;
eabs = 0;
erel = 0;
g = model.gobj(x);
if check_all
    gfd = zeros(n,1);
    for i=1:n
        h(i) = e;
        gfd(i) = (model.fobj(x+h) - model.fobj(x-h))/(2*e);    
        h(i) = 0;

        err = abs(gfd(i) - g(i));
        erel = max(erel, err/max(abs(gfd(i)),1));
        eabs = max(eabs, err);
        gobj = gobj & (err < atol + rtol*abs(gfd(i)));
    end
else
    for i=1:k
        h = 2*rand(n,1)-1;
        h = h/norm(h);
        
        gfd = (model.fobj(x+e*h) - model.fobj(x-e*h))/(2*e);
        
        err = abs(gfd - g'*h);
        erel = max(erel, err/max(abs(gfd),1));
        eabs = max(eabs, err);
        gobj = gobj & (err < atol + rtol*abs(gfd));
    end
end

fprintf(logB, 'gobj', erel, eabs, gobj);

hobj = true;
if second_deriv
% Check the objective Hessian
eabs = 0;
erel = 0;
h = zeros(n,1);
H = model.hobj(x);
if check_all
    for i=1:n
        h(i) = e;
        Hfd = (model.gobj(x+h) - model.gobj(x-h))/(2*e);
        h(i) = 0;

        err = norm(Hfd - H(:,i),inf);
        erel = max(erel, err/max(norm(Hfd,inf),1));
        eabs = max(eabs, err);
        hobj = hobj & (err < atol + rtol*norm(Hfd,inf));
    end
else
    for i=1:k
        h = 2*rand(n,1)-1;
        h = h/norm(h);
        
        Hfd = (model.gobj(x+e*h) - model.gobj(x-e*h))/(2*e);
        
        err = norm(Hfd - H*h,inf);
        erel = max(erel, err/max(norm(Hfd,inf),1));
        eabs = max(eabs, err);
        hobj = hobj & (err < atol + rtol*norm(Hfd,inf));
    end
end
fprintf(logB, 'hobj', erel, eabs, hobj);
end % if second_deriv

else
    gobj = true;
    hobj = true;
end % if check_obj

if check_con
% Check the constraint gradient
gcon = true;
eabs = 0;
erel = 0;
J = model.gcon(x);
if check_all
    for i=1:n
        h(i) = e;
        Jfd = (model.fcon(x+h) - model.fcon(x-h))/(2*e);
        h(i) = 0;

        err = norm(Jfd - J(:,i),inf);
        erel = max(erel, err/max(norm(Jfd,inf),1));
        eabs = max(eabs, err);   
        gcon = gcon & (err < atol + rtol*norm(Jfd,inf));
    end
else
    for i=1:k
        h = 2*rand(n,1)-1;
        h = h/norm(h);
        
        Jfd = (model.fcon(x+e*h) - model.fcon(x-e*h))/(2*e);

        err = norm(Jfd - J*h,inf);
        erel = max(erel, err/max(norm(Jfd,inf),1));
        eabs = max(eabs, err);
        gcon = gcon & (err < atol + rtol*norm(Jfd,inf));
    end    
end

fprintf(logB, 'gcon', erel, eabs, gcon);

hcon = true;
if second_deriv
% Check constraint Hessians
eabs = 0;
erel = 0;
HC = model.hcon(x,y);
if check_all
    for i=1:n          
        h(i) = e;
        HCfd = (model.gcon(x+h) - model.gcon(x-h))/(2*e);
        h(i) = 0;
        HCfd = HCfd'*y;
        
        err = norm(HCfd - HC(:,i),inf);
        erel = max(erel, err/max(norm(HCfd,inf),1));
        eabs = max(eabs, err);
        hcon = hcon & (err < atol + rtol*norm(HCfd,inf));    
    end
else
    for i=1:k
        h = 2*rand(n,1)-1;
        h = h/norm(h);
        
        HCfd = (model.gcon(x+e*h) - model.gcon(x-e*h))/(2*e);
        HCfd = HCfd'*y;
        
        err = norm(HCfd - HC*h,inf);
        erel = max(erel, err/max(norm(HCfd,inf),1));
        eabs = max(eabs, err);
        hcon = hcon & (err < atol + rtol*norm(HCfd,inf));    
    end
end
fprintf(logB, 'hcon', erel, eabs, hcon);

end % if second_deriv

else
    gcon = true;
    hcon = true;
end % if check_con


result = gobj & hobj & gcon & hcon;
end