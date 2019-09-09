function [ eparam, logL, fit_fcst, optimoutput] = ...
    caw_dpc_estim_3step( ...
        data_RC, p_l, q_l, p_c, q_c, fcst_steps, enf_pd ...
    )
%CAW Estimate the DPC-CAW via 3step procedure.
%
% USAGE:
%   [ eparam, logL, fit_fcst, optimoutput] = ...
%       caw_dpc_estim_3step( data_RC, p_l, q_l, p_c, q_c, x0 )
%
% INPUTS:
%
% OUTPUTS:
%
% COMMENTS:
% 
% REFERENCES:
%
% DEPENDENCIES:
%
% See also
%
% Michael Stollenwerk
% michael.stollenwerk@live.com

[N,~,T] = size(data_RC);
K = N*(N+1)/2;
%% Input Checking
if isempty(enf_pd)
    enf_pd = 1;
end
%
eparam = struct();
%% Step 1 and 2 
% Estimate a Scalar CAW(p_l,q_l) model (Intercept via targting). Store
% parameters in struct "eparam", fields are "loadings.intrcpt", 
% "loadings.chol_intrcpt", "loadings.arch_param", "loadings.garch_param"
% "loadings.intrcpt" is (1-a-b)*mean(rc,3), NOT mean(rc,3); 
% "loadings.chol_intrcpt is sqrt(1-a-b)*chol(mean(rc,3),'lower') accordingly. 
% Code is in different file. 
[ eparam.loadings, ~, ~, fit_fcst_l, optimoutput.loading_recursion] = ...
    caw_scalar_estim_targeting( ...
        data_RC, p_l, q_l, 0, [], enf_pd, ...
        'Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15 ...
    );

%% Step 3
L = NaN(N,N,T);
for tt = 1:T
    % take estimated Sigma's from Scalar CAW of Step 1 and 2 (in DPC-CAW 
    % they are called Q) and create time-series of loading matrices "L".
    % Each L is created s.th. the first eigenvector corresponds to the
    % biggest eigenvalue, the second to the second biggest, and so on...
    [L(:,:,tt), ~] = sorteig(fit_fcst_l.Sigma_(:,:,tt)); 
end

diagLRL = NaN(T,N);
for tt = 1:T
    % take time series of loading matrices and create data input to
    % component recursions (in DPC-CAW row i of diagLRL is called g_i).
    diagLRL(tt,:) = diag(L(:,:,tt)'*data_RC(:,:,tt)*L(:,:,tt));
end

eparam.components.all = NaN(N, 1 + p_c + q_c); % Create field "components" in struct "eparam".
[~, gammas] = sorteig(mean(data_RC,3)); % Intercepts of eigenvalue (i.e. component) recursions. (In DPC-CAW they are called gamma_i)
d = NaN(T,N); % Storage for eigenvalue recursion.
fcst_d = NaN(fcst_steps,N);
x0 = [.05*ones(1,p_c)/p_c, .9*ones(1,q_c)/q_c];
% Restrictions-------------------------------------------------------------
A = ones(1,p_c + q_c); % This is to ensure pd unconditional mean (=(1-arch-garch)*Intrcpt). sufficient and necessary. Sigma is enforced to be pd in sample, since otherwise the likeRec returns inf.
b = 1;
lb = [];
if enf_pd
    lb = zeros(1,p_c + q_c); % This is to ensure Sigma has 0 probability to be < 0, sufficient but not necessary.
end
% Optimoptions-------------------------------------------------------------
options = optimoptions( ...
    'fmincon', 'Display', 'off', 'MaxFunEvals', 1e15, 'MaxIter', 1e15 ...
);

optimoutput.components = cell(N,1);
for ii = 1:N
    % Do unconstraint optimization over component recursion arch and garch
    % parameters. The optimized function is components_qlikeRec (see below
    % in Helper functions). Store estimated arch/garch parameters in
    % positions (2:end) of struct eparam, field "components.all".
    % Objective Function-------------------------------------------------------
    fun_comp = @(param) components_qlikeRec(...
                    param, gammas(ii), p_c, q_c, diagLRL(:,ii), enf_pd...
    );
    % Optimization-------------------------------------------------------------
    warning('off') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    [eparam.components.all(ii,2:end),~,exitflag,...
     optimoutput.components{ii},grad,hessian] = ...
        fmincon( fun_comp, x0, A, b, [], [], lb, [], [], options );
    optimoutput.components{ii}.estimation_time = toc;
    optimoutput.components{ii}.exitflag = exitflag;
    optimoutput.components{ii}.grad = grad;
    optimoutput.components{ii}.hessian = hessian;
    warning('on') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Get eigenvalue recursion d-------------------------------------------
    [~,~,d(:,ii),fcst_d(:,ii)] = components_qlikeRec(...
        eparam.components.all(ii,2:end),...
        gammas(ii), p_c, q_c, diagLRL(:,ii), enf_pd, fcst_steps...
     );
end

% Store estimated parameters in struct "eparam"
eparam.components.arch_param = eparam.components.all(:,2:1+p_c);
eparam.components.garch_param = eparam.components.all(:,2+p_c:1+p_c+q_c);
% Stored component intercepts are = (1-alphas-betas)*gammas. NOT gammas
% themselves.
eparam.components.all(:,1) = ...
    (1-sum(eparam.components.all(:,2:end),2)).*gammas; 
eparam.components.intrcpt = eparam.components.all(:,1);

% Create Sigma_ time series of DPC-CAW.
Sigma_ = NaN(N,N,T);
for tt = 1:T
    Sigma_(:,:,tt) = L(:,:,tt)*diag(d(tt,:))*L(:,:,tt)';
    Sigma_(:,:,tt) = (Sigma_(:,:,tt)' + Sigma_(:,:,tt))/2;
end

% Create forecasts. The d recursion can be easily extended. The loadings
% recursion has to take Sigma (Expectation of R) as inputs and is thus a
% little more complicated to extend (see below).
% ACTUALLY THIS IS WRONG: I COULD JUST EXTEND THE Q_t RECURSION, THIS 
% WOULD YIELD THE SAME FORECASTS.
fcst_Sigma = NaN(N,N,fcst_steps);
fcst_L = NaN(N,N,fcst_steps);
Q = fit_fcst_l.Sigma_;
for ii = 1:fcst_steps
    Q(:,:,T+ii) = eparam.loadings.intrcpt;
    for jj = 1:p_l
        if (T+ii-jj) <= T
            Q(:,:,T+ii) = Q(:,:,T+ii) + eparam.loadings.arch_param(jj)*data_RC(:,:,T+ii-jj);
        else
            Q(:,:,T+ii) = Q(:,:,T+ii) + eparam.loadings.arch_param(jj)*fcst_Sigma(:,:,ii-jj);
        end
    end
    for jj = 1:q_l
        Q(:,:,T+ii) = Q(:,:,T+ii) + eparam.loadings.garch_param(jj)*Q(:,:,T+ii-jj);
    end
    [fcst_L(:,:,ii), ~] = sorteig(Q(:,:,T+ii)); 
    fcst_Sigma(:,:,ii) = fcst_L(:,:,ii)*diag(fcst_d(ii,:))*fcst_L(:,:,ii)';
    fcst_Sigma(:,:,ii) = (fcst_Sigma(:,:,ii)' + fcst_Sigma(:,:,ii))/2;
end 

% Estimate wishart degrees of freedom and store in struct "eparam"
% wishlike see below.
eparam.df = fmincon( ...
    @(df) wishlike( Sigma_/df, df, data_RC ), 2*N, -1, -(N-1),...
    [], [], [], [], [], optimoptions('fmincon', 'Display', 'off') ...
);

% create field "all" to store all defining parameters (for quick access).
eparam.all = [eparam.loadings.chol_intrcpt,...
              eparam.loadings.arch_param,...
              eparam.loadings.garch_param,...
              eparam.components.arch_param(:)',...
              eparam.components.garch_param(:)',...
              eparam.df];

% get full log-likelihood values.
[nLogL, logLcontr] = wishlike( Sigma_/eparam.df, eparam.df, data_RC);

% get AIC, BIC and store it together with logL and logLcontr in struct
% "logL".
aic = 2*nLogL + 2*(K + p_l + q_l + N*(p_c + q_c) + 1);
bic = 2*nLogL + log(K*T)*(K + p_l + q_l + N*(p_c + q_c) + 1);
logL = struct(...
    'logL', -nLogL,...
    'aic', aic,...
    'bic', bic,...
    'logLcontr', logLcontr...
);

% store Sigma_, L, d, diagLRL and Q in struct "fit_fcst".
fit_fcst = struct(...
    'Sigma_', Sigma_,...
    'Loadings', L,...
    'd', d,...
    'diagLRL', diagLRL,...
    'Q', fit_fcst_l.Sigma_,...
    'fcst_Sigma', fcst_Sigma,...
    'fcst_Loadings', fcst_L,...
    'fcst_d', fcst_d,...
    'fcst_Q', fit_fcst_l.fcst_Sigma...
);

end

%% Helper Functions
function [nLogL, logLcontr, d, fcst_d] = ...
    components_qlikeRec(param, intrcpt, p_c, q_c, g, enf_pd, varargin)

if isempty(varargin)
    fcst_steps = 0;
else
    fcst_steps = varargin{1};
end
T = size(g,1);

arch_param = param(1:p_c);
garch_param = param(1+p_c : p_c+q_c);
ini_rec = mean(g); %Initialization of the recursion

d = NaN(T+fcst_steps,1);
logLcontr = NaN(T,1);
for tt = 1:T+fcst_steps
    d(tt) = (1-sum(arch_param)-sum(garch_param))*intrcpt;
    for jj = 1:p_c
        if (tt-jj) <= 0 % if true, use initialization
            d(tt) = d(tt) + arch_param(jj)*ini_rec;
        elseif (tt-jj) <= T
            d(tt) = d(tt) + arch_param(jj)*g(tt-jj);
        else
            d(tt) = d(tt) + arch_param(jj)*d(tt-jj);
        end
    end
    for jj = 1:q_c
        if (tt-jj) <= 0
            d(tt) = d(tt) + garch_param(jj)*ini_rec;
        else
            d(tt) = d(tt) + garch_param(jj)*d(tt-jj);
        end
    end
    if ~enf_pd % To avoid unnecessary computation if pd is ensured by optimizaitinon constraints anyways
        if d(tt) < 0 % To allow for unconstraint optimization
            nLogL = inf;
            return
        end
    end    
    if tt <= T
        logLcontr(tt) = -log(d(tt)) -g(tt)/d(tt); %quasi-log-likelihood
    end 
end
fcst_d = d(T+1:end);
d = d(1:T);

nLogL = -sum(logLcontr);

end



function [ L, d ] = sorteig( X, L_ )
%SORTEIG sorts eigenvalues big2small (eigenvectors accordingly). Diagonal
%elements of eigenvectors matrix are restricted to be positive (1 input) or
%are restricted s.th. diagonal elements of L_*X are positive (2 inputs).

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 29.08.2016

[L, d]=eig(X,'vector');
[d,perm]=sort(d,'descend');
L=L(:,perm);
if nargin==1
    L = bsxfun(@times,L,sign(diag(L)'));
elseif nargin==2
    L = bsxfun(@times,L,sign(diag(L_*L)'));
end

end



function [ nLogL, logLcontr ] = wishlike( sigma, df, data )
%WISHLIKE Negative log-likelihood for the Wishart distribution.
%
% USAGE:
%   NLOGL = wishlike(sigma,df,data)
%
% INPUTS:
%   SIGMA        - K by K by N array of scale matrices of the Wishart
%                  distribution
%   DF           - Degree of freedom of the Wishart distribution
%   DATA         - K by K by N array of covariance matrix realizations
%
% OUTPUTS:
%   NLOGL        - Negative log-likelihood value
%   LOGLCONTR    - Log-likelihood contributions
%
% COMMENTS:
%
%  See also 
%
% REFERENCES:

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 11.02.2017
%% Error checking
narginchk(3,3);

[k,k2,n] = size(sigma);
if k~=k2
    error('Covariance matrix parameters of t-Wishart distribution must be square, symmetric, and positive semi-definite.')
end

[k,k2,~] = size(data);
if k~=k2
    error('Covariance matrix data of t-Wishart distribution must be square, symmetric, and positive semi-definite.')
end

if size(sigma)~=size(data)
    error('Covariance matrix data and parameters of t-Wishart distribution must have the same dimension size')
end
%% Log-likelihood computation

logLcontr=NaN(n,1);

if isnumeric(sigma) && isnumeric(data) % Double input
    for i = 1:n 
        logLcontr(i) = -df/2*log(det(sigma(:,:,i))) - ...
                       1/2*trace(sigma(:,:,i)\data(:,:,i)) + ...
                       (df-k-1)/2*log(det(data(:,:,i)));
    end
elseif iscell(sigma) && iscell(data) % Cell input
    for i = 1:n
        logLcontr(i) = -df/2*log(det(sigma{i})) - ...
                       1/2*trace(sigma{i}\data{i}) + ...
                       (df-k-1)/2*log(det(data{i}));
    end
end

logLcontr = logLcontr - df*k/2*log(2) - mvgammaln(df/2,k);

nLogL=-sum(logLcontr);
end


function y = mvgammaln(x,d)
%MVGAMMALC computes the natural logarithm of the multivariate gamma function 
%
% USAGE:
%  Y = mvgammaln(x,d)
%
% INPUTS:
%   X            - x-value
%   D            - "number of sums" parameter
%
% OUTPUTS:
%   Y            - y-value
%
% COMMENTS:
%   Used in the probability density function of the Wishart and inverse 
%   Wishart distributions.
%   Gamma_d(x) = pi^(d(d-1)/4) \prod_(j=1)^d Gamma(x+(1-j)/2)
%   log(Gamma_d(x)) = d(d-1)/4 log(pi) + \sum_(j=1)^d log(Gamma(x+(1-j)/2))
%
% REFERENCES:
%      [1] James (1964) - Distributions of Matrix Variates and Latent 
%      Roots Derived from Normal Samples. 

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 12.02.2017

y = d*(d-1)/4*log(pi)+sum(gammaln(x+(1-(1:d))/2));

end
