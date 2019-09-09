function [ eparam, tstats, logL, fit_fcst, optimoutput] = ...
    caw_scalar_estim_targeting( ...
        data_RC, p, q, fcst_steps, x0, enf_pd, varargin )
%CAW Estimate the scalar Conditional Autoregressive Wishart (CAW) model.
%
% USAGE:
%  [ eparam, tstats, logL, fit_fcst, optimoutput] = ...
%     caw_scalar_estim_targeting( data_RC, p, q, enf_pd, x0 )
%
% INPUTS:
%   DATA_RC    - K by K by T array of covariance estimators.
%   P          - Number of lagged covariance matrix data.
%   Q          - Number of lagged recursion matrices, i.e. conditional covariance matrices.
%   X0         - [Optional] Starting point of optimization
%   FCST_STEPS - [Optional] Number of steps to forecast.
%   ENF_PD     - [Optional] Should Sigma_ be enforced to be theoretically pd in Optimization?
%   VARARGIN   - [Optional] Optional Input to Optimizer.
%
% OUTPUTS:
%   EPARAM          - Struct of estimated parameters with following fields:
%       INTRCPT         - Recursion intercept matrix.
%       ARCH_PARAM      - ARCH parameters of the recursion.
%       GARCH_PARAM     - GARCH parameters of the recursion.
%   TSTATS          - Struct of t-stats with the fields as eparam.
%   LOGL            - Struct containing
%                       - logL value of log-likelihood at optimum.
%                       - logLcontr log-likelihood contributios at optimum.
%                       - aic Akaike Information Criterion.
%                       - bic Bayes Information Criterion.
%   FIT_FCST        - Struct containing ...
%   OPTIMOUTPUT     - Optimization Output from fmincon with fields:
%                       - estimation_time 
%
% COMMENTS:
%   Estimation is done by QMLE. This implies that the nuisance paramter
%   "nu" (degrees of freedom of Wishart distribution) is not estimated.
%   If we would estimate "nu" as well, estimation does not work for
%   high-dimensional realized covariance matrices, because the true "nu"
%   paramter often lies outside of the possible parameter space, namely it
%   is smaller than the cross-sectional dimension. This distorts the
%   estimation of the dynamic paramters.
% 
% REFERENCES:
%      [1] Golosnoy, Gribisch and Liesenfeld (2012) - The conditional 
%      autoregressive Wishart model for multivariate stock volatility
%
% DEPENDENCIES:
%   vcv, (robustvcv)
%
% See also scaw11, caw
%
% Michael Stollenwerk
% michael.stollenwerk@live.com
% 21.06.2019

%% Input Checking
% Will be added later
[k,~,~] = size(data_RC);
if isempty(enf_pd)
    enf_pd = 1;
end
%% Optimization
% Objective Function (see below)-------------------------------------------
fun = @( param ) scalarcaw_likeRec( param, p, q, data_RC, enf_pd );
% x0-----------------------------------------------------------------------
if isempty(x0) || isstruct(x0)
    if isfield(x0,'arch_param')
        arch_param = x0.arch_param;
    else
        arch_param = ones(1,p)*.05/p;
    end
    if isfield(x0,'garch_param')
        garch_param = x0.garch_param;
    else
        garch_param = ones(1,q)*.90/q;        
    end
    x0 = [ arch_param';     % p
           garch_param'     % p + q
    ]';
    try
        fun(x0);
    catch
        warning('User supplied x0 is not valid. Using default x0 instead.')
        x0 = [ ones(p,1)*.05/p;
               ones(q,1)*.90/q
        ]';
    end
else
    error('x0 must be empty or struct.')
end
% Restrictions
A = ones(1,p+q); % This is to ensure pd unconditional mean (=(1-arch-garch)*Intrcpt). sufficient and necessary. Sigma is enforced to be pd in sample, since otherwise the likeRec returns inf.
b = 1;
lb = [];
if enf_pd
    lb = zeros(1,p+q); % This is to ensure Sigma has 0 probability to be < 0, sufficient but not necessary.
end         
% Optimoptions-------------------------------------------------------------
options = optimoptions(optimoptions('fmincon'), varargin{:});
% Optimization-------------------------------------------------------------
warning('off') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[eparam,~,exitflag,optimoutput,grad,hessian] = fmincon( ...
    fun, x0, A, b, [], [], lb, [], [], options ...
);
optimoutput.estimation_time = toc;
optimoutput.exitflag = exitflag;
optimoutput.grad = grad;
optimoutput.hessian = hessian;
warning('on') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tstats (vcv see below) --------------------------------------------------
%[VCV,A,B,scores,hess,gross_scores] = robustvcv(fun, eparam, 3);
[VCV,scores,gross_scores] = vcv(fun, eparam);
tstats = eparam./sqrt(diag(VCV)');

% Output Creation----------------------------------------------------------
[ nLogL, logLcontr, Sigma_, fcst_Sigma ] = scalarcaw_likeRec( ...
    eparam, 1, 1, data_RC, enf_pd, fcst_steps ...
);

intrcpt = (1-sum(eparam))*mean(data_RC,3);
intrcpt = (intrcpt + intrcpt')/2;

eparam = struct(...
    'intrcpt', intrcpt,...
    'chol_intrcpt', vech(chol(intrcpt, 'lower'))',... 
    'arch_param', eparam(1:p),...
    'garch_param', eparam(p+1:p+q),...
    'all', [vech(chol(intrcpt, 'lower'))', eparam]...
);
aic = 2*nLogL + 2*(numel(x0) + k*(k+1)/2);
bic = 2*nLogL + log(numel(data_RC))*(numel(x0) + k*(k+1)/2);
logL = struct(...
    'logL', -nLogL,...
    'aic', aic,...
    'bic', bic,...
    'logLcontr', logLcontr...
);
fit_fcst = struct('Sigma_', Sigma_, 'fcst_Sigma', fcst_Sigma);

tstats = struct(...
    'chol_intrcpt', NaN(1,k*(k+1)/2),...
    'arch_param', tstats(1:p),...
    'garch_param', tstats(p+1:p+q),...
    'all', [NaN(1,k*(k+1)/2), tstats]...
);
end

%% Helper Functions

function [ nLogL, logLcontr, Sigma_, fcst_Sigma ] = scalarcaw_likeRec( ...
    param, p, q, data_RC, enf_pd, varargin ...
)

if isempty(varargin)
    fcst_steps = 0;
else
    fcst_steps = varargin{1};
end
[k,~,T] = size(data_RC);
arch_param = param(1:p);
garch_param = param(p+1:p+q);
%% Data Storage
Sigma_ = NaN(k,k,T+fcst_steps);
%% Recursion
meanRC = mean(data_RC,3);
intrcpt = (1-sum(param))*meanRC;
intrcpt = (intrcpt + intrcpt')/2; % force exact symmetry
for tt=1:T+fcst_steps
    Sigma_(:,:,tt) = intrcpt;
    for jj = 1:p
        if (tt-jj) <= 0
            Sigma_(:,:,tt) = Sigma_(:,:,tt) + arch_param(jj)*meanRC;
        elseif (tt-jj) <= T
            Sigma_(:,:,tt) = Sigma_(:,:,tt) + arch_param(jj)*data_RC(:,:,tt-jj);
        else
            Sigma_(:,:,tt) = Sigma_(:,:,tt) + arch_param(jj)*Sigma_(:,:,tt-jj);
        end
    end
    for jj = 1:q
        if (tt-jj) <= 0
            Sigma_(:,:,tt) = Sigma_(:,:,tt) + garch_param(jj)*meanRC;
        else
            Sigma_(:,:,tt) = Sigma_(:,:,tt) + garch_param(jj)*Sigma_(:,:,tt-jj);
        end
    end
    if ~enf_pd % To avoid computing time if pd is enforced by opt constraints anyways.
        if any(eig(Sigma_(:,:,tt))<0) % To allow for unconstraint optimization
            nLogL = inf;
            return
        end
    end
end
fcst_Sigma = Sigma_(:,:,T+1:T+fcst_steps);
Sigma_ = Sigma_(:,:,1:T);
%% Likelihood
[ nLogL, logLcontr ] = wishqlike( Sigma_, data_RC ); % (see below)
end


function [VCV,scores,gross_scores]=vcv(fun,theta,varargin)
% Compute Variance Covariance matrix numerically only based on gradient
%
% USAGE:
%     [VCV,A,SCORES,GROSS_SCORES]=vcv(FUN,THETA,VARARGIN)
%
% INPUTS:
%     FUN           - Function name ('fun') or function handle (@fun) which will
%                       return the sum of the log-likelihood (scalar) as the 1st output and the individual
%                       log likelihoods (T by 1 vector) as the second output.
%     THETA         - Parameter estimates at the optimum, usually from fmin*
%     VARARGIN      - Other inputs to the log-likelihood function, such as data
%
% OUTPUTS:
%     VCV           - Estimated robust covariance matrix (see White 1994)
%     SCORES        - T x num_parameters matrix of scores
%     GROSS_SCORES  - Numerical scores (1 by num_parameters) of the objective function, usually for diagnostics
%
% COMMENTS:
%     For (Q)MLE estimation

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 05.02.2019

% Adapted from robustvcv by Kevin Sheppard:
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 9/1/2005

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(theta,1)<size(theta,2)
    theta=theta';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


k=length(theta);
h=abs(theta*eps^(1/3));
h=diag(h);

[~,like]=feval(fun,theta,varargin{:});

t=length(like);

LLFp=zeros(k,1);
LLFm=zeros(k,1);
likep=zeros(t,k);
likem=zeros(t,k);
for i=1:k
    thetaph=theta+h(:,i);
    [LLFp(i),likep(:,i)]=feval(fun,thetaph,varargin{:});
    thetamh=theta-h(:,i);
    [LLFm(i),likem(:,i)]=feval(fun,thetamh,varargin{:});
end

scores=zeros(t,k);
gross_scores=zeros(k,1);
h=diag(h);
for i=1:k
    scores(:,i)=(likep(:,i)-likem(:,i))./(2*h(i));
    gross_scores(i)=(LLFp(i)-LLFm(i))./(2*h(i));
end

B=cov(scores);
VCV=inv(B)/t;
end



function vechMatrix = vech(Matrix)
% Half-vectorizes a matrix
% 
%% Inputs/Outputs
%
% Inputs:
%   MATRIXDATA   - a K by K symmetric matrix
%
% Outputs:
%   STACKEDDATA - A K(K+1)/2 vector of stacked data
%
% Comments:
%   The data is stacked according to 
%     [ data(1) ...        ...         ...               ...
%       data(2) data(K+1)  ...         ...               ...
%       data(3) data(K+2)  data(2K)    ...               ...
%       ...     ....       ...         ...               ...
%       data(K) data(2K-1) ...         data(K(K+1)/2-1)  data(K(K+1)/2) ]

%% Input Checking
[k,l] = size(Matrix);
pl = ~tril(true(k));
if k~=l 
    error('MATRIXDATA must be a square matrix');
end
if ~issymmetric(Matrix) && any(Matrix(pl)~=0) && ~all(isnan(Matrix(:)))
    error('MATRIXDATA must be a lower triangular matrix or symmetric');
end

%% Core Code
sel = tril(true(k));
vechMatrix = Matrix(sel);
end



function [ nLogL, logLcontr ] = wishqlike( sigma, data )
%WISHLIKE Negative quasi log-likelihood for the Wishart distribution.
%
% USAGE:
%   NLOGL = wishqlike(sigma,df,data)
%
% INPUTS:
%   SIGMA        - K by K by N array of scale matrices of the Wishart
%                  distribution.
%   DATA         - K by K by N array of matrix data realizations.
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
%      [1] Sutradhar and Ali (1989) - A Generalization of the Wishart
%      Distribution for the Elliptical Model and its Moments for the
%      Multivariate t Model.

% Michael Stollenwerk
% michael.stollenwerk@live.com
% 21.06.2019
%% Error checking
narginchk(2,2);

[k,k2,n] = size(sigma);
if k~=k2
    error('Covariance matrix parameters of Wishart distribution must be square, symmetric, and positive semi-definite.')
end

[k,k2,~] = size(data);
if k~=k2
    error('Covariance matrix data of Wishart distribution must be square, symmetric, and positive semi-definite.')
end

if size(sigma)~=size(data)
    error('Covariance matrix data and parameters of Wishart distribution must have the same dimension size')
end
%% Log-likelihood computation

logLcontr=NaN(n,1);

if isnumeric(sigma) && isnumeric(data) % Double input
    for i = 1:n 
        logLcontr(i) = -.5*log(det(sigma(:,:,i))) ...
                       -.5*trace(sigma(:,:,i)\data(:,:,i));
    end
elseif iscell(sigma) && iscell(data) % Cell input
    for i = 1:n
        logLcontr(i) = -.5*log(det(sigma{i})) ...
                       -.5*trace(sigma{i}\data{i});
    end
end

nLogL=-sum(logLcontr);
end

