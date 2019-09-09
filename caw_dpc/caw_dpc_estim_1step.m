function [ eparam, tstats, logL, fit, fcst, optimoutput] = ...
    caw_dpc_estim_1step( ...
        data_RC, p_l, q_l, p_c, q_c, fcst_steps, x0, enf_pd, varargin ...
    )
%CAW Estimate the DPC-CAW via QMLE.
%
% USAGE:
%   [ eparam, logL, fit_fcst, optimoutput] = ...
%       caw_dpc_estim_1step( data_RC, p_l, q_l, p_c, q_c, x0 )
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
if isempty(enf_pd)
    enf_pd = 1;
end
%% Optimization
% Objective Function (see below)-------------------------------------------
fun = @(param) qlikeRec( ...
    param, p_l, q_l, p_c, q_c, data_RC, fcst_steps, enf_pd ...
);
% x0-----------------------------------------------------------------------
if isempty(x0) || isstruct(x0)
    try
        arch_param_l = x0.loadings.arch_param;
    catch
        arch_param_l = ones(1,p_l)*..05/p_l;
    end
    try
        garch_param_l = x0.loadings.garch_param;
    catch
        garch_param_l = ones(1,q_l)*.9/q_l;        
    end
    try
        arch_param_c = x0.components.arch_param(:)';
    catch
        arch_param_c = ones(1,N*p_c)*.05/p_c;
    end
    try
        garch_param_c = x0.components.garch_param(:)';
    catch
        garch_param_c = ones(1,N*q_c)*.9/q_c;        
    end 
    try
        chol_intrcpt = x0.loadings.chol_intrcpt / ...
                    sqrt(1-sum(arch_param_l+garch_param_l));
    catch
        chol_intrcpt = vech(chol(mean(data_RC,3), 'lower'))';
    end    
    x0 = [ chol_intrcpt';   % N*(N+1)/2
           arch_param_l';   % N*(N+1)/2 + p_l
           garch_param_l';  % N*(N+1)/2 + p_l + q_l
           arch_param_c';   % N*(N+1)/2 + p_l + q_l + p_c
           garch_param_c'   % N*(N+1)/2 + p_l + q_l + p_c + q_c
    ]';
    try
        fun(x0);
    catch
        warning('User supplied x0 is not valid. Using default x0 instead.')
        x0 = [ vech(chol(mean(data_RC,3), 'lower'))
               ones(p_l,1)*.05/p_l;
               ones(q_l,1)*.90/q_l;
               ones(N*p_c,1)*.05/p_c;
               ones(N*q_c,1)*.90/q_c
        ]';
    end
else
    error('x0 must be empty or struct.')
end
% Optimoptions-------------------------------------------------------------
options = optimoptions(optimoptions('fmincon'), varargin{:});
% Optimization Constraints-------------------------------------------------
% Stationarity Constraints
A = [ zeros(1,K) ones(1,p_l+q_l)  zeros(1,N*(p_c+q_c))    ; % This is to ensure pd unconditional mean (=(1-arch-garch)*Intrcpt). sufficient and necessary. Sigma is enforced to be pd in sample, since otherwise the likeRec returns inf.
      zeros(N,K) zeros(N,p_l+q_l) repmat(eye(N),1,p_c+q_c);
    ];
b = ones(1,1+N);
lb = [];
if enf_pd
    lb = [ -inf(1,K) zeros(1,p_l+q_l) zeros(1, N*(p_c+q_c) )]; % This is to ensure Sigma has 0 probability to be < 0, sufficient but not necessary.
end   
% Optimization-------------------------------------------------------------
warning('off') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[eparam_all,~,exitflag,optimoutput,grad,hessian] = ...
    fmincon( fun,x0,A,b,[],[],lb,[],[],options );
optimoutput.estimation_time = toc;
optimoutput.exitflag = exitflag;
optimoutput.grad = grad;
optimoutput.hessian = hessian;
warning('on') %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tstats (vcv see below) --------------------------------------------------
%[VCV,A,B,scores,hess,gross_scores] = robustvcv(fun, eparam, 3);
[VCV,scores,gross_scores] = vcv(fun, eparam_all);
tstats = eparam_all./sqrt(diag(VCV)');

[~, ~, fit, fcst] = ...
    qlikeRec(eparam_all, p_l, q_l, p_c, q_c, data_RC, fcst_steps, enf_pd);

df = fmincon( ...
    @(df) wishlike( fit.Sigma_/df, df, data_RC ), 2*N, -1, -(N-1),...
    [], [], [], [], [], optimoptions('fmincon', 'Display', 'off') ...
);

% Output Creation----------------------------------------------------------

intrcpt_ = ivech(eparam_all(1:K),'lower')*ivech(eparam_all(1:K),'lower')';
[~, gammas] = sorteig(intrcpt_);
intrcpt = (1-sum(eparam_all(K+1:K+p_l+q_l)))*intrcpt_;

loadings = struct(...
    'intrcpt', intrcpt,...
    'chol_intrcpt', vech(chol(intrcpt, 'lower'))',... 
    'arch_param', eparam_all(K+1:K+p_l),...
    'garch_param', eparam_all(K+p_l+1:K+p_l+q_l),...
    'all', [vech(chol(intrcpt, 'lower'))', eparam_all(K:K+p_l+q_l)]...
);

components = struct(...
    'arch_param' , reshape( ...
                       eparam_all(K+p_l+q_l+1:K+p_l+q_l+N*p_c), ...
                       N,p_c ...
                   ), ...
    'garch_param', reshape( ...
                       eparam_all(K+p_l+q_l+N*p_c+1:K+p_l+q_l+N*p_c+N*q_c), ...
                       N,q_c ...
                   ) ...
);
components.intrcpt = ...
    (1-sum(components.arch_param,2)-sum(components.garch_param,2)).*gammas;
components.all = [components.intrcpt, components.arch_param, components.garch_param];

all = [loadings.chol_intrcpt,...
       loadings.arch_param,...
       loadings.garch_param,...
       components.arch_param(:)',...
       components.garch_param(:)',...
       df];

eparam.loadings = loadings;
eparam.components = components;
eparam.df = df;
eparam.all = all;

% Estimate wishart degrees of freedom and store in struct "eparam"
% wishlike see below.

% get full log-likelihood values.
[nLogL, logLcontr] = wishlike( fit.Sigma_/eparam.df, eparam.df, data_RC);

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

end

%% Helper Functions
function [nLogL, logLcontr, fit, fcst] = ...
    qlikeRec(param, p_l, q_l, p_c, q_c, data_RC, fcst_steps, enf_pd)

if size(param,2)<size(param,1)
    param = param';
end
[N,~,T] = size(data_RC);
K = N*(N+1)/2;

if size(param,2) ~= K+p_l+q_l+N*p_c+N*q_c
    error('size(param,2) is incorrect.')
end

if isempty(fcst_steps)
    fcst_steps = 0;
end

intrcpt = ivech(param(1:K),'lower');
intrcpt = intrcpt*intrcpt';

[~, eig_intrcpt] = sorteig(intrcpt);

arch_param_l  = param(K+1               : K+p_l                 );
garch_param_l = param(K+p_l+1           : K+p_l+q_l             );
arch_param_c  = param(K+p_l+q_l+1       : K+p_l+q_l+N*p_c       );
arch_param_c  = reshape(arch_param_c, N, p_c);
garch_param_c = param(K+p_l+q_l+N*p_c+1 : K+p_l+q_l+N*p_c+N*q_c );
garch_param_c  = reshape(garch_param_c, N, q_c);

% There are two possibilities to initialize the eigenvalue recursion.
% Either via mean(diagLRL), or via sorteig(mean(data_RC,3)). I have to decide
% which one to use.
ini_rec_l = mean(data_RC,3); 

Q = NaN(N,N,T+fcst_steps);
L = NaN(N,N,T+fcst_steps);
Sigma_ = NaN(N,N,T+fcst_steps);
diagLRL = NaN(T,N);
d = NaN(T+fcst_steps,N);

% In-Sample ---------------------------------------------------------------
logLcontr = NaN(T,1);
for tt = 1:T
    Q(:,:,tt) = (1-sum(arch_param_l)-sum(garch_param_l))*intrcpt;
    for jj = 1:p_l
        if (tt-jj) <= 0 % if true, use initialization
            Q(:,:,tt) = Q(:,:,tt) + arch_param_l(jj)*ini_rec_l;
        elseif (tt-jj) <= T
            Q(:,:,tt) = Q(:,:,tt) + arch_param_l(jj)*data_RC(:,:,tt-jj);
        end
    end
    for jj = 1:q_l
        if (tt-jj) <= 0 % if true, use initialization
            Q(:,:,tt) = Q(:,:,tt) + garch_param_l(jj)*ini_rec_l;
        else
            Q(:,:,tt) = Q(:,:,tt) + garch_param_l(jj)*Q(:,:,tt-jj);
        end
    end
    [L(:,:,tt), eig_] = sorteig(Q(:,:,tt)); % To allow for unconstraint optimization
    if any(eig_) < 0
        nLogL = inf;
        return
    end
    diagLRL(tt,:) = diag(L(:,:,tt)'*data_RC(:,:,tt)*L(:,:,tt));
end

ini_rec_c = mean(diagLRL);
for tt = 1:T 
    for nn = 1:N
        d(tt,nn) = ...
            (1-sum(arch_param_c(nn,:))-sum(garch_param_c(nn,:))) * ...
                eig_intrcpt(nn);
        for jj = 1:p_c
            if (tt-jj) <= 0 % if true, use initialization
                d(tt,nn) = d(tt,nn) + arch_param_c(nn,jj)*ini_rec_c(nn);
            elseif (tt-jj) <= T
                d(tt,nn) = d(tt,nn) + arch_param_c(nn,jj)*diagLRL(tt-jj,nn);
            end
        end
        for jj = 1:q_c
            if (tt-jj) <= 0 % if true, use initialization
                d(tt,nn) = d(tt,nn) + garch_param_c(nn,jj)*ini_rec_c(nn);
            else
                d(tt,nn) = d(tt,nn) + garch_param_c(nn,jj)*d(tt-jj,nn);
            end
        end
        if ~enf_pd % To avoid computing time if pd is enforced by opt constraints anyways.
            if d(tt,nn) < 0 % To allow for unconstraint optimization
                nLogL = inf;
                return
            end
        end
    end
    logLcontr(tt) = -.5*sum(log(d(tt,:))+diagLRL(tt,:)./d(tt,:));
    Sigma_(:,:,tt) = L(:,:,tt)*diag(d(tt,:))*L(:,:,tt)';
    Sigma_(:,:,tt) = (Sigma_(:,:,tt)' + Sigma_(:,:,tt))/2;    
end
nLogL = -sum(logLcontr);

% Out-Of-Sample -----------------------------------------------------------
for tt = T+1:T+fcst_steps
    Q(:,:,tt) = (1-sum(arch_param_l)-sum(garch_param_l))*intrcpt;
    for jj = 1:p_l
        if (tt-jj) <= T
            Q(:,:,tt) = Q(:,:,tt) + arch_param_l(jj)*data_RC(:,:,tt-jj);
        elseif (tt-jj) > T
            Q(:,:,tt) = Q(:,:,tt) + arch_param_l(jj)*Sigma_(:,:,tt-jj);
        end
    end
    for jj = 1:q_l
        Q(:,:,tt) = Q(:,:,tt) + garch_param_l(jj)*Q(:,:,tt-jj);
    end
    [L(:,:,tt), ~] = sorteig(Q(:,:,tt)); 
    
    for nn = 1:N
        for jj = 1:p_c
            if (tt-jj) <= T
                d(tt,nn) = d(tt,nn) + arch_param_c(nn,jj)*diagLRL(tt-jj,nn);
            elseif (tt-jj) > T
                d(tt,nn) = d(tt,nn) + arch_param_c(nn,jj)*d(tt-jj,nn);
            end
        end
        for jj = 1:q_c
            d(tt,nn) = d(tt,nn) + garch_param_c(nn,jj)*d(tt-jj,nn);
        end 
    end
    Sigma_(:,:,tt) = L(:,:,tt)*diag(d(tt,:))*L(:,:,tt)';    
end

% Output Creation----------------------------------------------------------
fcst_Q = Q(:,:,T+1:end);
Q = Q(:,:,1:T);

fcst_L = L(:,:,T+1:end);
L = L(:,:,1:T);

fcst_d = d(T+1:end,:);
d = d(1:T,:);

fcst_Sigma = Sigma_(:,:,T+1:end);
Sigma_ = Sigma_(:,:,1:T);

fit = struct(...
    'Sigma_', Sigma_,...
    'Loadings', L,...
    'd', d,...
    'diagLRL', diagLRL,...
    'Q', Q...
);

fcst = struct(...
    'fcst_Sigma', fcst_Sigma,...
    'fcst_Loadings', fcst_L,...
    'fcst_d', fcst_d,...
    'Q', fcst_Q ...    
);
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



function Matrix=ivech(vechMatrix,type)
%% Transform a vector into 'sym' symmetric or 'lower' lower triangluar matrix
% 
% USAGE:
%   MATRIX = ivech(vechMatrix)
% 
% INPUTS:
%   vechMatrix   - A K(K+1)/2 vector of data to be transformed 
%   type         - 1 for symmetric, 2 for lower triangular
%
% OUTPUTS:
%   Matrix       - 'sym' K by K symmetric matrix of the form 
%                  [ data(1) data(2)    data(3)     ...               data(K)
%                    data(2) data(K+1)  data(K+2)   ...               ...
%                    data(3) data(K+2)  data(2K)    ...               ...
%                    ...     ....       ...         ...               data(K(K+1)/2-1)
%                    data(K) data(2K-1) ...         data(K(K+1)/2-1)  data(K(K+1)/2) ]
%
%                  OR 'lower' K by K lower triangular matrix of the form 
%                  [ data(1) 0          0           ...               0
%                    data(2) data(K+1)  0           ...               0
%                    data(3) data(K+2)  data(2K)    ...               ...
%                    ...     ....       ...         ...               0
%                    data(K) data(2K-1) ...         data(K(K+1)/2-1)  data(K(K+1)/2) ]

% Author: Kevin Sheppard
% Modified: Michael Stollenwerk
% Modification: Added option to create lower triangular matrix.
if nargin==1
    type='sym';
end
%% Input Checking
if size(vechMatrix,2)>size(vechMatrix,1)
    vechMatrix=vechMatrix';
end

if size(vechMatrix,2)~=1
    error('STACKED_DATA must be a column vector.')
end

K2=size(vechMatrix,1);%checking if numel in stackedData fits
K=(-1+sqrt(1+8*K2))/2; %checking if numel in stackedData fits

if floor(K)~=K %checking if numel in stackedData fits
    error(['The number of elements in STACKED_DATA must be conformable to' ...
    'the inverse chol2vec operation.'])
end

if ~strcmp(type,'sym') && ~strcmp(type,'lower')
    error('Please specify type: sym or lower')
end

%%
% Initialize the output data
Matrix=zeros(K);

% Use a logical trick to inverse the vech
pl=tril(true(K));
Matrix(pl)=vechMatrix;

if strcmp(type,'sym')
    diag_matrixData=diag(diag(Matrix));
    Matrix=Matrix+Matrix'-diag_matrixData;
end
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