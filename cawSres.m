function [ vechSres ] = cawSres( rdata, edata, df, distr )
%CAWSRES calculates the standardized residuals from (t-)CAW model
%   See formula (24) in Golosnoy et al. (2012) - CAW model paper.
%   
%   RDATA - realized covariance matrix data (n,n,T).
%   EDATA - expected value of realized covariance matrices (n,n,T).
%   DF    - In case of Distribution=='n': Scalar, Wishart degrees of freedom (dof).
%           In case of Distribution=='t': Array [df_w df_t], Wishart degrees of freedom (dof) and dof of the underlying t-distribution.                     
%   DISTR - [Optional] Type of Wishart distribution, one of 'n' (Default) or 't'.
%
%   VECHSRES - standardized residuals in vech format (n*(n+1)/2,T)

[N,~,T] = size(rdata);
%%
L = sparse(ELmatrix(N));
K = sparse(Cmatrix(N,N));
eye_ = sparse(eye(N^2));
vechSres = NaN(N*(N+1)/2,T);
%% Input Checking
if size(rdata)~=size(edata)
    error('rdata and edata must be same size')
end
%% Calculating Residuals
res = rdata-edata;
parfor t=1:T
	%% Calculating Standardized Residuals
	if isempty(distr) || strcmpi(distr,'n')
		vmat = 1/df*L*(eye_ + K)*kron(edata(:,:,t),edata(:,:,t))*L';
	elseif strcmpi(distr,'t')
		vmat = df(2)^2/(df(2)-2)/(df(2)-4) * df(1) * (vmat + 2*df(1)/(df(2)-2) * vech(edata(:,:,t))*vech(edata(:,:,t))');
	end
    vechSres(:,t) = chol(vmat,'lower')\vech(res(:,:,t));
end
end




%% Helper Functions
function [ L ] = ELmatrix( n )
%Ematrix creates Elimination matrix as in Magnus and Neudecker (1988)
% Follows: Mathematics for Econometrics, p.131, Phoebus J. Dhrymes (2013)

storage = cell(n,1);
for i=1:n
    storage(i)= {[zeros(n+1-i,i-1), eye(n+1-i)]};
end

L=blkdiag(storage{:});  
  
end


function Y = Cmatrix(m,n)
% Calculate commutation matrix as in Magnus Neuendecker 2005.
% Code take from stackexchange. 
% Input m = nrows, n = ncols.
I = reshape(1:m*n, [m, n]); % initialize a matrix of indices of size(A)
I = I'; % Transpose it
I = I(:); % vectorize the required indices
Y = eye(m*n); % Initialize an identity matrix
Y = Y(I,:); % Re-arrange the rows of the identity matrix
end