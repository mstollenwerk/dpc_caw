function [ R, S, L, Q, d, g ] = caw_dpc_sim_scalar( param, T )

%% Extend T for initialization period
burn_in = 1000;
T = T + burn_in;
%% Parameter
intrcpt_l = param.loadings.intrcpt;
a = param.loadings.arch_param;
b = param.loadings.garch_param;
intrcpt_c = param.components.intrcpt;
alphas = param.components.arch_param;
betas = param.components.garch_param;
df = param.df;

p_l = size(a,2); % eigenvector (loadings) recursion (p,q)
q_l = size(b,2);
p_c = size(alphas,2); % eigenvalue (component) recursion (p,q)
q_c = size(betas,2);

N = size(intrcpt_l,1);
%% Storage
% cell recursions are faster
R = cell(1,T);
S = cell(1,T);
L = cell(1,T);
d = NaN(N,T);
g = NaN(N,T);
Q = cell(1,T);

%% Simulation
uncMean_l = intrcpt_l/(1-sum(a)-sum(b));
uncMean_c = intrcpt_c./(1-sum(alphas,2)-sum(betas,2));

% Initialization
Q{1} = uncMean_l;
[L{1}, ~] = sorteig(Q{1});
d(:,1) = uncMean_c;
g(:,1) = uncMean_c;
S{1} = L{1}*diag(d(:,1))*L{1}';
R{1} = wishrnd(S{1}/df,df);

% Recursion
for tt=2:T
    Q{tt} = intrcpt_l;
    for jj = 1:p_l
        if (tt-jj) > 0
            Q{tt} = Q{tt} + a(jj)*R{tt-jj};
        else
            Q{tt} = Q{tt} + a(jj)*uncMean_l;
        end
    end
    for jj = 1:q_l
        if (tt-jj) > 0
            Q{tt} = Q{tt} + b(jj)*Q{tt-jj};
        else
            Q{tt} = Q{tt} + b(jj)*uncMean_l;           
        end
    end
    [L{tt},] = sorteig(Q{tt}); % use sorteig (see below)
    
    for ii = 1:N
        d(ii,tt) = intrcpt_c(ii);
        for jj = 1:p_c
            if (tt-jj) > 0 
                d(ii,tt) = d(ii,tt) + alphas(ii,jj)*g(ii,tt-jj);
            else
                d(ii,tt) = d(ii,tt) + alphas(ii,jj)*uncMean_c(ii);
            end
        end
        for jj = 1:q_c
            if (tt-jj) > 0 
                d(ii,tt) = d(ii,tt) + betas(ii,jj)*d(ii,tt-jj);
            else
                d(ii,tt) = d(ii,tt) + betas(ii,jj)*uncMean_c(ii);
            end
        end        
    end

    S{tt} = L{tt}*diag(d(:,tt))*L{tt}';
    S{tt} = (S{tt} + S{tt}')/2; % to force exact symmetry
    R{tt} = wishrnd(S{tt}/df,df);
    g(:,tt) = diag(L{tt}'*R{tt}*L{tt});
end
%% Output
% (discard burn_in)
R = R(burn_in+1:end);
R = reshape(cell2mat(R),N,N,[]);
S = S(burn_in+1:end);
S = reshape(cell2mat(S),N,N,[]);
L = L(burn_in+1:end);
L = reshape(cell2mat(L),N,N,[]);
Q = Q(burn_in+1:end);
Q = reshape(cell2mat(Q),N,N,[]);
d = d(:,burn_in+1:end);
g = g(:,burn_in+1:end);
end



%% Helper Functions

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