function [w_k_0,p_k_t,gamma_k,psi_k_t,phi_k_t,xita_k] = cal_init_w_v14(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan)


alpha = ones(1,M);
gamma2 = cal_gamma2(D,n,epsilon); %% 最小信噪比

cvx_begin quiet
variable w_k_init(K,M) complex 
sum_w = 0;
for ii = 1:M
    sum_w = sum_w + w_k_init(:,ii)' * w_k_init(:,ii);   % (46a)
end
minimize sum_w
subject to
for ii = 1:M
    for jj = 1:M
        o_k(ii,jj) = h_k(ii,:) * w_k_init(:,jj);
    end
end
o_k(:,M+1) = 1;
for ii = 1:M
    norm(o_k(ii,:)) <= sqrt((1+1/gamma2)) *real((h_k(ii,:) * w_k_init(:,ii)))
end
cvx_end


for ii = 1:M
    p_k_0(ii) = norm(w_k_init(:,ii))^2;
end
w_k_t = w_k_init;
for ii = 1:M
    p_h_w = 0;
    for jj = 1:M
        if jj ~= ii
            p_h_w = p_h_w + abs(h_k(ii,:)*w_k_t(:,jj))^2;
        end
    end
    gamma_k(ii) = abs(h_k(ii,:)*w_k_t(:,ii))^2/(p_h_w + 1);
end
gamma_k

for ii = 1:M
    w_k_init(:,ii) = w_k_init(:,ii)/norm(w_k_init(:,ii));
end

w_k_t = w_k_init;

%% 验算总功率是否满足要求
p_k_t = p_k_0;
% w_k_t = w_k_init;
w_k_0 = w_k_init;

phi_k_t = gamma_k; % varphi
theta_k = gamma_k; % phi
V_theta_k = 1-1./(1+theta_k).^2;
psi_k_t = V_theta_k;  % psi
xita_k = sqrt(psi_k_t); % theta

% %% 验算速率是否为正
for ii  = 1:M
    R_gamma(ii) = log(1+gamma_k(ii)) - qfuncinv(epsilon)/sqrt(n) * sqrt(1-1/(1+gamma_k(ii))^2);
end
% R_gamma
sum(R_gamma);

tao_value = 0;
for jj = 1:M
    tao_value = tao_value + alpha(jj)*( log(1+phi_k_t(jj))- qfuncinv(epsilon)/sqrt(n)* xita_k(jj));
end
tao_value;



if sum(p_k_0)<=Pm && sum(R_gamma>0) == M
    sprintf('dual初始化成功---');
else
    sprintf('dual初始化失败');
    w_k_ZFBF = nan;
    w_k_t = nan;
    w_k_0 = nan;
end

