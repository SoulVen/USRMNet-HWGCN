function gamma2 = cal_gamma2(D,n,epsilon)

%% 按照论文中方法计算gamma2

alpha = D/n*log(2); % alpha = D/n*log(2);
theta = qfuncinv(epsilon)/sqrt(n);
beta = exp(-alpha);

l1 = 2*theta;
l2 = -2*theta;
mu = -4*beta^2*theta^2;
w_minus = 0;
for m = 1:10
    B_m_1 = 0;
    for k = 0:m-1
        B_m_1 = B_m_1 + prod(1:m-1+k)/prod(1:k)/prod(1:m-1-k)* (-1/m/(l2-l1) )^k;
    end
    w_minus = w_minus + 1/(m*prod(1:m))*(mu*m*exp(-l1)/(l2-l1))^m * B_m_1;
end
w = l1-w_minus;
gamma2 = exp(alpha+w/2)-1;

%% 以论文计算方法为初始点，调用matlab算法求精确解
% alpha = D/n*log(2);
% theta = qfuncinv(epsilon)/sqrt(n);
% f=@(x) log(1+x) -theta*sqrt(1-1/(1+x)^2)-alpha;
% gamma2 = fzero(f,gamma2)

