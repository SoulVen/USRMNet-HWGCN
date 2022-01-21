%%%%%        该函数用于生成训练与测试样本，包含初始化后的w0和p0        %%%%%%
clear all
clc
close all
digits(40)                                      %% 规定运算精度是40位有效数字
warning off

cvx_setup
%% 参数配置
K = 32;                                         %% K个天线 （4:10）
Pm = [10^(11/10)];                              %% 最大功率约束
D = 32*8;                                       %% 传输的比特数
n = 128;                                        %% 有限块长
M = 4;                                          %% 用户数目

epsilon = [1e-5];                               %% BER误差
epsilon2 = 1e-2;                                %% 迭代误差停止条件
iter_cur = 1;                                   %% 迭代变量
num_H = 20;                                   %% 样本个数
H = [];                                         %% 存储所有信道信息
w0 = [];                                        %% 存储所有样本初始化的w
p0 = [];                                        %% 存储所有样本初始化的p

%% 生成信道系数及初始化波束向量，以及初始化功率
while(iter_cur <= num_H)                        %% 对每一个样本进行初始化
    sigma_k = ones(1,M);
    while(1)   %% 直到当前迭代找到一个可以初始化的信道信息
        h_k_ori=[];
        h_k=[];
        h_k_ori = channel(K,M);                 %% h_k的形式是M*K，即用户数*天线数
        h_k = h_k_ori/sigma_k(1);
        gamma_k_wan=[];
        for kk = 1:M
            gamma_k_wan(kk) = Pm*norm(h_k_ori(kk,:))^2./ (sigma_k(kk))^2;
        end

        [w_k_0,p_k_0,gamma_k,psi_k_t,phi_k_t,xita_k_t] = cal_init_w_v14(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan);

        if isnan(w_k_0)
            sprintf('Inf错误');
            continue;
        else
            sprintf('第%d个样本初始化成功', iter_cur)
            H(iter_cur, :, :) = h_k_ori;                %% 保存当前信道系数
            w0(iter_cur, :, :) = w_k_0;                 %% 保存当前信道初始化的w0
            p0(iter_cur, :) = p_k_0;                    %% 保存当前信道初始化的p0
            iter_cur = iter_cur + 1;
            break;
        end
        clear q_k*  w_k* ze* xi* t*
    end
end

%% 保存所有的数据
save_file = strcat('.\dataset\channel',num2str(M),'_',num2str(K),'.mat');
save(save_file,'H','w0', 'p0');
