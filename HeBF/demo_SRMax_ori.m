% K-user,  M-BS
clear all
clc
close all
digits(40)
warning off

cvx_setup
K = 4;    % K个天线 （4:10）
Pm = [10^(11/10)];
D = 32*8;

n_all = [128,256,512]; %%取值多少合适？是否为D均分K份 [64,128,256,512]
M_all = [4] ;     % M个用户 （2:10）         [2,4,6,8] 

epsilon = [1e-5];   %% BER误差
epsilon2 = 1e-2;  %% 迭代误差停止条件
zeros = 0;

for mm = 1:length(M_all)
    for nn = 1:length(n_all)

        M = M_all(mm);
        n = n_all(nn);
        alpha = ones(1,M)/M;
        counter = 1;   % 计数器
        SR_dual_value_record = [];
        SR_ZFBF_value_record = [];
        
        gamma2 = cal_gamma2(D,n,epsilon)  %% 最小信噪比
        gamma_4 = gamma2;
        
        sigma_k = ones(1,M);
        while counter < 401 %同一组参数重复做100次实验
            nn
            mm
            h_k_ori=[];
            h_k=[];
%             h_k_ori = channel(K,M);  %% h_k的形式是M*K，即用户数*天线数
%             h_k = h_k_ori/sigma_k(1);
            if M<=4
                save_file = strcat('D:\代码\论文实验代码\uRLLCUnfolding\uRLLCUnfolding-v03\dataset\test_channel_4_32.mat');
                load(save_file)

            elseif M >=6
                save_file = strcat('F:\05 多目标优化通信项目\公式推导\对偶方法\1018\test_SR\fig2_data\nn_',num2str(128),'_M_',num2str(M),'_第',num2str(counter),'次生成信道数据.mat');
                load(save_file)
            end
            
            h_k = squeeze(H(1,:,:));
            h_k_ori = h_k;
            
            gamma_k_wan=[];
            for kk = 1:M
                gamma_k_wan(kk) = Pm*norm(h_k_ori(kk,:))^2./(sigma_k(kk))^2;
            end
            [w_k_0_ZFBF,p_k_0_ZFBF,gamma_k_ZFBF,psi_k_t_ZFBF,phi_k_t_ZFBF,xita_k_t_ZFBF] = cal_init_w_p(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan);
            if ~isnan(w_k_0_ZFBF)
                [w_k_0,p_k_0,gamma_k,psi_k_t,phi_k_t,xita_k_t] = cal_init_w_p(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan);
            else
                [w_k_0,p_k_0,gamma_k,psi_k_t,phi_k_t,xita_k_t] = cal_init_w_v14(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan);
            end
%             [w_k_0,p_k_0,gamma_k,psi_k_t,phi_k_t,xita_k_t] = cal_init_w_v14(K,M,Pm,D,n,epsilon,h_k,sigma_k,gamma_k_wan);
            if isnan(w_k_0)
                counter = counter + 1;
                continue;
            end
            %% 比较ZFBF
            
            %% 计算big_phi，（41）
            big_phi=[];
            for kk = 1:M
                for ll = 1:M
                    if kk~=ll
                        big_phi(kk,ll) = -(abs(h_k(ll,:)*w_k_0(:,kk)))^2;
                    else
                        big_phi(kk,ll) = (abs(h_k(kk,:)*w_k_0(:,kk)))^2/gamma_k(kk);
                    end
                end
            end
            q_k_0 = ones(1,M)*pinv(big_phi);
            
            display(strcat('第',num2str(counter),'次求解'))
            iter_num = 0;
            max_iter = 50;
            w_k_l = w_k_0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            iter_num = 0;
            q_k_t = q_k_0;
            %% 计算zeta_l，通过公式（26）
            for kk = 1:M
                fenmu_h_w = 0;
                for ll = 1:M
                    if ll ~= kk
                        fenmu_h_w = fenmu_h_w + q_k_t(ll)*abs(h_k(kk,:)*w_k_0(:,ll))^2;
                    end
                end
                gamma_k(kk) = q_k_t(kk)*abs(h_k(kk,:)*w_k_0(:,kk))^2/(fenmu_h_w+1);
            end
            
            zeta_l = 0;
            for ii = 1:M
                zeta_l = zeta_l + alpha(ii)*( log(1+gamma_k(ii))- qfuncinv(epsilon)/sqrt(n)* sqrt(1-1/(1+gamma_k(ii))^2) ); %%% gamma_k的计算要不要（27）
            end
            zeta_l_save = zeta_l
            zeta_l_bak = zeta_l*rand(1);
            if isnan(w_k_0)
                continue
            end
            while ((zeta_l-zeta_l_bak)/zeta_l_bak)>epsilon2
                abs((zeta_l-zeta_l_bak)/zeta_l_bak)>epsilon2
                %% 计算初始tao_value， （39）
                tao_value = 0;
                for jj = 1:M
                    tao_value = tao_value + alpha(jj)*( log(1+phi_k_t(jj))- qfuncinv(epsilon)/sqrt(n)* xita_k_t(jj));
                end
                
                tao_value_bak = rand(1)*tao_value;
                tao_num = 0;
                while abs(tao_value-tao_value_bak)/tao_value_bak > epsilon2  && tao_num < 10
                    %% 初始化各个参数，第一次求解给出初始点
                    cvx_begin
                    variable q_k(M,1)
                    variable psi_k(1,M)
                    variable phi_k(1,M)
                    variable theta_k(1,M)
                    variable xita_k(1,M)
                    variable a_k_l(M,M)
                    variable b_k_l(M,M)
                    
                    SR_max = 0;
                    for jj = 1:M
                        SR_max = SR_max + alpha(jj)*( log(1+phi_k(jj)) - qfuncinv(epsilon)/sqrt(n)* xita_k(jj) ); %% concave-concave
                    end
                    maximize SR_max;
                    subject to
                    %% 给a_k_l和b_k_l对角线赋值
                    %% 35(b)
                    for jj = 1:M
                        gamma_4<=phi_k(jj);
                        %                         gamma_k_ZFBF(jj)<=phi_k(jj);
                    end
                    
                    %% 35(d)
                    for jj = 1:M
                        theta_k(jj)<=gamma_k_wan(jj);
                    end
                    %% 33(h)
                    for jj = 1:M
                        psi_k(jj)<=1-1/(1+gamma_k_wan(jj))^2;  % 27(f)
                    end
                    for jj = 1:M
                        1-1/(1+gamma_4)^2<=psi_k(jj);  % 27(f)
                    end
                    %%  33(i)
                    sum(q_k) <= Pm  % 27(g)
                    %                     %% 33(j)  %%%%%%%%%% 用哪个公式？(32a) or (39a)
                    %                     for jj = 1:M
                    %                         D/n*log(2) - log(1+phi_k(jj)) + qfuncinv(epsilon)/sqrt(n) *  xita_k(jj) <= 0
                    %                     end
                    %% 37(a)
%                     for kk=1:M
%                         for ll = 1:M
%                             if ll~=kk
%                                 a_k_l(kk,ll) = phi_k_t(kk)*q_k_0(ll);
%                             end
%                         end
%                     end
%                     
%                     for kk=1:M
%                         for ll = 1:M
%                             if ll~=kk
%                                 b_k_l(kk,ll) = phi_k_t(kk)*q_k_0(ll);
%                             end
%                         end
%                     end
%                     
                    for kk=1:M
                        sum_a_h_w = 0;
                        for ll = 1:M
                            if ll ~= kk
                                sum_a_h_w = sum_a_h_w + a_k_l(kk,ll)*norm(h_k(ll,:)*w_k_l(:,kk))^2;
                            end
                        end
                        (sum_a_h_w - (q_k(kk))*norm(h_k(kk,:)*w_k_l(:,kk))^2 + (phi_k(kk)))<=zeros^3;
                    end
                    %% 37(b)
                    for kk=1:M
                        sum_b_h_w = 0;
                        for ll = 1:M
                            if ll ~= kk
                                sum_b_h_w = sum_b_h_w + b_k_l(kk,ll)*norm(h_k(ll,:)*w_k_l(:,kk))^2;
                            end
                        end
                        (-sum_b_h_w + (q_k(kk))*norm(h_k(kk,:)*w_k_l(:,kk))^2 - (theta_k(kk)))<=zeros^3;   %% 是否可以取real
                    end
                    
                    %% 40(c)
                    for jj = 1:M
                        (1-psi_k_t(jj))^2*(1+theta_k(jj))^2 - (1-2*psi_k_t(jj)) - psi_k(jj) <= zeros^3;
                    end
                    %% 40(d)
                    for jj = 1:M
                        0.5*sqrt(psi_k_t(jj)) + psi_k(jj)/2/sqrt(psi_k_t(jj))-xita_k(jj)<=zeros^3;
                    end
                    %% 41(c)---(j)
                    for kk = 1:M
                        for ll = 1:M
                            if kk~=ll
                                big_phi(kk,ll) = -(abs(h_k(ll,:)*w_k_l(:,kk)))^2;
                            else
                                big_phi(kk,ll) = (abs(h_k(kk,:)*w_k_l(:,kk)))^2/gamma2;
                            end
                        end
                    end
                    q_k_wan = ones(1,M)*pinv(big_phi);
                    
                    for kk = 1:M
                        for ll = 1:M
                            if kk ~= ll
                                gamma_4*q_k(ll) + q_k_wan(ll)*phi_k(kk) - a_k_l(kk,ll)<= gamma_4*q_k_wan(ll);                     % 39c
                                gamma_k_wan(kk)*q_k(ll) + Pm*phi_k(kk) - a_k_l(kk,ll)<=Pm*gamma_k_wan(kk);                        % 39d
                                a_k_l(kk,ll)-gamma_k_wan(kk)*q_k(ll) - q_k_wan(ll)*phi_k(kk)<=-gamma_k_wan(kk)*q_k_wan(ll);       % 39e
                                a_k_l(kk,ll)-Pm*phi_k(kk)-gamma_4*q_k(ll) <= -Pm*gamma_4;                                         % 39f
                                
                                gamma_4*q_k(ll) + q_k_wan(ll)*theta_k(kk)- b_k_l(kk,ll)<=q_k_wan(ll)*gamma_4;                     % 39g
                                gamma_k_wan(kk)*q_k(ll) + Pm*theta_k(kk)-b_k_l(kk,ll)<=Pm*gamma_k_wan(kk);                        % 39h
                                b_k_l(kk,ll) - gamma_k_wan(kk)*q_k(ll) - q_k_wan(ll)*theta_k(kk)<= -q_k_wan(ll)*gamma_k_wan(kk);  % 39i
                                b_k_l(kk,ll) - Pm*theta_k(kk)-gamma_4*q_k(ll)<=-Pm*gamma_4;                                       % 39j
                            end
                        end
                    end
                    cvx_end
                    if isnan(psi_k)
                        sprintf('Inf错误')
                        tao_value = nan;
                        tao_value_bak = nan;
                        %                         counter = counter + 1;
                        break;
                        %                         counter = counter + 1;
                    end
                    if ~isreal(psi_k)
                        psi_k_t = real(psi_k);
                        q_k_t = q_k;
                        phi_k_t = phi_k;
                        xita_k_t = xita_k;
                    else
                        psi_k_t = max(psi_k,zeros^2);
                        psi_k_t = min(psi_k,1);
                        q_k_t = q_k;
                        phi_k_t = phi_k;
                        xita_k_t = xita_k;
                    end
                    
                    tao_value_bak = tao_value;
                    tao_value = 0;
                    for jj = 1:M
                        tao_value = tao_value + alpha(jj)*( log(1+phi_k(jj))- qfuncinv(epsilon)/sqrt(n)* xita_k(jj));
                    end
                    tao_num = tao_num + 1;
                end
                if isnan(psi_k)
                        sprintf('Inf错误')
                        break;
                end
                %% 公式（27），计算gamma_k
                %% (31)式，利用q更新w_k_l
                sum_q_h_h = 0;
                for ii = 1:M
                    sum_q_h_h = sum_q_h_h + q_k_t(ii) * h_k(ii,:)'*h_k(ii,:);
                end
                sum_q_h_h_inv = pinv(eye(K)+sum_q_h_h);
                %                 for kk=1:M
                %                     w_k_l(:,kk) =  sum_q_h_h_inv*h_k(kk,:)'/norm(sum_q_h_h_inv*h_k(kk,:)');
                %                 end
                
                for kk=1:M
                    w_k_l(:,kk) =  (h_k(kk,:)*sum_q_h_h_inv/norm(h_k(kk,:)*sum_q_h_h_inv))';
                end
                
                for kk = 1:M
                    fenmu_h_w = 0;
                    for ll = 1:M
                        if ll ~= kk
                            fenmu_h_w = fenmu_h_w + q_k_t(ll)*abs(h_k(ll,:)*w_k_l(:,kk))^2;
                        end
                    end
                    gamma_k(kk) = q_k_t(kk)*abs(h_k(kk,:)*w_k_l(:,kk))^2/(fenmu_h_w+1);
                end
                
                zeta_l_bak = zeta_l;
                zeta_l = 0;
                for ii = 1:M
                    zeta_l = zeta_l + alpha(ii)*( log(1+gamma_k(ii))- qfuncinv(epsilon)/sqrt(n)* sqrt(1-1/(1+gamma_k(ii))^2) ); %%% gamma_k的计算要不要（27）
                end
                
            end
            
            if isnan(w_k_0)
                sprintf('Inf错误')
                counter = counter + 1;
                continue;
            else
                SR_dual_value_record(counter) = zeta_l;
                save_file = strcat('.\fig2_data\nn','_M_',num2str(M),'_第',num2str(counter),'次生成信道数据.mat');
%                 save(save_file,'h_k','h_k_ori');
                counter = counter + 1;
            end
            clear q_k*  w_k* ze* xi* t*
        end
        res_dual{mm,nn} = SR_dual_value_record;
        %         res_ZFBF{mm,nn} = SR_ZFBF_value_record;
        save('param1_dual_res_200_M8.mat','res_dual')
        %         save('param1_ZFBF_res500.mat','res_ZFBF')
        %         clear q_k* s* w_k* ze* xi* t* a* c*
    end
end