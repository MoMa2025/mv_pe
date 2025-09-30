
clear;
close all;
clc;
T = 30;

% to simulate new data uncomment this
%{
sigma = 0.15;
theta = 1;
sigma_obs = 0.15;
delta_t_fine = 1/2^10;
particle_count_fine = 1000;
x0 = 1;
X_0_fine = x0*ones(particle_count_fine, 1);           % assumption: X0 is constant.
kuramoto_exact_all = simulate_modkuramoto(delta_t_fine, T, X_0_fine, theta, sigma);
kuramoto_exact = kuramoto_exact_all(1,:);
kuramoto_obs = kuramoto_exact(1+1/delta_t_fine:1/delta_t_fine:T/delta_t_fine+1) + sigma_obs * randn(1,T);

writematrix(kuramoto_obs, 'kuramoto_obs_T_30_S3_L8.txt');
writematrix(kuramoto_exact, 'kuramoto_exact_T_30_S3_L8.txt');
%}


kuramoto_obs = readmatrix('kuramoto_obs_T_30_S3.txt');
L_max = 4;
L_min = 2;

p_max = 5;      
p_min = 1;      
N_0 = 30;
M_0 = 2;
particleCount = 50;
iterCount = 5000; % this is the maximum \bar{M} in the numerical section.
theta0 = 0.5;
x0 = 1;
sigma = 0.15;
sigmaObs = 0.15;

Cost_MSA_try_iter = zeros(iterCount, 1);
Time_MSA_try_iter = zeros(iterCount, 1);
used_L_try_iter = zeros(iterCount, 1);
used_p_try_iter = zeros(iterCount, 1);
used_p_prob_try_iter = zeros(iterCount, 1);
used_L_prob_try_iter = zeros(iterCount, 1);
theta_iter = zeros(iterCount, 1);
theta_weighted_iter = zeros(iterCount, 1);
%iterCounts = zeros(1, L_max - L_start + 1);
theta_trace1 = cell(iterCount, 1);
theta_trace2 = cell(iterCount, 1);
theta_p_trace1 = cell(iterCount, 1);
theta_p_trace2 = cell(iterCount, 1);
theta_trace = cell(iterCount, 1);
theta_p_trace = cell(iterCount, 1);


for i=1:iterCount
    [L, L_density] = sample_l(L_max, L_min);
    delta_t = 2^(-L);
    [p, p_density] = sample_p_given_l(L-1, L_max-1, p_max);
    %p = 3;  % adhoc value, change it later.
    N = N_0 * 2^(p+2) + 20;
    M = ceil(M_0 * L * L_max + 2);
    used_p_try_iter(i) = p;
    used_L_try_iter(i) = L;
    used_p_prob_try_iter(i) = p_density(p-p_min+1);
    used_L_prob_try_iter(i) = L_density(L-L_min+1);
    theta_level = 0;
    if mod(i,1) == 0
        disp(['i = ', num2str(i), ', p = ', num2str(p), ', L = ', num2str(L), ', N = ', num2str(N), ', M = ', num2str(M)]);
    end
    if L == L_min
        iter_time_start = tic;
        theta_p = zeros(N+1,1);
        theta_p(1,:) = theta0;
        theta = theta_p(1,1);
        X0_m = x0 * ones(M, 1);
        X_law = simulate_modkuramoto(delta_t, T, X0_m, theta, sigma);

        Xs = simulate_modkuramoto_given_law(delta_t, T, x0, theta, sigma, X_law);
        for n=1:N
            gamma = get_gamma(n,L);
            theta = theta_p(n,1);
            [Xs,X] = Conditional_Particle_Filter(delta_t, T, particleCount, x0, kuramoto_obs, theta, sigma, sigmaObs, Xs, X_law);
            h = H(Xs, X, kuramoto_obs, theta_p(n), sigma, delta_t, X_law);
            theta_p(n+1,:) = theta_p(n,:) + gamma .* h;            
            if any(abs(gamma.*h) > 0.1)
                theta_p(n+1,:) = theta_p(n,:);
            end
        end
        theta_p_level =  theta_p(end,:);
        if p > p_min
            theta_p_level =  theta_p_level - theta_p(round(N/2),:);
        end
        theta_level = theta_p_level;
        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = N/delta_t;
        theta_p_trace{i,1} = theta_p;
    else
        iter_time_start = tic;
        theta1 = zeros(N+1,1);
        theta2 = zeros(N+1,1);
        theta1(1) = theta0;
        theta2(1) = theta0;
        X0_m = x0 * ones(M, 1);
        [X_1_law,X_2_law] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_m, X0_m, theta, theta, sigma);
        [Xs1, Xs2] = simulate_coupled_discrete_modkuramoto_given_laws(delta_t, T, x0, x0, theta, theta, sigma, X_1_law, X_2_law);
        for n=1:N
            [Xs1,X1,Xs2,X2] = Coupled_Conditional_Particle_Filter(delta_t, T, particleCount, x0, x0, ...
                kuramoto_obs, theta1(n), theta2(n), sigma, sigmaObs, Xs1, Xs2, X_1_law, X_2_law);
            gamma = get_gamma(n,L);
            h1 = H(Xs1, X1, kuramoto_obs, theta1(n), sigma, delta_t, X_1_law);
            theta1(n+1) = theta1(n) + gamma * h1;
            h2 = H(Xs2, X2, kuramoto_obs, theta2(n), sigma, 2*delta_t, X_2_law);
            theta2(n+1) = theta2(n) + gamma * h2;
            if abs(gamma*h1) > 0.1 || abs(gamma*h2) > 0.1
                theta1(n+1) = theta1(n);
                theta2(n+1) = theta2(n);
            end

        end
        theta_level = theta1(end) - theta2(end);
        % if abs(theta1(end) - exact_theta) > 0.5
        %     disp(["theta1 exceeded threshold. theta1 = ", num2str(theta1(end)), " L = ", num2str(L), " i = ", num2str(i)]);
        % end
        % if abs(theta2(end) - exact_theta) > 0.5
        %     disp(["theta2 exceeded threshold. theta1 = ", num2str(theta2(end)), " L = ", num2str(L), " i = ", num2str(i)]);
        % end

        if p > p_min
            theta_level = theta_level - (theta1(round(N/2)) - theta2(round(N/2)));
        end
        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = 3/2*N*(1/delta_t);
        theta_trace1{i,1} = theta1;
        theta_trace2{i,1} = theta2;
    end
    theta_iter(i,:) = theta_level; 
    theta_weighted_iter(i,:) = theta_level/(L_density(L-L_min+1)*p_density(p-p_min+1));
    disp(['theta_level = ', num2str(abs(theta_level))]);
    disp(['current final estimate = ', num2str(abs(mean(theta_weighted_iter(1:i), 1)))])
end

estimated_theta = abs(mean(theta_weighted_iter, 1));
% Define block sizes for averaging
 MSE_Ms = [8, 16, 32, 64, 128];  % number of iterations per block (2^k with 3<=k<=12)
MSEs = zeros(length(MSE_Ms),1);
Costs = zeros(length(MSE_Ms),1);
theta_star = -estimated_theta;



for i = 1:length(MSE_Ms)
    M = MSE_Ms(i);
    numBlocks = floor(iterCount / M);  % number of complete blocks
    s = 0;
    avg_cost = 0;
    
    for j = 1:numBlocks
        idx_start = (j-1)*M + 1;
        idx_end   = j*M;
        
        % Average parameter estimate over block
        theta_block = mean(theta_weighted_iter(idx_start:idx_end));
        
        % Block cost = sum of iteration times
        block_cost = sum(Time_MSA_try_iter(idx_start:idx_end));
        
        s = s + (theta_block - theta_star)^2;
        avg_cost = avg_cost + block_cost;
    end
    
    MSEs(i) = s / numBlocks;
    Costs(i) =  avg_cost / numBlocks;  % average cost per block
end

% Log-log plot: MSE vs computational cost
figure;
loglog(MSEs, Costs, '-o','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','m');
xlabel('MSE','FontSize',14);
ylabel('Cost','FontSize',14);
title('\theta','FontSize',50);
grid on;
set(gca, 'FontSize',14, 'Linewidth', 1.5);






%% functions

function  indx  = resampleSystematic(w)
    N = length(w);
    Q = cumsum(w);
    indx = zeros(1,N);
    T = linspace(0,1-1/N,N) + rand(1)/N;
    T(N+1) = 1;
    i=1;
    j=1;
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j;
            i=i+1;
        else
            j=j+1;        
        end
    end
end


function a = log_normpdf(x,m,s)
    a = -log(2*pi)/2 - log(s) - (x-m).^2/(2*s^2);
end


function [l, density] = sample_l(L_max, L_min)
    %density = 2.^(-1*(L_min:L_max)).*((L_min:L_max)+1).*log2((L_min:L_max)+2).^2;
    epsilon = 0.2;
    density = 2.^(-1/epsilon/2*(L_min:L_max));
    density = density / sum(density);
    l = randsample(L_max - L_min + 1, 1, true, density);
    l = l + L_min - 1;
end


function gamma = get_gamma(n,L)
    n_threhold = 1;
    if n < n_threhold
        n = n_threhold;
    end
    C = 3*1e-3;
    gamma =  C * 1/((n - (n_threhold-1) + 100)^(1 - 0.0));
end


function [p, density] = sample_p_given_l(l, L, p_max)
    density = zeros(1,L);
    %for i = 1:min(4, L-l+1)
    for i = 1:min(4, L-l+1) 
        density(i) = 2^(4-i);
    end
    for i = (min(4, L-l+1)+1):(L-l+1)
        density(i) = 2^(-i)*i*(log(i))^2;
    end
    density = density / sum(density);
    p = randsample(1:L, 1, true, density);
end


function X = simulate_modkuramoto(delta_t, T, X0, theta, sigma)
    particle_count = length(X0);
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count);
    for i = 1:steps_count
        X(:,i+1) = X(:,i) + (theta + mean(sin(X(:,i) - X(:,i)'),2)) * delta_t + ...
                          + sigma * delta_W(:,i);
    end
end

function X = simulate_modkuramoto_given_law(delta_t, T, X0, theta, sigma, X_law)
    particle_count = length(X0);
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count);
    for i = 1:steps_count
        X(:,i+1) = X(:,i) + (theta + mean(sin(X(:,i) - X_law(:,i)'),2)) * delta_t + ...
                          + sigma * delta_W(:,i);
    end
end


function X = simulate_kuramoto_given_noise_and_law(delta_t, T, X0, theta, sigma, delta_W, X_law)
    particle_count = length(X0);
    steps_count = round(T/delta_t);
    X = zeros(particle_count, steps_count+1);
    X(:,1) = X0;
    for i = 1:steps_count
        X(:,i+1) = X(:,i) + (theta + mean(sin(X(:,i) - X_law(:,i)'),2)) * delta_t + ...
                          + sigma * delta_W(:,i);
    end
end


function [X_1, X_2] = simulate_coupled_discrete_modkuramoto(delta_t, T, X0_1, X0_2, theta_1, theta_2, sigma)
    particle_count = length(X0_1);     % is this good?
    steps_count_2 = round(T/delta_t/2);       % suppose it is an integer.
    steps_count_1 = 2*steps_count_2;
    X_1 = zeros(particle_count, steps_count_1+1);
    X_2 = zeros(particle_count, steps_count_2+1);
    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count_1);
    for i = 2:steps_count_1+1
        X_1(:,i) = X_1(:,i-1) + (theta_1 + mean(sin(X_1(:,i-1) - X_1(:,i-1)'),2)) * delta_t + ...
                          + sigma * delta_W(:,i-1);
        if mod(i,2) == 1
            j = ceil(i/2);
            X_2(:,j) = X_2(:,j-1) + (theta_2 + mean(sin(X_2(:,j-1) - X_2(:,j-1)'),2)) * 2*delta_t + ...
                          + sigma * (delta_W(:,i-1) + delta_W(:,i-2));         
        end
    end
end

function [X_1, X_2] = simulate_coupled_discrete_modkuramoto_given_laws(delta_t, T, X0_1, X0_2, theta_1, theta_2, sigma, X_1_law, X_2_law)
    particle_count = length(X0_1);     % is this good?
    steps_count_2 = round(T/delta_t/2);       % suppose it is an integer.
    steps_count_1 = 2*steps_count_2;
    X_1 = zeros(particle_count, steps_count_1+1);
    X_2 = zeros(particle_count, steps_count_2+1);
    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;
    delta_W = sqrt(delta_t)*randn(particle_count, steps_count_1);
    for i = 2:steps_count_1+1
        X_1(:,i) = X_1(:,i-1) + (theta_1 + mean(sin(X_1(:,i-1) - X_1_law(:,i-1)'),2)) * delta_t + ...
                          + sigma * delta_W(:,i-1);
        if mod(i,2) == 1
            j = ceil(i/2);
            X_2(:,j) = X_2(:,j-1) + (theta_2 + mean(sin(X_2(:,j-1) - X_2_law(:,j-1)'),2)) * 2*delta_t + ...
                          + sigma * (delta_W(:,i-1) + delta_W(:,i-2));         
        end
    end
end


function [Xs, X] = Conditional_Particle_Filter(delta_t, T, particleCount, X0, Y, theta, sigma, sigmaObs, ref_path, X_law)
    N = T/delta_t;
    Xs = zeros(particleCount,N+1);
    X = zeros(particleCount,T+1);
    ref_path_at_unit_times = ref_path((0:T)/delta_t+1);
    log_like = 0;

    X(:,1) = X0;
    Xs(:,1) = X0;
    log_W = zeros(particleCount,1);
    %W = ones(particleCount,1);
    for i=1:T
        loop_X_laws = X_law(:,((i-1)/delta_t+1):(i/delta_t+1));
        A = simulate_modkuramoto_given_law(delta_t, 1, X(:,i), theta, sigma, loop_X_laws);
        B = A(:,(1/delta_t+1));
        A(end,:) = ref_path(((i-1)/delta_t+1):(i/delta_t+1));
        B(end,:) = [ref_path_at_unit_times(i+1)];
        Xs(:,((i-1)/delta_t+1):(i/delta_t+1)) = A;
        X(:,i+1) = B(:,end);

        log_w = log_normpdf(Y(i), X(:,i+1), sigmaObs);
        log_W = log_W + log_w;
        W = exp(log_W - max(log_W));
        W = W/sum(W);
        
        I = resampleSystematic(W);
        Xs = Xs(I,:);
        X = X(I,:);
        Xs(end,:) = ref_path;
        X(end,:) = ref_path_at_unit_times;
        log_W = zeros(particleCount,1);
    end
    index = randsample(particleCount, 1, true, W);
    Xs = Xs(index,:);
    X = X(index,:);
end

function [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter(delta_t, T, particleCount, X0_1, X0_2, Y, theta_1, theta_2, sigma, sigmaObs, ref_path1, ref_path2, X_1_law, X_2_law)
    N1 = T/delta_t;
    N2 = T/(2*delta_t);
    Xs1 = zeros(particleCount,N1+1);
    X1 = zeros(particleCount,T+1);
    Xs2 = zeros(particleCount,N2+1);
    X2 = zeros(particleCount,T+1);

    ref_path_1_at_unit_times = ref_path1((0:T)/delta_t+1);
    ref_path_2_at_unit_times = ref_path2((0:T)/(2*delta_t)+1);

    X1(:,1) = X0_1;
    Xs1(:,1) = X0_1;
    X2(:,1) = X0_2;
    Xs2(:,1) = X0_2;
    %W1 = ones(particleCount,1)/particleCount;
    %W2 = ones(particleCount,1)/particleCount;
    log_W1 = zeros(particleCount,1);
    log_W2 = zeros(particleCount,1);
    for i=1:T
        loop_X_1_laws = X_1_law(:,((i-1)/delta_t+1):(i/delta_t+1));
        loop_X_2_laws = X_2_law(:,((i-1)/delta_t/2+1):(i/delta_t/2+1));
        [A1, A2] = simulate_coupled_discrete_modkuramoto_given_laws(delta_t, 1, X1(:,i), X2(:,i), theta_1, theta_2, sigma, loop_X_1_laws, loop_X_2_laws);
        B1 = A1(:,(1/delta_t+1));
        B2 = A2(:,(1/delta_t/2+1));

        A1(end,:) = ref_path1(((i-1)/delta_t+1):(i/delta_t+1));
        B1(end,:) = [ref_path_1_at_unit_times(i+1)];
        Xs1(:,((i-1)/delta_t+1):(i/delta_t+1)) = A1;
        X1(:,i+1) = B1(:,end);

        A2(end,:) = ref_path2(((i-1)/(2*delta_t)+1):(i/(2*delta_t)+1));
        B2(end,:) = [ref_path_2_at_unit_times(i+1)];
        Xs2(:,((i-1)/(2*delta_t)+1):(i/(2*delta_t)+1)) = A2;
        X2(:,i+1) = B2(:,end);


        % resampling first for the previous step
        log_w1 = log_normpdf(Y(i), X1(:,i+1), sigmaObs);
        log_W1 = log_W1 + log_w1;
        W1 = exp(log_W1 - max(log_W1));
        W1 = W1/sum(W1);
        log_w2 = log_normpdf(Y(i), X2(:,i+1), sigmaObs);
        log_W2 = log_W2 + log_w2;
        W2 = exp(log_W2 - max(log_W2));
        W2 = W2/sum(W2);

        I1 = zeros(particleCount,1);
        I2 = zeros(particleCount,1);
        alpha = sum(min(W1,W2));
        U = rand();
        if U < alpha
            I1 = resampleSystematic(min(W1,W2)/alpha);
            I2 = I1;
        else
            I1 = resampleSystematic((W1-min(W1,W2))/(1-alpha));
            I2 = resampleSystematic((W2-min(W1,W2))/(1-alpha));
        end

        Xs1 = Xs1(I1,:);
        X1 = X1(I1,:);
        Xs1(end,:) = ref_path1;
        X1(end,:) = ref_path_1_at_unit_times;
        Xs2 = Xs2(I2,:);
        X2 = X2(I2,:);
        Xs2(end,:) = ref_path2;
        X2(end,:) = ref_path_2_at_unit_times;
        log_W1 = zeros(particleCount,1);
        log_W2 = zeros(particleCount,1);
    end    
    index = 1;
    Xs1 = Xs1(index,:);
    X1 = X1(index,:);
    %index = randsample(particleCount, 1, true, W2);
    Xs2 = Xs2(index,:);
    X2 = X2(index,:);
end






%%%%%% H function, grad of discrete likelihood
function h = H(Xs, X, Y, theta, sigma, delta_t, X_law)
    %h = -delta_t/sigma^2 * sum(theta + mean(sin(Xs(:,:) - X_law(:,:)),1)) - 1/sigma^2 * sum((Xs(2:end)-Xs(1:end-1)));
    sum_of_means = 0;
    for j = 1:length(Xs)
        sum_of_means = sum_of_means + mean(sin(Xs(j) - X_law(:,j)));
    end
    h = -theta*length(X)/sigma^2 -delta_t/sigma^2 * sum_of_means - 1/sigma^2 * ((Xs(end)-Xs(1)));
end


