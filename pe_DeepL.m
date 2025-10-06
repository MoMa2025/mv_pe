
clear;
close all;
clc;
T = 100;                    % total simulation time

d = 100;                     % number of neurons
alpha0 = 0.5;                % initial alpha
beta0 = 0.35;                % initial beta
w_bar = 1;                   % mean weight
b = 0;                        % bias
delta_t_fine = 0.01;          % fine time step
x0 = 0.5*ones(d,1); 
sigma = 0.15;
sigmaObs = 0.15;
f = @(x) 1./(1+exp(-x));








nn_obs = readmatrix('nn_obs_T_100.txt');


L_max = 4;
L_min = 2;

p_max = 5;      % useless now
p_min = 1;      % p_min is always 1.
N_0 = 30;
M_0 = 2;
particleCount = 50;
iterCount = 5000;




% Initialize storage
alpha_iter = zeros(iterCount,1);
beta_iter  = zeros(iterCount,1);
alpha_trace = cell(iterCount,1);
beta_trace = cell(iterCount,1);



Cost_MSA_try_iter = zeros(iterCount, 1);
Time_MSA_try_iter = zeros(iterCount, 1);
used_L_try_iter = zeros(iterCount, 1);
used_p_try_iter = zeros(iterCount, 1);
used_p_prob_try_iter = zeros(iterCount, 1);
used_L_prob_try_iter = zeros(iterCount, 1);

alpha_weighted_iter = zeros(iterCount, 1);
beta_weighted_iter  = zeros(iterCount, 1);

alpha_trace1 = cell(iterCount, 1);  % fine
alpha_trace2 = cell(iterCount, 1);  % coarse
beta_trace1  = cell(iterCount, 1);
beta_trace2  = cell(iterCount, 1);



for i=1:iterCount
    [L, L_density] = sample_l(L_max, L_min);
    delta_t = 2^(-L);
    [p, p_density] = sample_p_given_l(L-1, L_max-1, p_max);
    N = N_0 * 2^(p+2) + 20;
    M = ceil(M_0 * L * L_max + 2);

    used_p_try_iter(i) = p;
    used_L_try_iter(i) = L;
    used_p_prob_try_iter(i) = p_density(p-p_min+1);
    used_L_prob_try_iter(i) = L_density(L-L_min+1);

    if mod(i,1) == 0
        disp(['i = ', num2str(i), ', p = ', num2str(p), ', L = ', num2str(L), ', N = ', num2str(N), ', M = ', num2str(M)]);
    end

    if L == L_min
        % ================== Single-level (fine) ==================
        iter_time_start = tic;

        % law path under current nominal params (seed law)
        X_law = simulate_meanfieldNN(delta_t, T, x0, alpha0, beta0, w_bar, b, sigma);

        % SA trajectories for alpha & beta
        alpha_p = zeros(N+1,1); alpha_p(1) = alpha0;
        beta_p  = zeros(N+1,1); beta_p(1)  = beta0;

        % Optional warm-start conditional path (not required by CPF implementation below)
        % Xs = simulate_meanfieldNN_given_law(delta_t, T, x0, alpha_p(1), beta_p(1), w_bar, b, sigma, X_law);

        for n=1:N
            gamma = get_gamma(n,L);

            % ---------- Use Conditional Particle Filter (CPF) ----------
            % CPF returns one conditional trajectory Xs (d x (steps+1)) and X for obs-handling
            % Signature:
            % [Xs, X] = Conditional_Particle_Filter_NN(delta_t, T, particleCount, X0, Y, ...
            %                                           alpha, beta, sigma, sigmaObs, ref_path, X_law, w_bar, b, f)
            %
            % If you DO have observations, pass nn_obs as Y; otherwise pass [] and ensure
            % the CPF implementation can handle no-observation mode (or use the simulate fallback).
            %
            try
                [Xs, X] = Conditional_Particle_Filter_NN(delta_t, T, particleCount, x0, nn_obs, ...
                    alpha_p(n), beta_p(n), sigma, sigmaObs, Xs, X_law, w_bar, b, f);
            catch
                % If your CPF signature differs, try without ref Xs:
                [Xs, X] = Conditional_Particle_Filter_NN(delta_t, T, particleCount, x0, nn_obs, ...
                    alpha_p(n), beta_p(n), sigma, sigmaObs, [], X_law, w_bar, b, f);
            end
            % ---------- End CPF ----------

            % compute H_l (gradient vector) using the returned conditional trajectory and the law
            H = H_NN(Xs, X_law, sigma, w_bar, b, f, delta_t, alpha_p(n), beta_p(n));

            % update parameters
            alpha_p(n+1) = alpha_p(n) + gamma * H(1);
            beta_p(n+1)  = beta_p(n)  + gamma * H(2);

            % stability guard
            if any(abs(gamma.*H) > 0.1)
                alpha_p(n+1) = alpha_p(n);
                beta_p(n+1)  = beta_p(n);
            end
        end

        % level output with optional Romberg correction
        alpha_level = alpha_p(end);
        beta_level  = beta_p(end);
        if p > p_min
            alpha_level = alpha_level - alpha_p(round(N/2));
            beta_level  = beta_level  - beta_p(round(N/2));
        end

        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = N/delta_t;

        alpha_trace1{i} = alpha_p;
        beta_trace1{i}  = beta_p;

    else
        % ================== Multilevel (fine vs coarse) ==================
        iter_time_start = tic;

        % coupled laws (fine & coarse) under nominal params (for conditional sims)
        [X_1_law, X_2_law] = simulate_coupled_discrete_meanfieldNN(delta_t, T, x0, x0, ...
            alpha0, beta0, w_bar, b, alpha0, beta0, w_bar, b, sigma);

        % SA trajectories
        alpha1 = zeros(N+1,1); alpha2 = zeros(N+1,1);
        beta1  = zeros(N+1,1); beta2  = zeros(N+1,1);
        alpha1(1) = alpha0; alpha2(1) = alpha0;
        beta1(1)  = beta0;  beta2(1)  = beta0;

        % optional warm-start conditional paths for coupled CPF:
        [Xs1, Xs2] = simulate_coupled_discrete_meanfieldNN_given_laws(delta_t, T, x0, x0, ...
            alpha1(1), beta1(1), w_bar, b, alpha2(1), beta2(1), w_bar, b, sigma, X_1_law, X_2_law);

        for n=1:N
            gamma = get_gamma(n,L);

            % ---------- Use Coupled Conditional Particle Filter (CCPF) ----------
            % Signature:
            % [Xs1,X1,Xs2,X2] = Coupled_Conditional_Particle_Filter_NN(delta_t, T, particleCount, ...
            %    X0_1, X0_2, Y, alpha_1, beta_1, alpha_2, beta_2, sigma, sigmaObs, ...
            %    ref_path1, ref_path2, X_1_law, X_2_law, w_bar, b, f)
            %
            try
                [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter_NN(delta_t, T, particleCount, ...
                    x0, x0, nn_obs, alpha1(n), beta1(n), alpha2(n), beta2(n), sigma, sigmaObs, ...
                    Xs1, Xs2, X_1_law, X_2_law, w_bar, b, f);
            catch
                % fallback without providing ref Xs1/Xs2 if your implementation doesn't expect them
                [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter_NN(delta_t, T, particleCount, ...
                    x0, x0, nn_obs, alpha1(n), beta1(n), alpha2(n), beta2(n), sigma, sigmaObs, ...
                    [], [], X_1_law, X_2_law, w_bar, b, f);
            end
            % ---------- End CCPF ----------

            % compute H on the coupled conditional trajectories for fine & coarse
            H1 = H_NN(Xs1, X_1_law, sigma, w_bar, b, f, delta_t, alpha1(n), beta1(n));
            H2 = H_NN(Xs2, X_2_law, sigma, w_bar, b, f, delta_t, alpha2(n), beta2(n));


            % update fine & coarse SA chains
            alpha1(n+1) = alpha1(n) + gamma * H1(1);
            beta1(n+1)  = beta1(n)  + gamma * H1(2);
            alpha2(n+1) = alpha2(n) + gamma * H2(1);
            beta2(n+1)  = beta2(n)  + gamma * H2(2);

            if any(abs(gamma*H1) > 0.1) || any(abs(gamma*H2) > 0.1)
                alpha1(n+1) = alpha1(n); beta1(n+1) = beta1(n);
                alpha2(n+1) = alpha2(n); beta2(n+1) = beta2(n);
            end
        end

        % level difference (fine - coarse) and Romberg correction
        alpha_level = alpha1(end) - alpha2(end);
        beta_level  = beta1(end)  - beta2(end);
        if p > p_min
            alpha_level = alpha_level - (alpha1(round(N/2)) - alpha2(round(N/2)));
            beta_level  = beta_level  - (beta1(round(N/2))  - beta2(round(N/2)));
        end

        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = 3/2*N*(1/delta_t);

        alpha_trace1{i} = alpha1; alpha_trace2{i} = alpha2;
        beta_trace1{i}  = beta1;  beta_trace2{i}  = beta2;
    end

    % store & importance weighting
    alpha_iter(i) = alpha_level;
    beta_iter(i)  = beta_level;

    w_imp = 1./(L_density(L-L_min+1)*p_density(p-p_min+1));
    alpha_weighted_iter(i) = alpha_level * w_imp;
    beta_weighted_iter(i)  = beta_level  * w_imp;

 
end

estimated_alpha = mean(alpha_weighted_iter, 1);
estimated_beta  = mean(beta_weighted_iter, 1);







% MSE evaluation for different batch sizes
MSE_Ms = [8, 16, 32, 64, 128];  % number of iterations per block (2^k with 3<=k<=12)
MSEs_alpha = zeros(length(MSE_Ms),1);
MSEs_beta  = zeros(length(MSE_Ms),1);
Costs      = zeros(length(MSE_Ms),1);

total_iters = length(alpha_weighted_iter);


alpha_proxy = estimated_alpha;
beta_proxy = estimated_beta;

for i = 1:length(MSE_Ms)
    M = MSE_Ms(i);
    mse_alpha_sum = 0;
    mse_beta_sum  = 0;
    avg_cost_time = 0;

    % compute number of available batches (include partial last one)
    num_batches = ceil(total_iters / M);

    for j = 1:num_batches
        start_idx = (j-1)*M + 1;
        end_idx   = min(j*M, total_iters);  % clamp to avoid exceeding length

        % average parameter estimates over batch
        alpha_M = mean(alpha_weighted_iter(start_idx:end_idx));
        beta_M  = mean(beta_weighted_iter(start_idx:end_idx));

        % cost/time for this block
        cost_time_block = sum(Time_MSA_try_iter(start_idx:end_idx));

        % squared error vs ground truth
        %mse_alpha_sum = mse_alpha_sum + (alpha_M - alpha_true)^2;
        %mse_beta_sum  = mse_beta_sum  + (beta_M - beta_true)^2;
        mse_alpha_sum = mse_alpha_sum + (alpha_M - alpha_proxy)^2;
        mse_beta_sum  = mse_beta_sum  + (beta_M - beta_proxy)^2;

        avg_cost_time = avg_cost_time + cost_time_block;
    end

    % average over batches
    MSEs_alpha(i) = mse_alpha_sum / num_batches;
    MSEs_beta(i)  = mse_beta_sum  / num_batches;
    Costs(i)      = avg_cost_time / num_batches;
end

% --- Plot results ---
figure;
loglog( MSEs_alpha, Costs, '-o','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','m');
xlabel('MSE','FontSize',14); ylabel('Cost','FontSize',14);
title('\alpha','FontSize',50);
grid on;
set(gca, 'FontSize',14, 'Linewidth', 1.5);


figure;
loglog( MSEs_beta, Costs, '-o','LineWidth',2,'MarkerSize',8,'MarkerFaceColor','m');
xlabel('MSE'); ylabel('Cost');
title('\beta','FontSize',50);
grid on;
set(gca, 'FontSize',14, 'Linewidth', 1.5);






%% functions

function  indx  = resampleSystematic(w)
    N = length(w);
    Q = cumsum(w);
    indx = zeros(1,N);
    T = linspace(0,1-1/N,N) + rand(1)/N;
    T(N+1) = 1;
    i=1; j=1;
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j; i=i+1;
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


function X = simulate_meanfieldNN(delta_t, T, X0, alpha, beta, w, b, sigma)
    d = length(X0);
    steps_count = round(T/delta_t);
    X = zeros(d, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(d, steps_count);

    f = @(x) 1 ./ (1 + exp(-x));  % sigmoid activation

    for i = 1:steps_count
        Xi = X(:,i);
        mean_act = mean(f(Xi));
        drift = alpha * (mean_act - f(Xi)) + beta * (w * Xi - b);
        X(:,i+1) = Xi + drift * delta_t + sigma * delta_W(:,i);
    end
end


function X = simulate_meanfieldNN_given_law(delta_t, T, X0, alpha, beta, w, b, sigma, X_law)
    d = length(X0);
    steps_count = round(T/delta_t);
    X = zeros(d, steps_count+1);
    X(:,1) = X0;
    delta_W = sqrt(delta_t)*randn(d, steps_count);

    f = @(x) 1 ./ (1 + exp(-x));

    for i = 1:steps_count
        Xi = X(:,i);
        mean_act = mean(f(X_law(:,i)));   % use provided law
        drift = alpha * (mean_act - f(Xi)) + beta * (w * Xi - b);
        X(:,i+1) = Xi + drift * delta_t + sigma * delta_W(:,i);
    end
end


function [X_1, X_2] = simulate_coupled_discrete_meanfieldNN(delta_t, T, X0_1, X0_2, ...
                                                            alpha_1, beta_1, w_1, b_1, ...
                                                            alpha_2, beta_2, w_2, b_2, sigma)
    % Coupled fine (delta_t) vs coarse (2*delta_t) simulators for mean-field NN model
    d = numel(X0_1);     
    steps_coarse = round(T/(2*delta_t));  
    steps_fine   = 2*steps_coarse;

    X_1 = zeros(d, steps_fine+1);
    X_2 = zeros(d, steps_coarse+1);

    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;

    delta_W = sqrt(delta_t)*randn(d, steps_fine);
    f = @(x) 1 ./ (1 + exp(-x));

    for i = 1:steps_fine
        % fine update
        Xi = X_1(:,i);
        mean_act = mean(f(Xi));
        drift = alpha_1 * (mean_act - f(Xi)) + beta_1 * (w_1 * Xi - b_1);
        X_1(:,i+1) = Xi + drift * delta_t + sigma * delta_W(:,i);

        % coarse update every 2 fine steps
        if mod(i,2) == 0
            j = i/2;
            Xj = X_2(:,j);
            mean_act2 = mean(f(Xj));
            drift2 = alpha_2 * (mean_act2 - f(Xj)) + beta_2 * (w_2 * Xj - b_2);
            X_2(:,j+1) = Xj + drift2 * (2*delta_t) + sigma * (delta_W(:,i) + delta_W(:,i-1));
        end
    end
end


function [X_1, X_2] = simulate_coupled_discrete_meanfieldNN_given_laws(delta_t, T, X0_1, X0_2, ...
                                                                       alpha_1, beta_1, w_1, b_1, ...
                                                                       alpha_2, beta_2, w_2, b_2, ...
                                                                       sigma, X_1_law, X_2_law)
    d = numel(X0_1);     
    steps_coarse = round(T/(2*delta_t));  
    steps_fine   = 2*steps_coarse;

    X_1 = zeros(d, steps_fine+1);
    X_2 = zeros(d, steps_coarse+1);

    X_1(:,1) = X0_1;
    X_2(:,1) = X0_2;

    delta_W = sqrt(delta_t)*randn(d, steps_fine);
    f = @(x) 1 ./ (1 + exp(-x));

    for i = 1:steps_fine
        % fine update using law for mean-field term
        Xi = X_1(:,i);
        mean_act = mean(f(X_1_law(:,i)));
        drift = alpha_1 * (mean_act - f(Xi)) + beta_1 * (w_1 * Xi - b_1);
        X_1(:,i+1) = Xi + drift * delta_t + sigma * delta_W(:,i);

        % coarse update every 2 fine steps
        if mod(i,2) == 0
            j = i/2;
            Xj = X_2(:,j);
            mean_act2 = mean(f(X_2_law(:,j)));
            drift2 = alpha_2 * (mean_act2 - f(Xj)) + beta_2 * (w_2 * Xj - b_2);
            X_2(:,j+1) = Xj + drift2 * (2*delta_t) + sigma * (delta_W(:,i) + delta_W(:,i-1));
        end
    end
end



function [Xs, X] = Conditional_Particle_Filter_NN(delta_t, T, particleCount, X0, Y, ...
    alpha, beta, sigma, sigmaObs, ref_path, X_law, w_bar, b, f)

    % Number of steps for the fine simulation
    N = round(T/delta_t);
    d = numel(X0);

    % Initialize particle arrays
    Xs = zeros(particleCount, N+1);  % neuron 1 trajectory for weights
    X  = zeros(particleCount, T+1);  % neuron 1 at integer times for observations

    % Precompute reference at integer times if available
    if isempty(ref_path)
        ref_at_unit = zeros(1, T+1);  % placeholder
        use_ref = false;
    else
        % Ensure indices are integers
        idx = round((0:T)/delta_t + 1);
        idx(idx > size(ref_path, 2)) = size(ref_path, 2); % clamp to max
        ref_at_unit = ref_path(idx);
        use_ref = true;
    end

    % Initialize particles
    X(:,1) = X0(1);
    Xs(:,1) = X0(1);
    log_W = zeros(particleCount,1);

    for i = 1:T
        % Indices in fine grid corresponding to this integer step
        fine_start = round((i-1)/delta_t + 1);
        fine_end   = round(i/delta_t + 1);

        % simulate one unit of time using the law
        if use_ref
            law_slice = X_law(:, fine_start:fine_end);
        else
            law_slice = X_law(:, fine_start:fine_end); % still needed for drift
        end

        A = simulate_meanfieldNN_given_law(delta_t, 1, X0, alpha, beta, w_bar, b, sigma, law_slice);
        B = A(1, end);  % neuron 1 at next integer

        % Overwrite conditional path if available
        if use_ref
            A(end,:) = ref_path(:, fine_start:fine_end);
            B = ref_at_unit(i+1);
        end

        % Store neuron 1 only
        Xs(:, fine_start:fine_end) = repmat(A(1,:), particleCount, 1);
        X(:, i+1) = B;

        % Update weights from observation
        if ~isempty(Y)
            log_w = log_normpdf(Y(i), X(:,i+1), sigmaObs);
            log_W = log_W + log_w;
            W = exp(log_W - max(log_W)); 
            W = W / sum(W);

            % Resample particles
            I = resampleSystematic(W);
            Xs = Xs(I,:); X = X(I,:);
            if use_ref
                Xs(end,:) = ref_at_unit;  % keep conditional path in last particle
                X(end,:)  = ref_at_unit;
            end
            log_W = zeros(particleCount,1);
        end

        % Update X0 for next unit step
        X0(1) = Xs(1,end);
    end

    % Return one particle (conditional)
    index = randsample(particleCount, 1);
    Xs = Xs(index,:);
    X  = X(index,:);
end


function [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter_NN(delta_t, T, particleCount, ...
    X0_1, X0_2, Y, alpha_1, beta_1, alpha_2, beta_2, sigma, sigmaObs, ref_path1, ref_path2, ...
    X_1_law, X_2_law, w_bar, b, f)

    N1 = round(T/delta_t);
    N2 = round(T/(2*delta_t));

    Xs1 = zeros(particleCount, N1+1); 
    X1  = zeros(particleCount, T+1);
    Xs2 = zeros(particleCount, N2+1); 
    X2  = zeros(particleCount, T+1);

    % Safely handle empty reference paths
    if isempty(ref_path1)
        ref1_unit = zeros(1, T+1);
        use_ref1 = false;
    else
        idx1 = round((0:T)/delta_t + 1);
        idx1(idx1 > size(ref_path1,2)) = size(ref_path1,2); % clamp
        ref1_unit = ref_path1(idx1);
        use_ref1 = true;
    end

    if isempty(ref_path2)
        ref2_unit = zeros(1, T+1);
        use_ref2 = false;
    else
        idx2 = round((0:T)/(2*delta_t) + 1);
        idx2(idx2 > size(ref_path2,2)) = size(ref_path2,2); % clamp
        ref2_unit = ref_path2(idx2);
        use_ref2 = true;
    end

    X1(:,1)  = X0_1(1); Xs1(:,1) = X0_1(1);
    X2(:,1)  = X0_2(1); Xs2(:,1) = X0_2(1);

    log_W1 = zeros(particleCount,1);
    log_W2 = zeros(particleCount,1);

    for i = 1:T
        % Fine & coarse law slices
        fine_start  = round((i-1)/delta_t + 1);
        fine_end    = round(i/delta_t + 1);
        coarse_start = round((i-1)/(2*delta_t) + 1);
        coarse_end   = round(i/(2*delta_t) + 1);

        [A1, A2] = simulate_coupled_discrete_meanfieldNN_given_laws( ...
            delta_t, 1, X0_1, X0_2, alpha_1, beta_1, w_bar, b, alpha_2, beta_2, w_bar, b, sigma, ...
            X_1_law(:, fine_start:fine_end), X_2_law(:, coarse_start:coarse_end));

        B1 = A1(1,end);  % neuron 1 next integer
        B2 = A2(1,end);

        % overwrite conditional path if available
        if use_ref1
            A1(end,:) = ref_path1(:, fine_start:fine_end);
            B1 = ref1_unit(i+1);
        end
        if use_ref2
            A2(end,:) = ref_path2(:, coarse_start:coarse_end);
            B2 = ref2_unit(i+1);
        end

        Xs1(:, fine_start:fine_end) = repmat(A1(1,:), particleCount, 1);
        X1(:, i+1) = B1;

        Xs2(:, coarse_start:coarse_end) = repmat(A2(1,:), particleCount, 1);
        X2(:, i+1) = B2;

        % Coupled resampling
        log_w1 = log_normpdf(Y(i), X1(:, i+1), sigmaObs);
        log_W1 = log_W1 + log_w1;
        W1 = exp(log_W1 - max(log_W1)); W1 = W1 / sum(W1);

        log_w2 = log_normpdf(Y(i), X2(:, i+1), sigmaObs);
        log_W2 = log_W2 + log_w2;
        W2 = exp(log_W2 - max(log_W2)); W2 = W2 / sum(W2);

        alp = sum(min(W1,W2));
        U = rand();
        if U < alp
            I1 = resampleSystematic(min(W1,W2)/alp);
            I2 = I1;
        else
            I1 = resampleSystematic((W1-min(W1,W2))/(1-alp));
            I2 = resampleSystematic((W2-min(W1,W2))/(1-alp));
        end

        Xs1 = Xs1(I1,:); X1 = X1(I1,:);
        Xs2 = Xs2(I2,:); X2 = X2(I2,:);

        if use_ref1
            Xs1(end,:) = ref1_unit;
            X1(end,:)  = ref1_unit;
        end
        if use_ref2
            Xs2(end,:) = ref2_unit;
            X2(end,:)  = ref2_unit;
        end

        log_W1 = zeros(particleCount,1);
        log_W2 = zeros(particleCount,1);
    end

    % Return conditional particle
    index = 1;
    Xs1 = Xs1(index,:); X1 = X1(index,:);
    Xs2 = Xs2(index,:); X2 = X2(index,:);
end





%%%%%% H function, grad of discrete likelihood
function H = H_NN(Xs, X_law, sigma, w_bar, b, f, delta_t, alpha, beta)
[d, Tplus1] = size(Xs);
T = Tplus1-1;
sum_alpha = 0; sum_beta = 0; sum_alpha_2 = 0; sum_beta_2 = 0;
for k=2:T
mean_f_law = mean( f(X_law(:,k)) );
vec_alpha = mean_f_law - f(Xs(:,k));
vec_beta = w_bar*Xs(:,k) - b;
A = vec_alpha.*(alpha*vec_alpha + beta*vec_beta);
sum_alpha = sum_alpha + sum(A);
B = vec_beta.*(alpha*vec_alpha + beta*vec_beta);
sum_beta = sum_beta + sum(B);

 sum_alpha_2 = sum_alpha_2 + vec_alpha' * (Xs(:,k) - Xs(:,k-1));
 sum_beta_2  = sum_beta_2  + vec_beta'  * (Xs(:,k) - Xs(:,k-1));


end

H_alpha = -delta_t/(sigma^2) * sum_alpha - sum_alpha_2/(sigma^2);
H_beta = -delta_t/(sigma^2) * sum_beta - sum_beta_2/(sigma^2);
H = [H_alpha, H_beta];
end




