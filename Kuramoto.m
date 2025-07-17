% Kuramoto SDE
clear;
close all;
clc;

% Parameters
theta = 0.50;      % Natural frequency
sigma = 0.15;      % Noise intensity
T = 100;           % Total time
dt = 0.01;        % Time step
N = 100;          % Number of oscillators
steps = T/dt;     % Number of time steps

% Initialization
X = rand(N, 1) * 2 * pi; % Initial phases uniformly distributed in [0, 2*pi]
time = 0:dt:T;           % Time vector
X_all = zeros(N, length(time)); % To store phases at each time step
X_all(:, 1) = X;

% Simulation
for t = 2:length(time)
    % Mean-field interaction term
    interaction_term = zeros(N, 1);
    for i = 1:N
        interaction_term(i) = mean(sin(X(i) - X));  % Empirical mean
    end
    
    % Update phases using the stochastic Kuramoto model
    dX = (theta + interaction_term) * dt + sigma * sqrt(dt) * randn(N, 1);
    X = X + dX;
    
    % Store the updated phases
    X_all(:, t) = X;
end

% Plotting the results

% Plot 1: Phases of the oscillators over time
figure;
hold on;
for i = 1:N
    plot(time, mod(X_all(i, :), 2*pi));
end
xlabel('Time');
ylabel('Phase X_t');
title('Phases of Oscillators in the Stochastic Kuramoto Model');
grid on;
hold off;

% Plot 2: Mean phase over time
% mean_phase = mean(X_all, 1);
% figure;
% plot(time, mod(mean_phase, 2*pi), 'LineWidth', 2);
% xlabel('Time');
% ylabel('Mean Phase');
% title('Mean Phase Over Time');
% grid on;
% Plot 3: Standard deviation of phases over time
std_phase = std(X_all, 0, 1);
figure;
plot(time, std_phase, 'LineWidth', 2);
xlabel('Time');
ylabel('Standard Deviation of Phases');
title('Standard Deviation of Phases Over Time');

% Plot 4: Phase distribution at final time
figure;
histogram(mod(X_all(:, end), 2*pi), 20);
xlabel('Phase');
ylabel('Frequency');
title('Phase Distribution at Final Time');
grid on;

% Plot 5: Evolution of one oscillator's phase over time
figure;
plot(time, mod(X_all(1, :), 2*pi), 'LineWidth', 2);
xlabel('Time');
ylabel('Phase X_t');
title('Evolution of One Oscillator''s Phase Over Time');
grid on;
