% McKean-Vlasov SDE for Neural Networks Simulation
clear;
close all;
clc;

% Parameters
alpha = 0.5;       % Coupling strength (mean-field interaction influence)
beta = 0.35;       % Rate of change in activation (self-input influence)
sigma = 0.15;      % Noise intensity (stochastic fluctuations)
N = 100;           % Number of neurons in the network
T = 100;           % Total simulation time
dt = 0.01;         % Time step for numerical integration
steps = T / dt;    % Total number of time steps

% Initialization
X_initial = 2 * pi * rand(N, 1); % Initial activations uniformly distributed in [0, 2*pi]
time_vector = 0:dt:T;             % Time vector for plotting
X_all = zeros(N, length(time_vector)); % Matrix to store activations at each time step
X_all(:, 1) = X_initial;          % Store initial activations

% Random weights and biases for each neuron
weights = randn(N, 1); % Random weights drawn from a normal distribution
biases = randn(N, 1);  % Random biases drawn from a normal distribution

% Simulation Loop
for t = 2:length(time_vector)
    % Compute mean-field interaction: average of sine of activations
    mean_field_interaction = mean(sin(bsxfun(@minus, X_all(:, t-1), X_all(:, t-1)')), 2);
    
    % Update activations using the stochastic McKean-Vlasov model
    % Discretization scheme for the SDE
    dX = (alpha * (mean_field_interaction - sin(X_all(:, t-1))) + ...
          beta * (weights .* X_all(:, t-1) - biases)) * dt + ...
          sigma * sqrt(dt) * randn(N, 1); % Stochastic term
    
    % Update neuron activations
    X_all(:, t) = X_all(:, t-1) + dX; % Store updated activations
end

% Plotting Results

% 1. Activations of all neurons over time
figure('Position', [100, 100, 900, 700]);
hold on;
plot(time_vector, mod(X_all, 2*pi), 'LineWidth', 0.5);
xlabel('Time', 'Interpreter', 'latex');
ylabel('Activation $X_i(t)$', 'Interpreter', 'latex');
title('Activations of Neurons in the Stochastic McKean-Vlasov Model', 'Interpreter', 'latex');
grid on;
legend(arrayfun(@(i) sprintf('Neuron %d', i), 1:N, 'UniformOutput', false), 'Location', 'best');
hold off;

% 2. Activation distribution at final time
figure('Position', [100, 100, 900, 700]);
histogram(mod(X_all(:, end), 2*pi), 30, 'Normalization', 'pdf');
xlabel('Activation', 'Interpreter', 'latex');
ylabel('Probability Density', 'Interpreter', 'latex');
title('Activation Distribution at Final Time', 'Interpreter', 'latex');
grid on;

% 3. Evolution of one neuron's activation over time
figure('Position', [100, 100, 900, 700]);
plot(time_vector, mod(X_all(1, :), 2*pi), 'LineWidth', 2, 'Color', 'm');
xlabel('Time', 'Interpreter', 'latex');
ylabel('Activation $X_1(t)$', 'Interpreter', 'latex');
title('Evolution of One Neuron''s Activation Over Time', 'Interpreter', 'latex');
grid on;

% 4. Histogram of weights and biases
figure('Position', [100, 100, 900, 700]);
subplot(1, 2, 1);
histogram(weights, 30, 'Normalization', 'pdf');
xlabel('Weights', 'Interpreter', 'latex');
ylabel('Probability Density', 'Interpreter', 'latex');
title('Distribution of Weights', 'Interpreter', 'latex');

subplot(1, 2, 2);
histogram(biases, 30, 'Normalization', 'pdf');
xlabel('Biases', 'Interpreter', 'latex');
ylabel('Probability Density', 'Interpreter', 'latex');
title('Distribution of Biases', 'Interpreter', 'latex');
grid on;
