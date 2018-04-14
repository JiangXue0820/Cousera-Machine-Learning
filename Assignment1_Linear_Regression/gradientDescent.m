function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
            
                               
           if  iter == 1 || computeCost(X,y,theta) < J_history(iter - 1)
                sum = [0;0];
                for i = 1:m
                    sum(1) = sum(1) + (X(i,:) * theta - y(i))*X(i,1);
                    sum(2) = sum(2) + (X(i,:) * theta - y(i))*X(i,2);
                end
               J_history(iter) = computeCost(X, y, theta);
               theta = theta - alpha / m * sum; 
           else
               theta = theta + alpha / m * sum; 
           end
           
           
        % ============================================================

        

end

