   function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% This one was really tough to do.
K = num_labels;			% This is the different types of labels that we can have
temp_y = zeros(m, K);	% Mapping vector y to this one

for i = 1:m,
	idx = y(i, 1);
	temp_y(i, idx) = 1;
end;

X = [ones(m, 1) X];		% X = 5000 * 401

y = temp_y;				% Now we have mapped y to binary vector of 1's and 0's
hidden = sigmoid(Theta1 * X');		% 'hidden = 25 * 5000
hidden = hidden';					% '
hidden_grad = sigmoidGradient(Theta1 * X'); % 'hidden_grad = 25 * 5000
hidden_grad = hidden_grad';					% '
hidden_grad = [ones(m, 1) hidden_grad];


hidden = [ones(m, 1) hidden];		% hidden = 5000 * 26
output = sigmoid(Theta2 * hidden'); % 'output = 10 * 5000
hx     = output';					% Now' hx = 5000 * 10



for i = 1:m,
	for k = 1:K,
		val = y(i, k) * log(hx(i, k)) + (1 - y(i, k)) * log(1 - hx(i, k));
		J = J + val;
	end;
end;

J = (-1 * J) / m;

% Regularization part begins now
% input_layer_size = 400, hidden_layer_size = 25, output_layer_size = 10 (Assuming)

reg_part = 0;

for j = 1:hidden_layer_size,
	for k = 1:input_layer_size,
		reg_part = reg_part + (Theta1(j, k+1) * Theta1(j, k+1));
	end;
end;

for j = 1:K,
	for k = 1:hidden_layer_size,
		reg_part = reg_part + (Theta2(j, k+1) * Theta2(j, k+1));
	end;
end;


J = J + (lambda * reg_part) / (2 * m);



% backpropagation part begins now

% Here first we have to do Feedforward propogation which we have already done
% Hidden and output have already got everything calculated so we don't 
% 'need to do any Feedforward propogation.

delta_3 = hx - y;		% delta_3 = 5000 * 10
delta_2 = (Theta2' * delta_3') .* hidden_grad' ;	% 'delta_2 = 26 * 5000
delta_2 = delta_2(2:end, :);						% delta_2 = 25 * 5000

Theta2_grad = (Theta2_grad + (delta_3' * hidden)) / m;		% 'Theta2_grad = 10 * 26
Theta1_grad = (Theta1_grad + (delta_2  * X)) / m;			% ''Theta1_grad = 25 * 401

% Now we add the regularization terms to the backpropagation
reg2_grad = zeros(K, hidden_layer_size);	% reg2_grad = 10 * 25
reg2_grad = reg2_grad + ((Theta2(:, 2:end) .* lambda) / m);
reg2_grad = [zeros(K, 1) reg2_grad];		% reg2_grad = 10 * 26

Theta2_grad = Theta2_grad + reg2_grad;		

reg1_grad = zeros(hidden_layer_size, input_layer_size);		% reg1_grad = 25 * 400
reg1_grad = reg1_grad + ((Theta1(:, 2:end) .* lambda) / m);
reg1_grad = [zeros(hidden_layer_size, 1) reg1_grad];		% reg1_grad = 25 * 401

Theta1_grad = Theta1_grad + reg1_grad;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
