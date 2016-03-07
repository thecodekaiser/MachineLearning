function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m,
	h = (1 /(1 + e ^ (-1 * (theta'*X(i,:)'))));
	J = J + (y(i)*log(h) + (1-y(i))*log(1-h));
end;

J = J /(-1*m);
val = 0;

for j = 2:size(theta),
	val = val + theta(j)^2;
end;

J = J + (lambda * val / (2 * m));	% We have computed the final J

for j = 1:size(theta),
	val = 0;
	for i = 1:m,
		h = (1 / (1 + e ^ (-1 * (theta'*X(i,:)'))));
		val = val + (h - y(i)) * (X(i, j));
	end;
	grad(j) = val / m ;

	if j > 1,
		grad(j) = grad(j) + (lambda * theta(j) / m);
	end;
end;





% =============================================================

end
