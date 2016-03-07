function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

szs = size(z);

for i = 1:szs(1),
	if length(szs) > 1,
		for j = 1:szs(2),
			g(i,j) = 1 / (1 + e^(-1 * z(i,j)));
		end;
	else
		g(i) = 1 / (1 + e^(-1 * z(i)));
	end;
end;


% =============================================================

end
