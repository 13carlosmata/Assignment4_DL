function num_grads = ComputeGradsNum(X, Y, RNN, h)
for f = fieldnames(RNN)'
    num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
end

function grad = ComputeGradNum(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = Getloss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = Getloss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end

