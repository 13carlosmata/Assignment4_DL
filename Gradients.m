function [grads,h] = Gradients(RNN,h0,X,Y)
n = size(X,2);
[at,ht,ot,pt,ii] = synthesize(RNN,h0,X,n,0);
% [loss] = Getloss(X,Y,RNN,h0);
grad_V = 0;
grad_o = zeros(size(X));
grad_h = zeros(size(RNN.V,2),n);
grad_a = zeros(size(RNN.V,2),n);
grad_W = 0;
grad_U = 0;
grad_b = 0;
grad_c = 0;

for i=1:n
    grad_o(:,i)=-(Y(:,i)-pt(:,i))';
    grad_V = grad_o(:,i)*ht(:,i)'+grad_V;
    grad_c = grad_c + grad_o(:,i);
end
grad_h(:,n) = grad_o(:,n)'*RNN.V;
grad_a(:,n) = grad_h(:,n)'*diag(1-(tanh(at(:,n))).^2);

for i=n-1:-1:1
    grad_h(:,i) = grad_o(:,i)'*RNN.V+grad_a(:,i+1)'*RNN.W;
    grad_a(:,i) = grad_h(:,i)'*diag(1-(tanh(at(:,i))).^2);    
end
h_0=zeros(size(ht(:,1)));
for i=1:n
    if i==1
        grad_W = grad_a(:,i)*h_0(:,i)' +  grad_W;
    else
        grad_W = grad_a(:,i)*ht(:,i-1)' +  grad_W;
    end
    grad_U = grad_a(:,i)*X(:,i)'+grad_U;
    grad_b = grad_b + grad_a(:,i);
end
grads = Grads(grad_b,grad_c,grad_U,grad_W,grad_V);
h=ht(:,i);
end