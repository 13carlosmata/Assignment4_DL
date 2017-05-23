function [loss] = Getloss(X,Y,RNN,h0)
[a,h,o,p,i]= synthesize(RNN,h0,X);
n=size(X,2);
loss = 0;
for i=1:n
  loss=-log(Y(:,i)'*p(:,i))+loss;
end
end