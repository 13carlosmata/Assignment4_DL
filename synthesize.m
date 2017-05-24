function[A,H,O,P,I]= synthesize(RNN,h,x,n)
I = [];
%n = size(x,2);
K = size(RNN.c,1);
val = size(h,1);

A = zeros(val,n);
H = zeros(val,n+1);
O = zeros(K,n);
P = zeros(K,n);
for i=1:n
    at = RNN.W*h +RNN.U*x(:,i) + RNN.b;
    size(x);
    A(:,i) = at;
    h = tanh(at);
    H(:,i+1) = h; 
    ot = RNN.V*h + RNN.c;
    O(:,i) = ot; 
    pt = softmax(ot);
    P(:,i)=pt;
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp-a>0);
    ii = ixs(1);
    I = [I;ii];
%     Y(ii,i+1) = 1;
%     size(Y(ii,i+1));
%     x(:,i+1) = Y(ii,i+1);
end
H(:,1)=[];
end