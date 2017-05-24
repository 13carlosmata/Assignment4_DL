function[H,P,I,Y]= nueva(RNN,h,x,n)
I = [];
%n = size(x,2);
K = size(RNN.c,1);
val = size(h,1);

%     x = zeros(K,1);
%     x(char) = 1;
Y(:,1)=x;
H = zeros(val,n+1);
P = zeros(K,n);
for i=1:n
    at = RNN.W*h +RNN.U*x + RNN.b;
    h = tanh(at);
    ot = RNN.V*h + RNN.c;
    pt = softmax(ot);
    cp = cumsum(pt);
    a = rand;
    ixs = find(cp-a>0);
    ii = ixs(1);
    
    I = [I;ii];
    Y(ii,i+1) = 1;
    size(Y(ii,i+1));
    x = Y(:,i+1);
end
end