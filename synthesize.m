function[A,H,O,P,Y]= synthesize(RNN,h,x,n,text)
K = size(RNN.c,1);
val = size(h,1);
A = zeros(val,n);
H = zeros(val,n+1);
O = zeros(K,n);
P = zeros(K,n);
Y(:,1)=x(:,1);
for i=1:n
    size(x);
    at = RNN.W*h +RNN.U*x(:,i) + RNN.b;
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
    Y(ii,i+1) = 1;
    if text == 1
        x(:,i+1) = Y(:,i+1);
    end
end
H(:,1)=[];
end