function[at,ht,ot,pt,ii]= synthesize(RNN,h0,x0,n)
I = [];
at = RNN.W*h0 +RNN.U*x0 + RNN.b;
ht = tanh(at);
ot = RNN.V*ht + RNN.c;
pt = softmax(ot);
cp = cumsum(pt);
a = rand;
ixs = find(cp-a>0);
ii= ixs(1);
end