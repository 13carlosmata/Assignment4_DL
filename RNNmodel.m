classdef RNNmodel
   properties
      b;c;U;W;V
   end
   methods
      function RNN = RNNmodel(m,K,sig)
         RNN.b = zeros(m,1);
         RNN.c = zeros(K,1);
         RNN.U = randn(m,K)*sig;
         RNN.W = randn(m,m)*sig;
         RNN.V = randn(K,m)*sig;
      end
   end
end