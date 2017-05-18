classdef RNNmodel
   properties
      b;c;W;V;U
   end
   methods
      function RNN = RNNmodel(m,K,sig)
         RNN.b = zeros(m,1);
         RNN.c = zeros(K,1);
         RNN.W = randn(m,m)*sig;
         RNN.V = randn(K,m)*sig;
         RNN.U = randn(m,K)*sig;
      end
   end
end