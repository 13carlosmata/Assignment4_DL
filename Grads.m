classdef Grads
   properties
      b;c;U;W;V
   end
   methods
      function grad = Grads(b,c,U,W,V)
         grad.b = b;
         grad.c = c;
         grad.U = U;
         grad.W = W;
         grad.V = V;
      end
   end
end