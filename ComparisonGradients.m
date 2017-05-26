function ComparisonGradients(book_data,seq_length,char_to_ind,K,m_grad,sig)
RNN_grad = RNNmodel(m_grad,K,sig);
h_grad = zeros(m_grad,1); %hidden state at time 0
X_chars= book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

[X_grad,Y_grad]= ConversiontoMatrices(X_chars,Y_chars,char_to_ind,K);
fprintf('Computing analytical gradients')
[grads_c,~] = Gradients(RNN_grad,h_grad,X_grad,Y_grad); fprintf(' - done \n');
fprintf('Computing numerical gradients ')
num_grads = ComputeGradsNum(X_grad, Y_grad, RNN_grad, 1e-4); fprintf(' - done \n');
diff = zeros(size(m_grad));
fprintf('Numerical-Analitical comparison done using m = 5\n');
str=['b','c','W','V','U'];
for i=1:m_grad
    temp = max(abs(grads_c.(str(i))-num_grads.(str(i)))/(max(0,abs(grads_c.(str(i)))+abs(num_grads.(str(i))))));
    [value local]=max(temp);
    diff(i) = temp(local);
    fprintf(['  Worst approximation gradient ', str(i), ' is ', num2str(diff(i)), '\n']);
end
end