close all
clear all
clc
%% 
fprintf('Running Code \n');
fprintf('Loading Text');

filename = 'goblet_book.txt';
% [char_to_ind, ind_to_char,K] = ReadInData(filename);
book_fname = filename;
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data)';
K = length(book_chars); 
array_chars = {};
for i=1:K
    array_chars{i}=book_chars(i);
end
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
char_to_ind = containers.Map(array_chars,int32(1:K));
ind_to_char = containers.Map(int32(1:K),array_chars);
fprintf(' - done \n');
%%
eta = .1;   %learning rate
seq_length = 25;    %length of the input sequences
sig = .01;
e = 1; % puntero de posicion
%%
m_grad=5;
h_grad = zeros(m_grad,1); %hidden state at time 0
RNN_grad = RNNmodel(m_grad,K,sig);
%% Comparison of Gradients
% fprintf('    --> Comparison of Numerical and Analytical Gradients using m=5 \n')
% X_chars= book_data(1:seq_length);
% Y_chars = book_data(2:seq_length+1);
% [X_grad,Y_grad]= ConversiontoMatrices(X_chars,Y_chars,char_to_ind,K);
% % [A,H,O,P,I]= synthesize(RNN_grad,h_grad,X_grad);
% fprintf('Computing analytical gradients')
% [grads_c,h_grad] = Gradients(RNN_grad,h_grad,X_grad,Y_grad); fprintf(' - done \n');
% fprintf('Computing numerical gradients ')
% num_grads = ComputeGradsNum(X_grad, Y_grad, RNN_grad, 1e-4); fprintf(' - done \n');
% 
% diff = zeros(size(m_grad));
% fprintf('Numerical-Analitical comparison done using m = 5\n');
% str=['b','c','W','V','U'];
% for i=1:m_grad
%     diff(i) = abs(max(max(num_grads.(str(i))-grads_c.(str(i)))));
%     fprintf(['  Worst approximation gradient ', str(i), ' is ', num2str(diff(i)), '\n']);
% end

%%  Training the RNN with AdaGrad
fprintf(' \n Inititating training \n');

m=100;
h = zeros(m,1); %hidden state at time 0
RNN = RNNmodel(m,K,sig);
X_in = book_data(e:e+seq_length-1);
Y_in = book_data(e+1:e+seq_length+1);
[Xi,Yi]= ConversiontoMatrices(X_in,Y_in,char_to_ind,K);
eta = 0.1;
epsilon = 1e-8;
iterations = 10000;
L = [];
wb = waitbar(0,'1','Name','Iterations');    
for iter=1:iterations
    waitbar(iter/iterations,wb,strcat('Iteration # ',num2str(iter)));
    if e+1>=length(book_data)
        e=1
    end
    X_in = book_data(e:e+seq_length-1);
    Y_in = book_data(e+1:e+seq_length+1);
    [X,Y]= ConversiontoMatrices(X_in,Y_in,char_to_ind,K);
    [grads,h] = Gradients(RNN,h,X,Y);
    for f = fieldnames(RNN)'
        if iter == 1
            mAG.(f{1}) = zeros(size(grads.(f{1})));            
        end
        mAG.(f{1}) = mAG.(f{1}) + (grads.(f{1})).^2;
        RNN.(f{1}) = RNN.(f{1}) - (grads.(f{1})*eta)./(mAG.(f{1}) + epsilon).^.5;
    end
    loss = Getloss(X,Y,RNN,h);
    if iter == 1
        smooth_loss = loss;
    else
        smooth_loss = .999*smooth_loss + .001*loss;
    end
    L = [L;smooth_loss];
    seq_learn=200;
    last_text = '';
%     last_text = char(zeros(1,seq_learn));
    xi=zeros(K,1);
    if mod(iter,1000) == 0
        [hl,pl,Il,Yl]= nueva(RNN,h,xi,seq_learn);
%         [Aig,Hig,Oig,Pig,Yl]= synthesize(RNN,h,x,n)
        for ind=1:seq_learn
            [one,place]=max(Yl(:,ind));
            chars=ind_to_char(place);
            last_text=[last_text,chars];
        end
        fprintf(['\n\n Iteration #:', num2str(iter), ' | text: \n' ]);
        fprintf(last_text);
        fprintf (['smooth loss: ',num2str(smooth_loss),'\n'])
    end
    e=e+seq_length;
end
close(wb);
%%
plot(L);
fprintf("\n done \n");





