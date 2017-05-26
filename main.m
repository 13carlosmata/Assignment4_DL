close all
clear all
clc
%% 
fprintf('Running Code \n');
fprintf('Loading Text');

filename = 'goblet_book.txt';
[char_to_ind, ind_to_char,K,book_data]  = ReadInData(filename);
fprintf(' - done \n');
%%
eta = .1;   %learning rate
seq_length = 25;    %length of the input sequences
sig = .01;
e = 1; % puntero de posicion
%% Comparison of Gradients
m_grad=5;
fprintf('    --> Comparison of Numerical and Analytical Gradients using m=5 \n')
ComparisonGradients(book_data,seq_length,char_to_ind,K,m_grad,sig);
%%  Training the RNN with AdaGrad
fprintf(' \n Inititating training \n');
m=100;
h = zeros(m,1); %hidden state at time 0
RNN = RNNmodel(m,K,sig);
X_in = book_data(e:e+seq_length-1);
Y_in = book_data(e+1:e+seq_length+1);
[Xi,Yi]= ConversiontoMatrices(X_in,Y_in,char_to_ind,K);
eta = 0.1;
epsilon = 1e-10;
iterations = 100000; %run foe 3 epochs
L = [];
wb = waitbar(0,'1','Name','Iterations');    
epoch=1;
fprintf(['-------------------------\n']);
fprintf(['Running epoch: ',num2str(epoch),'\n']);
fprintf(['-------------------------\n']);
name = ['Iterations_',num2str(iterations),'_h',num2str(hour(datetime)),'m',num2str(minute(datetime)),'s',num2str(second(datetime))];
path=['fig/',name,'.txt'];
index_file=1;
for iter=1:iterations
    waitbar(iter/iterations,wb,strcat('Iteration # ',num2str(iter)));
    if e+seq_length>=length(book_data)
        e=1;
        epoch = epoch+1;
        fprintf(['-------------------------\n']);
        fprintf(['Running epoch ',num2str(epoch),'\n']);
        fprintf(['-------------------------\n']);
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
    xi=zeros(K,seq_learn);
    if mod(iter,iterations/10) == 0 || iter==1
        [~,~,~,~,Yl]= synthesize(RNN,h,xi,seq_learn,1);
        for ind=1:seq_learn
            [one place]=max(Yl(:,ind));
            chars=ind_to_char(place);
            last_text=[last_text,chars];
        end
        header = ['\n iter = ', num2str(iter), ', smooth_loss=',num2str(smooth_loss),'\n']; 
        fprintf(header);
        text_to_print = ['Text: \n',last_text,'\n'];
        fprintf (text_to_print);
        fid=fopen(path,'a+');
        fprintf(fid, header);
        fprintf(fid, [text_to_print '\n']);
        fclose(fid);
    end
    e=e+seq_length;
end
close(wb);
%%
Figure = plot(L);
title(['Iterations: ', num2str(iterations), '  epoch:', num2str(epoch)])
xlabel(['Final value of loss: ',num2str(smooth_loss)]);
saveas(Figure,['fig/','h',num2str(hour(datetime)),'m',num2str(minute(datetime)),'s',num2str(second(datetime),2),'.jpg']);
%%
last_text = '';
xi=zeros(K,1000);
[~,~,~,~,Yl]= synthesize(RNN,h,xi,seq_learn,1);
for ind=1:seq_learn
    [one place]=max(Yl(:,ind));
    chars=ind_to_char(place);
    last_text=[last_text,chars];
end
text1 = ['Passage of length 1000 characters synthesized from the best model (loss:',num2str(smooth_loss),') : \n'];
fprintf('\n\n',text1);
fprintf(last_text);
fid=fopen(path,'a+');
fprintf(fid, ['\n' text1]);
fprintf(fid, [last_text '\n']);
fclose(fid);

fprintf("\n ---> Code ran successfully <---\n");
beep


