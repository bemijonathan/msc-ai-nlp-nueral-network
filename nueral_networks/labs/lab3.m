X = [ -0.5 -0.5 +0.3 -0.1;
    -0.5 +0.5 -0.5 +1.0];
T = [1 1 0 0];
% plotpv(X,T);


net = perceptron;
net = configure(net,X,T);


plotpv(X,T);
plotpc(net.IW{1},net.b{1});

XX = repmat(con2seq(X),1,3);
TT = repmat(con2seq(T),1,3);
net = adapt(net,XX,TT);
plotpc(net.IW{1},net.b{1});
