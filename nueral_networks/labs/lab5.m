%We define a problem , with a set of fourteen
%2− element input vectors P and the corresponding
% fourteen
%1− element targets T
P = [ 2 1 2 5 7 2 3 6 1 2 5 4 6 5 6 7 8 7  ;
    2 3 3 3 3 4 4 4 5 5 5 6 6 7 6 6 7 7 ] ;
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 1 0 0 0 0 ] ;


figure ( 1 ) ;
plotpv(P ,T)

% calculate min and max

minMaxVal = minmax(P);

%  This code create a perceptron layer with one 2 - element
net = newp( minMaxVal , 1 ) ;
simT = sim ( net , P);
net.trainParam.epochs = 1000 ;
net = train( net, P , T) ;
simT = sim ( net , P);

% print simt
figure ( 2 ) ;
plotpc(net.iw{ 1 , 1 }, net.b { 1 })



%%