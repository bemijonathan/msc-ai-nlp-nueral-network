% define the variables to be tested

P = [ 2 1 2 5 7 2 3 6 1 2 5 4 6 5 ; 
      2 3 3 3 3 4 4 4 5 5 5 6 6 7 ] ;
T = [ 0 0 0 1 1 0 0 1 0 0 1 1 1 1 ];

figure(1)

plotpv(P, T)


minMaxVal = minmax(P);

net = newp(minMaxVal, 1);

simT = sim(net, P);
net.trainParam.epochs = 10;
net = train(net, P, T);
simT = sim(net, P);


figure(1);
plotpc(net.iw{1 ,1} , net.b{1});

net = train(net,P,T) ;

figure(1);
plotpc(net.iw{1 ,1}, net.b{1}, 'r+');
