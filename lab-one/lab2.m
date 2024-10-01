x = 0:0.5:10;
y = sin(x);

plot(x, y, '-o', 'MarkerSize', 8, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'g');

figure
figure
plot(x, y, "Color", 'm')
figure(2)   %selects figure 2
plot(x, y, "Color", 'y')