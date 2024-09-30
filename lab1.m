A = [ 1 2 3; 4 5 6; 7 8 9 ];

for index = A
    for z = index
        disp(z + '---')
    end
end

x = zeros([10, 0]);
for i = 1:10
    x(i) = i ^ 2;
end

disp(x)