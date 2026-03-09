function [x0, peak_response, y0, all_roots]=alpFit_chris(x, y, dbg)
if nargin==0
    close all;
    clear all;
    dbg=1;
%     x=[1.20 1.52 1.92 2.43 3.07]; y=[10 14 13 9 6];
%     x=[1.20 1.52 1.92 2.43 3.07]; y=[18 14 13 9 5];
    x=[1.20 1.52 1.92 2.43 3.07]; y=[8 10 9 7 5];
else
    dbg=0;
end

[p, s]=polyfit(x, y, 3); % y = a*x^3 + b*x^2 + c*x +d
delta=0.02;
x_ = [x(1):delta:x(end)];
fx_plot = polyval(p, x_);

dp = [3*p(1), 2*p(2), p(3)];
all_roots = roots(dp);
fx = polyval(p, all_roots);
[peak_response, indx] = max(fx);
x0 = all_roots(indx);

% check the first and 2nd order of diff:
y0  = 3*p(1)*x0^2 + 2*p(2)*x0 + p(3);
y0_ = 6*p(1)*x0 + 2*p(2);
if (dbg)
    % analytical solution: critical point
    x1 = (-2*p(2)-sqrt(4*p(2)*p(2)-12*p(1)*p(3)))/6*p(1); 
    x2 = (-2*p(2)+sqrt(4*p(2)*p(2)-12*p(1)*p(3)))/6*p(1);
end

if (dbg)
    figure(41); hold on; grid on; stem(x, y, 'ob', 'LineWidth', 2);
    plot(x_, fx_plot, ':b', 'LineWidth', 2); stem(x0, peak_response, '*r', 'LineWidth', 2); %
    xlabel('\sigma', 'FontSize', 14); ylabel('LoG(x,y)', 'FontSize', 14);
    ax = gca;
    ax.FontSize = 14;  % Set the desired font size
    fig = gcf;
    exportgraphics(fig, 'alphfit_Chris.png');
    % return
    
    figure(42); subplot(1,2,1); hold on; grid on; stem(x, y, 'ob');
    plot(x_, fx_plot, ':b'); stem(x0, peak_response, '*r');
    %title('LoG response');
    xlabel('\sigma', 'FontSize', 16); ylabel('LoG(x,y)', 'FontSize', 14);
    
    subplot(1,2,2); hold on; grid on; 
    plot(x_, 3*p(1)*x_.^2 + 2*p(2)*x_ + p(3), ':b'); plot(x0, y0, '*r');
    %title('LoG response diff');
    xlabel('\sigma', 'FontSize', 16); ylabel('Value', 'FontSize', 14);
end  
return;