%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [k_pos]=edgeRemoval(pyramid, octaves, scales, sigma)
% Implementation of hessian edge removal
% input:
%   pyramid: Laplacian of Gaussian pyramid
%   octaves: number of octaves
%   scales: number of scales
%   sigma: initial value for sigma
% output:
%   points: collection of non-edge keypoint locations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [k_points, k_subs, e_points, e_subs] = edgeRemoval(pyramid, points, scales)
    k_points = [];
    k_subs = [];
    e_points = [];
    e_subs = [];
    r_curvature = 10;% 
    
    for p=1:size(points,1)
        [~,row,col]= size(pyramid);
        x = points(p, 1); y = points(p, 2);
        if x == 1 || x == col || y == 1 || y == row
            continue
        else
            for s=2:scales-1
                sub = pyramid(s-1:s+1,y-1:y+1,x-1:x+1);
                if sub(2,2,2) > max([sub(1:13),sub(15:end)]) || sub(2,2,2) < min([sub(1:13),sub(15:end)])
                    % Calculating trace and determinant of hessian matrix.
                    Dxx = sub(1,2,2)+sub(3,2,2)-2*sub(2,2,2);
                    Dyy = sub(2,1,2)+sub(2,3,2)-2*sub(2,2,2);
                    Dxy = sub(1,1,2)+sub(3,3,2)-2*sub(1,3,2)-2*sub(3,1,2);
                    trace = Dxx+Dyy;
                    determinant = Dxx*Dyy-Dxy*Dxy;
                    curvature = trace*trace/determinant;
                    if curvature > (r_curvature+1)^2/r_curvature
                        new_point = [x, y];
                        e_points = [e_points; new_point];
                        e_subs = [e_subs, sub(:)];
                    else
                        new_point = [x, y];
                        k_points = [k_points; new_point];
                        k_subs = [k_subs, sub(:)];
                    end
                else
                    new_point = [x, y];
                    e_points = [e_points; new_point];
                    e_subs = [e_subs, sub(:)];
                end
            end
        end
    end
end