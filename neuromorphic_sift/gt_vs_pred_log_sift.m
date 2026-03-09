%  E2SIFT Keypoint Evaluation Script
%  This script compares keypoints extracted from:
%  1) Ground truth LoG pyramid
%  2) Predicted LoG pyramid
%
%  It computes matching keypoints within a spatial tolerance and
%  optionally saves visualization plots.

clear all;
close all;

% Paths
image_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\pred\calibration\images'; % Path to frames
gt_log_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\pred\calibration\log'; % Path to ground truth LoG pyramid
pred_log_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\pred\calibration\pred_log'; % Path to ground truth LoG pyramid
output_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\output\calibration/'; % Path to output files
vl_feat_dir = 'vlfeat-0.9.21-bin\vlfeat-0.9.21\toolbox\vl_setup';  % Path to VLFeat

% Parameters
opt.log_peak_thres = 0.03; % Minimum LoG response magnitude required to consider a candidate keypoint
opt.log_grad_thres = 0.05; % Maximum allowed absolute gradient at the scale-space peak (used to filter unstable extrema)
tolerance = 5; % Spatial tolerance (in pixels) for matching GT and predicted keypoints
plot_var = 0; % Set to 1 to save visualization plots, 0 to disable plotting
cropSize = [160, 160];

% Load VLFeat
run(vl_feat_dir);

image_files = dir(fullfile(image_dir, '**', '*.png'));
gt_log_files = dir(fullfile(gt_log_dir, '**', '*.mat'));
pred_log_files = dir(fullfile(pred_log_dir, '**', '*.mat'));

if ~exist(fullfile(output_dir, 'plots'), 'dir')
    mkdir(fullfile(output_dir, 'plots'));
end

% Create results file
results_file = fullfile(output_dir, 'results.txt');
fid = fopen(results_file, 'w');

fprintf(fid, 'E2SIFT Keypoint Matching Results\n');
fprintf(fid, '=================================\n\n');

total = 0;
for image_idx = 1:numel(gt_log_files)
    % Load image
    image_path = fullfile(image_files(image_idx).folder, image_files(image_idx).name);
    im = imread(image_path);
    
    % Load LoG pyramids    
    gt_log_pyramid = load(fullfile(gt_log_files(image_idx).folder, gt_log_files(image_idx).name)).log_pyramid;
    gt_log_pyramid = double(gt_log_pyramid);
    
    pred_log_pyramid = load(fullfile(pred_log_files(image_idx).folder, pred_log_files(image_idx).name)).log_pyramid;
    pred_log_pyramid = double(pred_log_pyramid);
    
    % Center crop image and pyramids
    startIdx = floor((size(im) - cropSize) / 2) + 1;
    im = im(startIdx(1):startIdx(1) + cropSize(1) - 1, startIdx(2):startIdx(2) + cropSize(2) - 1, :);
    gt_log_pyramid = gt_log_pyramid(:, startIdx(1):startIdx(1) + cropSize(1) - 1, startIdx(2):startIdx(2) + cropSize(2) - 1, :);
    
    [ht, wid]=size(im);
    
    a=1.2; b=2; n_scales = 4; log_mask_size=11;
    for k=1:n_scales 
        scales(k)=a*sqrt(b)^(k-1); 
    end

    % Detect keypoints from Ground Truth LoG pyramid
    log_max = reshape(max(reshape(abs(gt_log_pyramid(:,:,:)), [4, wid*ht])), [ht, wid]);
    alp_offs  = find(log_max>=opt.log_peak_thres); 
    n_peak_log = length(alp_offs);

    log_sift_pos = zeros(size(log_max)); n_log_sift=0;
    for k=1:length(alp_offs)
        [alp_i, alp_j] = ind2sub(size(log_max), alp_offs(k));
        [x0_log(k), peak_log(k), y0(k), all_roots]= alpFit_chris(scales, abs(gt_log_pyramid(:, alp_i, alp_j)'), 0);
        if isreal(all_roots)
            if abs(y0(k)) < opt.log_grad_thres
                log_sift_pos(alp_i, alp_j) = 1; n_log_sift=n_log_sift+1;
                k_pos_gt(n_log_sift, 1) = alp_j; k_pos_gt(n_log_sift, 2) = alp_i; 
                k_scale(n_log_sift) = x0_log(k);
                k_grad(n_log_sift) = y0(k);
            end
        else
            continue;
        end
    end
    
    % Detect keypoints from Predicted LoG pyramid
    log_max = reshape(max(reshape(abs(pred_log_pyramid(:,:,:)), [4, wid*ht])), [ht, wid]);
    alp_offs  = find(log_max>=opt.log_peak_thres); 
    n_peak_log = length(alp_offs);

    log_sift_pos = zeros(size(log_max)); n_log_sift=0;
    for k=1:length(alp_offs)
        [alp_i, alp_j] = ind2sub(size(log_max), alp_offs(k));
        [x0_log(k), peak_log(k), y0(k), all_roots]= alpFit_chris(scales, abs(pred_log_pyramid(:, alp_i, alp_j)'), 0);
        if isreal(all_roots)
            if abs(y0(k)) < opt.log_grad_thres
                log_sift_pos(alp_i, alp_j) = 1; n_log_sift=n_log_sift+1;
                k_pos_pred(n_log_sift, 1) = alp_j; k_pos_pred(n_log_sift, 2) = alp_i; 
                k_scale(n_log_sift) = x0_log(k);
                k_grad(n_log_sift) = y0(k);
            end
        else
            continue;
        end
    end
    
    % Edge removal
    [k_pos_final_gt, ~, ~, ~] = edgeRemoval(gt_log_pyramid, k_pos_gt, 4);
    [k_pos_final_pred, ~, ~, ~] = edgeRemoval(pred_log_pyramid, k_pos_pred, 4);
    
    % Compute matching accuracy
    distances = pdist2(k_pos_final_gt, k_pos_final_pred, 'euclidean');
    matches = sum(any(distances <= tolerance, 2));
    acc_alpfit = matches/size(k_pos_final_gt,1);
    disp(['Number of matching rows: ' num2str(matches) ' precision_alpfit: ' num2str(acc_alpfit)]);
    fprintf(fid, '%s  ->  accuracy: %.6f  matches: %d\n', image_files(image_idx).name, acc_alpfit, matches);
    total = total + acc_alpfit;
    
    % Plot results (optional)
    if plot_var == 1
        % Create figure without displaying it
        fig = figure('visible','off');
        colormap('gray')
        set(fig, 'Position', [100 100 1200 500]);   % [x y width height]

        % Intensity Image
        subplot(1,3,1)
        imagesc(im)
        axis image off
        title('Intensity Image','FontSize',12)

        % Ground Truth Keypoints
        subplot(1,3,2)
        imagesc(im)
        axis image off
        hold on
        plot(k_pos_final_gt(:,1), k_pos_final_gt(:,2), ...
            'xy','LineWidth',1.5,'MarkerSize',7)
        title(sprintf('GT LoG SIFT (#kp: %d)', length(k_pos_final_gt)), 'FontSize',12)

        % Predicted Keypoints
        subplot(1,3,3)
        imagesc(im)
        axis image off
        hold on
        plot(k_pos_final_pred(:,1), k_pos_final_pred(:,2), ...
            'xy','LineWidth',1.5,'MarkerSize',7)
        title(sprintf('Predicted LoG SIFT (#kp: %d)', length(k_pos_final_pred)), 'FontSize',12)
        
        % Save figure
        save_path = fullfile(output_dir, 'plots', image_files(image_idx).name);
        exportgraphics(fig, save_path, 'BackgroundColor', 'white');
%         exportgraphics(fig, save_path, 'Resolution', 300, 'BackgroundColor', 'white');

        close(fig)

    end
    disp(['Processed file ' num2str(image_idx) '/ ' num2str(length(gt_log_files))]);
end

final_acc = total/length(gt_log_files);

disp(['Final matching accuracy = ' num2str(final_acc)])

fprintf(fid, '\n=================================\n');
fprintf(fid, 'Final Average Accuracy: %.5f\n', final_acc);

fclose(fid);

