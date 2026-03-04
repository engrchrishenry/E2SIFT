clear all;
close all;

% Set paths
image_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\final_data_ecd (copy)\train\images';  % Replace with your images directory
output_dir = 'D:\Chris Temp\Code\LoGSIFTGitHub\final_data_ecd (copy)\train\log';  % Specify the output directory
n_cores = 5; % Number of cores to use

% LoG Pyramid parameters
a = 1.2;
b = 2;
n_scales = 4;
log_mask_size = 11;

image_files = dir(fullfile(image_dir, '**', '*.png'));

% Create the output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if isempty(gcp('nocreate'))
    parpool(n_cores); % use only n_cores workers (cores)
end

queue = createTerminalProgressBar(numel(image_files));

parfor image_idx = 1:numel(image_files)
    image_path = fullfile(image_files(image_idx).folder, image_files(image_idx).name);
    im = imread(image_path);
    if size(im, 3) == 3
        im = rgb2gray(im);
    end
    
    % LoG Pyramid
    scales = zeros(1, n_scales);
    f_log = cell(1, n_scales);
    log_pyramid = zeros(n_scales, size(im, 1), size(im, 2));

    for k = 1:n_scales
        scales(k) = a * sqrt(b)^(k - 1);
        f_log{k} = fspecial('log', log_mask_size, scales(k));
    end

    for k = 1:n_scales
        im_log = imfilter(im2double(im), f_log{k});
        log_pyramid(k, :, :) = (scales(k)^2) * im_log;
%         Min = min(min(im_log));
%         Max = max(max(im_log));
%         fprintf('Scale %d: Min = %.4f, Max = %.4f\n', k, Min, Max);
    end

    % Save the log_pyramid as a .mat file in the specified output directory
    [~, image_name, ~] = fileparts(image_files(image_idx).name);
    mat_file_path = fullfile(output_dir, [image_name '.mat']);
    parsave(mat_file_path, log_pyramid);
    
    % Print progress
    % fprintf('Processed image %d/%d\n', image_idx, numel(image_files));   
    
    % Update progress bar
    send(queue, 1);
end

