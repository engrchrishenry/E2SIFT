function queue = createTerminalProgressBar(totalIterations)
    % Initializes a DataQueue to track progress in terminal
    % Returns a queue to send progress updates from parfor

    queue = parallel.pool.DataQueue;  % create the queue
    progress = 0;

    % Define the update function
    afterEach(queue, @updateProgress);

    function updateProgress(~)
        progress = progress + 1;
        percent = (progress / totalIterations) * 100;
        fprintf('Progress: %.2f%% (%d/%d)\n', percent, progress, totalIterations);
    end
end