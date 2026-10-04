function paths = hqt_paths()
% HQT_PATHS Return repository-relative locations as absolute paths.
%   No directory is created by this function. Existing data are kept under
%   data/, and generated figures, movies, and logs belong under outputs/.

    % Anchor every path to the checkout so scripts can be launched from any folder.
    paths.root = fileparts(mfilename('fullpath'));
    % Separate stationary inputs, processed arrays, and generated checkpoints.
    paths.reference = fullfile(paths.root, 'data', 'reference');
    paths.processed = fullfile(paths.root, 'data', 'processed');
    paths.raw = fullfile(paths.root, 'data', 'raw');
    paths.metadata = fullfile(paths.root, 'data', 'metadata');
    % Keep presentation outputs apart from the numerical data collection.
    paths.figures = fullfile(paths.root, 'outputs', 'figures');
    paths.movies = fullfile(paths.root, 'outputs', 'movies');
    paths.logs = fullfile(paths.root, 'outputs', 'logs');
end
