function path = hqt_data_path(category, varargin)
% HQT_DATA_PATH Build a path under a named project data directory.
%   PATH = HQT_DATA_PATH('processed', 'average2.mat') locates a compact result.
%   PATH = HQT_DATA_PATH('raw', runName, checkpointName) locates a checkpoint.
%   This helper does not load, save, or create files.

    % Validate the data category, then join optional run folders and filenames.
    paths = hqt_paths();
    category = validatestring(category, {'reference', 'processed', 'raw', 'metadata'});
    path = fullfile(paths.(category), varargin{:});
end
