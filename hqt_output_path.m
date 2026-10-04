function path = hqt_output_path(category, varargin)
% HQT_OUTPUT_PATH Build an output path and create its parent directory.
%   PATH = HQT_OUTPUT_PATH('figures', 'spectrum.pdf') is independent of pwd.
%   Categories are 'figures', 'movies', and 'logs'.

    % Resolve the requested output category against the repository root.
    paths = hqt_paths();
    category = validatestring(category, {'figures', 'movies', 'logs'});
    path = fullfile(paths.(category), varargin{:});
    % Create only the parent directory; the caller decides whether to write a file.
    parent = fileparts(path);
    if ~isfolder(parent)
        mkdir(parent);
    end
end
