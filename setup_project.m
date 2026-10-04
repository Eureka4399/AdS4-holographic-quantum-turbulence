function paths = setup_project()
% SETUP_PROJECT Add the project code to the current MATLAB session.
%   PATHS = SETUP_PROJECT() returns absolute data and output directories.
%   Call this once from the repository root. It does not run a simulation,
%   load data, select a GPU, or change the current working directory.
%   Historical exploratory scripts are intentionally not added to the path.

    % Resolve locations from this file rather than from the current working folder.
    root = fileparts(mfilename('fullpath'));
    addpath(root);
    directories = {'src/numerics', 'src/solvers', 'src/diagnostics', ...
        'src/utilities', 'third_party/dmsuite', 'scripts/simulation', ...
        'scripts/analysis', 'examples'};
    % Add only executable code folders; data and legacy notebooks are excluded.
    for i = 1:numel(directories)
        addpath(fullfile(root, directories{i}));
    end
    paths = hqt_paths();
end
