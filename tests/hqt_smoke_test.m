function hqt_smoke_test()
% HQT_SMOKE_TEST Check project paths and short workflows without saving data.
%   From the repository root: addpath('tests'); hqt_smoke_test
%   Production scripts are read into memory, shortened there, and evaluated
%   with file writers disabled. Their on-disk scientific settings are kept.
%   Missing analysis checkpoints are replaced by explicitly synthetic fields.

    % Preserve session settings and keep figures hidden during batch testing.
    oldPath = path;
    oldFolder = pwd;
    oldVisible = get(groot, 'DefaultFigureVisible');
    oldRandom = rng;
    cleanup = onCleanup(@() restore_session(oldPath, oldFolder, oldVisible, oldRandom));
    set(groot, 'DefaultFigureVisible', 'off');
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root);
    paths = setup_project();
    rng(11);

    % The shared test state contains only in-memory data and write counters.
    global hqtSmokeState
    hqtSmokeState = struct('paths', paths, 'discardedWrites', 0, ...
        'syntheticLoads', 0, 'passed', 0, 'failed', 0, 'seed', [], 'zeroSeed', []);
    baseline = data_snapshot(paths.root);
    fprintf('MATLAB %s\n', version);

    % Check all source files with MATLAB's parser before evaluating notebooks.
    files = [dir(fullfile(root, '*.m')); dir(fullfile(root, 'src', '**', '*.m')); ...
        dir(fullfile(root, 'scripts', '**', '*.m')); ...
        dir(fullfile(root, 'examples', '*.m')); ...
        dir(fullfile(root, 'third_party', '**', '*.m'))];
    errors = 0;
    for i = 1:numel(files)
        diagnostics = checkcode(fullfile(files(i).folder, files(i).name), '-id');
        for j = 1:numel(diagnostics)
            if strcmp(diagnostics(j).id, 'SYNER') || contains(diagnostics(j).message, 'Parse error')
                fprintf('PARSE ERROR %s:%d %s\n', files(i).name, ...
                    diagnostics(j).line, diagnostics(j).message);
                errors = errors + 1;
            end
        end
    end
    check_result('MATLAB source parsing', @() assert(errors == 0));
    check_result('Portable paths from a different working directory', @() test_paths(root));

    % Enable GPU compatibility before deserializing any saved gpuArray objects.
    enable_gpu;
    device = gpuDevice();
    fprintf('GPU: %s\n', device.Name);

    % Load only the compact files; large boundary time series are not loaded.
    seedFile = hqt_data_path('reference', 'mu=6,nz=30,vy=7_20.mat');
    hqtSmokeState.seed = load(seedFile);
    fprintf('Stationary input fields: %s\n', strjoin(fieldnames(hqtSmokeState.seed), ', '));
    average = load(hqt_data_path('processed', 'average2.mat'));
    structure = load(hqt_data_path('processed', 'structure_factor.mat'));
    fprintf('Compact averaged fields: %s\n', strjoin(fieldnames(average), ', '));
    check_result('Compact processed data paths', @() assert(~isempty(fieldnames(structure))));

    % Test the numerical helpers after the real compact inputs are available.
    check_result('Differentiation matrices and Fourier filters', @test_numerics);
    check_result('Vortex, tracking, and structure-function helpers', @test_diagnostics);
    check_result('Stationary Newton solver', @test_stationary);

    % Each solver retains its equations but runs on a 33-by-33-by-10 grid.
    simulations = {'tevol_driven.m', 'tevol_driven_cluster.m', 'tevol_lattice.m'};
    for i = 1:numel(simulations)
        source = fullfile(root, 'scripts', 'simulation', simulations{i});
        check_result(['Short evolution: ' simulations{i}], @() test_evolution(source));
    end

    % Run checkpoint-reading initialization and the first analysis frame.
    analyses = {'movie_show.m', 'movie_show_large.m', 'movie_show_decay.m', ...
        'movie_show_lattice.m', 'plots_cluster.m', 'plots_for_paper.m', 'movie_show_pair.m'};
    for i = 1:numel(analyses)
        source = fullfile(root, 'scripts', 'analysis', analyses{i});
        check_result(['Analysis entry with an in-memory checkpoint: ' analyses{i}], ...
            @() test_analysis_entry(source));
    end
    source = fullfile(root, 'scripts', 'analysis', 'plots_for_paper.m');
    check_result('Paper figure sections using real compact averaged data', ...
        @() test_cached_figures(source));
    source = fullfile(root, 'scripts', 'legacy', 'Statinary_state_with_v.m');
    check_result('Stationary notebook first section', @() test_stationary_notebook(source));
    source = fullfile(root, 'scripts', 'legacy', 'statistics.m');
    check_result('Legacy statistics entry with an in-memory checkpoint', ...
        @() test_analysis_entry(source));
    source = fullfile(root, 'examples', 'structure_function_vs_energy_spectrum.m');
    check_result('Standalone spectrum/structure-function example', @() execute_source(fileread(source)));

    % Prove that the test has neither created nor changed data or output files.
    check_result('No data or output files created or modified', ...
        @() assert(isequal(baseline, data_snapshot(root))));
    fprintf('SUMMARY: %d passed, %d failed; %d writes suppressed; %d synthetic checkpoint loads.\n', ...
        hqtSmokeState.passed, hqtSmokeState.failed, hqtSmokeState.discardedWrites, hqtSmokeState.syntheticLoads);
    assert(hqtSmokeState.failed == 0, 'hqt:SmokeFailure', 'One or more smoke checks failed.');
end

function check_result(label, action)
% CHECK_RESULT Run an independent check and keep the rest of the suite useful.
    global hqtSmokeState
    try
        action();
        hqtSmokeState.passed = hqtSmokeState.passed + 1;
        fprintf('PASS %s\n', label);
    catch exception
        hqtSmokeState.failed = hqtSmokeState.failed + 1;
        fprintf('FAIL %s\n%s\n', label, getReport(exception, 'extended', 'hyperlinks', 'off'));
    end
    close all force;
end

function test_paths(root)
% TEST_PATHS Resolve a real reference after moving away from the project root.
    original = pwd;
    cleanup = onCleanup(@() cd(original));
    cd(tempdir);
    setup_project();
    assert(isfile(hqt_data_path('reference', 'mu=6,nz=30,vy=7_20.mat')));
    assert(strcmp(which('fourdif'), fullfile(root, 'third_party', 'dmsuite', 'fourdif.m')));
    assert(strcmp(which('tevol_driven_cluster'), fullfile(root, 'scripts', 'simulation', 'tevol_driven_cluster.m')));
    paths = hqt_paths();
    assert(strcmp(hqt_data_path('raw', 'example', 'check_point_t=1.mat'), ...
        fullfile(paths.raw, 'example', 'check_point_t=1.mat')));
end

function test_numerics()
% TEST_NUMERICS Compare derivatives to known analytic functions and filter limits.
    [x, D] = fourdif(33, 1);
    assert(max(abs(D*sin(3*x) - 3*cos(3*x))) < 1e-10);
    [z, Dz] = chebdif(20, 2);
    assert(max(abs(Dz(:,:,1)*z.^4 - 4*z.^3)) < 1e-10);
    assert(max(abs(Dz(:,:,2)*z.^4 - 12*z.^2)) < 1e-8);
    [x, D1, D6, D2, Dv1, Dv2] = FD(33);
    interior = 4:30;
    derivative = D1*x.^3;
    assert(max(abs(derivative(interior)-3*x(interior).^2)) < 1e-10);
    assert(all(isfinite([D6(:); D2(:); Dv1(:); Dv2(:)])));
    field = randn(17, 19) + 1i*randn(17, 19);
    emptyMask = false(size(field));
    assert(max(abs(ff(field, emptyMask)-real(field)), [], 'all') < 1e-12);
    assert(max(abs(ff_diss(field, emptyMask, 1)-field), [], 'all') < 1e-12);
    assert(max(abs(ff_diss(field, true(size(field)), 1)-exp(-1)*field), [], 'all') < 1e-12);
end

function test_diagnostics()
% TEST_DIAGNOSTICS Exercise periodic defects, empty fields, and boundary tracking.
    [x, y] = ndgrid((0:32)*2*pi/33, (0:32)*2*pi/33);
    phi = sin(x+0.271) + 1i*sin(y+0.613);
    [n, np, nm] = countN(phi);
    assert(n == 4 && np == 2 && nm == 2);
    [n2, np2, nm2, correlation] = countN_C(phi);
    assert(n2 == n && np2 == np && nm2 == nm && isfinite(correlation));
    [positive, negative] = find_vortices(phi, 0.4, 2);
    assert(isequal(size(positive), [2, 2]) && isequal(size(negative), [2, 2]));
    [positive, negative] = find_vortices(ones(9), 0.4, 2);
    assert(isempty(positive) && isempty(negative));
    [k, charge, number, vortices, spacing] = compute_vortex_structure_factor(phi.', 2*pi, 2*pi);
    assert(numel(k) == numel(charge) && numel(k) == numel(number));
    assert(size(vortices, 1) == n && isfinite(spacing.d_nnd));
    [centers, info] = find_vortex_dpc_pbc(vortices(:, [2, 1, 3]), 2*pi, 2*pi, 1, 0);
    assert(size(centers, 1) == info.num_clusters && info.num_clusters > 0);
    ux = sin(x); uy = cos(y);
    [S2, rx, ry] = structure_function_2d(ux, uy);
    assert(abs(S2(rx == 0, ry == 0)) < 1e-12);
    [lx, tx, ly, ty] = structure_functions(ux, uy, [2, 3], 4);
    assert(isequal(size(lx), [4, 2]) && all(lx(:) >= 0) && all(ty(:) >= 0));
    assert(max(abs(tx), [], 'all') < 1e-12 && max(abs(ty), [], 'all') < 1e-12);
    raw = zeros(2, 2, 2);
    raw(1,:,:) = reshape([9.8, 5, 1, 5], [1, 2, 2]);
    raw(2,:,:) = reshape([5.1, 0.1, 5, 1], [1, 2, 2]);
    tracked = track_vortices(raw, 10, 1);
    assert(abs(tracked(2,1,1)-0.1) < 1e-12);
end

function test_stationary()
% TEST_STATIONARY Validate the included seed and obtain a zero-flow seed in RAM.
    global hqtSmokeState
    seed = hqtSmokeState.seed;
    evalc('[z1,z2,vy1,it,update] = NewtonRaphson_vy(30,1,0,0,gather(seed.Z1),gather(seed.Z2),gather(seed.Vy1),-2*pi*7/20);');
    assert(all(isfinite([z1; z2; vy1])) && it < 1000 && isfinite(update));
    evalc('[z1,z2,vy1,it,update] = NewtonRaphson_vy(30,1,0,0,gather(seed.Z1),gather(seed.Z2),gather(seed.Vy1),0);');
    assert(all(isfinite([z1; z2; vy1])) && it < 1000 && isfinite(update));
    [~, Dz] = chebdif(30, 1);
    hqtSmokeState.zeroSeed = struct('mu', 6, 'Z1', z1, 'Z2', z2, ...
        'Vy1', vy1, 'rhoi', -2*Dz*z2);
end

function test_evolution(source)
% TEST_EVOLUTION Shorten only test-grid settings and time in the in-memory copy.
    text = fileread(source);
    text = regexprep(text, 'nx=\d+;', 'nx=33;');
    text = regexprep(text, 'nz=\d+;', 'nz=10;');
    text = regexprep(text, 'while\s+t<\d+', 'while t<0.08');
    % Run the output branch every step, but intercept every writer before evaluation.
    text = strrep(text, 'rem(n/50,1)==0', 'true');
    execute_source(disable_writes(text));
end

function test_analysis_entry(source)
% TEST_ANALYSIS_ENTRY Evaluate real initialization expressions up to first use.
    text = fileread(source);
    [~, name] = fileparts(source);
    if strcmp(name, 'movie_show_pair')
        % The pair notebook constructs its grid before its first checkpoint read.
        text = regexprep(text, 'nx=\d+;', 'nx=33;');
        text = regexprep(text, 'nz=\d+;', 'nz=10;');
        marker = '[vortices_temp, antivortices_temp]';
        index = strfind(text, marker);
        text = text(1:index(1)-1);
        text = strrep(text, 'for t=1:775', 'for t=1:1');
        % Finish the open checkpoint-time loop and chemical-potential loop.
        text = [text newline 'assert(all(isfinite(gather(O1(:)))));' newline 'end' newline 'end'];
    else
        % Stop after the first checkpoint is consumed and its dimensions are known.
        marker = regexp(text, '(?m)^[^\r\n]*load\(fileName\)[^\r\n]*$', 'end', 'once');
        if isempty(marker)
            % The older statistics notebook has a direct checkpoint filename expression.
            marker = regexp(text, '(?m)^[ \t]*load\(hqt_data_path\(''raw''[^\r\n]*$', 'end', 'once');
        end
        assert(~isempty(marker), 'No checkpoint read found in analysis entry.');
        tail = text(marker+1:end);
        dimensionEnd = regexp(tail, '(?m)^nx=NN\(1\);[^\r\n]*$', 'end', 'once');
        if ~isempty(dimensionEnd)
            marker = marker + dimensionEnd;
        end
        % Include the derivative setup but stop before checkpoint loops or large loads.
        if ~strcmp(name, 'statistics')
            loopStart = regexp(text, '(?m)^[ \t]*for t=', 'start', 'once');
            processedStart = strfind(text, 'load(hqt_data_path(''processed'',');
            boundaries = [loopStart, processedStart];
            assert(~isempty(boundaries), 'No analysis initialization boundary found.');
            marker = min(boundaries) - 1;
        end
        text = text(1:marker);
        if strcmp(name, 'statistics')
            text = regexprep(text, 'nx=\d+;', 'nx=33;');
            text = regexprep(text, 'ny=\d+;', 'ny=33;');
            text = regexprep(text, 'nz=\d+;', 'nz=10;');
            text = [text newline 'end'];
        end
        text = [text newline 'O1=func_Lz(Z1); assert(all(isfinite(gather(O1(:)))));'];
    end
    execute_source(disable_writes(text));
end

function test_stationary_notebook(source)
% TEST_STATIONARY_NOTEBOOK Run the continuation section and discard its save call.
    text = fileread(source);
    marker = strfind(text, '%% free energy');
    text = text(1:marker(1)-1);
    execute_source(disable_writes(text));
end

function test_cached_figures(source)
% TEST_CACHED_FIGURES Exercise real processed data and fitting without checkpoint reads.
    text = fileread(source);
    first = strfind(text, '%% figure2');
    last = strfind(text, '%% figure S2');
    assert(~isempty(first) && ~isempty(last), 'Expected paper-figure sections not found.');
    % These sections use the documented domain length of the archived large-domain run.
    preamble = ['L0=500; L=500; xx=(0:1000)''*500/1001;' newline];
    selected = text(first(1):last(1)-1);
    % Mark standalone analysis statements so an eval failure identifies its source line.
    lines = splitlines(selected);
    annotated = cell(numel(lines), 1);
    offset = count(text(1:first(1)-1), newline);
    for i = 1:numel(lines)
        line = char(lines{i});
        if ~isempty(regexp(line, '^[ \t]*(plot\(|scatter\(|errorbar\(|ylim\(|xlim\(|a1 =|std_c=|idx =|x_v=|load\()', 'once'))
            annotated{i} = sprintf('hqtSmokeLine=%d;\n%s', offset+i, line);
        else
            annotated{i} = line;
        end
    end
    execute_source(disable_writes([preamble strjoin(annotated, newline)]));
end

function text = disable_writes(text)
% DISABLE_WRITES Redirect all notebook I/O writers before evaluating any code.
    % No global MATLAB functions are shadowed; only the in-memory notebook changes.
    writers = {'save', 'imwrite', 'exportgraphics', 'saveas', 'diary', 'mkdir', 'writeVideo'};
    for i = 1:numel(writers)
        text = regexprep(text, ['\<' writers{i} '\>\s*\('], 'discard_write(');
    end
    text = regexprep(text, '\<load\>\s*\(', 'load_test_input(');
    text = regexprep(text, '\<getframe\>\s*\(', 'test_frame(');
    text = regexprep(text, '(?m)^\s*exit\s*$', 'error(''hqt:NonfiniteEvolution'', ''Evolution produced NaNs.'');');
    text = regexprep(text, '(?m)^\s*pause\s*$', 'error(''hqt:UnexpectedPause'', ''Unexpected parameter branch.'');');
    assert(isempty(regexp(text, '\<(save|imwrite|exportgraphics|saveas|diary|mkdir|VideoWriter|fopen)\>\s*\(', 'once')), ...
        'hqt:UnsafeTestInput', 'A writer remains in the selected test source.');
end

function execute_source(sourceCode)
% EXECUTE_SOURCE Keep notebook variables private and discard console chatter.
    % Avoid a variable named text, which would shadow MATLAB's annotation function.
    try
        evalc(sourceCode);
    catch exception
        if exist('hqtSmokeLine', 'var')
            fprintf('Paper source statement near line %d\n', hqtSmokeLine);
        end
        rethrow(exception);
    end
    if exist('Z1', 'var') && exist('Z3', 'var') && exist('Z4', 'var')
        assert(all(isfinite(gather([Z1(:); Z2(:); Z3(:); Z4(:)]))));
    end
end

function discard_write(varargin)
% DISCARD_WRITE Count a suppressed output operation without accessing the disk.
    global hqtSmokeState
    hqtSmokeState.discardedWrites = hqtSmokeState.discardedWrites + 1;
end

function frame = test_frame(varargin)
% TEST_FRAME Supply an in-memory RGB frame so no visible window capture is needed.
    frame = struct('cdata', zeros(4, 4, 3, 'uint8'), 'colormap', []);
end

function load_test_input(filename)
% LOAD_TEST_INPUT Load real compact inputs or explicitly generate an in-memory field.
    global hqtSmokeState
    paths = hqtSmokeState.paths;
    assert(startsWith(filename, paths.root), 'Input path is outside the repository.');
    if isfile(filename)
        % Restrict real reads to the compact files so tests cannot load gigabytes.
        info = dir(filename);
        assert(info.bytes < 2e6, 'A smoke test attempted to load a large dataset.');
        input = load(filename);
    elseif startsWith(filename, paths.reference) && contains(filename, 'vy=0_20.mat')
        input = hqtSmokeState.zeroSeed;
        assert(~isempty(input), 'The in-memory zero-flow seed is unavailable.');
    elseif startsWith(filename, paths.raw)
        % Determine dimensions from the notebook if it defined them before loading.
        nx = 33; ny = 33; nz = 10;
        if evalin('caller', 'exist(''nx'',''var'')'), nx = evalin('caller', 'nx'); end
        if evalin('caller', 'exist(''ny'',''var'')'), ny = evalin('caller', 'ny'); end
        if evalin('caller', 'exist(''nz'',''var'')'), nz = evalin('caller', 'nz'); end
        [x,y] = ndgrid((0:nx-1)*2*pi/nx, (0:ny-1)*2*pi/ny);
        phi = sin(x+0.271) + 1i*sin(y+0.613);
        [z,~] = chebdif(nz,1);
        z = (z+1)/2;
        input = struct('Z1', phi.*reshape(z.*(1-z), [1,1,nz]), ...
            'Z2', repmat(reshape(6*(1-z), [1,1,nz]), [nx,ny,1]), ...
            'Z3', zeros(nx,ny,nz), 'Z4', zeros(nx,ny,nz), 'rhoi', 6*ones(nx,ny,nz));
        hqtSmokeState.syntheticLoads = hqtSmokeState.syntheticLoads + 1;
    else
        error('hqt:MissingTestInput', 'No in-memory substitute defined for %s.', filename);
    end
    % Preserve MATLAB load-without-output semantics in the notebook's workspace.
    names = fieldnames(input);
    for i = 1:numel(names)
        assignin('caller', names{i}, input.(names{i}));
    end
end

function snapshot = data_snapshot(root)
% DATA_SNAPSHOT Record file names, sizes, and modification times without writes.
    files = [dir(fullfile(root, 'data', '**', '*')); dir(fullfile(root, 'outputs', '**', '*'))];
    files = files(~[files.isdir]);
    snapshot = cell(numel(files), 3);
    for i = 1:numel(files)
        snapshot(i,:) = {fullfile(files(i).folder, files(i).name), files(i).bytes, files(i).datenum};
    end
    snapshot = sortrows(snapshot, 1);
end

function restore_session(oldPath, oldFolder, oldVisible, oldRandom)
% RESTORE_SESSION Restore the caller's path, folder, graphics setting, and RNG.
    close all force;
    set(groot, 'DefaultFigureVisible', oldVisible);
    rng(oldRandom);
    cd(oldFolder);
    path(oldPath);
    clear global hqtSmokeState
end
