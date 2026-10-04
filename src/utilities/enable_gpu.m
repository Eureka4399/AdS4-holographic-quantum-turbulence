% ENABLE_GPU Opt in to MATLAB CUDA forward compatibility.
% Requires Parallel Computing Toolbox; call explicitly for a newer GPU if needed.

% Opt in before creating or loading gpuArray objects on a newer GPU architecture.
parallel.gpu.enableCUDAForwardCompatibility(true)
