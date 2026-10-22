classdef FastKANLayer < nnet.layer.Layer & nnet.layer.Formattable
    % FastKANLayer  Gaussian-RBF Kolmogorov-Arnold Network layer.
    %
    % Each input coordinate is expanded on NumGrids Gaussian RBFs. The
    % learned spline/RBF edge contributions are summed into each output,
    % with an additional learned SiLU base path. A learnable LayerNorm is
    % applied to the input of the RBF branch, matching the FastKAN design.

    properties
        InputSize
        OutputSize
        NumGrids
        GridMin
        GridMax
        Denominator
        Grid
        Epsilon = 1e-5
    end

    properties (Learnable)
        SplineWeights   % [OutputSize x (InputSize*NumGrids)]
        BaseWeights     % [OutputSize x InputSize]
        BaseBias        % [OutputSize x 1]
        NormScale       % [InputSize x 1]
        NormOffset      % [InputSize x 1]
    end

    methods
        function layer = FastKANLayer(inputSize, outputSize, numGrids, name)
            if nargin < 4
                name = '';
            end
            layer.Name = name;
            layer.Description = sprintf('FastKAN RBF layer %d -> %d (%d grids)', ...
                inputSize, outputSize, numGrids);

            layer.InputSize = inputSize;
            layer.OutputSize = outputSize;
            layer.NumGrids = numGrids;
            layer.GridMin = -2;
            layer.GridMax = 2;
            layer.Grid = single(linspace(layer.GridMin, layer.GridMax, numGrids));
            layer.Denominator = single((layer.GridMax-layer.GridMin)/(numGrids-1));

            layer.SplineWeights = single(0.02 * randn(outputSize, inputSize*numGrids));
            layer.BaseWeights = single(sqrt(2/max(inputSize,1)) * randn(outputSize, inputSize));
            layer.BaseBias = zeros(outputSize,1,'single');
            layer.NormScale = ones(inputSize,1,'single');
            layer.NormOffset = zeros(inputSize,1,'single');
        end

        function Z = predict(layer, X)
            % X is features x batch. stripdims preserves autodiff while
            % avoiding dependence on formatted dlarray labels.
            X0 = stripdims(X);
            B = size(X0,2);

            % LayerNorm across features for each sample.
            mu = mean(X0,1);
            v = mean((X0-mu).^2,1);
            Xn = (X0-mu) ./ sqrt(v + layer.Epsilon);
            Xn = Xn .* layer.NormScale + layer.NormOffset;

            % RBF expansion: [InputSize x NumGrids x Batch].
            X3 = reshape(Xn,[layer.InputSize 1 B]);
            G = reshape(layer.Grid,[1 layer.NumGrids 1]);
            % Keep fixed RBF grid on the same device as the minibatch.
            if isa(extractdata(X0),'gpuArray')
                G = gpuArray(G);
            end
            Phi = exp(-((X3-G)./layer.Denominator).^2);
            Phi = reshape(Phi,[layer.InputSize*layer.NumGrids B]);

            % Learned KAN edge functions.
            Zs = layer.SplineWeights * Phi;

            % Residual/base SiLU path.
            A = X0 ./ (1 + exp(-X0));
            Zb = layer.BaseWeights * A + layer.BaseBias;

            Z = Zs + Zb;
        end

        function Z = forward(layer, X)
            % Same differentiable computation during training.
            Z = predict(layer,X);
        end
    end
end
