classdef Conv_Excitation_Backprop < dagnn.Conv
    properties
        % none
    end
    
    methods
                
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        % Implemented according to Alg. 1 description in Zhang et al., 2016
        % and based on Caffe implementation here: 
        % https://github.com/jimmie33/Caffe-ExcitationBP/blob/master/src/caffe/layers/conv_layer.cpp

            if ~obj.hasBias, params{2} = [] ; end

            % differentiate w.r.t. parameters as typical
            [~, derParams{1}, derParams{2}] = vl_nnconv(...
                inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;

            Wp = params{1};
            Wp(Wp < 0) = 0;
            
            X = vl_nnconv(inputs{1}, Wp, [], ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:});
            
            Y = derOutputs{1} ./ X;
            if length(find(isnan(Y))) > 0
                warning('NaNs found in Conv_Excitation_Backprop layer with size %s', mat2str(obj.size));
                Y(isnan(Y)) = 0;
            end

            [Z, ~, ~] = vl_nnconv(inputs{1}, Wp, [], Y, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:});
            
            derInputs{1} = inputs{1} .* Z;
        end
        
        function obj = Conv_Excitation_Backprop(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            obj.size = obj.size ;
            obj.stride = obj.stride ;
            obj.pad = obj.pad ;
        end

    end
end