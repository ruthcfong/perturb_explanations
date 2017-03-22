classdef ReLU_custom < dagnn.ReLU
    properties
        custom_type = 'deconvnet';
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if obj.leak ~= 0
                error('ReLU_custom type %s does not support leak', obj.custom_type);
            end
            outputs{1} = vl_nnrelu(inputs{1}, [], ...
                                 'leak', obj.leak, obj.opts{:}) ;
        end
        
        function forwardAdvanced(obj, layer)
            if obj.leak ~= 0
                error('ReLU_custom type %s does not support leak', obj.custom_type);
            end
            
            if ~obj.useShortCircuit || ~obj.net.conserveMemory
                forwardAdvanced@dagnn.Layer(obj, layer) ;
                return ;
            end
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            net.vars(out).value = vl_nnrelu(net.vars(in).value, [], ...
                                          'leak', obj.leak, ...
                                          obj.opts{:}) ;
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
            if ~net.vars(in).precious & net.numPendingVarRefs(in) == 0
                net.vars(in).value = [] ;
            end
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            if obj.leak ~= 0
                error('ReLU_custom type %s does not support leak', obj.custom_type);
            end

            switch obj.custom_type
                case 'deconvnet'
                    derInputs{1} = max(derOutputs{1}, 0);
                case 'nobackprop'
                    derInputs{1} = derOutputs{1};
                case {'guidedbackprop', 'eccv16'}
                    derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}, ...
                        'leak', obj.leak, ...
                        obj.opts{:}) ;
                    derInputs{1} = max(derInputs{1}, 0);
                otherwise
                    error('ReLU_custom type %s is not supported', obj.custom_type);
            end
            derParams = {} ;
        end
        
        function backwardAdvanced(obj, layer)
            if obj.leak ~= 0
                error('ReLU_custom type %s does not support leak', obj.custom_type);
            end

            if ~obj.useShortCircuit || ~obj.net.conserveMemory
                backwardAdvanced@dagnn.Layer(obj, layer) ;
                return ;
            end
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;

            if isempty(net.vars(out).der), return ; end

            switch obj.custom_type
                case 'deconvnet'
                    derInput = max(net.vars(out).der, 0);
                case 'nobackprop'
                    derInput = net.vars(out).der;
                case {'guidedbackprop','eccv16'}
                    derInput = vl_nnrelu(net.vars(out).value, net.vars(out).der, ...
                        'leak', obj.leak, obj.opts{:}) ;
                    derInput = max(derInput, 0);
                otherwise
                    error('ReLU_custom type %s is not supported', obj.custom_type);
            end

            if ~net.vars(out).precious
                net.vars(out).der = [] ;
                net.vars(out).value = [] ;
            end

            if net.numPendingVarRefs(in) == 0
                net.vars(in).der = derInput ;
            else
                net.vars(in).der = net.vars(in).der + derInput ;
            end
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
        end
                
        function obj = ReLU_custom(varargin)
            obj.load(varargin) ;
            % normalize field by implicitly calling setters defined in
            % dagnn.Filter and here
            %obj.custom_type = obj.custom_type ;
            %obj.stride = obj.stride ;
            %obj.pad = obj.pad ;
        end
    end
end