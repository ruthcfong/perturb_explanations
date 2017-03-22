classdef LRN_custom < dagnn.LRN
  properties
      custom_type = 'nobackprop';
  end

  methods
%     function outputs = forward(obj, inputs, params)
%       outputs{1} = vl_nnnormalize(inputs{1}, obj.param) ;
%     end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
        %derInputs{1} = vl_nnnormalize(inputs{1}, obj.param, derOutputs{1}) ;
        switch obj.custom_type
            case {'hacked','nobackprop'}
                derInputs{1} = derOutputs{1};
            otherwise % TODO: implement normalizelp?
                error('LRN_custom type %s is not implemented', obj.custom_type);
        end
        derParams = {} ;
    end

    function obj = LRN_custom(varargin)
        obj.load(varargin) ;
    end
  end
end
