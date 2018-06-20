function myval = getFromStruct(mystruct,myfield,defaultval)
% Inherit specified field from a struct if the field exists,
% otherwise set to default value.
%
% INPUTS:
% - mystruct: a struct variable
% - myfield: a string (supposed to be the field name)
% - defaultval: default value to set when field is nonexisting
% OUTPUT:
% - myval: value (of the specified field, or the default)
% ------------------------------------------------------------------------

% Copyright 2018 Ji Hyun Bak

if(isfield(mystruct,myfield))
    myval = mystruct.(myfield);
else
    myval = defaultval; % set default value
end

end
