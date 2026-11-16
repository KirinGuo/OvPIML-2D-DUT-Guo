function env=v9_environment
% Discover the repository-local Python and the installed CUDA LibTorch.
root=fileparts(fileparts(mfilename('fullpath')));
exe=string(getenv('V9_PORTABLE_PYTHON'));
if strlength(exe)==0,exe=string(fullfile(root,'.venv','Scripts','python.exe'));end
assert(isfile(exe),'V9:Environment','Run SETUP.cmd to create .venv with Python 3.10 and CUDA PyTorch.');
torch=string(getenv('V9_PORTABLE_TORCH'));
if strlength(torch)==0
    command='"'+exe+'" -I -c "import importlib.util;print(importlib.util.find_spec(''torch'').origin)"';
    [status,out]=system(command);
    assert(status==0,'V9:Environment','Cannot find PyTorch in the selected Python.');
    torch=string(fileparts(strtrim(out)));
end
assert(isfolder(fullfile(torch,'lib')),'V9:Environment','LibTorch DLL directory is missing.');
paths=strsplit(getenv('PATH'),pathsep);lib=char(fullfile(torch,'lib'));
paths=paths(~strcmpi(paths,lib));setenv('PATH',strjoin([{lib},paths],pathsep));
env=struct('python',exe,'torch',torch,'root',root);
end
