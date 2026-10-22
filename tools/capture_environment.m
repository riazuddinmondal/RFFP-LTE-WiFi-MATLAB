function capture_environment(outputFile)
% Run this on the machine that performs the clean-release MATLAB check.
if nargin<1,outputFile='environment_report.txt';end
fid=fopen(outputFile,'w');assert(fid>=0);c=onCleanup(@()fclose(fid));
fprintf(fid,'MATLAB version: %s\nRelease: %s\nComputer: %s\n',version,version('-release'),computer);
v=ver;
for k=1:numel(v),fprintf(fid,'%s | %s | %s\n',v(k).Name,v(k).Version,v(k).Release);end
fprintf(fid,'\nGPU details (if available):\n');
try
 g=gpuDevice;fprintf(fid,'Name: %s\nCompute capability: %s\nTotal memory bytes: %.0f\n',g.Name,g.ComputeCapability,g.TotalMemory);
catch ME
 fprintf(fid,'GPU not queried: %s\n',ME.identifier);
end
end
