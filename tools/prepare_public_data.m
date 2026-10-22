function outputFile = prepare_public_data(inputFile, outputFile)
% Create a minimal evaluation MAT without the original source-file path or
% redundant raw parser arrays. Does not overwrite the source input.
arguments
    inputFile (1,1) string
    outputFile (1,1) string
end
assert(~strcmpi(char(inputFile),char(outputFile)), ...
    'Choose a different output path; preserve the source snapshot.');
S=load(inputFile,'reps','parsed');
assert(isfield(S,'reps') && isfield(S,'parsed'));
reps=S.reps;
assert(istable(reps.metadata));
assert(all(ismember({'LogID','Campaign'},reps.metadata.Properties.VariableNames)));
reps.metadata=reps.metadata(:,{'LogID','Campaign'});
% Retain only fields required by the archived evaluation code.
keep={'A_rankPairs','B_topK','C_alignedSparse','formatVersion','metadata','targetXY_m'};
remove=setdiff(fieldnames(reps),keep);if ~isempty(remove),reps=rmfield(reps,remove);end
parsed=struct('WLANIDStart',S.parsed.WLANIDStart); %#ok<NASGU>
assert(size(reps.targetXY_m,1)==118930);
assert(size(reps.C_alignedSparse.PowerObserved_dBm,2)==5415);
save(outputFile,'reps','parsed','-v7.3');
fprintf('Wrote minimal evaluation data: %s\n',outputFile);
fprintf('Retained numeric transmitter IDs, coordinates, LogID and Campaign.\n');
fprintf('Confirm the applicable data-sharing permission before publication.\n');
end
