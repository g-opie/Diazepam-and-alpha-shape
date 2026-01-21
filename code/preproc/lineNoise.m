clearvars;

mPath = ''; % location of input data, in EEGlab format
load([mPath 'working\' 'cond.mat'])
load([mPath 'working\' 'time.mat'])

cd([mPath 'eeglab_dat\']);
subjs = dir('0*');

eeglab;

for sub = 1:length(subjs)

    path_in = [mPath 'eeglab_dat\' subjs(sub).name '\'];
    for cnd = 1:length(cond)
        for tme = 1:length(time)
            
            %load data
            EEG = pop_loadset('filename',[subjs(sub).name '_' cond{1, cnd} time{1, tme} '.set'], 'filepath',path_in); 

            %downsample
            EEG = pop_resample(EEG, 1000);

            %zapline
            [EEG, fig] = clean_data_with_zapline_plus_eeglab_wrapper(EEG, struct('noisefreqs', 'line'));
            close(fig);

            %save
            EEG = pop_saveset(EEG, 'filename', [subjs(sub).name '_' cond{1, cnd} time{1, tme} '_line.set'], 'filepath', path_in);
        end
    end
end





