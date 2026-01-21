clearvars;

main_path = ''; %location of raw data (from NeurOne device files)
exChan = {'FT9', 'FT10', 'APBr', 'FDIr'};

datIn = [main_path 'data\'];
datOut = [main_path 'eeglab_dat\'];

cd(datIn);

files = dir;

%load channel lcos
load([main_path 'scripts\chans.mat']);

for a = 1:length(files)
    if regexp(string(files(a).name), regexptranslate('wildcard', 'OSIRIS_*'))
        
        %load dataset
        load(files(a).name);

        EEG = {};

        EEG.filepath = '';
        EEG.filename = '';
        EEG.subject = data.Session.TablePerson.PersonID;
        EEG.setname = data.Session.TableSession.ProtocolName;
        EEG.srate = data.properties.samplingRate;        
        EEG.icawinv = [];
        EEG.icaweights =[];
        EEG.icasphere = [];
        EEG.icaact = [];
        EEG.ref = 'FCz';
        EEG.trials = 1;

        dat = [];

        for b = 1:length(data.Protocol.TableInput)
            if ~contains(exChan, data.signalTypes{b})
                dat = [dat; data.signal.(data.signalTypes{b}).data'];                
            end
        end
        
        EEG.data = dat;
        EEG.chanlocs = chanlocs;
        EEG.nbchan = length(dat);
        EEG.pnts = numel(EEG.data)/numel(EEG.nbchan);
        EEG.xmin = 0;
        EEG.xmax = (EEG.pnts-1)/EEG.srate;

        fname = files(a).name;
        fname = fname(1:strfind(fname, '.')-1);
        fname = fname(strfind(fname, '_')+1:end);

        dirName = fname(1:strfind(fname, '_')-1);

        if ~isfolder([datOut dirName])
            mkdir([datOut dirName])
        end

        EEG = pop_saveset(EEG, 'filename', fname, 'filepath', [datOut dirName '\']);
    end
end






