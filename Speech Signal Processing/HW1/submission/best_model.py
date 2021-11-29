from submission import model
#from submission import dataset as ds
from submission import features as ft

class BestAnalysis(ft.MFCCAnalysis):
    def perform(self,wavfile):
        ### BEGIN YOUR CODE
        ##
        x, sampling_rate = librosa.load(wavfile)
        window_duration_ms = 40
        n_fft = int((window_duration_ms / 1000.) * sampling_rate)

        hop_duration_ms = 10
        hop_length = int((hop_duration_ms / 1000.) * sampling_rate)
        mfcc_count = 13
        mfcc=librosa.feature.mfcc(x,sr=sampling_rate,n_mfcc=mfcc_count,n_fft=n_fft,hop_length=hop_length)
        mfccs_and_deltas=mfcc
        mfcc_delta1=librosa.feature.delta(mfcc,order=1)
        mfccs_and_deltas=np.concatenate((mfccs_and_deltas,mfcc_delta1),axis=0)
        mfcc_delta2=librosa.feature.delta(mfcc,order=2)
        mfccs_and_deltas=np.concatenate((mfccs_and_deltas,mfcc_delta2),axis=0)
        mfccs_and_deltas=mfccs_and_deltas.transpose(1,0)
        ##
        #### END YOUR CODE
        return mfccs_and_deltas, hop_length, n_fft
        
    
wav_files, label_files = read_data_folder("audio/part-2/train")
X_train, y_train, _, _  = prepare_data(wav_files, label_files, stanalysis=BestAnalysis())

X_train, scaler_mean ,scaler_std= normalize_mean(X_train)
    
wav_files, label_files = read_data_folder("audio/part-2/val")
print (len(wav_files), len(label_files))
X_test, y_test, test_seg_ids, test_seg2labels  = prepare_data(
        wav_files, label_files, stanalysis=BestAnalysis())

X_test = apply_normalize_mean(X_test, scaler_mean,scaler_std)
cls = model.PhonemeClassifier()
cls.train(X_train, y_train)
y_pred = cls.test(X_test, y_test)

model.segment_based_evaluation(y_pred, test_seg_ids, test_seg2labels)
