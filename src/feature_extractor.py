import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        
    def extract_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    
    def extract_chroma(self, audio):
        """Extract chroma features"""
        chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate)
        return np.mean(chroma.T, axis=0)
    
    def extract_mel(self, audio):
        """Extract mel spectrogram features"""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        return np.mean(mel.T, axis=0)
    
    def extract_contrast(self, audio):
        """Extract spectral contrast features"""
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        return np.mean(contrast.T, axis=0)
    
    def extract_tonnetz(self, audio):
        """Extract tonnetz features"""
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        return np.mean(tonnetz.T, axis=0)
    
    def extract_zcr(self, audio):
        """Extract zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(audio)
        return np.mean(zcr)
    
    def extract_spectral_centroid(self, audio):
        """Extract spectral centroid"""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        return np.mean(centroid)
    
    def extract_spectral_rolloff(self, audio):
        """Extract spectral rolloff"""
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        return np.mean(rolloff)
    
    def extract_all_features(self, audio):
        """Extract all features from audio signal"""
        features = []
        
        # MFCC features
        mfcc = self.extract_mfcc(audio)
        features.extend(mfcc)
        
        # Chroma features
        chroma = self.extract_chroma(audio)
        features.extend(chroma)
        
        # Mel spectrogram features
        mel = self.extract_mel(audio)
        features.extend(mel)
        
        # Spectral contrast
        contrast = self.extract_contrast(audio)
        features.extend(contrast)
        
        # Tonnetz features
        tonnetz = self.extract_tonnetz(audio)
        features.extend(tonnetz)
        
        # Additional features
        features.append(self.extract_zcr(audio))
        features.append(self.extract_spectral_centroid(audio))
        features.append(self.extract_spectral_rolloff(audio))
        
        return np.array(features)
    
    def extract_features_batch(self, audio_batch):
        """Extract features from a batch of audio signals"""
        feature_batch = []
        for audio in audio_batch:
            features = self.extract_all_features(audio)
            feature_batch.append(features)
        return np.array(feature_batch)
    
    def fit_scaler(self, features):
        """Fit the scaler on training features"""
        self.scaler.fit(features)
    
    def transform(self, features):
        """Transform features using fitted scaler"""
        return self.scaler.transform(features)