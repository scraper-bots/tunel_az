import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf

class SpeechEmotionDataLoader:
    def __init__(self, data_path, sample_rate=22050, duration=3.0):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.label_encoder = LabelEncoder()
        
    def load_ravdess_data(self):
        """
        Load RAVDESS dataset
        Filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
        Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        """
        data = []
        labels = []
        
        emotion_mapping = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract emotion from filename
                    emotion_code = int(file.split('-')[2])
                    emotion = emotion_mapping.get(emotion_code)
                    
                    if emotion:
                        try:
                            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
                            # Pad or trim to fixed length
                            target_length = int(self.sample_rate * self.duration)
                            if len(audio) < target_length:
                                audio = np.pad(audio, (0, target_length - len(audio)))
                            else:
                                audio = audio[:target_length]
                            
                            data.append(audio)
                            labels.append(emotion)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
        
        return np.array(data), np.array(labels)
    
    def load_tess_data(self):
        """Load TESS dataset"""
        data = []
        labels = []
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract emotion from filename or folder structure
                    # TESS typically has emotion in filename: "word_emotion.wav"
                    emotion = file.split('_')[-1].split('.')[0].lower()
                    
                    try:
                        audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
                        target_length = int(self.sample_rate * self.duration)
                        if len(audio) < target_length:
                            audio = np.pad(audio, (0, target_length - len(audio)))
                        else:
                            audio = audio[:target_length]
                        
                        data.append(audio)
                        labels.append(emotion)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        
        return np.array(data), np.array(labels)
    
    def prepare_data(self, dataset_type='ravdess'):
        """Load and prepare data for training"""
        if dataset_type == 'ravdess':
            X, y = self.load_ravdess_data()
        elif dataset_type == 'tess':
            X, y = self.load_tess_data()
        else:
            raise ValueError("Unsupported dataset type. Use 'ravdess' or 'tess'")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, self.label_encoder.classes_