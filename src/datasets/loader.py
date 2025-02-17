from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        # Load data from the specified directory
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mp4'):  # Assuming video files
                data.append(os.path.join(self.data_dir, filename))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        video = self._load_video(video_path)

        if self.transform:
            video = self.transform(video)

        return video

    def _load_video(self, video_path):
        # Load video and convert to a suitable format (e.g., numpy array)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames)  # Convert list of frames to numpy array

def get_dataloader(data_dir, batch_size=1, shuffle=True, transform=None):
    dataset = CustomDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)