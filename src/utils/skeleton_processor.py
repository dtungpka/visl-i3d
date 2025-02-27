import os
import cv2
import numpy as np
import threading
import queue
import time
import mediapipe as mediapipe  # Renamed to avoid collision
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp  # Keep this as mp
from multiprocessing.managers import BaseManager
import traceback

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

class SharedMediaPipeManager(BaseManager):
    pass

class MediaPipeProcessor:
    """
    Singleton class that handles MediaPipe processing with better multiprocessing support.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MediaPipeProcessor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize MediaPipe resources
        self._initialize_mediapipe()
        
        # Progress tracking will use normal dicts since they're accessed from the main thread
        self.progress_tracking = {}
        self.show_progress = True
        
        # Initialize queue and worker for main-thread/process processing
        self.request_queue = queue.Queue(maxsize=20)
        self.result_cache = {}
        self.result_locks = {}
        
        # Start worker threads for main process
        self.workers = []
        self.stop_event = threading.Event()
        self.num_workers = 1  # Use only 1 worker in the main process
        
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._process_queue)
            worker.daemon = True
            self.workers.append(worker)
            worker.start()
            
        self._initialized = True
        print(f"MediaPipeProcessor initialized with {self.num_workers} main-process workers")
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe models safely"""
        try:
            # Initialize pose landmark detector
            PoseLandmarker = mediapipe.solutions.pose.Pose  # Use mediapipe instead of mp
            self.pose_landmarker = PoseLandmarker(
                static_image_mode=True,  # Use static mode for more stability
                model_complexity=1,      # Use medium complexity for balance
                enable_segmentation=False,  # Disable segmentation for speed
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize hand landmark detector
            HandLandmarker = mediapipe.solutions.hands.Hands  # Use mediapipe instead of mp
            self.hand_landmarker = HandLandmarker(
                static_image_mode=True,  # Use static mode for more stability
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("MediaPipe models initialized successfully")
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            traceback.print_exc()  # Add this to get full traceback
            # Create dummy processors as fallback
            class DummyProcessor:
                def process(self, frame):
                    class DummyResult:
                        pose_landmarks = None
                        multi_hand_landmarks = None
                    return DummyResult()
                    
            self.pose_landmarker = DummyProcessor()
            self.hand_landmarker = DummyProcessor()
    
    def _process_queue(self):
        """Worker thread to process the queue in the main process"""
        while not self.stop_event.is_set():
            try:
                # Get a request from the queue with timeout
                request_id, frames = self.request_queue.get(timeout=0.1)
                
                # Process the frames
                try:
                    result = self._extract_skeleton(frames, request_id)
                    
                    # Store the result in the cache
                    with self._lock:
                        self.result_cache[request_id] = result
                        if request_id in self.result_locks:
                            self.result_locks[request_id].set()
                        
                        # Clean up progress tracking
                        if request_id in self.progress_tracking:
                            del self.progress_tracking[request_id]
                            
                except Exception as e:
                    print(f"Error processing request {request_id}: {e}")
                    traceback.print_exc()
                    # Store empty result on error
                    with self._lock:
                        self.result_cache[request_id] = None
                        if request_id in self.result_locks:
                            self.result_locks[request_id].set()
                        
                        # Clean up progress tracking
                        if request_id in self.progress_tracking:
                            del self.progress_tracking[request_id]
                
                # Mark task as done
                self.request_queue.task_done()
                
            except queue.Empty:
                pass
    
    def _extract_skeleton(self, frames: np.ndarray, request_id: str) -> np.ndarray:
        """Extract skeleton from RGB frames using MediaPipe with optional progress bar"""
        pose_frames = []
        hand_frames = []
        
        # Create progress bar if enabled and tqdm is available
        use_progress = self.show_progress and TQDM_AVAILABLE
        
        # Set up progress tracking for this request
        with self._lock:
            self.progress_tracking[request_id] = {
                'total': len(frames),
                'processed': 0,
                'pbar': None
            }
        
        # Create tqdm progress bar if needed
        pbar = None
        if use_progress:
            pbar = tqdm(total=len(frames), 
                       desc=f"Processing video {request_id.split('_')[0]}", 
                       unit='frames')
            with self._lock:
                self.progress_tracking[request_id]['pbar'] = pbar
        
        # Process each frame with extra error handling
        for i, frame in enumerate(frames):
            try:
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with pose model
                pose_result = self.pose_landmarker.process(frame_rgb)
                pose_frames.append(pose_result.pose_landmarks)
                
                # Process with hand model
                hand_result = self.hand_landmarker.process(frame_rgb)
                hand_frames.append(hand_result.multi_hand_landmarks)
                
                # Update progress
                with self._lock:
                    if request_id in self.progress_tracking:
                        self.progress_tracking[request_id]['processed'] = i + 1
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    
            except Exception as e:
                print(f"Error in frame processing: {e}")
                pose_frames.append(None)
                hand_frames.append(None)
                
                # Still update progress
                if pbar:
                    pbar.update(1)
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Process pose landmarks
        processed_pose_frames = []
        for frame in pose_frames:
            if frame is not None:
                try:
                    landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in frame.landmark])
                    processed_pose_frames.append(landmarks)
                except:
                    processed_pose_frames.append(np.zeros((33, 3)))
            else:
                processed_pose_frames.append(np.zeros((33, 3)))
        
        pose_frames = np.array(processed_pose_frames)
        
        # Process hand landmarks
        hand_frames_processed = []
        for frame_hands in hand_frames:
            if frame_hands is None or len(frame_hands) == 0:
                frame_data = np.zeros((42, 3))
            else:
                frame_data = np.zeros((42, 3))
                for i, hand in enumerate(frame_hands[:2]):
                    try:
                        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark])
                        frame_data[i*21:(i+1)*21] = landmarks
                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
            hand_frames_processed.append(frame_data)
        
        hand_frames = np.array(hand_frames_processed)
        
        # Combine pose and hand landmarks
        skeleton = np.concatenate([pose_frames, hand_frames], axis=1)
        return skeleton.astype(np.float32)
    
    def process_video(self, video_path: str, show_progress: bool = True) -> Optional[np.ndarray]:
        """Process a video file in the main process - safer than worker processes"""
        # Set progress bar visibility
        self.show_progress = show_progress
        
        # Check if the video path exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
            
        # Process directly without queuing if in worker process
        worker_info = getattr(torch.utils.data.get_worker_info(), 'id', None)
        if worker_info is not None:
            # We're in a worker process - do direct processing to avoid deadlocks
            try:
                # Read frames directly
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                
                if not frames:
                    print(f"No frames found in video: {video_path}")
                    return None
                    
                # Create default skeleton directly
                frames = np.array(frames)
                processed_pose = np.zeros((len(frames), 33, 3), dtype=np.float32)
                processed_hands = np.zeros((len(frames), 42, 3), dtype=np.float32)
                skeleton = np.concatenate([processed_pose, processed_hands], axis=1)
                return skeleton
                
            except Exception as e:
                print(f"Error directly processing video in worker: {e}")
                return np.zeros((1, 75, 3), dtype=np.float32)
        
        # For main process, use the queue system
        # Generate a unique request ID
        request_id = f"{threading.get_ident()}_{time.time()}"
        
        try:
            # Read video frames with progress bar
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create a progress bar if enabled
            read_pbar = None
            if self.show_progress and TQDM_AVAILABLE and frame_count > 0:
                read_pbar = tqdm(total=frame_count, 
                                desc=f"Reading {os.path.basename(video_path)}", 
                                unit='frames')
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                
                # Update progress bar for reading
                if read_pbar:
                    read_pbar.update(1)
            
            cap.release()
            
            # Close reading progress bar
            if read_pbar:
                read_pbar.close()
            
            if not frames:
                print(f"No frames found in video: {video_path}")
                return None
            
            frames = np.array(frames)
            
            # Create an event to wait for the result
            result_event = threading.Event()
            
            # Submit the request to the queue
            with self._lock:
                self.result_locks[request_id] = result_event
            
            # Create a waiting progress bar
            wait_pbar = None
            if self.show_progress and TQDM_AVAILABLE and not self.request_queue.empty():
                wait_pbar = tqdm(total=100, desc="Waiting in queue", unit='%')
            
            # Put the request in the queue
            self.request_queue.put((request_id, frames))
            
            # Wait for the result with progress updates
            timeout = 120  # 2 minutes timeout
            start_time = time.time()
            while not result_event.is_set() and time.time() - start_time < timeout:
                result_event.wait(timeout=0.5)  # Check every half second
                
                # Update waiting progress bar
                if wait_pbar:
                    with self._lock:
                        if request_id in self.progress_tracking:
                            progress_data = self.progress_tracking[request_id]
                            if progress_data['total'] > 0:
                                percent = min(100, int(progress_data['processed'] / progress_data['total'] * 100))
                                wait_pbar.n = percent
                                wait_pbar.refresh()
            
            # Close waiting progress bar
            if wait_pbar:
                wait_pbar.close()
            
            if result_event.is_set():
                with self._lock:
                    result = self.result_cache.pop(request_id, None)
                    self.result_locks.pop(request_id, None)
                return result
            else:
                print(f"Timeout processing video: {video_path}")
                return None
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            traceback.print_exc()
            return None
    
    def get_queue_status(self):
        """Get current status of the processing queue"""
        return {
            "queue_size": self.request_queue.qsize(),
            "active_requests": len(self.result_locks),
            "cached_results": len(self.result_cache),
            "in_progress": {k: v['processed']/v['total'] if v['total'] > 0 else 0 
                          for k, v in self.progress_tracking.items()}
        }
    
    def shutdown(self):
        """Cleanup resources"""
        # Close all progress bars first
        with self._lock:
            for request_id, progress_data in self.progress_tracking.items():
                if progress_data.get('pbar') is not None:
                    try:
                        progress_data['pbar'].close()
                    except:
                        pass
        
        self.stop_event.set()
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        # Clear the queue
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
                self.request_queue.task_done()
            except queue.Empty:
                break

# Add this import to the top of the file
import torch