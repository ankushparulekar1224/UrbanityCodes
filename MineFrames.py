import os 
import cv2
from pathlib import Path

vidoes_folder = "c:/Urbanity-data/VideosLatest"
output_folder="c:/Urbanity-data/ankush8/images"
frame_interval=2
g_frame_count=0

os.makedirs(output_folder,exist_ok=True)

extensions=(".mp4",".avi",".mov",".mkv",".mpeg")

for video_file in Path(vidoes_folder).iterdir():
    if video_file.suffix.lower() not in extensions:
        continue

    print(f"\nprocessing:{video_file.name}")

    cap=cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        print(f"not able to open :{video_file.name}")
        continue

    fps=cap.get(cv2.CAP_PROP_FPS)
    frames_interval=fps*frame_interval

    frame_count=0
    saved_frames=0

    video_name=video_file.stem

    while(True):
        success,frame=cap.read()

        if not success:
            break

        if frame_count%frames_interval==0:
            output_path=os.path.join(output_folder,str(g_frame_count)+".jpg")
            cv2.imwrite(output_path,frame)
            g_frame_count+=1
        
            saved_frames+=1
        frame_count+=1
        

    cap.release()
    print(f"saved {saved_frames} from {video_name}")
print(f"Done extracting {g_frame_count}") 
        




    
    





    


    


    
