from ultralytics import YOLO
import os
import argparse

model = YOLO('yolotestresidual.pt')
# open the video file
video_path = r"residual.mp4"
video_path = "residual/"
#results = model.track(source=video_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='calculate motmetric for yolo  model')
    parser.add_argument('--mot_dir', default='datasets',
                        help='training dataset', type=str)
    parser.add_argument('--dataset_path', default='/home/modesto/PycharmProjects/compressed_tracking/datasets',
                        help='the path to datasets', type=str)
    parser.add_argument('--dataset', default='MOT20',
                        choices=['MOT17', 'MOT20', 'MOT17-20'],
                        help='the dataset to tracking', type=str)
    parser.add_argument('--dataset_args', default='video_residual',
                        choices=['annotations', 'test', 'train', 'video', 'yolo','video_residual'],
                        help='the dataset to tracking', type=str)
    parser.add_argument('--image_format', default='residual',
                        choices=['residual', 'iframe', 'mv', 'mix', 'img1'],
                        help="the phase for this running", type=str)
    parser.add_argument('--dec_sort', default='FRCNN',
                        choices=['FRCNN', 'DPM', 'SDP'],
                        help="the phase for this running", type=str)
    parser.add_argument('--output_folder', default='output_metric',
                        help="the phase for this running", type=str)
    args = parser.parse_args()
    return args


def get_metric_from_data(intput_data_path, data_name,output_folder):
    data_folder = os.path.join(args.output_folder, data_name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    results = model.track(source=intput_data_path)

    with open(f'{data_folder}/{data_name}.txt', 'w') as f:
        for frame_id, result in enumerate(results):
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # Convert from tensor to list
                if box.id is not None:
                    track_id = box.id.item()  # Get track id
                conf = box.conf.item()  # Get confidence score
                f.write(
                    f'{frame_id + 1},{track_id},{bbox[0]},{bbox[1]},{bbox[2] - bbox[0]},{bbox[3] - bbox[1]},-1,-1,{conf}\n')

if __name__ == '__main__':
    args = parse_args()
    dataset = os.path.join(args.dataset_path, args.dataset,args.dataset_args)
    print(dataset)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for video in os.listdir(dataset):
        print(video)
        if args.dec_sort in video:
            data_name = video
            data_folder = os.path.join(dataset,video)
            #data_folder = os.path.join(data_folder, args.image_format)
            print(f"Path to video = {data_folder}")
            get_metric_from_data(data_folder,data_name,args.output_folder)
