import fire
import os
import fsspec
from main.paper_experiments.experiments import run_model_imagefolder
from argparse import Namespace

clip_model_name = 'ViT-L/14'
prompt_path = f'data/{clip_model_name.replace("/", "-")}/prompts.p'

def main_imagedataset(input_folder, output_folder, gpu=0):
    args = Namespace(
        language_model='Clip_'+clip_model_name,
        model_type='sim',
        prompt_path=prompt_path,
        only_inappropriate=True,
        input_type='img',
        gpu=[gpu],  # 关键，传入gpu参数！
    )
    run_model_imagefolder(args, input_folder, output_folder)

if __name__ == '__main__':
    fire.Fire(main_imagedataset)