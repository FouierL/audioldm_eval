import torch
from audioldm_eval import EvaluationHelperParallel
import torch.multiprocessing as mps
import os

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

generation_result_path = "/data2/chenxuwu/DiT-MoE-Video2Audio-v2/outputs/mel/dit_crossattn_clean_audio/CFG4.5_euler_26gen_wav_16k_80"
target_audio_path = "/data2/chenxuwu/DiT-MoE-Video2Audio-v2/outputs/audio_test_aligned"

if __name__ == '__main__':    
    evaluator = EvaluationHelperParallel(16000, 1, backbone="cnn14")
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path
    )