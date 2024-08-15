# import the necessary packages
from keras.callbacks import Callback
import os
import nvidia_smi
#import GPUtil as GPU

class GPUStats(Callback):


	def on_train_batch_begin(self,batch,logs=None):
        
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            res1 = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            #GPUs = GPU.getGPUs()
            #gpu = GPUs[0]

            print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
            #print(f'total_mem: {res1.total}, used_mem: {res1.used}')
            #print(f"GPU RAM Free: {gpu.memoryFree}MB | Used: {gpu.memoryUsed}MB | Util {gpu.memoryUtil*100}% | Total {gpu.memoryTotal}MB")

