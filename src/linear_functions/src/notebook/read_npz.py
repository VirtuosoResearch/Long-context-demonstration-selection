import numpy as np

data = np.load("./results/noisy_LR2.npz")
print(data.keys())
print(data["x"])
print("loss_full_label_list: ", np.mean(data["loss_full_label_list"], axis=0))
print("loss_fs_inference_list: ", np.mean(data["loss_fs_inference_list"], axis=0))
print("loss_random_list: ", np.mean(data["loss_random_list"], axis=0))
print("loss_kv_final_list: ", np.mean(data["loss_kv_final_list"], axis=0))
print("loss_beta_list: ", np.mean(data["loss_beta_list"], axis=0))

print("loss_full_label_list: ", data["loss_full_label_list"])
print("loss_fs_inference_list: ", data["loss_fs_inference_list"])
print("loss_random_list: ", data["loss_random_list"])
print("loss_kv_final_list: ", data["loss_kv_final_list"])
print("loss_beta_list: ", data["loss_beta_list"])