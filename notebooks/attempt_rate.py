from datasets import load_dataset

dataset = load_dataset("bigcode/humanevalpack", "python", split="test")

path = "./cmd_results/self_cpp_qwen-7b.log"
num=cnt=0
with open(path) as f:
    for line in f.readlines():
        if "Passed on" in line:
            cnt+=1
            num+=int(line.split("attempt ")[1])

print(num/cnt)