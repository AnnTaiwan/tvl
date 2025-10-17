import os
from tvl_enc import tacvis
from util.eval_util import load_model, EVAL_PROMPT, get_evaluator
import llama

# 設定路徑
model_path = r"D:\NSYSU_Fourth_grade\ADV_ML_Final_Project\checkpoints\tvl_llama_vittiny.pth"
llama_dir = r"D:\NSYSU_Fourth_grade\ADV_ML_Final_Project\Prompt-Highlighter\base_models\LLaVA_checkpoints\llava-v1.5-7b"

# 載入模型
model_args = type('', (), {})()  # 空的 namespace object
model_args.llama_type = "llama-2-7b"
model_args.has_lora = True
model_args.active_modality_names = ["tactile", "vision"]
model_args.tactile_model = "vit_tiny_patch16_224"
model_args.lora_rank = 4
model_args.lora_layer_idxs = None
model_args.checkpoint_path = None
model_args.lora_modality_names = []

model = load_model(model_path, llama_dir, model_args)
model.eval()

# 使用者輸入圖片路徑
IMAGE_NAME = '93-0.0398557186126709'
TACTILE_IMAGE_PATH = "tactile/"+IMAGE_NAME+".jpg"
IMAGE_PATH = "image/"+IMAGE_NAME+".jpg"

# vision_path = input("請輸入 vision image 檔案路徑：")
vision_path = IMAGE_NAME
while not os.path.exists(vision_path):
    vision_path = input("找不到檔案，請再輸入 vision image 路徑：")

tactile_path = TACTILE_IMAGE_PATH
# tactile_path = input("請輸入 tactile image 檔案路徑：")
while not os.path.exists(tactile_path):
    tactile_path = input("找不到檔案，請再輸入 tactile image 路徑：")

# 載入資料
vision_data = tacvis.load_vision_data(vision_path, device="cuda").unsqueeze(0)
tactile_data = tacvis.load_tactile_data(tactile_path, device="cuda").unsqueeze(0)

# 組成 inputs
inputs = {
    "vision": [vision_data, 1],
    "tactile": [tactile_data, 1]
}

# 設定 prompt
prompt = "This image gives tactile feelings of?"

# 產生結果
results = model.generate(
    inputs,
    [llama.format_prompt(prompt)],
    max_gen_len=256
)

# 輸出
assistant_response = results[0].strip()
print(f"\n\n🖥️ 模型預測：{assistant_response}")

# 用內建 evaluator 打分數
evaluator = get_evaluator("gpt-4", EVAL_PROMPT)
gt_label = input("請輸入 ground truth label (for evaluation)：")
score = evaluator(prompt=prompt, assistant_response=assistant_response, correct_response=gt_label)
print(f"💯 評分結果：{score}")
