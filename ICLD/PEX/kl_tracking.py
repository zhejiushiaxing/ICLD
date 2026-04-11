from config.cfg import cfg
from core.model_infer import ModelInfer
import copy


def main():
    # 1.初始化推理引擎
    model_infer = ModelInfer()

    # 2.配置相关信息
    test_cfg = copy.deepcopy(cfg)
    # test_cfg['single_model_name'] = ""
    test_cfg['single_model_path'] = "mistralai/Mistral-7B-Instruct-v0.3"
    test_cfg['dataset_name'] = "mmlu" # 选择数据集的名称
    test_cfg['dataset_path'] = "/mnt/Data/zhoujiaxing/code/ICLD/dataset/mmlu/all/test-00000-of-00001.parquet" # 选择的数据集文件路径
    
    # 3.执行入口
    model_infer.single_model_infer(test_cfg)

if __name__ == "__main__":
    main()

    # --- 运行示例 ---
    # 假设 self, cfg, model, tokenizer 已经准备好
    # kl_data, token_labels = generate_with_kl_tracking(self, prompt, cfg, model, tokenizer)
    # plot_kl_divergence(kl_data, token_labels)