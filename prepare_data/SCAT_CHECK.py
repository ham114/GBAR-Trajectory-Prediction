import torch

TAKEOFF_IC   = "takeoff_initial_climb"
CLIMB        = "climb"
HIGH_CRUISE  = "high_cruise"
MIDLOW_LEVEL = "midlow_level"
DESCENT      = "descent"
APPROACH     = "approach"

if __name__ == '__main__':
    pt_path = "/home/h3c/dataset/SCAT/train.pt"


    val_path = '/home/h3c/dataset/SCAT/valid.pt'
    data = torch.load(val_path, map_location="cpu")

    print(data.keys())


    print(data['geo_codes_bits'].shape)
    print(data['geo_deltas'].shape)
    print(data['geo_deltas_int'].shape)

    # X = data['geo_codes_bits'] # [N,L + H, BIN_sizes]
    #
    # y = data["labels_onehot"].float()  # [N, num_classes]
    #
    # num_classes = y.size(1)
    # counts = y.sum(dim=0)             # 每个类的1的数量
    # total = y.size(0)
    #
    # print(f"样本总数: {total}")
    # print("各类别出现次数及比例:")
    # for i in range(num_classes):
    #     c = int(counts[i].item())
    #     ratio = c / total
    #     print(f"  类别 {i}: {c} ({ratio:.4%})")
    #
    # # 自动识别稀有类（出现比例 < 5%）
    # rare_threshold = 0.05
    # rare_classes = [i for i, c in enumerate(counts) if c / total < rare_threshold]
    # print(f"\n稀有类别 (比例<{rare_threshold*100:.1f}%): {rare_classes}")
