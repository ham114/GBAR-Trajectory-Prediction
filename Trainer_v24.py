import torch
import torch.nn.functional as F
import datetime
import json
from load_data.dataloader_v17 import DataGenerator
from models.FlightBert_v3 import FlightBERT_PP
from utils import load_config_from_json, print_attrs, file_print, load_torch_model, perturb_X_bin_tailbits_per_feature
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np
import time
import argparse
import torch._dynamo
torch._dynamo.config.suppress_errors = True

default_config = 'config_v20.json'
torch.manual_seed(97)
scaler = GradScaler()
data_worker = 0
iscuda = torch.cuda.is_available()
# gpuid = 'cpu'
gpuid = 0  # 目标 GPU 编号
device = torch.device(f"cuda:{gpuid}")
if iscuda:
    print(f"Using GPU: {torch.cuda.get_device_name(gpuid)}")
else:
    print("Using CPU")


def worker_init_fn(worker_id):
    np.random.seed(7 + worker_id)


class TrainClient():
    def __init__(self, json_path='config.json'):
        self.configs = load_config_from_json(json_path)
        self.features = self.configs.features
        ATTRS = print_attrs(self.configs)
        self.log_path = self.configs.save_dir
        self.log_path = os.path.join(
            self.configs.save_dir,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        )
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        with open(os.path.join(self.log_path, 'readme'), 'w+') as fw:
            fw.write(ATTRS)


        self.train_tensor_path = self.configs.train_tensor_path
        self.test_tensor_path = self.configs.test_tensor_path


        self.log_file = "train.log" if self.configs.is_training else "test.log"
        self.logger(str(self.configs), self.log_file)
        # self.train_costs, self.val_costs = [], []
        self.keep_model_list = []

    def load_data(self):
        print("Loading data...")
        self.eval_gen = None
        self.train_gen = None

        if self.configs.is_training:
            self.train_gen = DataGenerator(self.configs)
            self.eval_gen = DataGenerator(self.configs)

            self.train_gen.load_from_tensor_file(self.train_tensor_path)
            self.eval_gen.load_from_tensor_file(self.test_tensor_path)


        else:
            self.eval_gen = DataGenerator(self.configs)
            self.eval_gen.load_from_tensor_file(self.test_tensor_path)

    def resume_training(self):
        if not self.configs.resume_training:
            print('重新训练中...')
            return

        model_path = self.configs.resume_model_path
        if not os.path.exists(model_path):
            print(f"恢复训练失败：模型文件 {model_path} 不存在，重新训练！")
            return

        try:
            print(f"加载模型参数、优化器和学习率调度器状态: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

            # 把 optimizer 中的 lr 改成当前 config 里写的
            if self.configs.lr_reset:
                for g in self.optimiser.param_groups:
                    g['lr'] = self.configs.learning_rate

                # 同时重置 scheduler 的状态和 base_lrs
                self.opt_lr_scheduler.last_epoch = -1
                self.opt_lr_scheduler._step_count = 0
                self.opt_lr_scheduler.base_lrs = [
                    g['lr'] for g in self.optimiser.param_groups
                ]
                print(f'学习率已重置为 {self.configs.learning_rate}')
            else:
                if 'scheduler_state_dict' in checkpoint:
                    self.opt_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    print('无调度器信息，学习率重置！')

            # 调整优化器 state 的设备
            for state in self.optimiser.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            print("模型恢复成功！继续训练...")
        except Exception as e:
            print(f"恢复训练时发生错误: {e}，重新训练！")

    def init_model(self):
        print("构建模型...")

        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_math_sdp(True)  # 使用纯 math-based 实现

        self.model = FlightBERT_PP(self.configs)
        self.model.to(device)
        self.model = torch.compile(self.model)



        total_pa = self.model.get_num_params()
        print("Total params: %.2f M" % (total_pa / 1e6))

        # 初始化优化器
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.configs.learning_rate)

        self.opt_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=10, gamma=0.7)

        # 仅在训练时尝试恢复
        if self.configs.is_training:
            self.resume_training()



    def run(self):
        self.init_model()
        self.load_data()
        if self.configs.is_training:
            print("进入训练模式...")
            self.run_train()
        else:
            print("进入测试模式...")
            # 判断是否可用 GPU
            iscuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if iscuda else "cpu")
            # device = torch.device("cpu")

            model_path = self.configs.model_path
            if model_path:
                self.logger(f"Test: Reload params from {model_path}", self.log_file)
                try:
                    # 加载模型到正确的设备
                    checkpoint = torch.load(model_path, map_location=device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("模型加载成功！")
                except Exception as e:
                    print(f"模型加载失败：{e}")
                    return
            else:
                print("未指定模型路径，无法进行测试！")
                return
            self.run_test(999, self.configs.batch_size, full_batch=True)

    def run_train(self):

        model_path = self.configs.model_path

        if model_path != '' and (not self.configs.is_training):
            self.logger("Reload params from " + model_path, self.log_file)
            # load_torch_model(self.model, self.optimiser, model_path)
            load_torch_model(self.model, model_path)

        for p in self.model.parameters():
            p.requires_grad = True

        print(self.model)

        print("Starting training process...")
        print(self.opt_lr_scheduler.state_dict())
        for epoch in range(self.configs.epochs):
            self.run_train_epoch(epoch, self.configs.batch_size, debug=self.configs.debug)
            self.opt_lr_scheduler.step()
            print(f"Training lr at epoch {epoch} is {self.opt_lr_scheduler.get_last_lr()[0]}")
            print(self.opt_lr_scheduler.state_dict())


    def logger(self, info, log_file, debug=True):
        file_print(info, logfilename=log_file, savepath=self.log_path, debug=debug)

    def run_train_epoch(self, epoch, batch_size=1024, debug=False):
        self.model.train()
        self.logger("Run the training epoch {}\n".format(epoch), log_file=self.log_file)

        num_batches = self.train_gen.data_num // batch_size

        data_loader = DataLoader(
            self.train_gen,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_worker,
            collate_fn=self.train_gen.prepare_minibatch,
            pin_memory=iscuda,
            worker_init_fn=worker_init_fn
        )

        tq = tqdm(iter(data_loader), desc='Training epoch {}'.format(epoch), total=num_batches, dynamic_ncols=True)

        start_time = time.time()
        total_loss = 0.0

        for i, batch in enumerate(tq):
            if len(batch['geo_codes_bits']) < batch_size:
                continue

            # 只提取需要的字段
            X_bin = batch['geo_codes_bits'][:, :self.configs.inp_seq_len, :].float().to(device)
            Δ_bin = batch['geo_deltas'].float().to(device)

            de_inputs = Δ_bin[:, :self.configs.inp_seq_len - 1, :]
            diff_output = Δ_bin[:, -self.configs.horizon:, :]
            en_inputs = X_bin

            # with autocast('cuda'):
            with torch.amp.autocast(device_type='cuda'):
                pred = self.model(en_inputs, de_inputs)
                loss = F.mse_loss(pred, diff_output)

            # -----------
            # 反向传播
            # -----------
            self.optimiser.zero_grad()

            # torch.autograd.set_detect_anomaly(True)

            scaler.scale(loss).backward()
            scaler.step(self.optimiser)
            scaler.update()

            total_loss += loss.item()

            # 每100个批次打印一次
            if i % 100 == 0 and i > 0:
                avg_loss = total_loss / (i + 1)
                log_msg = "Epoch {} | Iter {}/{} | Avg MSE Loss: {:.8f} | Time: {:.2f}s".format(
                    epoch, i, num_batches, avg_loss, time.time() - start_time
                )
                self.logger(log_msg, log_file=self.log_file, debug=False)
                tq.set_postfix_str(log_msg)
                start_time = time.time()

            if debug: break

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        del batch, en_inputs, de_inputs, diff_output, pred

        self.run_test(epoch, batch_size=self.configs.batch_size, full_batch=True)

        # if (epoch + 1) % 5 == 0 or epoch == 0:
        #     self.run_test(epoch, batch_size=self.configs.batch_size, full_batch=True)

    def run_test(self, epoch, batch_size=64, full_batch=False):
        print("Test process...")
        self.model.eval()
        loader = DataLoader(
            self.eval_gen,
            batch_size=batch_size,
            num_workers=data_worker,
            collate_fn=self.eval_gen.prepare_minibatch,
            pin_memory=iscuda,
            worker_init_fn=worker_init_fn)

        feat_names = self.features
        C = len(feat_names)

        sum_rmse = torch.zeros(C, device=device)
        sum_mae = torch.zeros(C, device=device)
        sum_mde, sum_loss, batches = 0.0, 0.0, 0

        all_pred_list, all_true_list, all_cls_list = [], [], []
        all_raw_pred_list, all_raw_tgt_list = [], []  # 新增：保存原始模型输出和真实标签

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Eval", unit="batch"):
                if full_batch and len(batch['geo_codes_bits']) < batch_size:
                    continue

                X_bin = batch['geo_codes_bits'][:, :self.configs.inp_seq_len, :].to(device)
                Δ_bin = batch['geo_deltas'].to(device)
                raw = batch['windows'].to(device)
                cls = batch['labels_onehot'].to(device)

                # X_bin = perturb_X_bin_tailbits_per_feature(X_bin, self.configs, feat_names)

                en = X_bin.float()
                de = Δ_bin[:, :self.configs.inp_seq_len - 1, :].float()
                tgt = Δ_bin[:, -self.configs.horizon:, :].float()

                # with autocast('cuda'):
                with torch.amp.autocast(device_type='cuda'):
                    # 你的代码
                    pred = self.model(en, de)
                    loss = F.mse_loss(pred, tgt)

                # 保存原始模型输出和真实标签
                all_raw_pred_list.append(pred.detach().cpu())
                all_raw_tgt_list.append(tgt.detach().cpu())

                metrics = self.evaluate_predictions(
                    pred,
                    last_step_bin=en[:, -1, :],
                    raw_future=raw[:, -self.configs.horizon:, :],
                    raw_obs=raw[:, :self.configs.inp_seq_len, :],
                    feat_names=feat_names,
                    epoch=epoch,
                    save_all_pred=True,
                    all_pred_list=all_pred_list,
                    all_true_list=all_true_list,
                    all_cls_list=all_cls_list,
                    cls=cls
                )

                sum_rmse += metrics['rmse'].to(device)
                sum_mae += metrics['mae'].to(device)
                sum_mde += metrics['mde']
                sum_loss += loss.item()
                batches += 1

        avg_rmse = (sum_rmse / batches).cpu().numpy()
        avg_mae = (sum_mae / batches).cpu().numpy()
        avg_mde = sum_mde / batches
        avg_loss = sum_loss / batches

        log_lines = [
            f"Test epoch = {epoch}",
            f"[Total MSE Loss]    {avg_loss:.6f}",
            f"[RMSE]              " + "  ".join([f'{x:.4f}' for x in avg_rmse]),
            f"[MAE]               " + "  ".join([f'{x:.4f}' for x in avg_mae]),
            f"[MDE]               {avg_mde:.6f} km"
        ]
        for line in log_lines:
            print(line)
            self.logger(line, log_file=self.log_file)

        # === 保存全量预测 ===
        if not self.configs.is_training:
            save_dir = os.path.join(self.log_path, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)

            # 将所有列表转换为张量
            all_pred_tensor = torch.cat(all_pred_list, dim=0).cpu()  # [N, T, C]
            all_true_tensor = torch.cat(all_true_list, dim=0).cpu()
            all_raw_pred_tensor = torch.cat(all_raw_pred_list, dim=0).cpu()  # [N, H, sum(delta_bits)]
            all_raw_tgt_tensor = torch.cat(all_raw_tgt_list, dim=0).cpu()  # [N, H, sum(delta_bits)]

            save_dict = {
                "all_pred": all_pred_tensor,  # 解码后的实值预测 [N, T, C]
                "all_true": all_true_tensor,  # 解码后的实值真实值 [N, T, C]
                "all_raw_pred": all_raw_pred_tensor,  # 模型原始输出（二进制位）[N, H, sum(delta_bits)]
                "all_raw_tgt": all_raw_tgt_tensor,  # 真实二进制标签 [N, H, sum(delta_bits)]
                "all_cls": torch.cat(all_cls_list, dim=0).cpu() if all_cls_list else None,  # [N, ...]
                "feat_names": feat_names,
                "horizon": self.configs.horizon,
                "inp_seq_len": self.configs.inp_seq_len
            }

            if self.configs.save_results:
                torch.save(save_dict, os.path.join(save_dir, "all_predictions.pt"))
                print(f"[Saved all predictions to] {os.path.join(save_dir, 'all_predictions.pt')}")

            # 可选：打印一些统计信息
            print(f"Raw pred shape: {all_raw_pred_tensor.shape}")
            print(f"Raw tgt shape: {all_raw_tgt_tensor.shape}")
            print(f"Real pred shape: {all_pred_tensor.shape}")
            print(f"Real true shape: {all_true_tensor.shape}")

        elif self.configs.is_training:
            save_p = os.path.join(self.log_path, f"epoch_{epoch}.pt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'scheduler_state_dict': self.opt_lr_scheduler.state_dict(),
                'epoch': epoch
            }, save_p)
            self.keep_model_list.append(save_p)
            if len(self.keep_model_list) > 5:
                os.remove(self.keep_model_list.pop(0))

    def evaluate_predictions(
            self,
            pred, *,  # [B, H, sum(delta_bits)], 每位是概率或0/1
            last_step_bin,  # [B, sum(bits)] 观测段最后一步 X_bin
            raw_future, raw_obs,  # [B, H, C], [B, Tobs, C] 原始实值
            feat_names, epoch,
            save_prob: float = 0.1,
            save_all_pred: bool = False,
            all_pred_list=None, all_true_list=None, all_cls_list=None,
            cls=None
    ):
        """
        适配“Δ_bin 二分区间位串”输出的评估：
          - pred: [B,H,Σ delta_bits]  -> 先按 delta_bits 拆分，再按 [delta_lo,delta_hi] 解码为整数差分
          - last_step_bin: 仍按 bits 还原 index，并逐步叠加差分得到实值预测
        """
        cfg = self.configs
        device = pred.device

        B, H, Dd = pred.shape
        C = len(feat_names)

        # 1) 读取各特征的位宽与范围
        bit_sizes = [getattr(getattr(cfg, f), 'bits') for f in feat_names]  # 原值 bits
        delta_bits = [getattr(getattr(cfg, f), 'delta_bits') for f in feat_names]  # 差分 bits
        delta_los = [getattr(getattr(cfg, f), 'delta_lo') for f in feat_names]
        delta_his = [getattr(getattr(cfg, f), 'delta_hi') for f in feat_names]
        minmax = torch.tensor([[getattr(getattr(cfg, f), 'min'),
                                getattr(getattr(cfg, f), 'max')]
                               for f in feat_names], device=device)
        mins, maxs = minmax[:, 0], minmax[:, 1]
        nbins = torch.tensor([2 ** b for b in bit_sizes], device=device)  # 原值bit格数/特征
        nbins_delta = torch.tensor([2 ** b for b in delta_bits], device=device)  # 差分bit格数/特征

        # 2) 把 pred 的最后一维按 delta_bits 切分，并将位串→差分“整数值”
        #    先把概率阈值为0.5得到 0/1 位，再还原为索引，再映射为区间中点 → 四舍五入为整数差分
        #    解码公式： val = lo + (idx + 0.5)/2^db * (hi - lo)
        pred_bits = (pred > 0.5).to(torch.uint8)  # 若模型已输出0/1可省略这步
        delta_slices = []
        off = 0
        for db in delta_bits:
            delta_slices.append((off, off + db))
            off += db
        assert off == Dd, "pred 最后一维与配置 delta_bits 总和不一致"

        pred_delta = torch.zeros(B, H, C, device=device, dtype=torch.float32)
        for c, (f, db) in enumerate(zip(feat_names, delta_bits)):
            s, e = delta_slices[c]
            bits_fc = pred_bits[:, :, s:e].float()  # [B,H,db]
            weight = (2 ** torch.arange(db - 1, -1, -1, device=device)).float()  # [db]
            idx = (bits_fc * weight).sum(dim=-1)  # [B,H]
            lo, hi = float(delta_los[c]), float(delta_his[c])
            m = float(2 ** db)
            # 区间中点解码 → 整数差分
            val = lo + (idx + 0.5) / m * (hi - lo)  # [B,H]
            pred_delta[:, :, c] = val.round()  # 还原为“区间差分整数”

        # 3) 将 last_step_bin（原值位串）转为起始索引
        start_idx, offset = {}, 0
        for f, bits in zip(feat_names, bit_sizes):
            vec = last_step_bin[:, offset:offset + bits]  # [B,bits]
            weight = (2 ** torch.arange(bits - 1, -1, -1, device=device)).float()
            start_idx[f] = (vec * weight).sum(-1).long()  # [B]
            offset += bits

        # 4) 依次叠加差分索引 → clamp → 映射为实值（用区间中点）
        real_pred = torch.zeros(B, H, C, device=device)
        for c, f in enumerate(feat_names):
            cur = start_idx[f].clone()  # [B]
            for h in range(H):
                cur = cur + pred_delta[:, h, c].long()
                cur.clamp_(0, nbins[c] - 1)
                real_pred[:, h, c] = mins[c] + (cur.float() + 0.5) / nbins[c] * (maxs[c] - mins[c])

        real_true = raw_future.to(device)

        # 5) 拼接观测段（可视化/保存完整轨迹）
        reconstructed_pred = torch.cat([raw_obs.to(device), real_pred], dim=1)  # [B, Tobs+H, C]
        reconstructed_true = torch.cat([raw_obs.to(device), real_true], dim=1)

        if save_all_pred and all_pred_list is not None:
            all_pred_list.append(reconstructed_pred.detach().cpu())
            all_true_list.append(reconstructed_true.detach().cpu())
            if all_cls_list is not None and cls is not None:
                all_cls_list.append(cls.detach().cpu() if torch.is_tensor(cls) else cls)

        # 6) 误差指标
        diff = real_pred - real_true
        rmse = torch.sqrt((diff ** 2).mean((0, 1)))  # [C]
        mae = diff.abs().mean((0, 1))  # [C]

        # MDE：E/N（米）与高度（米→千米）
        R_EARTH = 6371000.0  # 地球半径，m

        lat_idx = feat_names.index("lat")
        lon_idx = feat_names.index("lon")

        lat_pre = real_pred[:, :, lat_idx]
        lon_pre = real_pred[:, :, lon_idx]
        lat_tru = real_true[:, :, lat_idx]
        lon_tru = real_true[:, :, lon_idx]

        deg2rad = torch.pi / 180.0
        lat1 = lat_pre * deg2rad
        lon1 = lon_pre * deg2rad
        lat2 = lat_tru * deg2rad
        lon2 = lon_tru * deg2rad

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-12))
        dist = R_EARTH * c  # [B,H], 米

        mde = (dist.mean() / 1000.0).item()  # 千米

        return {"rmse": rmse, "mae": mae, "mde": mde}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path for the training", default=default_config,required=False)
    args = parser.parse_args()
    tc = TrainClient(json_path=args.config)
    tc.run()

"""
FlightBert++
09-15 02:20:07 [RMSE]              0.2552  0.2610  10.5928  0.7523  3.2289  3.7801  3.6851  4.5538  2.5326  0.4141  2.3763  0.2147
09-15 02:20:07 [MAE]               0.1613  0.1594  0.8170  0.4103  1.3562  1.0670  0.9039  2.3394  1.3631  0.0779  0.6238  0.0314
09-15 02:20:07 [MDE]               0.253711 km

M2FlightNet
10-23 10:01:47 [RMSE]              224.6440  187.4727  22.3532  7.3142  8.2201  1.8069
10-23 10:01:47 [MAE]               128.3961  106.0374  6.0568  3.0692  3.7844  1.0208
10-23 10:01:47 [MDE]               0.186944 km

Ours(non-GCVR)
09-03 11:30:11 [RMSE]              0.2665  0.2667  9.2225  0.6855  3.8325  4.4988  4.4587  5.4032  1.8164  0.1264  0.7473  0.1162
09-03 11:30:11 [MAE]               0.1574  0.1547  0.7220  0.3121  2.1491  1.2976  1.2066  2.9605  0.8905  0.0322  0.2656  0.0259
09-03 11:30:11 [MDE]               0.245163 km
All-binary
09-21 01:24:29 [RMSE]              0.2577  0.2946  15.2249  0.8268  4.0411  3.8492  3.6715  7.2915  5.7052  0.1886  1.3837  0.2432
09-21 01:24:29 [MAE]               0.0991  0.1015  1.1353  0.4158  2.2371  1.1971  1.0008  2.7774  2.9070  0.0421  0.5303  0.0778
09-21 01:24:29 [MDE]               0.166485 km

Ours
09-10 18:13:25 [RMSE]              0.2195  0.2543  9.9998  0.7007  3.4384  4.4559  4.4563  5.4020  1.8139  0.1664  0.9957  0.1526
09-10 18:13:25 [MAE]               0.1291  0.1400  0.7705  0.3418  1.2398  1.2778  1.4095  3.0605  0.8847  0.0310  0.2688  0.0262
09-10 18:13:25 [MDE]               0.212387 km
All-binary
09-17 10:38:52 [RMSE]              0.1966  0.2141  16.3557  0.8110  3.5749  3.8068  3.6842  5.2119  4.1067  0.2146  1.3037  0.2286
09-17 10:38:52 [MAE]               0.0839  0.0825  1.1703  0.4052  1.8399  1.1192  0.9804  2.4214  2.5710  0.0520  0.4804  0.0704
09-17 10:38:52 [MDE]               0.135911 km

"""

"""
FlightBERT++
11-26 08:12:22 [RMSE]              0.0040  0.0401  618.9891  21.0740  21.5524  4.2059
11-26 08:12:22 [MAE]               0.0026  0.0163  233.5783  8.1635  8.2312  2.0474
11-26 08:12:22 [MDE]               1.887051 km

M2Flight
11-28 10:29:07 [RMSE]              0.0241  0.0460  393.4832  18.9482  19.5747  2.7730
11-28 10:29:07 [MAE]               0.0096  0.0189  200.3883  8.5656  8.3701  1.6123
11-28 10:29:07 [MDE]               1.686326 km

Flight2Vec
                                   1.49203575  1.65884387  2215.87988281,
                                   1.25361764  1.38112307  1851.03808594,
                                   167.93532433  km

Ours
11-27 11:45:16 [RMSE]              0.0305  0.0387  331.2303  16.5320  16.5464  2.7466
11-27 11:45:16 [MAE]               0.0097  0.0155  143.7471  6.9723  6.9818  1.5060
11-27 11:45:16 [MDE]               1.579255 km


"""

"""
12-01 15:23:06 [RMSE]              0.0305  0.0401  337.1994  16.6644  16.4759  2.7863
12-01 15:23:06 [MAE]               0.0098  0.0167  141.8017  6.9843  6.9631  1.5205
12-01 15:23:06 [MDE]               1.638296 km

12-01 12:56:12 [RMSE]              0.0305  0.0398  338.3895  16.7295  16.5839  2.7789
12-01 12:56:12 [MAE]               0.0099  0.0157  143.8891  7.0302  7.0852  1.5270
12-01 12:56:12 [MDE]               1.613139 km

12-01 16:00:01 [RMSE]              0.0305  0.0397  339.1517  16.5792  16.4378  2.7832
12-01 16:00:01 [MAE]               0.0097  0.0161  141.7330  6.8924  6.9211  1.5189
12-01 16:00:01 [MDE]               1.607197 km

12-01 16:36:47 [RMSE]              0.0305  0.0401  338.4511  16.8014  16.5624  2.8046
12-01 16:36:47 [MAE]               0.0098  0.0167  142.5835  7.0741  7.0023  1.5306
12-01 16:36:47 [MDE]               1.640723 km

12-01 09:52:33 [RMSE]              0.0303  0.0396  334.3600  16.7336  16.4309  2.7799
12-01 09:52:33 [MAE]               0.0095  0.0157  141.6096  6.9799  6.9425  1.5190
12-01 09:52:33 [MDE]               1.569500 km

"""


"""
E     -> min:     -133 | max:      377
N     -> min:     -328 | max:      302
RALT  -> min:     -500 | max:      500
PTCH  -> min:      -39 | max:       27
ROLL  -> min:     -130 | max:      107
MH    -> min:      -32 | max:       33
TRK   -> min:      -34 | max:       38
GS    -> min:     -108 | max:       10
ALTR  -> min:     -424 | max:      334
LONG  -> min:     -478 | max:      478
VRTG  -> min:     -220 | max:      228
LATG  -> min:     -449 | max:      459
"""
