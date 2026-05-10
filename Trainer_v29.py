import torch
import torch.nn.functional as F
import datetime
import json
from load_data.dataloader_v16 import DataGenerator
from models.FlightPatchNet import FlightPatch
from models.M2FlightNet import M2Flight
from utils import load_config_from_json, print_attrs, file_print, load_torch_model
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
import torch._logging



os.environ["TORCH_LOGS"] = ""
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
torch._dynamo.config.suppress_errors = True

default_config = 'config_v22.json'
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


# 禁用 Flash / Mem-efficient SDPA，强制用 math 实现
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

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
        """
        如果 `self.configs.resume_training` 为 True，则加载指定的模型、优化器和学习率调度器状态，
        以恢复训练进度。
        """
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

            # 恢复学习率调度器状态（如果存在）
            if 'scheduler_state_dict' in checkpoint and not self.configs.lr_reset:
                self.opt_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print('无调度器信息，学习率重置！')

            # 调整优化器的设备
            for state in self.optimiser.state.values():
                if isinstance(state, torch.Tensor):
                    state.data = state.data.to(device)

            print("模型恢复成功！继续训练...")
        except Exception as e:
            print(f"恢复训练时发生错误: {e}，重新训练！")

    def init_model(self):
        print("构建模型...")
        self.model = FlightPatch(self.configs)
        self.model = M2Flight(self.configs)
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
        for epoch in range(self.configs.epochs):
            self.run_train_epoch(epoch, self.configs.batch_size, debug=self.configs.debug)
            self.opt_lr_scheduler.step()
            print(f"Training lr at epoch {epoch} is {self.opt_lr_scheduler.get_last_lr()[0]}")


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
            if len(batch['state_norm']) < batch_size:
                continue

            # 只提取需要的字段
            state_norm = batch['state_norm'].to(device)

            input = state_norm[:, :self.configs.inp_seq_len, :]
            tgt = state_norm[:, -self.configs.horizon:, :]


            with autocast('cuda'):
            # with torch.cuda.amp.autocast(enabled=False):
                pred = self.model(input)
                loss = F.mse_loss(pred, tgt)

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
        del batch, input, tgt, state_norm
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

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Eval", unit="batch"):
                if full_batch and len(batch['state_norm']) < batch_size:
                    continue

                state_norm = batch['state_norm'].to(device)

                input = state_norm[:, :self.configs.inp_seq_len, :]
                tgt = state_norm[:, -self.configs.horizon:, :]
                raw = batch['state_raw'].to(device)

                with autocast('cuda'):
                # with torch.cuda.amp.autocast(enabled=False):
                    pred = self.model(input)  # pred: [B, H, 6] in [0,1]
                    loss = F.mse_loss(pred, tgt)  # 仍按归一化空间计算 MSE

                metrics = self.evaluate_predictions(
                    pred=pred,  # [B,H,6] 归一化
                    raw_future=raw[:, -self.configs.horizon:, :],  # [B,H,6] 真实原值
                    raw_obs=raw[:, :self.configs.inp_seq_len, :],  # [B,Tobs,6] 真实原值
                    feat_names=feat_names,
                    epoch=epoch,
                    save_all_pred=True,
                    all_pred_list=all_pred_list,
                    all_true_list=all_true_list,
                    all_cls_list=all_cls_list,
                    cls=None
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
            torch.save({
                "all_pred": torch.cat(all_pred_list, dim=0).cpu(),  # [N, T, C]
                "all_true": torch.cat(all_true_list, dim=0).cpu(),
                # "all_cls": torch.cat(all_cls_list, dim=0).cpu(),  # [N, ...]
                "feat_names": feat_names
            }, os.path.join(save_dir, "all_predictions.pt"))

            print(f"[Saved all predictions to] {os.path.join(save_dir, 'all_predictions.pt')}")

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
            pred, *,  # [B, H, 6]   归一化后的预测
            raw_future, raw_obs,  # [B, H, 6]、[B, Tobs, 6] 真实原值（lat, lon, RALT, Vx, Vy, ALTR）
            feat_names, epoch,
            save_prob: float = 0.1,
            save_all_pred: bool = False,
            all_pred_list=None, all_true_list=None, all_cls_list=None,
            cls=None
    ):
        """
        ENU 差分 → 经纬度重构 版本评测（6 通道）：
          feat_names 默认: ["dE","dN","RALT","Vx","Vy","ALTR"]

          - pred: 归一化值 ∈ [0,1]，先按 config 的 min/max 逆归一化
          - 前两通道为 E/N 差分 (m)，以观测段最后一步的纬度/经度为锚点重构未来 H 步的纬度/经度
          - 其它通道（RALT, Vx, Vy, ALTR）直接按逆归一化后的实值对齐
          - RMSE/MAE：在 [lat(°), lon(°), RALT, Vx, Vy, ALTR] 空间计算
          - MDE：只用重构后的纬度/经度，通过 haversine 公式计算平均球面距离 (km)
        """
        device = pred.device
        cfg = self.configs

        # ---- 0) 通道名与次序 ----
        names = feat_names
        assert len(names) == 6, f"expect 6 features, got {len(names)}"

        # ---- 1) 逆归一化 ---  pred ∈ [0,1] → 实值 (dE, dN, RALT, Vx, Vy, ALTR)
        lo = torch.tensor(
            [getattr(getattr(cfg, n), "min") for n in names],
            dtype=torch.float32,
            device=device
        ).view(1, 1, 6)
        hi = torch.tensor(
            [getattr(getattr(cfg, n), "max") for n in names],
            dtype=torch.float32,
            device=device
        ).view(1, 1, 6)
        pred_denorm = pred * (hi - lo) + lo  # [B,H,6] 实值

        B, H, C = pred_denorm.shape
        assert C == 6

        # ---- 2) 用 dE/dN 重构未来纬度/经度 ----
        # 差分 (ENU, m)
        dE_seq = pred_denorm[:, :, 0]  # [B,H], 东向位移
        dN_seq = pred_denorm[:, :, 1]  # [B,H], 北向位移

        # 其它通道直接取实值
        RALT_pred = pred_denorm[:, :, 2]  # m
        Vx_pred = pred_denorm[:, :, 3]  # m/s
        Vy_pred = pred_denorm[:, :, 4]  # m/s
        ALTR_pred = pred_denorm[:, :, 5]  # m/s

        # 观测段最后一步的纬度/经度（度）
        last_lat_deg = raw_obs[:, -1, 0].to(device)  # [B]
        last_lon_deg = raw_obs[:, -1, 1].to(device)  # [B]

        # 转成弧度
        last_lat_rad = torch.deg2rad(last_lat_deg)  # [B]
        last_lon_rad = torch.deg2rad(last_lon_deg)  # [B]

        # ENU → 经纬度，小角度局部平面近似
        R_earth_m = 6371000.0

        # 累积 ENU 位移 (相对 last_obs)
        csum_dE = torch.cumsum(dE_seq, dim=1)  # [B,H]
        csum_dN = torch.cumsum(dN_seq, dim=1)  # [B,H]

        # 北向：dN ≈ R * dφ  → dφ = dN / R
        dphi = csum_dN / R_earth_m  # [B,H]

        # 东向：dE ≈ R * cos(φ0) * dλ → dλ = dE / (R * cos(φ0))
        cos_lat0 = torch.cos(last_lat_rad).clamp(min=1e-6)  # [B]
        dlam = csum_dE / (R_earth_m * cos_lat0.unsqueeze(1))  # [B,H]

        lat_pred_rad = last_lat_rad.unsqueeze(1) + dphi  # [B,H]
        lon_pred_rad = last_lon_rad.unsqueeze(1) + dlam  # [B,H]

        # 转回度
        lat_pred_deg = torch.rad2deg(lat_pred_rad)  # [B,H]
        lon_pred_deg = torch.rad2deg(lon_pred_rad)  # [B,H]

        # ---- 3) 组合预测（实值域）：[lat(°), lon(°), RALT, Vx, Vy, ALTR] ----
        real_pred = torch.stack(
            [lat_pred_deg, lon_pred_deg, RALT_pred, Vx_pred, Vy_pred, ALTR_pred],
            dim=-1
        )  # [B,H,6]

        # 真实值：来自 raw_future（lat, lon, RALT, Vx, Vy, ALTR）
        real_true = raw_future.to(device)  # [B,H,6]

        # ---- 4) 保存完整轨迹（观测 + 预测）----
        reconstructed_pred = torch.cat([raw_obs.to(device), real_pred], dim=1)  # [B, Tobs+H, 6]
        reconstructed_true = torch.cat([raw_obs.to(device), real_true], dim=1)

        if save_all_pred and all_pred_list is not None:
            all_pred_list.append(reconstructed_pred.detach().cpu())
            all_true_list.append(reconstructed_true.detach().cpu())
            if all_cls_list is not None and cls is not None:
                all_cls_list.append(cls.detach().cpu() if torch.is_tensor(cls) else cls)

        # ---- 5) 指标：RMSE / MAE（逐通道）----
        diff = real_pred - real_true  # [B,H,6]
        rmse = torch.sqrt((diff ** 2).mean((0, 1)))  # [6]，前两维单位是“度”
        mae = diff.abs().mean((0, 1))  # [6]

        # ---- 6) MDE（单位 km）：只用纬度/经度做 haversine ----
        # 真实纬度/经度（弧度）
        lat_true_deg = real_true[:, :, 0]
        lon_true_deg = real_true[:, :, 1]
        lat_true_rad = torch.deg2rad(lat_true_deg)
        lon_true_rad = torch.deg2rad(lon_true_deg)

        # haversine 公式
        dlat = lat_pred_rad - lat_true_rad  # [B,H]
        dlon = lon_pred_rad - lon_true_rad  # [B,H]

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_true_rad) * torch.cos(lat_pred_rad) * torch.sin(dlon / 2) ** 2
        a = torch.clamp(a, 0.0, 1.0)
        c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a + 1e-12))

        R_earth_km = 6371.0
        dist_km = R_earth_km * c  # [B,H]
        mde = dist_km.mean().item()

        return {"rmse": rmse, "mae": mae, "mde": mde}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path for the training", default=default_config,required=False)
    args = parser.parse_args()
    tc = TrainClient(json_path=args.config)
    tc.run()
