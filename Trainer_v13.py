import torch
import torch.nn.functional as F
import datetime
import json
from load_data.dataloader_v11 import DataGenerator
from models.GeoHashBARTNet_v16 import GeohashBART
from utils import load_config_from_json, print_attrs, file_print, convert_to_latlon, save_torch_model, reconstruct_trajectory
from utils import calculate_average_distance_error, load_torch_model, binary_to_geohash_list, geohash_list_to_coordinates, haversine_distance_torch
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np
import time
import argparse
import geohash2
import torch._dynamo
torch._dynamo.config.suppress_errors = True

default_config = 'config_v15.json'
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

        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_math_sdp(True)  # 使用纯 math-based 实现

        self.model = GeohashBART(self.configs)
        self.model.to(device)
        self.model = torch.compile(self.model)



        total_pa = self.model.get_num_params()
        print("Total params: %.2f M" % (total_pa / 1e6))

        # 初始化优化器
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.configs.learning_rate)
        self.opt_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=20, gamma=0.7)

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
            self.run_train_epoch(epoch, self.configs.batch_size)
            self.opt_lr_scheduler.step()
            print(f"Training lr at epoch {epoch} is {self.opt_lr_scheduler.get_last_lr()[0]}")


    def logger(self, info, log_file, debug=True):
        file_print(info, logfilename=log_file, savepath=self.log_path, debug=debug)

    def run_train_epoch(self, epoch, batch_size=1024):
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
            if len(batch['X_bin']) < batch_size:
                continue

            # 只提取需要的字段
            X_bin = batch['X_bin'].float().to(device)
            Δ_norm = batch['Δ_norm'].float().to(device)

            de_inputs = Δ_norm[:, :self.configs.inp_seq_len - 1, :]
            diff_output = Δ_norm[:, -self.configs.horizon:, :]
            en_inputs = X_bin

            # with autocast('cuda'):
            with torch.cuda.amp.autocast(enabled=False):
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

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        del batch, en_inputs, de_inputs, diff_output, pred

        if (epoch + 1) % 5 == 0 or epoch == 0:
            self.run_test(epoch, batch_size=self.configs.batch_size, full_batch=True)


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
                if full_batch and len(batch['X_bin']) < batch_size:
                    continue

                X_bin = batch['X_bin'].to(device)
                Δ_norm = batch['Δ_norm'].to(device)
                raw = batch['raw_data'].to(device)
                cls = batch['y_cls'].to(device) # (B,4)

                en = X_bin.float()
                de = Δ_norm[:, :self.configs.inp_seq_len - 1, :]
                tgt = Δ_norm[:, -self.configs.horizon:, :]

                # with autocast('cuda'):
                with torch.cuda.amp.autocast(enabled=False):
                    pred = self.model(en, de)
                    loss = F.mse_loss(pred, tgt)

                metrics = self.evaluate_predictions(
                    pred, tgt,
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
            torch.save({
                "all_pred": torch.cat(all_pred_list, dim=0).cpu(),  # [N, T, C]
                "all_true": torch.cat(all_true_list, dim=0).cpu(),
                "all_cls": torch.cat(all_cls_list, dim=0).cpu(),  # [N, ...]
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
            pred_norm, tgt_norm, *,
            last_step_bin, raw_future, raw_obs,
            feat_names, epoch,
            save_prob: float = 0.1,
            save_all_pred: bool = False,
            all_pred_list=None, all_true_list=None, all_cls_list=None,
            cls=None
    ):

        cfg = self.configs
        B, H, C = pred_norm.shape
        device = pred_norm.device

        # 1. Δ反归一化
        lo_hi = torch.tensor([[getattr(getattr(cfg, f), 'delta_lo'),
                               getattr(getattr(cfg, f), 'delta_hi')]
                              for f in feat_names], device=device)
        lo = lo_hi[:, 0].view(1, 1, C)
        hi = lo_hi[:, 1].view(1, 1, C)
        pred_delta = pred_norm * (hi - lo) + lo
        true_delta = tgt_norm * (hi - lo) + lo

        # 2. 二进制索引 → 实值
        bit_sizes = [getattr(getattr(cfg, f), 'bits') for f in feat_names]
        start_idx, offset = {}, 0
        for f, bits in zip(feat_names, bit_sizes):
            vec = last_step_bin[:, offset:offset + bits]
            weight = (2 ** torch.arange(bits - 1, -1, -1, device=device)).float()
            start_idx[f] = (vec * weight).sum(-1).long()
            offset += bits

        minmax = torch.tensor([[getattr(getattr(cfg, f), 'min'),
                                getattr(getattr(cfg, f), 'max')]
                               for f in feat_names], device=device)
        mins, maxs = minmax[:, 0], minmax[:, 1]
        nbins = torch.tensor([2 ** b for b in bit_sizes], device=device)

        real_pred = torch.zeros(B, H, C, device=device)
        for c, f in enumerate(feat_names):
            cur = start_idx[f].clone()
            for h in range(H):
                cur += pred_delta[:, h, c].round().long()
                cur.clamp_(0, nbins[c] - 1)
                real_pred[:, h, c] = mins[c] + (cur + 0.5) / nbins[c] * (maxs[c] - mins[c])

        real_true = raw_future.to(device)

        # 3. 拼接观测段
        reconstructed_pred = torch.cat([raw_obs.to(device), real_pred], dim=1)  # [B, T, C]
        reconstructed_true = torch.cat([raw_obs.to(device), real_true], dim=1)

        if save_all_pred and all_pred_list is not None:
            all_pred_list.append(reconstructed_pred)
            all_true_list.append(reconstructed_true)
            if all_cls_list is not None and cls is not None:
                all_cls_list.append(cls)

        # 4. 误差
        diff = real_pred - real_true
        rmse = torch.sqrt((diff ** 2).mean((0, 1)))
        mae = diff.abs().mean((0, 1))

        e_pre, n_pre, h_pre = real_pred[:, :, 0], real_pred[:, :, 1], real_pred[:, :, 2] / 1000
        e_tru, n_tru, h_tru = real_true[:, :, 0], real_true[:, :, 1], real_true[:, :, 2] / 1000
        mde = torch.sqrt((e_pre - e_tru) ** 2 + (n_pre - n_tru) ** 2 + (h_pre - h_tru) ** 2).mean().item()

        return {
            "rmse": rmse,
            "mae": mae,
            "mde": mde
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path for the training", default=default_config,required=False)
    args = parser.parse_args()
    tc = TrainClient(json_path=args.config)
    tc.run()
