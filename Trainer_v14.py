import torch
import torch.nn.functional as F
import datetime
import json
from load_data.dataloader_v12 import DataGenerator
from models.FlightBert import FlightBERT_PP
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
import geopy.distance
import geohash2
from geopy.distance import geodesic
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 忽略 Dynamo 错误，回退到 Eager 模式

default_config = 'config_v16.json'
torch.manual_seed(97)
scaler = GradScaler()
data_worker = 0
iscuda = torch.cuda.is_available()
gpuid = 'cpu'
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

        self.train_datas = self.configs.train_data
        self.train_tensor_path = self.configs.train_tensor_path
        self.test_tensor_path = self.configs.test_tensor_path

        self.dev_datas = self.configs.dev_data
        self.test_datas = self.configs.test_data
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
            self.eval_gen.load_from_tensor_file(self.test_tensor_path)
            self.train_gen.load_from_tensor_file(self.train_tensor_path)



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

        self.model = FlightBERT_PP(self.configs)
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
            # 准备数据张量
            batch_tmp = {}
            for k, v in batch.items():
                if isinstance(v, str):
                    batch_tmp[k] = v
                else:
                    if isinstance(v, torch.Tensor):
                        v = v.float()
                    else:
                        v = torch.tensor(v, dtype=torch.float32)
                    batch_tmp[k] = v.to(device)
            batch = batch_tmp

            if len(batch['orig_bits']) < batch_size:
                # 若该批次数据不足 batch_size，则跳过
                continue

            orig_bits = batch['orig_bits']  # [B, seq_len, feature_dim]
            diff_bits = batch['diff_bits']

            de_inputs = diff_bits[:, :self.configs.inp_seq_len - 1, :]  # Decoder 输入
            diff_output = diff_bits[:, -self.configs.horizon:, :]  # 目标(真值)
            en_inputs = orig_bits[:, :self.configs.inp_seq_len].float()

            with autocast('cuda'):
                pred = self.model(en_inputs, de_inputs)
                loss = F.mse_loss(pred, diff_output)

            # -----------
            # 反向传播
            # -----------
            self.optimiser.zero_grad()
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
        del batch, batch_tmp, en_inputs, de_inputs, diff_output, pred
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
                if full_batch and len(batch['orig_bits']) < batch_size:
                    continue

                orig_bits = batch['orig_bits'].to(device)
                diff_bits = batch['diff_bits'].to(device)
                raw = batch['orig_int'].to(device)
                # cls = batch['y_cls'].to(device)  # (B, P)

                en = orig_bits[:, :self.configs.inp_seq_len].float()
                de = diff_bits[:, :self.configs.inp_seq_len - 1, :].float()
                tgt = diff_bits[:, -self.configs.horizon:, :].float()

                with autocast('cuda'):
                    pred = self.model(en, de)
                    loss = F.mse_loss(pred, tgt)

                metrics = self.evaluate_predictions(
                    pred,
                    last_step=raw[:, self.configs.inp_seq_len - 1],
                    raw_obs=raw[:, :self.configs.inp_seq_len, :],
                    raw_future=raw[:, -self.configs.horizon:, :],
                    feat_names=feat_names,
                    epoch=epoch,
                    save_all_pred=True,
                    all_pred_list=all_pred_list,
                    all_true_list=all_true_list,
                    all_cls_list=all_cls_list,
                    # cls=cls
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

        if not self.configs.is_training:
            save_dir = os.path.join(self.log_path, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "all_pred": torch.cat(all_pred_list, dim=0).cpu(),  # (N, T, C)
                "all_true": torch.cat(all_true_list, dim=0).cpu(),  # (N, T, C)
                # "all_cls": torch.cat(all_cls_list, dim=0).cpu(),  # (N, P)
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
            pred, *,
            last_step, raw_obs, raw_future,
            feat_names, epoch,
            save_prob: float = 0.1,
            save_all_pred=False,
            all_pred_list=None,
            all_true_list=None,
            all_cls_list=None,
            cls=None
    ):
        device = pred.device
        bits_list = [getattr(getattr(self.configs, f), "bits") for f in feat_names]
        scale_list = [getattr(getattr(self.configs, f), "scale") for f in feat_names]

        diff_bits_list = [b - 2 for b in bits_list]
        mag_bits_list = [b - 3 for b in bits_list]

        slices, st = [], 0
        for diff_b in diff_bits_list:
            slices.append((st, st + diff_b))
            st += diff_b

        pred_bin = (pred > 0.5).to(torch.int8)

        def decode(bin_tensor):
            outs = []
            for idx, (s, e) in enumerate(slices):
                chunk = bin_tensor[:, :, s:e]
                sign = chunk[:, :, 0]
                val_bits = chunk[:, :, 1:]
                mag_bits = mag_bits_list[idx]
                pw = (2 ** torch.arange(mag_bits - 1, -1, -1,
                                        device=bin_tensor.device,
                                        dtype=torch.int8))
                mag = (val_bits * pw).sum(-1)
                inc = mag * (1 - 2 * sign)
                outs.append(inc)
            return torch.stack(outs, dim=-1)

        pred_diff = decode(pred_bin)

        cur = last_step.to(device).long()
        B, H = pred.shape[:2]
        C = len(feat_names)
        pred_int = torch.empty((B, H, C), dtype=torch.long, device=device)
        for t in range(H):
            cur = cur + pred_diff[:, t, :]
            pred_int[:, t, :] = cur

        scale = torch.tensor(scale_list, device=device).view(1, 1, -1)
        pred_real = pred_int.float() / scale
        true_real = raw_future.to(device).float() / scale

        # === 拼接完整轨迹 ===
        reconstructed_pred = torch.cat([raw_obs.to(device).float() / scale, pred_real], dim=1)  # [B, T, C]
        reconstructed_true = torch.cat([raw_obs.to(device).float() / scale, true_real], dim=1)

        if save_all_pred and all_pred_list is not None:
            all_pred_list.append(reconstructed_pred)
            all_true_list.append(reconstructed_true)
            if all_cls_list is not None and cls is not None:
                all_cls_list.append(cls)

        diff = pred_real - true_real
        rmse = torch.sqrt((diff ** 2).mean(dim=(0, 1)))
        mae = diff.abs().mean(dim=(0, 1))

        idx_E, idx_N, idx_R = map(feat_names.index, ["E", "N", "RALT"])
        dE = (pred_real[..., idx_E] - true_real[..., idx_E])
        dN = (pred_real[..., idx_N] - true_real[..., idx_N])
        dH = (pred_real[..., idx_R] - true_real[..., idx_R]) / 1000
        mde = torch.sqrt(dE ** 2 + dN ** 2 + dH ** 2).mean().item()

        return {"rmse": rmse, "mae": mae, "mde": mde}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path for the training", default=default_config,required=False)
    args = parser.parse_args()
    tc = TrainClient(json_path=args.config)
    tc.run()
