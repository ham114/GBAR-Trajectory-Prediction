# train_vae_prior.py
import os
import math
import time
import json
import torch
import random
import logging
import datetime as dt
import numpy as np
from torch.utils.data import DataLoader
from utils import load_config_from_json
from models.PGM_v4 import VAEPrior
from load_data.dataloader_v15 import DataGenerator
from models.Classifier_v2 import PhaseClassifier
from tqdm.auto import tqdm


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(cfg):
    if getattr(cfg, "device", None) in ("cpu", "cuda"):
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_H_C_P(cfg):
    H = getattr(cfg, "horizon", None) or getattr(cfg, "H", None)
    C = sum(getattr(getattr(cfg, f), "delta_bits") for f in cfg.features)
    P = getattr(cfg, "num_classes", None) or getattr(cfg, "P", None) or getattr(cfg, "n_phases", None)
    assert H and C and P, "config 需包含 horizon/H、features(C) 与 num_classes/P/n_phases"
    return H, C, P

def build_loader(cfg, tensor_path, is_train=True):
    cfg.is_training = is_train
    dataset = DataGenerator(cfg)
    dataset.load_from_tensor_file(tensor_path)
    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg, "batch_size", 256),
        shuffle=is_train,
        num_workers=getattr(cfg, "num_workers", 4),
        pin_memory=True,
        collate_fn=DataGenerator.prepare_minibatch
    )
    return loader

def load_phase_classifier(cfg, device):
    clf = PhaseClassifier(cfg).to(device)
    ckpt = torch.load(cfg.classifier_path, map_location=device)
    state = ckpt.get("model", ckpt)
    clf.load_state_dict(state, strict=True)
    clf.eval()
    for p in clf.parameters(): p.requires_grad_(False)
    return clf

@torch.no_grad()
def classifier_probs(clf, batch, device):
    x = batch["X_bin"].to(device).float()
    return clf(x)  # [B,P], softmax 已在模型里做

def entropy(p, eps=1e-12):
    return -(p.clamp_min(eps) * (p.clamp_min(eps).log())).sum(dim=-1)

def sharpen(p, tau=1.0):
    if tau == 1.0: return p
    p_tau = (p + 1e-12).pow(1.0 / tau)
    return p_tau / p_tau.sum(dim=-1, keepdim=True)

def alpha_schedule(step, total_steps, alpha_max=1.0, mode="linear"):
    if mode == "linear":
        return alpha_max * min(1.0, step / float(max(1,total_steps)))
    elif mode == "cos":
        return alpha_max * 0.5 * (1 - math.cos(min(1.0, step/float(max(1,total_steps))) * math.pi))
    else:
        return alpha_max

def alpha_from_confidence(w_pred, alpha_max=1.0):
    B, P = w_pred.shape
    H = entropy(w_pred); Hmax = math.log(P)
    conf = 1.0 - (H / Hmax).clamp(0, 1)
    return (alpha_max * conf).unsqueeze(-1)

def get_phase_w_mixed(batch, device, cfg, clf=None, step=None, total_steps=None, is_eval: bool=False):
    w_gt = batch["y_cls"].to(device).float()
    mode = getattr(cfg, "w_mix_mode", "schedule")
    tau  = getattr(cfg, "w_tau", 1.0)

    if mode == "gt":
        return w_gt

    # 先算 w_pred
    if clf is None:
        # pred/mix 需要分类器
        if mode in ("pred", "fixed", "schedule", "adaptive"):
            raise AssertionError("mix/pred 模式需要传入分类器 clf")
    w_pred = sharpen(classifier_probs(clf, batch, device), tau=tau) if mode != "gt" else None

    if mode == "pred":
        return w_pred

    if mode == "fixed":
        alpha = torch.as_tensor(getattr(cfg, "w_alpha", 0.5), device=device)
        return (1 - alpha) * w_gt + alpha * w_pred

    if mode == "schedule":
        # ---- 评估/测试兜底：不用步数，直接用固定 alpha（默认1.0，可配 cfg.eval_alpha）----
        if is_eval or step is None or total_steps is None:
            alpha = torch.as_tensor(getattr(cfg, "eval_alpha", 1.0), device=device)
            return (1 - alpha) * w_gt + alpha * w_pred
        # ---- 训练时正常随步数调度 ----
        alpha = alpha_schedule(step, total_steps,
                               alpha_max=getattr(cfg, "alpha_max", 1.0),
                               mode=getattr(cfg, "alpha_mode", "linear"))
        alpha = torch.as_tensor(alpha, device=device)
        return (1 - alpha) * w_gt + alpha * w_pred

    if mode == "adaptive":
        alpha = alpha_from_confidence(w_pred, alpha_max=getattr(cfg, "alpha_max", 1.0))  # [B,1]
        return (1 - alpha) * w_gt + alpha * w_pred

    raise ValueError(f"Unknown w_mix_mode: {mode}")

def setup_run_dir_and_logger(cfg):
    # 创建 save_dir/YYYYMMDD-HHMM/
    base = getattr(cfg, "save_dir", "./runs")
    ts = dt.datetime.now().strftime("%Y-%-m-%d-%H-%M")
    run_dir = os.path.join(base, ts)
    os.makedirs(run_dir, exist_ok=True)

    # 保存超参数
    hparam_path = os.path.join(run_dir, "hparams.json")
    try:
        # load_config_from_json 通常返回 SimpleNamespace，直接 vars(cfg)
        with open(hparam_path, "w", encoding="utf-8") as f:
            json.dump(vars(cfg), f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"[WARN] 写入 hparams.json 失败：{e}")

    # 配置 logger -> train.log
    log_path = os.path.join(run_dir, "train.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    # 清理旧 handler（防止重复添加）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler();               sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)

    return run_dir, logger

def main():
    # ---------- 读取配置 ----------
    config_path = 'config_v17.json'
    cfg = load_config_from_json(config_path)

    # ---------- 设备/随机种子 ----------
    device = get_device(cfg)
    seed = getattr(cfg, "seed", 1919810893)
    set_seed(seed)

    # ---------- 运行目录 & 日志 ----------
    run_dir, logger = setup_run_dir_and_logger(cfg)
    logger.info(f"Run dir: {run_dir}")

    H, C, P = get_H_C_P(cfg)
    train_path = getattr(cfg, "train_tensor_path")
    val_path   = getattr(cfg, "val_tensor_path", None) or getattr(cfg, "valid_tensor_path", None)
    test_path  = getattr(cfg, "test_tensor_path", None)
    assert os.path.exists(train_path), f"train_tensor_path 不存在: {train_path}"

    # ---------- DataLoader ----------
    train_loader = build_loader(cfg, train_path, is_train=True)
    eval_loader = None
    eval_name = None
    if val_path and os.path.exists(val_path):
        eval_loader = build_loader(cfg, val_path, is_train=False); eval_name = "val"
    elif test_path and os.path.exists(test_path):
        eval_loader = build_loader(cfg, test_path, is_train=False); eval_name = "test"

    # ---------- 模型 & 优化器 ----------
    vae = VAEPrior(cfg).to(device)
    lr = getattr(cfg, "learning_rate", 1e-3)
    wd = getattr(cfg, "weight_decay", 1e-2)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(cfg, "fp16", False)))
    grad_clip = getattr(cfg, "grad_clip", 1.0)
    warmup_steps = getattr(cfg, "beta_warmup_steps", 5000)
    max_beta = getattr(cfg, "beta_max", 1.0)
    epochs = getattr(cfg, "epochs", 200)

    clf = load_phase_classifier(cfg, device)

    total_steps = len(train_loader) * epochs
    global_step = 0
    best_eval = math.inf

    # 自动生成的模型保存路径（与日志同目录）
    ckpt_best = os.path.join(run_dir, "vae_prior_best.pt")
    ckpt_last = os.path.join(run_dir, "vae_prior_last.pt")

    for ep in range(1, epochs + 1):
        # ===== Train epoch =====
        vae.train()
        t0 = time.time()
        tr_loss = tr_recon = tr_kl = 0.0

        train_pbar = tqdm(train_loader, desc=f"Train {ep:02d}", total=len(train_loader), dynamic_ncols=True)
        for step, batch in enumerate(train_pbar, start=1):
            delta_norm = batch['Δ_bin'].to(device).float()
            y = delta_norm[:, -H:, :]
            w = get_phase_w_mixed(
                batch, device, cfg, clf=clf,
                step=(ep - 1) * len(train_loader) + step,
                total_steps=total_steps
            )
            beta = float(max_beta) * min(1.0, global_step / float(warmup_steps)) if warmup_steps > 0 else max_beta

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                _, loss, info = vae(y, w, mode='elbo', beta=beta, reduction="mean")

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            scaler.step(opt);
            scaler.update()

            tr_loss += float(loss)
            tr_recon += float(info["recon"]) if isinstance(info["recon"], float) else float(info["recon"].mean())
            tr_kl += float(info["kl"]) if isinstance(info["kl"], float) else float(info["kl"].mean())
            global_step += 1

            # 进度条显示当前 batch 的快速指标 & 平均指标
            avg_loss = tr_loss / step
            avg_recon = tr_recon / step
            avg_kl = tr_kl / step
            train_pbar.set_postfix(loss=f"{float(loss):.2f}",
                                   avg_loss=f"{avg_loss:.2f}",
                                   recon=f"{float(info['recon']):.2f}",
                                   kl=f"{float(info['kl']):.2f}",
                                   beta=f"{beta:.2f}")

        n_train = len(train_loader)
        train_msg = (f"[Epoch {ep:02d}] train "
                     f"loss={tr_loss / n_train:.3f}  recon={tr_recon / n_train:.3f}  kl={tr_kl / n_train:.3f}  "
                     f"beta_now={beta:.2f}  time={time.time() - t0:.1f}s")
        logger.info(train_msg)
        scheduler.step()

        # ---------- 验证/测试 ----------
        # ===== Eval epoch =====
        if eval_loader is not None:
            vae.eval()
            ev_loss = ev_recon = ev_kl = 0.0
            seen = 0

            eval_pbar = tqdm(eval_loader, desc=f"{eval_name.capitalize()} {ep:02d}", total=len(eval_loader),
                             dynamic_ncols=True)
            with torch.no_grad():
                for batch in eval_pbar:
                    delta_norm = batch['Δ_bin'].to(device).float()
                    y = delta_norm[:, -H:, :]
                    # 评估推荐直接用分类器概率，避免 train/infer mismatch
                    w = get_phase_w_mixed(batch, device, cfg, clf=clf, is_eval=True)

                    _, loss, info = vae(y, w, mode='elbo', beta=beta, reduction="mean")
                    ev_loss += float(loss)
                    ev_recon += float(info["recon"]) if isinstance(info["recon"], float) else float(
                        info["recon"].mean())
                    ev_kl += float(info["kl"]) if isinstance(info["kl"], float) else float(info["kl"].mean())
                    seen += 1

                    eval_pbar.set_postfix(avg_loss=f"{(ev_loss / seen):.2f}",
                                          avg_recon=f"{(ev_recon / seen):.2f}",
                                          avg_kl=f"{(ev_kl / seen):.2f}")

            n_eval = len(eval_loader)
            avg_eval = ev_loss / n_eval
            eval_msg = (f"[Epoch {ep:02d}] {eval_name:>4s} "
                        f"loss={avg_eval:.3f}  recon={ev_recon / n_eval:.3f}  kl={ev_kl / n_eval:.3f}")
            logger.info(eval_msg)

            # …后续保存最优 ckpt 的逻辑保持不变 …

            # ---------- 保存最优 ----------
            if avg_eval < best_eval:
                best_eval = avg_eval
                torch.save(
                    {"model": vae.state_dict(),
                     "config_path": config_path,
                     "H": H, "C": C, "P": P, "best_eval": best_eval, "epoch": ep},
                    ckpt_best
                )
                logger.info(f"[INFO] Saved best checkpoint to {ckpt_best} (loss={best_eval:.3f})")

        # 总是保存最后一轮
        torch.save(
            {"model": vae.state_dict(),
             "config_path": config_path,
             "H": H, "C": C, "P": P, "epoch": ep},
            ckpt_last
        )

    logger.info("[Done] Training finished.")

if __name__ == "__main__":
    main()
