import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import argparse
from MSTA import MSTA
from dataset import SingleChannelDataset, collate_fn

TEST_FIXED_SEED = 0

def train_one_fold(
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        device:       torch.device,
        epochs:       int   = 50,
        n_class:      int   = 3,
        gumbel_tau:   float = 1.0,
):
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)
        for eeg, bands, label in loop:
            eeg   = eeg.to(device).float()
            bands = [b.to(device) for b in bands]
            label = label.to(device)
            if eeg.dim() == 2:
                eeg = eeg.unsqueeze(1)

            optimizer.zero_grad()
            logits, band_w = model(eeg, bands)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=f"{loss.item():.4f}")

    def evaluate(loader):
        preds, gts = [], []
        with torch.no_grad():
            for eeg, bands, label in loader:
                eeg   = eeg.to(device).float()
                bands = [b.to(device) for b in bands]
                label = label.to(device)
                if eeg.dim() == 2:
                    eeg = eeg.unsqueeze(1)
                logits, band_w = model(eeg, bands)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                gts.extend(label.cpu().numpy())
        acc         = accuracy_score(gts, preds)
        macro_f1    = f1_score(gts, preds, average="macro")
        weighted_f1 = f1_score(gts, preds, average="weighted")
        return acc, macro_f1, weighted_f1

    model.eval()
    val_acc,  val_macro_f1,  val_weighted_f1  = evaluate(val_loader)
    test_acc, test_macro_f1, test_weighted_f1 = evaluate(test_loader)

    return (
        val_acc,  val_macro_f1,  val_weighted_f1,
        test_acc, test_macro_f1, test_weighted_f1,
    )

def k_fold_cross_validation(
        model:        nn.Module,
        dataset:      SingleChannelDataset,
        channel_id,
        channel_name: str,
        total_count:  int,
        data_mean:    float,
        data_std:     float,
        epochs:       int   = 50,
        batch_size:   int   = 64,
        num_folds:    int   = 5,
        seed:         int   = 42,
        n_class:      int   = 3,
        device:       torch.device = None,
        gumbel_tau:   float = 1.0,
):
    n_samples = len(dataset)
    indices   = np.arange(n_samples)

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=TEST_FIXED_SEED,
        shuffle=True,
    )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
    )

    fold_val_results  = []
    fold_test_results = []

    for fold_idx, (tr_rel, val_rel) in enumerate(kf.split(trainval_idx), start=1):
        print(f"  Fold {fold_idx}/{num_folds} ...")
        actual_train_idx = trainval_idx[tr_rel]
        actual_val_idx   = trainval_idx[val_rel]

        train_loader = DataLoader(
            Subset(dataset, actual_train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, actual_val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, drop_last=False,
            collate_fn=collate_fn,
        )

        fold_model = copy.deepcopy(model)
        (
            val_acc,  val_macro_f1,  val_weighted_f1,
            test_acc, test_macro_f1, test_weighted_f1,
        ) = train_one_fold(
            fold_model, train_loader, val_loader, test_loader,
            device=device, epochs=epochs, n_class=n_class, gumbel_tau=gumbel_tau,
        )

        fold_val_results.append((val_acc,  val_macro_f1,  val_weighted_f1))
        fold_test_results.append((test_acc, test_macro_f1, test_weighted_f1))

    def _agg(results):
        arr = np.array(results)
        return (
            arr[:, 0].mean(), arr[:, 0].std(ddof=1),
            arr[:, 1].mean(), arr[:, 1].std(ddof=1),
            arr[:, 2].mean(), arr[:, 2].std(ddof=1),
        )

    (val_acc_m,  val_acc_s,  val_mf1_m,  val_mf1_s,  val_wf1_m,  val_wf1_s)  = _agg(fold_val_results)
    (test_acc_m, test_acc_s, test_mf1_m, test_mf1_s, test_wf1_m, test_wf1_s) = _agg(fold_test_results)

    return {
        "channel_id":            str(channel_id),
        "channel_name":          channel_name,
        "total":                 total_count,
        "test_size":             len(test_idx),
        "mean":                  data_mean,
        "std":                   data_std,
        "val_acc_mean":          val_acc_m,
        "val_acc_std":           val_acc_s,
        "val_macro_f1_mean":     val_mf1_m,
        "val_macro_f1_std":      val_mf1_s,
        "val_weighted_f1_mean":  val_wf1_m,
        "val_weighted_f1_std":   val_wf1_s,
        "test_acc_mean":         test_acc_m,
        "test_acc_std":          test_acc_s,
        "test_macro_f1_mean":    test_mf1_m,
        "test_macro_f1_std":     test_mf1_s,
        "test_weighted_f1_mean": test_wf1_m,
        "test_weighted_f1_std":  test_wf1_s,
        "fold_val_results":      fold_val_results,
        "fold_test_results":     fold_test_results,
    }

def get_unique_filepath(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def run_all_channels(
        root_dir:            str,
        channel_ids,
        channel_names_list:  list,
        sample_size:         int   = 400,
        fs:                  int   = 200,
        n_freq:              int   = 10,
        d_model:             int   = 64,
        nhead:               int   = 4,
        num_layers:          int   = 2,
        gumbel_tau:          float = 1.0,
        result_dir:          str   = "results",
        epochs:              int   = 50,
        batch_size:          int   = 64,
        num_folds:           int   = 5,
        seed:                int   = 42,
        n_class:             int   = 3,
        gpu_id:              int   = 0,
):
    pl.seed_everything(seed=seed, workers=True)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"new_{result_dir}", exist_ok=True)

    summary_file = get_unique_filepath(
        os.path.join(
            f"new_{result_dir}",
            f"test_SCTA_b{batch_size}_s{seed}_e{epochs}_tau{gumbel_tau}_cat.txt",
        )
    )
    print(f"结果将写入：{summary_file}")
    total_start         = time.time()
    all_channel_results = []

    W = {
        "rank":    6,
        "id":      8,
        "name":    12,
        "metrics": 24,
        "total":   10,
        "stat":    14,
    }

    header_base = (
        f"{'CH_id':<{W['id']}}{'CH_Name':<{W['name']}}"
        f"{'Val_Acc':<{W['metrics']}}{'Val_MacroF1':<{W['metrics']}}"
        f"{'Val_WtedF1':<{W['metrics']}}"
        f"{'Test_Acc':<{W['metrics']}}{'Test_MacroF1':<{W['metrics']}}"
        f"{'Test_WtedF1':<{W['metrics']}}"
        f"{'Total':<{W['total']}}{'TestSize':<{W['total']}}"
        f"{'DataMean':<{W['stat']}}{'DataStd':<{W['stat']}}"
    )
    line_sep = "-" * len(header_base)

    FOLD_PREFIX_W = W["id"] + W["name"]

    def _ms(m, s):
        return f"{m:.4f}±{s:.4f}"

    def _val(v):
        return f"{v:.4f}"

    def _write_channel_summary(f, r):
        f.write(
            f"{r['channel_id']:<{W['id']}}"
            f"{r['channel_name']:<{W['name']}}"
            f"{_ms(r['val_acc_mean'],          r['val_acc_std']):<{W['metrics']}}"
            f"{_ms(r['val_macro_f1_mean'],     r['val_macro_f1_std']):<{W['metrics']}}"
            f"{_ms(r['val_weighted_f1_mean'],  r['val_weighted_f1_std']):<{W['metrics']}}"
            f"{_ms(r['test_acc_mean'],         r['test_acc_std']):<{W['metrics']}}"
            f"{_ms(r['test_macro_f1_mean'],    r['test_macro_f1_std']):<{W['metrics']}}"
            f"{_ms(r['test_weighted_f1_mean'], r['test_weighted_f1_std']):<{W['metrics']}}"
            f"{str(r['total']):<{W['total']}}"
            f"{str(r['test_size']):<{W['total']}}"
            f"{r['mean']:<{W['stat']}.4f}"
            f"{r['std']:<{W['stat']}.4f}\n"
        )

    def _write_fold_details(f, r):
        num_folds_actual = len(r["fold_val_results"])
        for fi in range(num_folds_actual):
            va,  vmf1,  vwf1  = r["fold_val_results"][fi]
            ta,  tmf1,  twf1  = r["fold_test_results"][fi]

            fold_label = f"  Fold{fi + 1:<3}"
            prefix = f"{fold_label:<{FOLD_PREFIX_W}}"

            f.write(
                f"{prefix}"
                f"{_val(va):<{W['metrics']}}"
                f"{_val(vmf1):<{W['metrics']}}"
                f"{_val(vwf1):<{W['metrics']}}"
                f"{_val(ta):<{W['metrics']}}"
                f"{_val(tmf1):<{W['metrics']}}"
                f"{_val(twf1):<{W['metrics']}}"
                f"{'':>{W['total']}}"
                f"{'':>{W['total']}}"
                f"{'':>{W['stat']}}"
                f"{'':>{W['stat']}}\n"
            )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("模型：SCTA（时频并行单通道 EEG 分类）\n")
        f.write(
            f"配置：分类数={n_class} | Epochs={epochs} | Batch={batch_size} | "
            f"Seed={seed} | Sample={sample_size} | d_model={d_model} | "
            f"nhead={nhead} | num_layers={num_layers} | gumbel_tau={gumbel_tau}\n"
        )
        f.write(
            f"测试集划分种子：{TEST_FIXED_SEED} | "
            f"测试集比例：20%\n"
        )
        f.write(f"数据集路径：{root_dir}\n")
        f.write("说明：通道汇总行显示均值±标准差，其下缩进行为各折详细结果\n")
        f.write(line_sep + "\n")
        f.write(header_base + "\n")
        f.write(line_sep + "\n")

    for ch_id in channel_ids:
        print(f"\n>>> [通道 {ch_id}]")
        ch_name = (
            channel_names_list[ch_id - 1]
            if ch_id <= len(channel_names_list) else "N/A"
        )

        try:
            dataset   = SingleChannelDataset(
                root_dir=root_dir,
                channel_id=ch_id,
                sample_size=sample_size,
            )
            total_num = len(dataset)
            d_mean    = dataset.mean
            d_std     = dataset.std

            if total_num < num_folds * 2:
                print(f"  ⚠ 数据量不足以进行 {num_folds} 折，跳过。")
                continue

            model = MSTA(
                num_class              = n_class,
                n_freq                 = n_freq,
                d_model                = d_model,
                nhead                  = nhead,
                num_transformer_layers = num_layers,
                gumbel_tau             = gumbel_tau,
            )

            ch_result = k_fold_cross_validation(
                model=model,
                dataset=dataset,
                channel_id=ch_id,
                channel_name=ch_name,
                total_count=total_num,
                data_mean=d_mean,
                data_std=d_std,
                epochs=epochs,
                batch_size=batch_size,
                num_folds=num_folds,
                seed=seed,
                n_class=n_class,
                device=device,
                gumbel_tau=gumbel_tau,
            )
            all_channel_results.append(ch_result)

            with open(summary_file, "a", encoding="utf-8") as f:
                _write_channel_summary(f, ch_result)
                _write_fold_details(f, ch_result)
                f.write("\n")

            print(
                f"  ✔ Val Acc={ch_result['val_acc_mean']:.4f} | "
                f"Test Acc={ch_result['test_acc_mean']:.4f}"
            )

        except Exception as e:
            import traceback
            print(f"  ⚠ 通道 {ch_id} 出错: {e}")
            traceback.print_exc()

    total_elapsed  = time.time() - total_start
    sorted_results = sorted(
        all_channel_results, key=lambda x: x["test_acc_mean"], reverse=True
    )
    full_header = f"{'Rank':<{W['rank']}}" + header_base
    full_line   = "-" * len(full_header)
    FOLD_PREFIX_W_RANKED = W["rank"] + FOLD_PREFIX_W

    def _write_channel_summary_ranked(f, rank, r):
        f.write(
            f"{str(rank):<{W['rank']}}"
            f"{r['channel_id']:<{W['id']}}"
            f"{r['channel_name']:<{W['name']}}"
            f"{_ms(r['val_acc_mean'],          r['val_acc_std']):<{W['metrics']}}"
            f"{_ms(r['val_macro_f1_mean'],      r['val_macro_f1_std']):<{W['metrics']}}"
            f"{_ms(r['val_weighted_f1_mean'],   r['val_weighted_f1_std']):<{W['metrics']}}"
            f"{_ms(r['test_acc_mean'],          r['test_acc_std']):<{W['metrics']}}"
            f"{_ms(r['test_macro_f1_mean'],     r['test_macro_f1_std']):<{W['metrics']}}"
            f"{_ms(r['test_weighted_f1_mean'],  r['test_weighted_f1_std']):<{W['metrics']}}"
            f"{str(r['total']):<{W['total']}}"
            f"{str(r['test_size']):<{W['total']}}"
            f"{r['mean']:<{W['stat']}.4f}"
            f"{r['std']:<{W['stat']}.4f}\n"
        )

    def _write_fold_details_ranked(f, r):
        num_folds_actual = len(r["fold_val_results"])
        for fi in range(num_folds_actual):
            va,  vmf1,  vwf1  = r["fold_val_results"][fi]
            ta,  tmf1,  twf1  = r["fold_test_results"][fi]

            fold_label = f"  Fold{fi + 1:<3}"
            prefix = f"{'':>{W['rank']}}{fold_label:<{FOLD_PREFIX_W}}"

            f.write(
                f"{prefix}"
                f"{_val(va):<{W['metrics']}}"
                f"{_val(vmf1):<{W['metrics']}}"
                f"{_val(vwf1):<{W['metrics']}}"
                f"{_val(ta):<{W['metrics']}}"
                f"{_val(tmf1):<{W['metrics']}}"
                f"{_val(twf1):<{W['metrics']}}"
                f"{'':>{W['total']}}"
                f"{'':>{W['total']}}"
                f"{'':>{W['stat']}}"
                f"{'':>{W['stat']}}\n"
            )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("模型：SCTA（时频并行单通道 EEG 分类）| 最终统计（按测试集准确率降序）\n")
        f.write(
            f"配置：分类数={n_class} | Epochs={epochs} | Batch={batch_size} | "
            f"Seed={seed} | SampleSize={sample_size} | K-Fold={num_folds}\n"
            f"      d_model={d_model} | nhead={nhead} | "
            f"num_layers={num_layers} | gumbel_tau={gumbel_tau}\n"
        )
        f.write(
            f"测试集划分种子：{TEST_FIXED_SEED} | 测试集比例：20%\n"
        )
        f.write(f"总耗时：{total_elapsed / 60:.2f} min\n")
        f.write("说明：通道汇总行显示均值±标准差，其下缩进行为各折详细结果\n")
        f.write(full_line + "\n")
        f.write(full_header + "\n")
        f.write(full_line + "\n")

        for rank, r in enumerate(sorted_results, start=1):
            _write_channel_summary_ranked(f, rank, r)
            _write_fold_details_ranked(f, r)
            f.write("\n")

        if sorted_results:
            f.write(full_line + "\n")

    print(f"\n任务结束！结果已保存至: {summary_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="SCTA EEG Single Channel Classification")
    parser.add_argument("--dataset_name",  type=str,   default="stroop2")
    parser.add_argument("--sample_size",   type=int,   default=400)
    parser.add_argument("--fs",            type=int,   default=200)
    parser.add_argument("--n_freq",        type=int,   default=10)
    parser.add_argument("--d_model",       type=int,   default=64)
    parser.add_argument("--nhead",         type=int,   default=4)
    parser.add_argument("--num_layers",    type=int,   default=2)
    parser.add_argument("--gumbel_tau",    type=float, default=1.0)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--num_folds",     type=int,   default=5)
    parser.add_argument("--n_class",       type=int,   default=3)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--gpu_id",        type=int,   default=0)
    parser.add_argument(
        "--channel_names",
        type=str,
        default="",
    )
    return parser.parse_args()

def get_dataset_config(dataset_name: str):
    config = {
        "dataset": {
            "root_dir":    "/dataset",
            "channel_ids": list(range(1, 17)),
        }
    }
    if dataset_name not in config:
        raise ValueError(f"未定义的数据集名称：'{dataset_name}'")
    return config[dataset_name]

SEED_62_CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7',
    'FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3',
    'P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1',
    'OZ','O2','CB2',
]

STROOP_16_CHANNELS = [
    "Fpz","F7","F3","Fz","F4","F8","C3","Cz",
    "C4","P7","P3","Pz","P4","P8","O1","O2",
]

def filter_channel_ids_by_name(
        channel_ids:        list,
        channel_names_list: list,
        target_names:       list,
) -> list:
    target_upper = {n.upper() for n in target_names}
    result = []
    for ch_id in channel_ids:
        if ch_id <= len(channel_names_list):
            name = channel_names_list[ch_id - 1].upper()
            if name in target_upper:
                result.append(ch_id)
    if not result:
        raise ValueError(
            f"在当前数据集的通道列表中找不到指定通道：{target_names}\n"
            f"可用通道：{channel_names_list}"
        )
    return result

if __name__ == "__main__":
    args      = parse_args()
    ds_config = get_dataset_config(args.dataset_name)

    if not os.path.exists(ds_config["root_dir"]):
        raise FileNotFoundError(
            f"找不到路径 {ds_config['root_dir']}，请检查 --dataset_name 是否正确。"
        )

    full_channel_ids   = ds_config["channel_ids"]
    channel_names_list = (
        SEED_62_CHANNELS if len(full_channel_ids) > 16 else STROOP_16_CHANNELS
    )

    if args.channel_names.strip():
        target_names = [n.strip() for n in args.channel_names.split(",") if n.strip()]
        channel_ids  = filter_channel_ids_by_name(
            full_channel_ids, channel_names_list, target_names
        )
        print(f"按名称筛选后的通道 ID：{channel_ids}")
    else:
        channel_ids = full_channel_ids

    print("=" * 60)
    print(f"模型: SCTA  数据集: {args.dataset_name}")
    print(f"路径: {ds_config['root_dir']}")
    print(
        f"参数: Epochs={args.epochs} | Batch={args.batch_size} | "
        f"Sample={args.sample_size} | d_model={args.d_model} | "
        f"tau={args.gumbel_tau}"
    )
    print(f"测试集固定种子：{TEST_FIXED_SEED}，用户训练种子：{args.seed}")
    print("=" * 60)

    run_all_channels(
        root_dir           = ds_config["root_dir"],
        channel_ids        = channel_ids,
        channel_names_list = channel_names_list,
        sample_size        = args.sample_size,
        fs                 = args.fs,
        n_freq             = args.n_freq,
        d_model            = args.d_model,
        nhead              = args.nhead,
        num_layers         = args.num_layers,
        gumbel_tau         = args.gumbel_tau,
        result_dir         = args.dataset_name,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        num_folds          = args.num_folds,
        seed               = args.seed,
        n_class            = args.n_class,
        gpu_id             = args.gpu_id,
    )
