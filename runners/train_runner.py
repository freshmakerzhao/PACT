import argparse
import os

from constants import SIM_TASK_CONFIGS
from PACT.config.registry import StateDimRegistry
from PACT.io.stats import save_stats
from PACT.runners.bc_runner import build_training_checkpoint, train_bc
from utils import load_data, set_seed


def main(args):
    set_seed(1)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    equipment_model = args.get("equipment_model", "vx300s_bimanual")

    is_sim = task_name[:4] == "sim_"
    if is_sim:
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = args.get("dataset_dir") or task_config["dataset_dir"]
    num_episodes = args.get("num_episodes") or task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    state_dim = StateDimRegistry.get(equipment_model)

    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "equipment_model": equipment_model,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "equipment_model": equipment_model,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
    }

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        num_workers=args["num_workers"],
        prefetch_factor=args["prefetch_factor"],
        persistent_workers=bool(args["persistent_workers"]),
        pin_memory=bool(args["pin_memory"]),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_stats(stats, ckpt_dir)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, args)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    import torch

    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument("--policy_class", action="store", type=str, help="policy_class, capitalize", required=True)
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)
    parser.add_argument("--resume_ckpt", action="store", type=str, default=None, help="checkpoint path for resume")
    parser.add_argument("--start_epoch", action="store", type=int, default=None, help="override start epoch when resuming")

    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--dim_feedforward", action="store", type=int, help="dim_feedforward", required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--num_workers", action="store", type=int, default=4, help="dataloader workers")
    parser.add_argument("--prefetch_factor", action="store", type=int, default=2, help="dataloader prefetch factor")
    parser.add_argument("--persistent_workers", action="store", type=int, default=1, help="1 to keep workers persistent")
    parser.add_argument("--pin_memory", action="store", type=int, default=1, help="1 to enable pin memory")

    parser.add_argument(
        "--equipment_model",
        action="store",
        type=str,
        default="vx300s_bimanual",
        help="equipment model folder under assets (e.g., vx300s_bimanual)",
    )
    main(vars(parser.parse_args()))

