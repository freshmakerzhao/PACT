import argparse

from constants import SIM_TASK_CONFIGS
from PACT.config.registry import StateDimRegistry
from PACT.runners.bc_runner import eval_bc
from utils import set_seed


def main(args):
    set_seed(1)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    equipment_model = args.get("equipment_model", "vx300s_bimanual")

    is_sim = task_name[:4] == "sim_"
    if is_sim:
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    state_dim = StateDimRegistry.get(equipment_model)

    policy_config = {
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr": args["lr"],
        "camera_names": camera_names,
        "equipment_model": equipment_model,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
    }

    config = {
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "num_rollouts": args.get("num_rollouts", 50),
    }

    ckpt_names = ["policy_best.ckpt"]
    results = []
    for ckpt_name in ckpt_names:
        success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, equipment_model=equipment_model)
        results.append([ckpt_name, success_rate, avg_return])

    for ckpt_name, success_rate, avg_return in results:
        print(f"{ckpt_name}: {success_rate=} {avg_return=}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument("--policy_class", action="store", type=str, help="policy_class, capitalize", required=True)
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--dim_feedforward", action="store", type=int, help="dim_feedforward", required=False)
    parser.add_argument("--temporal_agg", action="store_true")

    parser.add_argument(
        "--equipment_model",
        action="store",
        type=str,
        default="vx300s_bimanual",
        help="equipment model folder under assets (e.g., vx300s_bimanual)",
    )
    parser.add_argument(
        "--num_rollouts",
        action="store",
        type=int,
        default=50,
        help="number of eval rollout episodes (default 50)",
    )
    main(vars(parser.parse_args()))

