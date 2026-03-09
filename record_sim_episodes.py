import argparse

from PACT.runners.collect_runner import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--dataset_dir", action="store", type=str, help="dataset saving dir", required=True)
    parser.add_argument("--num_episodes", action="store", type=int, help="num_episodes", required=False)
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--equipment_model",
        action="store",
        type=str,
        default="vx300s_bimanual",
        help="equipment model folder under assets (e.g., vx300s_bimanual)",
    )
    parser.add_argument(
        "--success_reward_threshold",
        action="store",
        type=float,
        default=None,
        help="optional success threshold on episode max reward (default: env max reward)",
    )
    parser.add_argument(
        "--fixed_excavator_box_pose",
        action="store_true",
        help="use fixed excavator box pose for deterministic smoke test",
    )
    parser.add_argument(
        "--excavator_pipeline",
        action="store",
        type=str,
        default="ee_replay",
        choices=["ee_replay", "direct_sim"],
        help="excavator data collection pipeline: ee_replay (full flow) or direct_sim",
    )
    parser.add_argument(
        "--only_save_success",
        action="store_true",
        help="only save episodes whose episode_max_reward >= threshold",
    )
    parser.add_argument(
        "--target_success_episodes",
        action="store",
        type=int,
        default=None,
        help="stop collecting once this many successful episodes have been saved",
    )

    main(vars(parser.parse_args()))

