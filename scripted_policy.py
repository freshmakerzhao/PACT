from PACT.policies.scripted import (
    BasePolicy,
    ExcavatorDigDumpPolicy,
    ExcavatorJointSpaceDigDumpPolicy,
    ExcavatorMocapLiftingPolicy,
    InsertionPolicy,
    LiftingAndMovingPolicy,
    PickAndTransferPolicy,
    test_policy,
)


__all__ = [
    "BasePolicy",
    "ExcavatorDigDumpPolicy",
    "ExcavatorJointSpaceDigDumpPolicy",
    "ExcavatorMocapLiftingPolicy",
    "InsertionPolicy",
    "LiftingAndMovingPolicy",
    "PickAndTransferPolicy",
    "test_policy",
]


if __name__ == "__main__":
    test_task_name = "sim_lifting_cube_scripted"
    equipment_model = "excavator_simple"
    test_policy(test_task_name, equipment_model)

