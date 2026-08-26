"""Runner support for expanding the Stage 1 policy observation interface."""

from __future__ import annotations

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner


class JumpStageTwoRunner(MjlabOnPolicyRunner):
    """Warm-start Stage 2 while zero-initializing its new command inputs."""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if (
            source_actor_dim == target_actor_dim
            and source_critic_dim == target_critic_dim
        ):
            return super().load(path, load_cfg, strict, map_location)
        if source_actor_dim > target_actor_dim:
            raise ValueError(
                f"Checkpoint actor has {source_actor_dim} observations, but "
                f"the Stage 2 actor has only {target_actor_dim}."
            )
        if source_critic_dim > target_critic_dim:
            raise ValueError(
                f"Checkpoint critic has {source_critic_dim} observations, but "
                f"the Stage 2 critic has only {target_critic_dim}."
            )
        if source_actor_dim < target_actor_dim:
            self._expand_model_inputs(
                loaded["actor_state_dict"],
                self.alg._raw_actor.state_dict(),
            )
            action_std = loaded["actor_state_dict"]["distribution.std_param"]
            action_std[-3:] = torch.clamp(action_std[-3:], min=0.5)
        if source_critic_dim < target_critic_dim:
            self._expand_model_inputs(
                loaded["critic_state_dict"],
                self.alg._raw_critic.state_dict(),
            )
        warm_start_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        self.alg.load(loaded, warm_start_cfg, strict=True)
        print(
            f"[INFO]: Warm-started actor observations {source_actor_dim}/"
            f"{target_actor_dim} and critic observations {source_critic_dim}/"
            f"{target_critic_dim}; new observation weights are zero-initialized."
        )
        return loaded.get("infos") or {}

    @staticmethod
    def _expand_model_inputs(
        source: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> None:
        source_dim = source["mlp.0.weight"].shape[1]
        for key in (
            "obs_normalizer._mean",
            "obs_normalizer._var",
            "obs_normalizer._std",
            "mlp.0.weight",
        ):
            expanded = target[key].clone()
            expanded[..., :source_dim] = source[key]
            source[key] = expanded


class JumpWarmStartRunner(MjlabOnPolicyRunner):
    """Load balance weights while resetting PPO state for a new jump objective."""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        del load_cfg, strict
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if source_actor_dim != target_actor_dim or source_critic_dim != target_critic_dim:
            raise ValueError(
                "Warm-start jump observations must match the balance checkpoint: "
                f"actor {source_actor_dim}/{target_actor_dim}, "
                f"critic {source_critic_dim}/{target_critic_dim}."
            )
        loaded["actor_state_dict"]["distribution.std_param"] = torch.clamp(
            loaded["actor_state_dict"]["distribution.std_param"],
            min=0.5,
        )
        self.alg.load(
            loaded,
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        print(
            "[INFO]: Warm-started balance weights and reset PPO state for "
            "small-jump exploration."
        )
        return loaded.get("infos") or {}


class PipelineContinueRunner(MjlabOnPolicyRunner):
    """Resume a full-pipeline checkpoint with a fresh iteration counter.

    The continuation env re-runs its step-scheduled phases from
    common_step_counter zero, so the learn-iteration counter must restart too
    (a plain resume would keep the old counter and exit immediately once it
    exceeds max_iterations). Network shapes must match exactly; trained action
    std is kept as-is, optimizer state is reset.
    """

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        del load_cfg, strict
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if source_actor_dim != target_actor_dim or source_critic_dim != target_critic_dim:
            raise ValueError(
                "Pipeline continuation requires matching observation sizes: "
                f"actor {source_actor_dim}/{target_actor_dim}, "
                f"critic {source_critic_dim}/{target_critic_dim}."
            )
        self.alg.load(
            loaded,
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        print(
            "[INFO]: Continued full-pipeline checkpoint with fresh PPO "
            "iteration counter and optimizer state."
        )
        return loaded.get("infos") or {}


class SpeedContinueRunner(PipelineContinueRunner):
    """Continue the speed ladder with action noise pulled back down.

    v23/v23b/v24 all showed the action std ratcheting upward from the
    resumed checkpoint (0.78 and climbing) until hops turned chaotic and
    velocity error broke down. Capping the displacement income (v24)
    removed the reward funding but the drift persisted, so the noise is
    clamped at load to the level of v22's best-tracking era and the stage
    runs with a zero entropy bonus (set in train.py).

    v25 showed the climb persists even with zero entropy (0.28 -> 0.50 in
    900 iterations): per-jump income means above-threshold noise on the
    trigger channel keeps earning positive advantages, so the surrogate
    gradient itself inflates the std. The runner therefore re-clamps the
    std parameter after every PPO update; the ceiling is slightly above
    the load clamp so the optimizer retains some slack.
    """

    # v28: load clamp keeps the resume out of the sloppy-hopping noise zone
    # (v27 ended at 0.62); the per-update ceiling is deliberately OFF as the
    # honest test of diet 2.0 -- with all hop-count-scaling income removed,
    # the std should have nothing left to climb on.
    MAX_LOAD_STD = 0.40
    MAX_RUN_STD = None

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        del load_cfg, strict
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if source_actor_dim != target_actor_dim or source_critic_dim != target_critic_dim:
            raise ValueError(
                "Speed continuation requires matching observation sizes: "
                f"actor {source_actor_dim}/{target_actor_dim}, "
                f"critic {source_critic_dim}/{target_critic_dim}."
            )
        loaded["actor_state_dict"]["distribution.std_param"] = torch.clamp(
            loaded["actor_state_dict"]["distribution.std_param"],
            max=self.MAX_LOAD_STD,
        )
        self.alg.load(
            loaded,
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        if self.MAX_RUN_STD is not None:
            original_update = self.alg.update

            def update_with_std_ceiling(*args, **kwargs):
                result = original_update(*args, **kwargs)
                with torch.no_grad():
                    self.alg._raw_actor.distribution.std_param.clamp_(
                        max=self.MAX_RUN_STD,
                    )
                return result

            self.alg.update = update_with_std_ceiling
        print(
            "[INFO]: Continued speed-ladder checkpoint with fresh PPO state, "
            f"action std clamped to <= {self.MAX_LOAD_STD} at load, "
            f"per-update ceiling: {self.MAX_RUN_STD}."
        )
        return loaded.get("infos") or {}


class VelocityEstimateWarmStartRunner(JumpStageTwoRunner):
    """Resume a direct-pipeline checkpoint that gains new actor inputs.

    New observation columns (the actor velocity estimate) are appended after
    the checkpoint's layout, so trained input weights copy into the leading
    columns and only the new trailing columns start at the fresh init. Unlike
    JumpStageTwoRunner, the trained action std is kept as-is: the policy is
    already competent and only needs to learn to read the new signal.
    Optimizer state and the iteration counter are reset.
    """

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        del load_cfg, strict
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if source_actor_dim > target_actor_dim or source_critic_dim > target_critic_dim:
            raise ValueError(
                "Checkpoint observations exceed the target layout: "
                f"actor {source_actor_dim}/{target_actor_dim}, "
                f"critic {source_critic_dim}/{target_critic_dim}."
            )
        if source_actor_dim < target_actor_dim:
            self._expand_model_inputs(
                loaded["actor_state_dict"],
                self.alg._raw_actor.state_dict(),
            )
        if source_critic_dim < target_critic_dim:
            self._expand_model_inputs(
                loaded["critic_state_dict"],
                self.alg._raw_critic.state_dict(),
            )
        self.alg.load(
            loaded,
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        print(
            f"[INFO]: Warm-started with actor inputs {source_actor_dim}->"
            f"{target_actor_dim} and critic inputs {source_critic_dim}->"
            f"{target_critic_dim}; action std preserved, PPO state reset."
        )
        return loaded.get("infos") or {}


class NavigationWarmStartRunner(JumpStageTwoRunner):
    """Warm-start balance while appending zero-weighted planar command inputs."""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        del load_cfg, strict
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        source_actor_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_actor_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        source_critic_dim = loaded["critic_state_dict"]["mlp.0.weight"].shape[1]
        target_critic_dim = self.alg._raw_critic.state_dict()["mlp.0.weight"].shape[1]
        if source_actor_dim > target_actor_dim or source_critic_dim > target_critic_dim:
            raise ValueError(
                "Navigation observations cannot be smaller than the balance "
                f"checkpoint: actor {source_actor_dim}/{target_actor_dim}, "
                f"critic {source_critic_dim}/{target_critic_dim}."
            )
        self._expand_model_inputs(
            loaded["actor_state_dict"],
            self.alg._raw_actor.state_dict(),
        )
        self._expand_model_inputs(
            loaded["critic_state_dict"],
            self.alg._raw_critic.state_dict(),
        )
        self._expand_actor_outputs(
            loaded["actor_state_dict"],
            self.alg._raw_actor.state_dict(),
        )
        source_action_dim = loaded["actor_state_dict"][
            "distribution.std_param"
        ].shape[0]
        target_action_dim = self.alg._raw_actor.state_dict()[
            "distribution.std_param"
        ].shape[0]
        if source_action_dim == target_action_dim == 3:
            output_layer = max(
                int(key.split(".")[1])
                for key in loaded["actor_state_dict"]
                if key.startswith("mlp.") and key.endswith(".weight")
            )
            loaded["actor_state_dict"][f"mlp.{output_layer}.weight"][2].zero_()
            loaded["actor_state_dict"][f"mlp.{output_layer}.bias"][2].zero_()
            loaded["actor_state_dict"]["distribution.std_param"][2] = 0.20
            print(
                "[INFO]: Reset the previously gated leg-action head for "
                "free-hop exploration (mean 0, std 0.20)."
            )
        else:
            loaded["actor_state_dict"]["distribution.std_param"] = torch.clamp(
                loaded["actor_state_dict"]["distribution.std_param"],
                min=0.5,
            )
        self.alg.load(
            loaded,
            {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
        )
        print(
            f"[INFO]: Warm-started navigation from {source_actor_dim}/"
            f"{source_critic_dim} balance observations; appended command "
            "weights are zero-initialized and PPO state is reset."
        )
        return loaded.get("infos") or {}

    @staticmethod
    def _expand_actor_outputs(
        source: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> None:
        """Grow the actor's action head when the target has extra actions.

        Copied action rows keep their trained weights; new rows (e.g. the
        jump_request channel) keep the target's fresh initialization.
        """
        source_action_dim = source["distribution.std_param"].shape[0]
        target_action_dim = target["distribution.std_param"].shape[0]
        if source_action_dim == target_action_dim:
            return
        if source_action_dim > target_action_dim:
            raise ValueError(
                f"Checkpoint actor has {source_action_dim} actions, but the "
                f"target actor has only {target_action_dim}."
            )
        output_layer = max(
            int(key.split(".")[1])
            for key in source
            if key.startswith("mlp.") and key.endswith(".weight")
        )
        for key in (
            f"mlp.{output_layer}.weight",
            f"mlp.{output_layer}.bias",
            "distribution.std_param",
        ):
            expanded = target[key].clone()
            expanded[: source[key].shape[0]] = source[key]
            source[key] = expanded
        print(
            f"[INFO]: Expanded actor outputs {source_action_dim} -> "
            f"{target_action_dim}; new action head rows keep fresh init."
        )
