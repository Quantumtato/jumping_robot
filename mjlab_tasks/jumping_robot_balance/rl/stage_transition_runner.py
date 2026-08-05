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
        source_dim = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        target_dim = self.alg._raw_actor.state_dict()["mlp.0.weight"].shape[1]
        if source_dim == target_dim:
            return super().load(path, load_cfg, strict, map_location)
        if source_dim > target_dim:
            raise ValueError(
                f"Checkpoint has {source_dim} observations, but the Stage 2 "
                f"policy has only {target_dim}."
            )

        self._expand_model_inputs(
            loaded["actor_state_dict"],
            self.alg._raw_actor.state_dict(),
        )
        self._expand_model_inputs(
            loaded["critic_state_dict"],
            self.alg._raw_critic.state_dict(),
        )
        action_std = loaded["actor_state_dict"]["distribution.std_param"]
        action_std[-3:] = torch.clamp(action_std[-3:], min=0.5)
        warm_start_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }
        self.alg.load(loaded, warm_start_cfg, strict=True)
        print(
            f"[INFO]: Warm-started {source_dim} of {target_dim} observation "
            "inputs; new command weights are zero-initialized and linear-action "
            "exploration is at least 0.5."
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
