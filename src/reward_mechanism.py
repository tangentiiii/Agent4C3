import math
from abc import ABC, abstractmethod
from collections import defaultdict


class RewardMechanism(ABC):
    """Base class for rewarding mechanisms."""

    @abstractmethod
    def compute_rewards(
        self,
        interactions: list[dict],
        creator_ids: list[int],
    ) -> dict[int, float]:
        """
        Compute rewards for each content creator based on user interactions.

        Args:
            interactions: flat list of all user interaction records for this round.
                Each record has: {title, creator_id, click: 0/1, like?: 0/1, user_name: str}
            creator_ids: list of all creator IDs.

        Returns:
            dict mapping creator_id -> reward value
        """
        pass


class DefaultMechanism(RewardMechanism):
    """Reward = total number of likes received."""

    def compute_rewards(
        self,
        interactions: list[dict],
        creator_ids: list[int],
    ) -> dict[int, float]:
        likes = defaultdict(int)
        for record in interactions:
            if record.get("click") == 1 and record.get("like") == 1:
                likes[record["creator_id"]] += 1

        return {cid: float(likes.get(cid, 0)) for cid in creator_ids}


# ---------------------------------------------------------
# Base Class for Expected Reward Mechanisms 
# (Computes E_x[M(sigma_i, sigma_{-i})] as defined in paper)
# ---------------------------------------------------------
class AveragedUserMechanism(RewardMechanism):
    def compute_rewards(
        self,
        interactions: list[dict],
        creator_ids: list[int],
    ) -> dict[int, float]:
        # 按用户聚合评分 (like为1，不like或没click为0)
        user_interactions = defaultdict(lambda: {cid: 0.0 for cid in creator_ids})
        for record in interactions:
            if "user_name" in record:
                user = record["user_name"]
                cid = record["creator_id"]
                # 只有当 click==1 并且 like==1 时算作有效匹配分数(1.0)
                if record.get("click") == 1 and record.get("like") == 1:
                    user_interactions[user][cid] = 1.0
                else:
                    user_interactions[user][cid] = 0.0

        total_rewards = {cid: 0.0 for cid in creator_ids}
        num_users = len(user_interactions)
        if num_users == 0:
            return total_rewards

        # 对每个用户独立计算其触发的奖励，然后累加
        for user, scores_dict in user_interactions.items():
            user_rewards = self.compute_user_rewards(scores_dict, creator_ids)
            for cid in creator_ids:
                total_rewards[cid] += user_rewards.get(cid, 0.0)

        # 最终取所有用户的平均值
        return {cid: total_rewards[cid] / num_users for cid in creator_ids}

    @abstractmethod
    def compute_user_rewards(self, scores_dict: dict[int, float], creator_ids: list[int]) -> dict[int, float]:
        pass


# ---------------------------------------------------------
# M^3 Class Mechanisms
# ---------------------------------------------------------
class M3ZeroMechanism(AveragedUserMechanism):
    """M^3(0): 机制，每个创作者的奖励等于其自身的匹配分数."""
    def compute_user_rewards(self, scores_dict: dict[int, float], creator_ids: list[int]) -> dict[int, float]:
        return scores_dict.copy()


class M3ExpoMechanism(AveragedUserMechanism):
    """M^3(expo.): 基于曝光度的机制 (K=5, beta=0.05)."""
    def __init__(self, K=5, beta=0.05):
        self.K = K
        self.beta = beta

    def compute_user_rewards(self, scores_dict: dict[int, float], creator_ids: list[int]) -> dict[int, float]:
        # 按匹配分数降序排序
        sorted_cids = sorted(creator_ids, key=lambda c: scores_dict[c], reverse=True)
        top_K_cids = sorted_cids[:self.K]

        exp_scores = {c: math.exp(scores_dict[c] / self.beta) for c in top_K_cids}
        denominator = sum(exp_scores.values())

        rewards = {cid: 0.0 for cid in creator_ids}
        for c in top_K_cids:
            rewards[c] = exp_scores[c] / denominator
        return rewards


class M3EngaMechanism(AveragedUserMechanism):
    """M^3(enga.): 基于参与度的机制 (K=5, beta=0.05)."""
    def __init__(self, K=5, beta=0.05):
        self.K = K
        self.beta = beta

    def compute_user_rewards(self, scores_dict: dict[int, float], creator_ids: list[int]) -> dict[int, float]:
        sorted_cids = sorted(creator_ids, key=lambda c: scores_dict[c], reverse=True)
        top_K_cids = sorted_cids[:self.K]

        exp_scores = {c: math.exp(scores_dict[c] / self.beta) for c in top_K_cids}
        denominator = sum(exp_scores.values())
        
        # 用户福利的对数项
        total_welfare = self.beta * math.log(denominator)

        rewards = {cid: 0.0 for cid in creator_ids}
        for c in top_K_cids:
            rewards[c] = (exp_scores[c] / denominator) * total_welfare
        return rewards


# ---------------------------------------------------------
# BRCM Class Mechanisms
# ---------------------------------------------------------
class BRCMMechanism(AveragedUserMechanism):
    """Backward Rewarding Mechanisms 的通用基类."""
    def __init__(self, f_vector: list[float]):
        self.f_vector = f_vector

    def compute_user_rewards(self, scores_dict: dict[int, float], creator_ids: list[int]) -> dict[int, float]:
        sorted_cids = sorted(creator_ids, key=lambda c: scores_dict[c], reverse=True)
        n = len(creator_ids)

        # 匹配分数序列: sigma_1 >= sigma_2 >= ... >= sigma_n, 并在最后补一个 sigma_{n+1} = 0
        sigma = [scores_dict[c] for c in sorted_cids] + [0.0]
        # 对 f_vector 补 0 (若创作者数量超过 f_vector 的长度)
        f = self.f_vector + [0.0] * max(0, n - len(self.f_vector))

        rewards = {cid: 0.0 for cid in creator_ids}
        for i in range(n):
            r = 0.0
            # M_i = \sum_{k=i}^n f_k * (\sigma_k - \sigma_{k+1})
            for k in range(i, n):
                r += f[k] * (sigma[k] - sigma[k+1])
            rewards[sorted_cids[i]] = r

        return rewards


class BRCMOptMechanism(BRCMMechanism):
    """BRCM_{opt}: 论文实验中的参数起点 f=(1, 1, 1, 1, 1, 0, ...)."""
    def __init__(self):
        super().__init__([1.0, 1.0, 1.0, 1.0, 1.0])


class BRCMStarMechanism(BRCMMechanism):
    """BRCM^*: 理论最优机制，f_i = r_i = 1 / log2(i+1) 对 i=1..5."""
    def __init__(self):
        f_vector = [1.0 / math.log2(i + 1) for i in range(1, 6)]
        super().__init__(f_vector)


class BRCM1Mechanism(BRCMMechanism):
    """BRCM_1: 用于对比的偏差参数组，f_i = 1 / i 对 i=1..5."""
    def __init__(self):
        super().__init__([1.0, 1/2, 1/3, 1/4, 1/5])


# ---------------------------------------------------------
# Mechanism Registry
# ---------------------------------------------------------
MECHANISMS = {
    "default": DefaultMechanism,
    "M3_0": M3ZeroMechanism,
    "M3_expo": M3ExpoMechanism,
    "M3_enga": M3EngaMechanism,
    "BRCM_opt": BRCMOptMechanism,
    "BRCM_star": BRCMStarMechanism,
    "BRCM_1": BRCM1Mechanism,
}


def get_mechanism(name: str, params: dict | None = None) -> RewardMechanism:
    cls = MECHANISMS.get(name)
    if cls is None:
        raise ValueError(f"Unknown mechanism: {name}. Available: {list(MECHANISMS.keys())}")
    if params:
        return cls(**params)
    return cls()
