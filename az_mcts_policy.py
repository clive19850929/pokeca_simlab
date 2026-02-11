# placeholder for AlphaZeroMCTSPolicy (to be implemented)

import os
import sys
import random
import time
import traceback
from typing import List, Tuple, Optional, Any, Protocol

import torch


class MCTSSimEnvProtocol(Protocol):
    """
    AlphaZero 型 MCTS が利用する環境インターフェース。

    必須:
    - clone(): 現在の状態から独立なコピーを返す
    - legal_actions(): 現状態の合法手 ID のリスト（このプロジェクトでは 5-int）を返す
    - step(action): 指定 action(5-int) を 1 手適用して状態を進める
    - is_terminal(): 終局かどうかを返す
    - result(): ルート視点の value（勝ち=1, 引き分け=0, 負け=-1）を返す（非終局では例外推奨）

    MCTS で prior/value を使うために要求:
    - get_obs_vec(): prior/value 用の観測ベクトル（list[float]）を返す
      ※本プロジェクトでは state_dict の主キーを obs_belief_root に統一する前提。
        env 側が get_obs_vec() で返すのは obs_belief_root 相当（root 基準の belief）を推奨。
    - value_to_root(value_current_player): モデル value（現在手番視点）を root 視点に変換して返す
    """

    def clone(self) -> "MCTSSimEnvProtocol":
        ...

    def legal_actions(self) -> List[Any]:
        ...

    def step(self, action: Any) -> None:
        ...

    def is_terminal(self) -> bool:
        ...

    def result(self) -> float:
        ...

    def get_obs_vec(self) -> List[float]:
        ...

    def value_to_root(self, value_current_player: float) -> float:
        ...


class MCTSNode:
    """
    AlphaZero 型 MCTS 用のノード構造。

    - parent: 親ノード（ルートの場合は None）
    - children: action_key -> MCTSNode の辞書（action_key は hashable 化したもの）
    - N: 訪問回数
    - W: 累積価値（root 視点の value の合計）
    - Q: 平均価値（W / N）
    - P: 事前確率（policy ネットからの prior、親→このノードの edge prior）
    - action_from_parent: 親からこのノードへ遷移する action_id（このプロジェクトでは 5-int）
    """

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
        state_key: Optional[Any] = None,
        action_from_parent: Optional[Any] = None,
    ) -> None:
        self.parent: Optional["MCTSNode"] = parent
        self.children: dict = {}
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = float(prior)
        self.state_key: Optional[Any] = state_key
        self.action_from_parent: Optional[Any] = action_from_parent

    def is_leaf(self) -> bool:
        return not self.children


class AlphaZeroMCTSPolicy:
    """
    PolicyValueNet の prior/value を leaf で使い、PUCT で探索する AlphaZero 型 MCTS。

    重要: フォールバック禁止
    - モデル未ロード
    - obs_vec 生成不能 / 次元不一致
    - cand_vecs 生成不能（cand_dim!=5 で action_encoder_fn 不在/失敗）
    - env の step/turn/合法手整合性不一致
    などは即 RuntimeError。
    """

    def __init__(
        self,
        model_dir: str,
        model_filename: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = model_dir or "."
        self.model_filename = model_filename or "selfplay_supervised_pv_gen000.pt"
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.obs_dim: Optional[int] = None
        self.cand_dim: Optional[int] = None
        self.hidden_dim: int = 256

        self.expected_obs_dim: Optional[int] = None
        try:
            v = os.getenv("AZ_EXPECT_OBS_DIM", None)
            if v is not None and str(v).strip() != "":
                self.expected_obs_dim = int(float(v))
        except Exception:
            self.expected_obs_dim = None

        self.belief_only: bool = True
        try:
            v = os.getenv("AZ_BELIEF_ONLY", None)
            if v is not None:
                self.belief_only = str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            self.belief_only = True

        self.belief_dim: int = 1542
        try:
            v = os.getenv("AZ_BELIEF_DIM", None)
            if v is not None and str(v).strip() != "":
                self.belief_dim = int(float(v))
        except Exception:
            self.belief_dim = 1542

        # ★MCTS評価器（PVモデル入力）の期待次元を 1542 に固定（未指定時）
        if self.expected_obs_dim is None:
            try:
                self.expected_obs_dim = int(self.belief_dim)
            except Exception:
                self.expected_obs_dim = 1542

        self.action_encoder_fn = None

        self.temperature: float = 1.0
        self.greedy: bool = False

        self.num_simulations: int = 50
        self.c_puct: float = 1.5
        self.dirichlet_alpha: float = 0.3
        self.dirichlet_eps: float = 0.25

        self.mcts_pi_temperature: float = 1.0

        self.use_mcts: bool = False
        try:
            v = os.getenv("USE_MCTS_POLICY", None)
            if v is None:
                v = os.getenv("AZ_USE_MCTS", None)
            if v is not None:
                self.use_mcts = str(v).strip().lower() in ("1", "true", "yes", "y", "on")
        except Exception:
            self.use_mcts = False

        try:
            v = os.getenv("AZ_MCTS_NUM_SIMULATIONS", None)
            if v is not None and str(v).strip() != "":
                self.num_simulations = int(float(v))
        except Exception:
            pass

        try:
            v = os.getenv("AZ_MCTS_PI_TEMPERATURE", None)
            if v is not None and str(v).strip() != "":
                self.mcts_pi_temperature = float(v)
        except Exception:
            self.mcts_pi_temperature = 1.0

        if os.getenv("AZ_DECISION_LOG", "1") == "1":
            try:
                print(f"[AZ][INIT_SNAPSHOT] use_mcts={int(self.use_mcts)} sims={int(self.num_simulations)} note=may_be_overridden", flush=True)
            except Exception:
                pass

        self._az_pv_call_id: int = 0
        self._az_pv_callsite: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        """
        model_dir 配下から PolicyValueNet のチェックポイントを探してロードする。
        失敗はフォールバックせず RuntimeError。
        """
        path = self.model_filename
        if not os.path.isabs(path):
            path = os.path.join(self.model_dir, path)

        if not os.path.exists(path):
            cand_path = None
            try:
                for name in sorted(os.listdir(self.model_dir)):
                    if name.lower().endswith(".pt"):
                        cand_path = os.path.join(self.model_dir, name)
                        break
            except Exception:
                cand_path = None

            if cand_path is None or not os.path.exists(cand_path):
                raise RuntimeError(f"[AlphaZeroMCTSPolicy] model file not found in {self.model_dir}")
            path = cand_path

        data = torch.load(path, map_location=self.device)
        self.model_meta = None
        self.model_summary = None
        self.model_belief = None

        state_dict = None
        if isinstance(data, dict):
            # --- Option A: accept both old/new checkpoint formats ---
            # new: {"state_dict": ..., "obs_dim":..., "cand_dim":..., "hidden_dim":..., "model_meta":..., "summary":..., "belief":...}
            # old: {"model_state_dict": ..., "obs_dim":..., "cand_dim":..., "hidden_dim":..., "model_meta":...}
            if "state_dict" in data:
                state_dict = data.get("state_dict")
            elif "model_state_dict" in data:
                state_dict = data.get("model_state_dict")
            else:
                # allow raw state_dict dict saved as checkpoint root
                state_dict = data

            self.obs_dim = int(data.get("obs_dim", 0) or 0)
            self.cand_dim = int(data.get("cand_dim", 0) or 0)
            self.hidden_dim = int(data.get("hidden_dim", 256) or 256)

            self.model_meta = data.get("model_meta")
            self.model_summary = data.get("summary")
            self.model_belief = data.get("belief")

            # dims fallback from model_meta (Option A)
            if (not isinstance(self.obs_dim, int) or self.obs_dim <= 0) and isinstance(self.model_meta, dict):
                try:
                    self.obs_dim = int(self.model_meta.get("obs_dim", 0) or 0)
                except Exception:
                    self.obs_dim = int(self.obs_dim or 0)
            if (not isinstance(self.cand_dim, int) or self.cand_dim <= 0) and isinstance(self.model_meta, dict):
                try:
                    self.cand_dim = int(self.model_meta.get("cand_dim", 0) or 0)
                except Exception:
                    self.cand_dim = int(self.cand_dim or 0)
            if (not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0) and isinstance(self.model_meta, dict):
                try:
                    self.hidden_dim = int(self.model_meta.get("hidden_dim", 256) or 256)
                except Exception:
                    self.hidden_dim = int(self.hidden_dim or 256)
        else:
            state_dict = data
            self.obs_dim = None
            self.cand_dim = None
            self.hidden_dim = 256

        if self.obs_dim is None or self.cand_dim is None or int(self.obs_dim) <= 0 or int(self.cand_dim) <= 0:
            raise RuntimeError("[AlphaZeroMCTSPolicy] obs_dim / cand_dim missing or invalid in checkpoint.")

        if bool(getattr(self, "belief_only", False)):
            _bd = int(getattr(self, "belief_dim", 1542) or 1542)
            if int(self.obs_dim) != int(_bd):
                raise RuntimeError(
                    "[AlphaZeroMCTSPolicy] belief-only requires checkpoint obs_dim==belief_dim (no fallback): "
                    f"belief_dim={int(_bd)} got={int(self.obs_dim)}"
                )

        if self.expected_obs_dim is not None:
            if int(self.obs_dim) != int(self.expected_obs_dim):
                raise RuntimeError(
                    "[AlphaZeroMCTSPolicy] checkpoint obs_dim mismatch (no fallback): "
                    f"expected={int(self.expected_obs_dim)} got={int(self.obs_dim)}"
                )

        from train_pv_supervised import PolicyValueNet

        model = PolicyValueNet(self.obs_dim, self.cand_dim, hidden_dim=self.hidden_dim)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        try:
            if os.getenv("AZ_MODEL_LOAD_LOG", "0") == "1":
                self._az_emit_decision(
                    f"[AlphaZeroMCTSPolicy] loaded model from {path} "
                    f"(obs_dim={self.obs_dim}, cand_dim={self.cand_dim}, hidden_dim={self.hidden_dim})",
                    player=None,
                )
        except Exception:
            pass

    def set_action_encoder(self, fn) -> None:
        self.action_encoder_fn = fn

    def _az_get_log_file_path(self, player: Any = None) -> Optional[str]:
        # 1) env var (親プロセス側で渡せるように複数候補を見る)
        for k in (
            "POKECASIMLAB_GAMELOG_PATH",
            "GAMELOG_PATH",
            "POKECASIMLAB_LOG_PATH",
            "POKECASIMLAB_LOG_FILE",
            "GAME_LOG_PATH",
            "LOG_PATH",
        ):
            try:
                v = os.getenv(k, None)
            except Exception:
                v = None
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 1.5) LOG_DIR + game_id から推定（D:\date\{uuid}.log 系に対応）
        m = getattr(player, "match", None) if player is not None else None
        gid = None
        try:
            gid = getattr(m, "game_id", None) if m is not None else None
        except Exception:
            gid = None
        if isinstance(gid, str) and gid.strip():
            base = gid.strip().split("_", 1)[0]
        else:
            base = None

        for dk in (
            "POKECASIMLAB_LOG_DIR",
            "POKECASIMLAB_GAMELOG_DIR",
            "GAMELOG_DIR",
            "GAME_LOG_DIR",
            "LOG_DIR",
        ):
            try:
                d = os.getenv(dk, None)
            except Exception:
                d = None
            if not (isinstance(d, str) and d.strip()):
                continue
            if isinstance(base, str) and base.strip():
                try:
                    import os as _os
                    return _os.path.join(d.strip(), f"{base.strip()}.log")
                except Exception:
                    pass

        # 2) match から拾う（既存実装がどれを使ってもよいように複数候補）
        if m is not None:
            for attr in (
                "log_path",
                "gamelog_path",
                "game_log_path",
                "logfile_path",
                "log_file_path",
                "gamelog_file_path",
                "log_abs_path",
            ):
                try:
                    v = getattr(m, attr, None)
                except Exception:
                    v = None
                if isinstance(v, str) and v.strip():
                    return v.strip()

            for obj_attr in ("gamelog", "logger", "log", "game_log"):
                try:
                    obj = getattr(m, obj_attr, None)
                except Exception:
                    obj = None
                if obj is None:
                    continue
                for attr in ("path", "log_path", "file_path", "logfile_path", "log_file_path", "filename", "name"):
                    try:
                        v = getattr(obj, attr, None)
                    except Exception:
                        v = None
                    if isinstance(v, str) and v.strip():
                        return v.strip()

        # 3) sys.stdout が tee ラッパの場合、内包 file の name を拾えることがある
        try:
            out = sys.stdout
        except Exception:
            out = None
        if out is not None:
            for attr in ("log_path", "file_path", "path"):
                try:
                    v = getattr(out, attr, None)
                except Exception:
                    v = None
                if isinstance(v, str) and v.strip() and v not in ("<stdout>", "<stderr>"):
                    return v.strip()

            for attr in ("file", "_file", "f", "fp", "_fp", "log_file", "_log_file"):
                try:
                    f = getattr(out, attr, None)
                except Exception:
                    f = None
                if f is None:
                    continue
                try:
                    name = getattr(f, "name", None)
                except Exception:
                    name = None
                if isinstance(name, str) and name.strip() and name not in ("<stdout>", "<stderr>"):
                    return name.strip()

        return None

    def _az_append_to_logfile(self, line: str, player: Any = None) -> bool:
        path = self._az_get_log_file_path(player=player)
        if not isinstance(path, str) or not path.strip():
            try:
                path = os.getenv("AZ_DECISION_FALLBACK_PATH", None)
            except Exception:
                path = None
        if not isinstance(path, str) or not path.strip():
            return False

        try:
            with open(path, "a", encoding="utf-8") as f:
                if line.endswith("\n"):
                    f.write(line)
                else:
                    f.write(line + "\n")
            return True
        except Exception:
            return False

    def _az_emit_decision(self, line: str, player: Any = None) -> None:
        # [CALL] は常にファイルへ残し、コンソール表示は抑止（必要なら env で復帰）
        try:
            if isinstance(line, str) and ("[AZ][DECISION][CALL]" in line):
                self._az_append_to_logfile(line, player=player)
                if os.getenv("AZ_DECISION_CALL_TO_CONSOLE", "0") == "1":
                    print(line, flush=True)
                return
        except Exception:
            pass

        if os.getenv("AZ_DECISION_LOG_FILE_ONLY", "1") == "1":
            self._az_append_to_logfile(line, player=player)
            return
        print(line, flush=True)

    def _as_list_vec(self, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        if isinstance(v, tuple):
            return list(v)
        try:
            tl = getattr(v, "tolist", None)
        except Exception:
            tl = None
        if callable(tl):
            try:
                vv = tl()
            except Exception:
                vv = None
            if isinstance(vv, list):
                return vv
            if isinstance(vv, tuple):
                return list(vv)
        return None

    def _coerce_5int_row(self, row):
        try:
            if row is None:
                return None
            if isinstance(row, tuple):
                row = list(row)
            elif isinstance(row, list):
                row = row[:]
            else:
                return None

            if len(row) < 5:
                row = row + [0] * (5 - len(row))
            elif len(row) > 5:
                row = row[:5]

            out = []
            for x in row:
                out.append(int(x))
            return out
        except Exception:
            return None

    def _coerce_5int_rows(self, rows):
        out = []
        if rows is None:
            return out
        if isinstance(rows, tuple):
            rows = list(rows)
        if not isinstance(rows, list):
            return out
        for r in rows:
            rr = self._coerce_5int_row(r)
            if rr is not None:
                out.append(rr)
        return out

    def _az_assert_no_duplicate_actions_5(self, vecs: Any, where: str = "") -> None:
        if not isinstance(vecs, list) or not vecs:
            return
        seen = {}
        dups = []
        for i, v in enumerate(vecs):
            k = None
            try:
                if isinstance(v, tuple):
                    v = list(v)
                if isinstance(v, list):
                    if len(v) != 5:
                        continue
                    k = tuple(int(x) for x in v)
                else:
                    continue
            except Exception:
                continue
            if k in seen:
                dups.append((seen[k], int(i), k))
                if len(dups) >= 6:
                    break
            else:
                seen[k] = int(i)
        if dups:
            raise RuntimeError(
                "[AZ][DUP_ACTION][FATAL] duplicate legal_actions_5 detected (no fallback). "
                f"where={where} dups={dups}"
            )

    def _extract_legal_actions_5(self, state_dict: Any, actions: List[Any], player: Any = None) -> List[List[int]]:
        la_5 = []

        if isinstance(state_dict, dict):
            for k in ("legal_actions_5", "legal_actions_vec", "legal_actions"):
                try:
                    v = state_dict.get(k, None)
                except Exception:
                    v = None
                la_5 = self._coerce_5int_rows(v)
                if la_5:
                    return la_5

        m = getattr(player, "match", None) if player is not None else None
        conv = None
        if m is not None:
            try:
                conv = getattr(m, "converter", None)
                if conv is None:
                    conv = getattr(m, "action_converter", None)
            except Exception:
                conv = None

        if conv is not None:
            fn_la = getattr(conv, "convert_legal_actions", None)
            if callable(fn_la):
                try:
                    la_raw = fn_la(actions or [], player=player)
                except TypeError:
                    la_raw = fn_la(actions or [])
                la_5 = self._coerce_5int_rows(la_raw)
                if la_5:
                    return la_5

        tmp = []
        for a in actions or []:
            fn = getattr(a, "to_id_vec", None)
            if callable(fn):
                tmp.append(fn())
                continue
            if isinstance(a, (list, tuple)):
                tmp.append(list(a) if isinstance(a, tuple) else a)

        la_5 = self._coerce_5int_rows(tmp)
        return la_5

    def _extract_cand_vecs_for_model(self, state_dict: Any, la_5: List[List[int]]) -> Optional[Any]:
        cdim = int(self.cand_dim or 0)
        if cdim <= 0:
            return None

        if cdim == 5:
            return la_5 if la_5 else None

        if isinstance(state_dict, dict):
            for k in ("action_candidates_vec", "action_candidates_vecs", "cand_vecs", "cand_vecs_32d"):
                try:
                    v = state_dict.get(k, None)
                except Exception:
                    v = None
                if isinstance(v, list) and v:
                    try:
                        v0 = v[0]
                        if isinstance(v0, (list, tuple)) and len(v0) == cdim:
                            return v
                    except Exception:
                        pass

        return None

    def get_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        target_dim = int(self.obs_dim or 0)

        if not isinstance(state_dict, dict):
            raise RuntimeError("[AZ] state_dict is not dict; obs_belief_root is required (no fallback).")

        vv = None
        try:
            vv = state_dict.get("obs_belief_root", None)
        except Exception:
            vv = None

        # ★追加: obs_belief_root が dict ラッパなら obs_belief_vec を unwrap（obs_vec は一切使わない）
        if isinstance(vv, dict):
            try:
                _inner = vv.get("obs_belief_vec", None)
            except Exception:
                _inner = None
            if _inner is None:
                try:
                    _inner = vv.get("belief_vec", None)
                except Exception:
                    _inner = None
            vv = _inner

        if vv is None:
            try:
                vv = state_dict.get("obs_belief_vec", None)
            except Exception:
                vv = None

        vv = self._as_list_vec(vv)
        if vv is None:
            raise RuntimeError("[AZ] obs_belief_root is missing or not list-like (no fallback).")

        if target_dim > 0:
            if not isinstance(vv, list):
                raise RuntimeError("[AZ] obs_belief_root must be list after coercion (no fallback).")
            if int(len(vv)) != int(target_dim):
                raise RuntimeError(f"[AZ] obs_belief_root dim mismatch: expected={int(target_dim)} got={int(len(vv))} (no fallback).")

        return vv

    def encode_obs_vec(self, state_dict: Any = None, actions: Optional[List[Any]] = None, player: Any = None):
        return self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)

    def _build_obs_tensor(self, obs_vec):
        if self.model is None or self.obs_dim is None:
            raise RuntimeError("[AZ] model or obs_dim is missing (no fallback).")

        vv = self._as_list_vec(obs_vec)
        if vv is None:
            print(f"[AZ][OBS_TENSOR][BAD] type={type(obs_vec).__name__} has_tolist={int(callable(getattr(obs_vec, 'tolist', None)))}", flush=True)
            raise RuntimeError("[AZ] obs_belief_vec is missing or not list-like for tensor build (no fallback).")

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"[AZ] numpy is required for obs tensor build: {e}")

        obs_arr = np.asarray(vv, dtype="float32").reshape(-1)
        if int(obs_arr.shape[0]) != int(self.obs_dim):
            raise RuntimeError(f"[AZ] obs_dim mismatch: expected={int(self.obs_dim)} got={int(obs_arr.shape[0])} (no fallback).")
        return torch.from_numpy(obs_arr).view(1, -1).to(self.device)

    def _build_cands_tensor_from_action_ids(self, legal_action_ids: List[Any], cand_vecs: Optional[Any] = None):
        if self.model is None or self.cand_dim is None:
            raise RuntimeError("[AZ] model or cand_dim is missing (no fallback).")

        try:
            import numpy as np
        except Exception as e:
            raise RuntimeError(f"[AZ] numpy is required for cand tensor build: {e}")

        cdim = int(self.cand_dim or 0)
        if cdim <= 0:
            raise RuntimeError("[AZ] cand_dim must be positive (no fallback).")

        cand_list = []

        if cand_vecs is not None:
            try:
                for v in cand_vecs:
                    enc_arr = np.asarray(v, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr.reshape(-1), pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr.reshape(-1)[:cdim]
                    cand_list.append(enc_arr)
            except Exception:
                cand_list = []

        if not cand_list:
            if cdim == 5:
                for aid in legal_action_ids:
                    enc_arr = np.asarray(aid, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr.reshape(-1), pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr.reshape(-1)[:cdim]
                    cand_list.append(enc_arr)
            else:
                if self.action_encoder_fn is None:
                    raise RuntimeError(f"[AZ] action_encoder_fn is required for cand_dim={int(cdim)} (no fallback).")
                for aid in legal_action_ids:
                    enc = self.action_encoder_fn(aid)
                    enc_arr = np.asarray(enc, dtype="float32").reshape(-1)
                    if enc_arr.shape[-1] < cdim:
                        pad = np.zeros(cdim - enc_arr.shape[-1], dtype="float32")
                        enc_arr = np.concatenate([enc_arr, pad], axis=0)
                    elif enc_arr.shape[-1] > cdim:
                        enc_arr = enc_arr[:cdim]
                    cand_list.append(enc_arr)

        if not cand_list:
            raise RuntimeError("[AZ] failed to build cand_list (no fallback).")
        if len(cand_list) != int(len(legal_action_ids)):
            raise RuntimeError("[AZ] cand_list length mismatch (no fallback).")

        cands_arr = np.stack(cand_list, axis=0)
        return torch.from_numpy(cands_arr).to(self.device)

    def _policy_value(self, obs_vec, legal_action_ids: List[Any], cand_vecs: Optional[Any] = None) -> Tuple[List[float], float]:
        if self.model is None:
            raise RuntimeError("[AZ] model is None (no fallback).")
        if self.obs_dim is None or self.cand_dim is None:
            raise RuntimeError("[AZ] obs_dim/cand_dim missing (no fallback).")

        _pv_id = None
        try:
            self._az_pv_call_id = int(getattr(self, "_az_pv_call_id", 0) or 0) + 1
            _pv_id = int(self._az_pv_call_id)
        except Exception:
            _pv_id = None

        # --- contract guard: env.get_obs_vec() must return list[float], not dict ---
        if isinstance(obs_vec, dict):
            _caller = None
            try:
                _caller = getattr(self, "_az_pv_callsite", None)
            except Exception:
                _caller = None

            _khead = None
            try:
                kk = list(obs_vec.keys())
                _khead = kk[:16]
            except Exception:
                _khead = None

            _snap = {}
            try:
                for k in ("obs_belief_root", "obs_belief_vec", "obs_vec", "obs_full_vec", "obs_kind", "last_obs_kind", "last_obs_vec_len"):
                    if k in obs_vec:
                        v = obs_vec.get(k, None)
                        if isinstance(v, (list, tuple)):
                            _snap[k] = f"{type(v).__name__}:len={len(v)}"
                        else:
                            _snap[k] = f"{type(v).__name__}"
            except Exception:
                _snap = {}

            _stack = None
            try:
                if os.getenv("AZ_OBS_BOUNDARY_STACK", "0") == "1":
                    import inspect
                    st = inspect.stack()
                    parts = []
                    for fr in st[1:8]:
                        parts.append(f"{fr.function}@{fr.filename}:{fr.lineno}")
                    _stack = " <- ".join(parts)
            except Exception:
                _stack = None

            print(
                f"[AZ][OBS_CONTRACT][FATAL] phase=policy_value_pre_guard"
                f" pv_id={_pv_id}"
                f" caller={_caller}"
                f" obs_type=dict keys_n={len(obs_vec)}"
                + (f" keys_head={repr(_khead)}" if _khead is not None else "")
                + (f" snap={repr(_snap)}" if isinstance(_snap, dict) and _snap else "")
                + (f" stack={_stack}" if _stack is not None else ""),
                flush=True,
            )
            raise RuntimeError(
                "[AZ][OBS_CONTRACT][FATAL] _policy_value received dict as obs_vec. "
                "This indicates an env.get_obs_vec() contract violation (must be list[float]). "
                f"caller={_caller} pv_id={_pv_id}"
            )

        if os.getenv("AZ_OBS_BOUNDARY_LOG", "1") == "1":
            try:
                _t = type(obs_vec).__name__
            except Exception:
                _t = "<type_err>"
            try:
                _is_dict = int(isinstance(obs_vec, dict))
            except Exception:
                _is_dict = -1
            try:
                _has_tolist = int(callable(getattr(obs_vec, "tolist", None)))
            except Exception:
                _has_tolist = -1
            try:
                _len = len(obs_vec) if isinstance(obs_vec, (list, tuple)) else ("keys=" + str(len(obs_vec)) if isinstance(obs_vec, dict) else "NA")
            except Exception:
                _len = "NA"

            _caller = None
            try:
                _caller = getattr(self, "_az_pv_callsite", None)
            except Exception:
                _caller = None

            _khead = None
            if isinstance(obs_vec, dict):
                try:
                    kk = list(obs_vec.keys())
                    _khead = kk[:8]
                except Exception:
                    _khead = None

            _stack = None
            try:
                if os.getenv("AZ_OBS_BOUNDARY_STACK", "0") == "1":
                    import inspect
                    st = inspect.stack()
                    parts = []
                    for fr in st[1:6]:
                        parts.append(f"{fr.function}@{fr.filename}:{fr.lineno}")
                    _stack = " <- ".join(parts)
            except Exception:
                _stack = None

            print(
                f"[AZ][OBS_BOUNDARY] phase=policy_value_pre_build"
                f" pv_id={_pv_id}"
                f" caller={_caller}"
                f" obs_type={_t} is_dict={_is_dict} has_tolist={_has_tolist} len={_len}"
                + (f" keys_head={repr(_khead)}" if _khead is not None else "")
                + (f" stack={_stack}" if _stack is not None else ""),
                flush=True,
            )

        obs_tensor = self._build_obs_tensor(obs_vec)
        cands_tensor = self._build_cands_tensor_from_action_ids(legal_action_ids, cand_vecs=cand_vecs)

        self.model.eval()
        with torch.no_grad():
            logits_list, values = self.model(obs_tensor, [cands_tensor])

        if not logits_list:
            raise RuntimeError("[AZ] model returned empty logits_list (no fallback).")

        logits = logits_list[0].detach().cpu()
        if logits.ndim != 1 or int(logits.shape[0]) != int(len(legal_action_ids)):
            raise RuntimeError(
                f"[AZ] logits shape mismatch: shape={tuple(logits.shape)} n_actions={int(len(legal_action_ids))} (no fallback)."
            )

        temp = float(getattr(self, "temperature", 1.0) or 1.0)
        if temp <= 0.0:
            temp = 1.0
        scaled_logits = logits / temp
        probs_t = torch.softmax(scaled_logits, dim=-1)
        probs = probs_t.numpy().astype("float64")
        s = float(probs.sum())
        if not (s > 0.0):
            raise RuntimeError("[AZ] probs_sum<=0 (no fallback).")
        priors = (probs / s).tolist()

        v = values
        try:
            v0 = float(v.detach().view(-1)[0].cpu().item())
        except Exception:
            try:
                v0 = float(v[0])
            except Exception:
                raise RuntimeError("[AZ] value extraction failed (no fallback).")

        return priors, v0

    def select_action(
        self,
        obs_vec,
        legal_action_ids: List[Any],
        env: Optional["MCTSSimEnvProtocol"] = None,
        cand_vecs: Optional[Any] = None,
    ) -> Tuple[Optional[Any], List[float]]:
        if not isinstance(legal_action_ids, list) or not legal_action_ids:
            raise RuntimeError("[AZ] legal_action_ids must be a non-empty list (no fallback).")

        if self.model is None or self.obs_dim is None or self.cand_dim is None:
            raise RuntimeError("[AZ] model or dims missing (no fallback).")

        # ★契約: モデル入力 obs は env.get_obs_vec() を唯一の正とする（外部 obs_vec は信用しない）
        if env is not None:
            try:
                obs_vec = env.get_obs_vec()
            except Exception as e:
                raise RuntimeError(f"[AZ] env.get_obs_vec() failed (no fallback): {type(e).__name__}:{e}")

        if os.getenv("AZ_OBS_LAYER_TRACE", "1") == "1":
            try:
                _t0 = type(obs_vec).__name__
            except Exception:
                _t0 = "<type_err>"
            try:
                _id0 = hex(id(obs_vec))
            except Exception:
                _id0 = "NA"
            try:
                _is_dict0 = int(isinstance(obs_vec, dict))
            except Exception:
                _is_dict0 = -1
            try:
                _len0 = len(obs_vec) if isinstance(obs_vec, (list, tuple)) else ("keys=" + str(len(obs_vec)) if isinstance(obs_vec, dict) else "NA")
            except Exception:
                _len0 = "NA"
            try:
                _keys_head0 = list(obs_vec.keys())[:8] if isinstance(obs_vec, dict) else None
            except Exception:
                _keys_head0 = None
            try:
                _env_name0 = type(env).__name__ if env is not None else None
            except Exception:
                _env_name0 = None
            print(f"[AZ][OBS_LAYER] phase=select_action_entry obs_id={_id0} obs_type={_t0} is_dict={_is_dict0} len={_len0} keys_head={_keys_head0} env={_env_name0}", flush=True)

        use_mcts = bool(getattr(self, "use_mcts", False))
        num_sims = int(getattr(self, "num_simulations", 0) or 0)
        if os.getenv("AZ_DECISION_LOG", "1") == "1":
            try:
                env_name_always = type(env).__name__ if env is not None else None
            except Exception:
                env_name_always = None
            print(
                f"[AZ][MCTS][PRECHECK_ALWAYS] model_ok=1 use_mcts={int(use_mcts)} env={env_name_always} sims={int(num_sims)} n_actions={int(len(legal_action_ids))}",
                flush=True,
            )

        try:
            self._az_pv_callsite = "select_action:model_pre"
        except Exception:
            pass
        priors_model, _v_model = self._policy_value(obs_vec, legal_action_ids, cand_vecs=cand_vecs)
        try:
            self._az_pv_callsite = None
        except Exception:
            pass
        probs = priors_model
        decision_src = "model"

        # ★追加: env があるなら、use_mcts の有無に関わらず
        # - 重複 fatal
        # - 集合一致（順不同）で整合を確認
        # - env の順に legal_action_ids を正規化
        # - cand_vecs も追従（与えられている場合）
        if env is not None:
            env_la0 = env.legal_actions()
            if not isinstance(env_la0, list) or not env_la0:
                raise RuntimeError("[AZ] env.legal_actions() returned empty at select_action entry (no fallback).")

            # 重複LA5は設計上あり得るためfatalにはしない（macro重複を保持する）
            # self._az_assert_no_duplicate_actions_5(env_la0, where="select_action:entry_env_la")
            # self._az_assert_no_duplicate_actions_5(legal_action_ids, where="select_action:entry_given_la")

            def _k5(v):
                try:
                    if isinstance(v, tuple):
                        v = list(v)
                    if isinstance(v, list):
                        if len(v) != 5:
                            return None
                        return tuple(int(x) for x in v)
                except Exception:
                    return None
                return None

            env_set = set()
            for v in env_la0:
                kk = _k5(v)
                if kk is not None:
                    env_set.add(kk)

            given_set = set()
            for v in legal_action_ids:
                kk = _k5(v)
                if kk is not None:
                    given_set.add(kk)

            if env_set != given_set:
                only_env = sorted(list(env_set - given_set))[:12]
                only_given = sorted(list(given_set - env_set))[:12]
                raise RuntimeError(
                    "[AZ] legal_action_ids set mismatch vs env.legal_actions() at select_action entry (no fallback). "
                    f"only_env_head={only_env} only_given_head={only_given} "
                    f"env_n={int(len(env_la0))} given_n={int(len(legal_action_ids))}"
                )

            # env の順へ正規化（以後の policy/mcts/pi の index 対応を env 基準に統一）
            # cand_vecs が与えられている場合も同じ並びへ追従させる
            idx_map = {}
            for i, v in enumerate(legal_action_ids):
                kk = _k5(v)
                if kk is not None and kk not in idx_map:
                    idx_map[kk] = int(i)

            reordered = []
            reordered_cand = [] if cand_vecs is not None else None
            for v in env_la0:
                kk = _k5(v)
                if kk is None:
                    raise RuntimeError("[AZ] env.legal_actions() contains non-5int vec at select_action entry (no fallback).")
                if kk not in idx_map:
                    raise RuntimeError("[AZ] env.legal_actions() contains vec not present in given legal_action_ids (no fallback).")
                src_i = int(idx_map[kk])
                reordered.append(legal_action_ids[src_i])
                if reordered_cand is not None:
                    try:
                        reordered_cand.append(cand_vecs[src_i])
                    except Exception:
                        raise RuntimeError("[AZ] cand_vecs reorder failed at select_action entry (no fallback).")

            legal_action_ids = reordered
            if reordered_cand is not None:
                cand_vecs = reordered_cand

        if use_mcts and int(num_sims) > 0:
            if env is None:
                raise RuntimeError("[AZ] use_mcts=1 but env is None (no fallback).")

            mcts_pi = None
            try:
                mcts_pi = self._run_mcts(env, legal_action_ids, num_simulations=num_sims)
            except Exception as e:
                import time
                import traceback
                from ..debug_dump import write_debug_dump

                payload = None

                m = getattr(env, "_match", None)
                p = getattr(env, "_player", None)
                forced_active = None
                try:
                    fa = getattr(m, "forced_actions", None) if m is not None else None
                    if isinstance(fa, (list, tuple)):
                        forced_active = len(fa) > 0
                except Exception:
                    forced_active = None

                legal_actions_serialized = []
                try:
                    for i, vec in enumerate(legal_action_ids):
                        if isinstance(vec, tuple):
                            vec = list(vec)
                        if not isinstance(vec, list):
                            vec = [0, 0, 0, 0, 0]
                        if len(vec) < 5:
                            vec = list(vec) + [0] * (5 - len(vec))
                        if len(vec) > 5:
                            vec = list(vec)[:5]
                        legal_actions_serialized.append(
                            {"i": i, "action_type": None, "name": None, "vec": vec}
                        )
                except Exception:
                    legal_actions_serialized = []

                try:
                    payload = {
                        "error_type": type(e).__name__,
                        "exception_class": type(e).__name__,
                        "error_message": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "run_context": {
                            "game_id": getattr(m, "game_id", None) if m is not None else None,
                            "turn": getattr(m, "turn", None) if m is not None else None,
                            "player": getattr(p, "name", None) if p is not None else None,
                            "forced_actions_active": forced_active,
                        },
                        "action_context": {
                            "selected_vec": None,
                            "selected_source": "mcts",
                            "legal_actions_serialized": legal_actions_serialized,
                        },
                        "mcts_context": {
                            "num_simulations": int(num_sims),
                            "n_actions": int(len(legal_action_ids)),
                        },
                        "traceback": traceback.format_exception(type(e), e, e.__traceback__),
                    }
                except Exception:
                    payload = None

                try:
                    dump_path = write_debug_dump(payload)
                except Exception:
                    dump_path = None

                if dump_path is not None:
                    try:
                        cwd = os.getcwd()
                        rel = os.path.relpath(dump_path, cwd)
                    except Exception:
                        cwd = None
                        rel = dump_path
                    print(f"[DEBUG_DUMP] wrote: {rel} cwd={cwd}", flush=True)

                try:
                    rc = payload.get("run_context", {}) if isinstance(payload, dict) else {}
                    print(
                        f"[AZ][MCTS][EXCEPTION_HANDLED] game_id={rc.get('game_id')} turn={rc.get('turn')} player={rc.get('player')} err={type(e).__name__} msg={str(e)}",
                        flush=True,
                    )
                except Exception:
                    pass

                mcts_pi = None

            if mcts_pi is not None and not (isinstance(mcts_pi, list) and int(len(mcts_pi)) == int(len(legal_action_ids))):
                raise RuntimeError("[AZ] _run_mcts returned invalid pi (no fallback).")
            if mcts_pi is not None:
                probs = mcts_pi
                decision_src = "mcts"

        if os.getenv("AZ_PI_DIAG", "1") == "1":
            try:
                _psum = 0.0
                for _x in probs:
                    try:
                        _psum += float(_x)
                    except Exception:
                        pass
                print(f"[AZ][PI_DIAG][SELECT_ACTION] decision_src={decision_src} probs_len={int(len(probs))} probs_sum={_psum:.6f} n_actions={int(len(legal_action_ids))}", flush=True)
            except Exception:
                pass

        if bool(getattr(self, "greedy", False)):
            idx = int(max(range(len(probs)), key=lambda i: float(probs[i])))
            pick = "argmax"
        else:
            r = random.random()
            acc = 0.0
            idx = len(legal_action_ids) - 1
            for i, p in enumerate(probs):
                acc += float(p)
                if r <= acc:
                    idx = i
                    break
            pick = "sample"

        if os.getenv("AZ_PI_DIAG", "1") == "1":
            try:
                print(f"[AZ][PI_DIAG][SELECT_ACTION_PICK] decision_src={decision_src} pick={pick} idx={int(idx)}", flush=True)
            except Exception:
                pass

        chosen_action = legal_action_ids[idx]

        try:
            self.last_decision_src = str(decision_src)
            self.last_pick = str(pick)
            self.last_selected_la5_index = int(idx)
        except Exception:
            pass

        if os.getenv("AZ_DECISION_LOG", "1") == "1":
            try:
                a_repr = repr(chosen_action)
                if len(a_repr) > 160:
                    a_repr = a_repr[:160] + "..."
            except Exception:
                a_repr = "<unrepr>"
            _p = None
            try:
                _p = getattr(env, "player", None)
                if _p is None:
                    _p = getattr(env, "_player", None)
            except Exception:
                _p = None
            self._az_emit_decision(
                f"[AZ][DECISION] src={decision_src} pick={pick} idx={int(idx)} n_actions={int(len(legal_action_ids))} action={a_repr}",
                player=_p,
            )

        return chosen_action, probs

    def select_action_index_online(self, state_dict: Any, actions: List[Any], player: Any = None, return_pi: bool = False):
        if os.getenv("AZ_DECISION_LOG", "1") == "1":
            try:
                m = getattr(player, "match", None) if player is not None else None
                gid = getattr(m, "game_id", None) if m is not None else None
                turn = getattr(m, "turn", None) if m is not None else None
                n_act = int(len(actions)) if isinstance(actions, list) else -1
                self._az_emit_decision(
                    f"[AZ][DECISION][CALL] entry=select_action_index_online game_id={gid} turn={turn} n_actions={n_act} return_pi={int(bool(return_pi))}",
                    player=player,
                )
            except Exception:
                pass

        return self.select_action_index(state_dict, actions, player=player, return_pi=return_pi)

    def select_action_index(self, state_dict: Any, actions: List[Any], player: Any = None, return_pi: bool = False) -> Any:
        _az_log = (os.getenv("AZ_DECISION_LOG", "1") == "1")
        _gid = None
        _turn = None
        try:
            m = getattr(player, "match", None) if player is not None else None
            _gid = getattr(m, "game_id", None) if m is not None else None
            _turn = getattr(m, "turn", None) if m is not None else None
        except Exception:
            _gid = None
            _turn = None

        if os.getenv("AZ_PI_DIAG", "1") == "1":
            try:
                _is_dict = int(isinstance(state_dict, dict))
            except Exception:
                _is_dict = -1
            try:
                _keys_head = list(state_dict.keys())[:12] if isinstance(state_dict, dict) else None
            except Exception:
                _keys_head = None
            try:
                _n_act = int(len(actions)) if isinstance(actions, list) else -1
            except Exception:
                _n_act = -1
            try:
                self._az_emit_decision(
                    f"[AZ][PI_DIAG][SELECT_ACTION_INDEX][ENTER] game_id={_gid} turn={_turn} state_dict_is_dict={_is_dict} keys_head={_keys_head} n_actions={_n_act} return_pi={int(bool(return_pi))}",
                    player=player,
                )
            except Exception:
                pass

        if _az_log:
            try:
                n_act = int(len(actions)) if isinstance(actions, list) else -1
                self._az_emit_decision(
                    f"[AZ][DECISION][CALL] entry=select_action_index game_id={_gid} turn={_turn} n_actions={n_act} return_pi={int(bool(return_pi))}",
                    player=player,
                )
            except Exception:
                pass

        if not isinstance(actions, list) or not actions:
            raise RuntimeError("[AZ] actions is empty (no fallback).")

        obs_belief_root = None
        if isinstance(state_dict, dict):
            _v = None
            try:
                _v = state_dict.get("obs_belief_root", None)
            except Exception:
                _v = None

            # ★追加: obs_belief_root が dict ラップなら obs_belief_vec/belief_vec を unwrap
            if isinstance(_v, dict):
                _inner = None
                try:
                    _inner = _v.get("obs_belief_vec", None)
                except Exception:
                    _inner = None
                if _inner is None:
                    try:
                        _inner = _v.get("belief_vec", None)
                    except Exception:
                        _inner = None
                _v = _inner

            if _v is None:
                try:
                    _v = state_dict.get("obs_belief_vec", None)
                except Exception:
                    _v = None

            try:
                _need = int(self.obs_dim or 0)
            except Exception:
                _need = 0
            if isinstance(_v, (list, tuple)) and (_need <= 0 or len(_v) == _need):
                obs_belief_root = list(_v) if isinstance(_v, tuple) else _v

        if obs_belief_root is None:
            obs_belief_root = self.get_obs_vec(state_dict=state_dict, actions=actions, player=player)

        if obs_belief_root is None:
            raise RuntimeError("[AZ] obs_belief_root is missing (no fallback).")

        if isinstance(state_dict, dict):
            try:
                state_dict["obs_belief_root"] = obs_belief_root
            except Exception:
                pass

        la_5 = self._extract_legal_actions_5(state_dict, actions, player=player)
        if not la_5:
            raise RuntimeError("[AZ] failed to extract legal_actions_5 (no fallback).")

        if isinstance(state_dict, dict):
            state_dict["legal_actions_5"] = la_5
            state_dict["legal_actions_vec"] = la_5
            state_dict["legal_actions"] = la_5
            state_dict["az_la5_n"] = int(len(la_5))

        actions_src = actions
        legal_action_ids: List[Any] = list(la_5)

        # actions と la_5 の対応は「同数・同順」を前提にする（崩れるなら曖昧性のため fatal）
        if int(len(legal_action_ids)) != int(len(actions_src)):
            raise RuntimeError(
                "[AZ] la_5 length mismatch vs actions (no fallback). "
                f"len_la5={int(len(legal_action_ids))} len_actions={int(len(actions_src))}"
            )

        la5_src_indices = list(range(int(len(legal_action_ids))))

        env = None
        use_mcts = bool(getattr(self, "use_mcts", False))
        num_sims = int(getattr(self, "num_simulations", 0) or 0)

        # ★追加: MCTS 中は cand_vecs を state_dict から供給できないため、
        # cand_dim!=5 なら action_encoder_fn が必須（フォールバック禁止）
        if use_mcts and int(num_sims) > 0:
            try:
                _cdim = int(self.cand_dim or 0)
            except Exception:
                _cdim = 0
            if int(_cdim) != 5 and self.action_encoder_fn is None:
                raise RuntimeError(
                    "[AZ] use_mcts=1 requires action_encoder_fn when cand_dim!=5 (no fallback). "
                    f"cand_dim={int(_cdim)}"
                )

        if player is None or getattr(player, "match", None) is None:
            raise RuntimeError("[AZ] player or player.match is missing (no fallback).")

        MatchPlayerSimEnv = None
        try:
            from .mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
            MatchPlayerSimEnv = _MatchPlayerSimEnv
        except Exception as e_rel:
            try:
                from pokeca_simlab.policy.mcts_env import MatchPlayerSimEnv as _MatchPlayerSimEnv
                MatchPlayerSimEnv = _MatchPlayerSimEnv
            except Exception as e_abs:
                raise RuntimeError(f"[AZ] MatchPlayerSimEnv import failed: rel={repr(e_rel)} abs={repr(e_abs)}")

        env = MatchPlayerSimEnv(player.match, player)

        # ★契約: obs は env.get_obs_vec() を唯一の正とする（state_dict 由来 obs は信用しない）
        try:
            obs_belief_root = env.get_obs_vec()
        except Exception as e:
            raise RuntimeError(f"[AZ] env.get_obs_vec() failed (no fallback): {type(e).__name__}:{e}")

        if _az_log: self._az_emit_decision(f"[AZ][RUNTIME] game_id={_gid} turn={_turn} obs_len={len(obs_belief_root) if isinstance(obs_belief_root, list) else 'NA'} la5_n={int(len(la_5))} use_mcts={int(use_mcts)} sims={int(num_sims)}", player=player)

        # ★変更: use_mcts の有無に関わらず、env が作れた時点で
        # - 重複 fatal
        # - 集合一致（順不同）で整合を確認
        # - env の順に正規化し、la5_src_indices も追従させる
        env_la = env.legal_actions()
        if not isinstance(env_la, list) or not env_la:
            raise RuntimeError("[AZ] env.legal_actions() returned empty at root (no fallback).")

        # 重複LA5は設計上あり得るため、fatalにはしない（macro重複を保持する）
        # self._az_assert_no_duplicate_actions_5(env_la, where="select_action_index:root_env_la")
        # self._az_assert_no_duplicate_actions_5(la_5, where="select_action_index:actions_la5")

        def _k5(v):
            try:
                if isinstance(v, tuple):
                    v = list(v)
                if isinstance(v, list):
                    if len(v) != 5:
                        return None
                    return tuple(int(x) for x in v)
            except Exception:
                return None
            return None

        # まず「同じ5-intが何回出たか」も含めて整合チェック（count一致）
        def _k5(v):
            try:
                if isinstance(v, tuple):
                    v = list(v)
                if isinstance(v, list):
                    if len(v) != 5:
                        return None
                    return tuple(int(x) for x in v)
            except Exception:
                return None
            return None

        def _count_map(vecs):
            mp = {}
            for v in vecs:
                kk = _k5(v)
                if kk is None:
                    continue
                mp[kk] = int(mp.get(kk, 0)) + 1
            return mp

        env_cnt = _count_map(env_la)
        src_cnt = _count_map(la_5)

        if env_cnt != src_cnt:
            only_env = []
            only_src = []
            try:
                keys = set(list(env_cnt.keys()) + list(src_cnt.keys()))
            except Exception:
                keys = set()
            for kk in sorted(list(keys))[:200]:
                ce = int(env_cnt.get(kk, 0))
                cs = int(src_cnt.get(kk, 0))
                if ce != cs:
                    if ce > cs:
                        only_env.append((kk, ce, cs))
                    else:
                        only_src.append((kk, ce, cs))
                if len(only_env) >= 12 or len(only_src) >= 12:
                    break
            raise RuntimeError(
                "[AZ] env.legal_actions() multiset mismatch vs actions-derived la_5 at root (no fallback). "
                f"only_env_head={only_env[:12]} only_src_head={only_src[:12]} "
                f"env_n={int(len(env_la))} src_n={int(len(la_5))}"
            )

        # env順へ正規化：重複は「出現順」で対応づけ（first occurrence潰しをしない）
        queues = {}
        for i, v in enumerate(la_5):
            kk = _k5(v)
            if kk is None:
                raise RuntimeError("[AZ] actions-derived la_5 contains non-5int vec at root (no fallback).")
            if kk not in queues:
                queues[kk] = []
            queues[kk].append(int(i))

        mapped = []
        for v in env_la:
            kk = _k5(v)
            if kk is None:
                raise RuntimeError("[AZ] env.legal_actions() contains non-5int vec at root (no fallback).")
            q = queues.get(kk)
            if not isinstance(q, list) or not q:
                raise RuntimeError("[AZ] env.legal_actions() contains vec not present in actions-derived la_5 (no fallback).")
            mapped.append(int(q.pop(0)))

        legal_action_ids = list(env_la)
        la5_src_indices = mapped
        if isinstance(state_dict, dict):
            try:
                state_dict["legal_actions_5"] = legal_action_ids
                state_dict["legal_actions_vec"] = legal_action_ids
                state_dict["legal_actions"] = legal_action_ids
                state_dict["az_la5_n"] = int(len(legal_action_ids))
            except Exception:
                pass

        cand_vecs = self._extract_cand_vecs_for_model(state_dict, legal_action_ids)

        chosen_action, pi = self.select_action(obs_belief_root, legal_action_ids, env=env, cand_vecs=cand_vecs)

        if chosen_action is None:
            raise RuntimeError("[AZ] chosen_action is None (no fallback).")

        idx_la5 = None
        try:
            idx_la5 = int(getattr(self, "last_selected_la5_index", None))
        except Exception:
            idx_la5 = None

        if idx_la5 is None:
            raise RuntimeError("[AZ] last_selected_la5_index is missing (no fallback).")
        if idx_la5 < 0 or idx_la5 >= int(len(legal_action_ids)):
            raise RuntimeError("[AZ] last_selected_la5_index out of range (no fallback).")

        if not (isinstance(pi, list) and len(pi) == len(legal_action_ids)):
            raise RuntimeError("[AZ] pi missing or length mismatch (no fallback).")

        if idx_la5 < 0 or idx_la5 >= int(len(la5_src_indices)):
            raise RuntimeError("[AZ] la5_src_indices out of range (no fallback).")

        idx = int(la5_src_indices[int(idx_la5)])
        if idx < 0 or idx >= int(len(actions_src)):
            raise RuntimeError("[AZ] selected idx out of range vs actions (no fallback).")

        pi_full = [0.0] * int(len(actions_src))
        for j, p in enumerate(pi):
            try:
                k = int(la5_src_indices[int(j)])
                if 0 <= k < int(len(pi_full)):
                    pi_full[k] += float(p)
            except Exception:
                pass

        if isinstance(state_dict, dict):
            state_dict["mcts_pi"] = [float(x) for x in pi_full]
            state_dict["pi"] = [float(x) for x in pi_full]
            state_dict["mcts_idx"] = int(idx)
            state_dict["mcts_pi_present"] = 1
            state_dict["mcts_pi_len"] = int(len(pi_full))
            state_dict["mcts_pi_type"] = type(pi_full).__name__
            state_dict["mcts_pi_from"] = "az_pi_full"
            try:
                state_dict["mcts_pi_la5"] = [float(x) for x in pi]
                state_dict["mcts_idx_la5"] = int(idx_la5)
            except Exception:
                pass
            try:
                state_dict["az_decision_src"] = str(getattr(self, "last_decision_src", "unknown"))
                state_dict["az_decision_pick"] = str(getattr(self, "last_pick", "unknown"))
                state_dict["az_mcts_debug"] = getattr(self, "_last_mcts_detail", None)
            except Exception:
                pass

        if os.getenv("AZ_PI_DIAG", "1") == "1":
            try:
                _psum = 0.0
                for _x in pi_full:
                    try:
                        _psum += float(_x)
                    except Exception:
                        pass
                self._az_emit_decision(
                    f"[AZ][PI_DIAG][SELECT_ACTION_INDEX][WROTE] game_id={_gid} turn={_turn} state_dict_is_dict={int(isinstance(state_dict, dict))} idx={int(idx)} pi_full_len={int(len(pi_full))} pi_full_sum={_psum:.6f} mcts_pi_present={1 if isinstance(state_dict, dict) else 0}",
                    player=player,
                )
            except Exception:
                pass

        try:
            self.last_pi = [float(x) for x in pi_full]
            self.mcts_pi = [float(x) for x in pi_full]
            self.last_mcts_pi = [float(x) for x in pi_full]
        except Exception:
            pass

        if _az_log:
            try:
                self._az_emit_decision(
                    f"[AZ][DECISION] src={str(getattr(self, 'last_decision_src', 'unknown'))} pick={str(getattr(self, 'last_pick', 'unknown'))} idx={int(idx)} n_actions={int(len(actions))} pi_from=az_pi_full",
                    player=player,
                )
            except Exception:
                pass

        return (int(idx), [float(x) for x in pi_full]) if bool(return_pi) else int(idx)

    def _run_mcts(
        self,
        env: "MCTSSimEnvProtocol",
        legal_action_ids: List[Any],
        num_simulations: Optional[int] = None,
    ) -> List[float]:
        try:
            _n0 = len(legal_action_ids) if isinstance(legal_action_ids, list) else -1
        except Exception:
            _n0 = -1
        try:
            _env_name = type(env).__name__ if env is not None else None
        except Exception:
            _env_name = None
        if os.getenv("AZ_MCTS_ENTER_LOG", "0") == "1":
            print(
                f"[AZ][MCTS][_run_mcts][ENTER_ALWAYS] env={_env_name} n_actions={int(_n0)} num_simulations={num_simulations}",
                flush=True,
            )

        try:
            self._mcts_enter_counter = int(getattr(self, "_mcts_enter_counter", 0) or 0) + 1
        except Exception:
            self._mcts_enter_counter = 1

        if env is None:
            raise RuntimeError("[AZ][MCTS] env is None (no fallback).")
        if self.cand_dim is None:
            raise RuntimeError("[AZ][MCTS] cand_dim is None (no fallback).")
        try:
            _cdim = int(self.cand_dim or 0)
        except Exception:
            _cdim = 0
        if int(_cdim) != 5 and self.action_encoder_fn is None:
            raise RuntimeError(
                "[AZ][MCTS] action_encoder_fn is required when cand_dim!=5 (no fallback). "
                f"cand_dim={int(_cdim)}"
            )
        if not isinstance(legal_action_ids, list) or not legal_action_ids:
            raise RuntimeError("[AZ][MCTS] legal_action_ids is empty (no fallback).")

        sims = num_simulations if num_simulations is not None else getattr(self, "num_simulations", 0)
        if int(sims) <= 0:
            raise RuntimeError(f"[AZ][MCTS] sims<=0 (sims={int(sims)}) (no fallback).")

        if os.getenv("AZ_DECISION_LOG", "1") == "1" and int(getattr(self, "_mcts_enter_counter", 0) or 0) == 1: print(f"[AZ][RUNTIME] use_mcts={int(bool(getattr(self, 'use_mcts', False)))} sims={int(sims)}", flush=True)

        try:
            self._mcts_t0_perf = time.perf_counter()
        except Exception:
            self._mcts_t0_perf = None

        def _akey(a):
            try:
                hash(a)
                return a
            except Exception:
                pass
            if isinstance(a, list):
                return tuple(_akey(x) for x in a)
            if isinstance(a, tuple):
                return tuple(_akey(x) for x in a)
            if isinstance(a, dict):
                try:
                    items = sorted(a.items(), key=lambda kv: str(kv[0]))
                except Exception:
                    items = list(a.items())
                return tuple((_akey(k), _akey(v)) for k, v in items)
            try:
                return ("repr", repr(a))
            except Exception:
                return ("id", id(a))

        root = MCTSNode(parent=None, prior=1.0, state_key=None, action_from_parent=None)

        root_actions_env = env.legal_actions()
        if not isinstance(root_actions_env, list) or not root_actions_env:
            raise RuntimeError("[AZ][MCTS] env.legal_actions() returned empty at root (no fallback).")

        # ★重複 fatal（MCTS 中は曖昧性を許容しない）
        self._az_assert_no_duplicate_actions_5(root_actions_env, where="_run_mcts:root_legal_actions")

        try:
            if bool(env.is_terminal()):
                raise RuntimeError("[AZ][MCTS] env.is_terminal()==True at root (no fallback).")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"[AZ][MCTS] env.is_terminal() check failed at root: {type(e).__name__}:{e} (no fallback).")

        if int(len(root_actions_env)) != int(len(legal_action_ids)):
            raise RuntimeError(
                f"[AZ][MCTS] root legal_actions length mismatch: env={int(len(root_actions_env))} given={int(len(legal_action_ids))} (no fallback)."
            )

        try:
            for i in range(len(legal_action_ids)):
                if _akey(root_actions_env[i]) != _akey(legal_action_ids[i]):
                    raise RuntimeError("[AZ][MCTS] root legal_actions content mismatch (no fallback).")
        except Exception as e:
            raise RuntimeError(f"[AZ][MCTS] root legal_actions mismatch detail: {e}")

        obs_root = env.get_obs_vec()
        try:
            self._az_pv_callsite = "mcts:root"
        except Exception:
            pass
        priors_root, v_cur_root = self._policy_value(obs_root, legal_action_ids, cand_vecs=None)
        try:
            self._az_pv_callsite = None
        except Exception:
            pass

        try:
            v_root = float(env.value_to_root(float(v_cur_root)))
        except Exception:
            raise RuntimeError("[AZ][MCTS] env.value_to_root is missing or failed (no fallback).")

        priors = [float(x) for x in priors_root]

        use_dir = os.getenv("AZ_MCTS_DIRICHLET", "0") == "1"
        if use_dir:
            try:
                import numpy as np
                alpha = float(self.dirichlet_alpha)
                eps = float(self.dirichlet_eps)
                if alpha > 0.0 and eps > 0.0:
                    noise = np.random.dirichlet([alpha] * len(priors)).astype("float64").tolist()
                    priors = [float((1.0 - eps) * float(p) + eps * float(n)) for p, n in zip(priors, noise)]
                    s = float(sum(priors))
                    if not (s > 0.0):
                        raise RuntimeError("[AZ][MCTS] dirichlet produced non-positive sum (no fallback).")
                    priors = [float(p) / s for p in priors]
            except Exception as e:
                raise RuntimeError(f"[AZ][MCTS] dirichlet failed: {e} (no fallback).")

        root_keys = []
        for i, (aid, p) in enumerate(zip(legal_action_ids, priors)):
            k = (int(i), _akey(aid))
            root_keys.append(k)
            root.children[k] = MCTSNode(parent=root, prior=float(p), state_key=None, action_from_parent=(int(i), aid))

        sim_ok = 0
        sim_err = 0

        mute = os.getenv("AZ_MCTS_MUTE", "1") == "1"
        echo = os.getenv("AZ_MCTS_ECHO", ("1" if mute else "0")) == "1"
        if mute:
            import contextlib
            import io

            @contextlib.contextmanager
            def _mute_stdio():
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    yield buf
        else:
            import contextlib

            @contextlib.contextmanager
            def _mute_stdio():
                yield None

        def _echo_from_buf(_buf_text: str):
            if not echo:
                return
            try:
                k = "_mcts_echo_budget"
                b = getattr(self, k, None)
                if b is None:
                    b = int(os.getenv("AZ_MCTS_ECHO_BUDGET", "40") or "40")
                b = int(b)
                if b <= 0:
                    return

                mismatch_hits = []
                la5_hits = []
                act_hits = []
                for ln in str(_buf_text).splitlines():
                    if "[AZ][MCTS][SELECTION_MISMATCH]" in ln:
                        mismatch_hits.append(ln)
                    elif "[MCTS_ENV][LA5]" in ln:
                        la5_hits.append(ln)
                    elif "[ACTION_SERIALIZE][MCTS]" in ln:
                        act_hits.append(ln)

                hits = (mismatch_hits + la5_hits + act_hits)[:60]

                if not hits:
                    return

                for ln in hits:
                    if b <= 0:
                        break
                    print(ln, flush=True)
                    b -= 1

                setattr(self, k, b)
            except Exception:
                pass

        def _format_vec_list(vecs, full: bool = False) -> str:
            import os

            def _safe_repr(x, limit=2000):
                try:
                    s = repr(x)
                except Exception:
                    s = f"<repr failed: {type(x).__name__}>"
                if len(s) > int(limit):
                    s = s[: int(limit)] + "..."
                return s

            if not isinstance(vecs, list):
                return _safe_repr(vecs)

            mode = str(os.getenv("MCTS_ENV_LA5_LIST_MODE", "headtail")).strip().lower()
            if full or mode == "full":
                try:
                    return repr(vecs)
                except Exception:
                    return _safe_repr(vecs, limit=6000)

            try:
                k = int(os.getenv("MCTS_ENV_LA5_LIST_K", "6") or "6")
            except Exception:
                k = 6

            if k <= 0 or len(vecs) <= 2 * k:
                return _safe_repr(vecs, limit=2400)

            head = vecs[:k]
            tail = vecs[-k:]
            return f"{_safe_repr(head, limit=2400)}...{_safe_repr(tail, limit=2400)}(len={len(vecs)})"

        def _hash_vec_list(vecs) -> Optional[str]:
            from .trace_utils import hash_vecs

            return hash_vecs(vecs)

        def _get_env_ctx(_env):
            try:
                fn = getattr(_env, "_get_log_context", None)
                if callable(fn):
                    return fn()
            except Exception:
                pass
            out = {}
            try:
                m = getattr(_env, "_match", None)
                out["game_id"] = getattr(m, "game_id", None) if m is not None else None
                out["turn"] = getattr(m, "turn", None) if m is not None else None
            except Exception:
                out["game_id"] = None
                out["turn"] = None
            try:
                p = getattr(_env, "_get_current_player", None)
                if callable(p):
                    out["player"] = getattr(p(), "name", None)
            except Exception:
                out["player"] = None
            try:
                fa = getattr(_env, "_forced_is_active", None)
                if callable(fa):
                    out["forced_active"] = bool(fa())
            except Exception:
                out["forced_active"] = None
            return out

        _timeline = (str(os.getenv("AZ_MCTS_TIMELINE", "0")).strip() == "1")
        try:
            _timeline_k = int(os.getenv("AZ_MCTS_TIMELINE_K", "8") or "8")
        except Exception:
            _timeline_k = 8

        def _sum_child_N(_node: MCTSNode) -> int:
            try:
                return int(sum(int(getattr(c, "N", 0) or 0) for c in _node.children.values()))
            except Exception:
                return 0

        def _child_N_head(_node: MCTSNode, _keys, k: int) -> list:
            out = []
            try:
                kk = int(k)
            except Exception:
                kk = 8
            if kk <= 0:
                kk = 8
            try:
                keys = list(_keys) if isinstance(_keys, (list, tuple)) else []
            except Exception:
                keys = []
            for _k in keys[:kk]:
                try:
                    ch = _node.children.get(_k)
                except Exception:
                    ch = None
                try:
                    out.append(int(getattr(ch, "N", 0) or 0) if ch is not None else 0)
                except Exception:
                    out.append(0)
            return out

        def _emit_timeline(phase: str, sim: int, selection_aborted: int, did_backup: int) -> None:
            if not _timeline:
                return
            try:
                print(
                    "[AZ][MCTS][TIMELINE]"
                    f" phase={phase}"
                    f" sim={int(sim)}"
                    f" sim_ok={int(sim_ok)}"
                    f" sim_err={int(sim_err)}"
                    f" root_N={int(getattr(root, 'N', 0) or 0)}"
                    f" sum_child_N={int(_sum_child_N(root))}"
                    f" child_N_head={_child_N_head(root, root_keys, _timeline_k)}"
                    f" selection_aborted={int(selection_aborted)}"
                    f" did_backup={int(did_backup)}",
                    flush=True,
                )
            except Exception:
                pass

        _mcts_diag_backup_pathlen1 = 0
        _mcts_diag_root_leaf_start = 0
        _mcts_diag_sel_mismatch = 0
        _mcts_diag_sel_mismatch_root = 0
        _mcts_diag_last_mismatch = None

        for sim_i in range(int(sims)):
            buf_text = None

            try:
                if bool(getattr(root, "is_leaf")()):
                    _mcts_diag_root_leaf_start += 1
            except Exception:
                pass

            sim_selection_aborted = False
            sim_did_backup = False

            _emit_timeline("pre", sim_i, selection_aborted=0, did_backup=0)
            with _mute_stdio() as buf:
                try:
                    sim_env = env.clone()
                    try:
                        _ctx0 = _get_env_ctx(sim_env)
                    except Exception:
                        _ctx0 = {}
                    try:
                        _term0 = bool(sim_env.is_terminal())
                    except Exception:
                        _term0 = None
                    _la0 = None
                    _la0_n = None
                    try:
                        _la0 = sim_env.legal_actions()
                        _la0_n = len(_la0) if isinstance(_la0, list) else -1
                    except Exception:
                        _la0 = None
                        _la0_n = None
                    try:
                        print(
                            "[AZ][MCTS][SIM_START]"
                            f" sim={int(sim_i)}"
                            f" game_id={_ctx0.get('game_id')}"
                            f" turn={_ctx0.get('turn')}"
                            f" player={_ctx0.get('player')}"
                            f" forced_active={_ctx0.get('forced_active')}"
                            f" terminal={_term0}"
                            f" n_actions={_la0_n}",
                            flush=True,
                        )
                    except Exception:
                        pass

                    try:
                        if _term0 is True:
                            raise RuntimeError(
                                "[AZ][MCTS] sim_env.is_terminal()==True at SIM_START (no fallback). "
                                f"sim={int(sim_i)} game_id={_ctx0.get('game_id')} turn={_ctx0.get('turn')} "
                                f"player={_ctx0.get('player')} forced_active={_ctx0.get('forced_active')} n_actions={_la0_n}"
                            )
                    except RuntimeError:
                        raise
                    except Exception as e:
                        raise RuntimeError(f"[AZ][MCTS] SIM_START terminal guard failed: {type(e).__name__}:{e} (no fallback).")

                    node = root
                    path = [root]

                    last_action = None
                    depth = 0
                    selection_aborted = False

                    while (not sim_env.is_terminal()) and (not node.is_leaf()):
                        depth += 1
                        total_N = sum(c.N for c in node.children.values()) or 1
                        best_child = None
                        best_score = None

                        for child in node.children.values():
                            u = self.c_puct * float(child.P) * (float(total_N) ** 0.5) / (1.0 + float(child.N))
                            score = float(child.Q) + float(u)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_child = child

                        if best_child is None:
                            raise RuntimeError("[AZ][MCTS] selection failed: best_child is None (no fallback).")

                        if best_child.action_from_parent is None:
                            raise RuntimeError("[AZ][MCTS] selection failed: action_from_parent is None (no fallback).")

                        node_actions = []
                        try:
                            node_actions = [c.action_from_parent for c in node.children.values()]
                        except Exception:
                            node_actions = None
                        selection_aborted = False

                        current_actions = sim_env.legal_actions()

                        # 重複LA5は設計上あり得るためfatalにはしない
                        # self._az_assert_no_duplicate_actions_5(current_actions, where=f"_run_mcts:current_legal_actions sim={int(sim_i)} depth={int(depth)}")

                        action_index = None
                        wanted_vec = None
                        try:
                            action_index = int(best_child.action_from_parent[0])
                            wanted_vec = best_child.action_from_parent[1]
                        except Exception:
                            action_index = None
                            wanted_vec = None

                        if action_index is None or action_index < 0 or action_index >= int(len(current_actions)):
                            selection_aborted = True
                            node.children = {}
                            break

                        # sanity check（同indexのLA5が一致しているはず。違うなら macro変化で selection mismatch 扱い）
                        try:
                            cur_vec = current_actions[action_index]
                            if isinstance(cur_vec, tuple):
                                cur_vec = list(cur_vec)
                            if isinstance(wanted_vec, tuple):
                                wanted_vec = list(wanted_vec)
                            if isinstance(cur_vec, list) and isinstance(wanted_vec, list):
                                if len(cur_vec) == 5 and len(wanted_vec) == 5:
                                    if tuple(int(x) for x in cur_vec) != tuple(int(x) for x in wanted_vec):
                                        selection_aborted = True
                                        node.children = {}
                                        break
                        except Exception:
                            selection_aborted = True
                            node.children = {}
                            break

                        last_action = current_actions[action_index]
                        pre_ctx = _get_env_ctx(sim_env)
                        try:

                            if str(os.getenv("MCTS_ENV_TRACE", "0")).strip() == "1":
                                import json

                                event_id = None
                                generated_new = 0
                                try:
                                    event_id = getattr(sim_env, "_la_cache_event_id", None)
                                except Exception:
                                    event_id = None
                                try:
                                    if event_id is None:
                                        fn_eid = getattr(sim_env, "_next_la5_event_id", None)
                                        if callable(fn_eid):
                                            event_id = fn_eid()
                                            generated_new = 1
                                            try:
                                                setattr(sim_env, "_la_cache_event_id", event_id)
                                            except Exception:
                                                pass
                                except Exception:
                                    event_id = None
                                state_fp = None
                                try:
                                    cur_p = getattr(sim_env, "_get_current_player", None)
                                    if callable(cur_p):
                                        cur_p = cur_p()
                                    fp_fn = getattr(sim_env, "_get_state_fingerprint", None)
                                    if callable(fp_fn):
                                        state_fp = fp_fn(cur_p)
                                except Exception:
                                    state_fp = None
                                selected_idx = None
                                try:
                                    if isinstance(node_actions, list):
                                        selected_idx = node_actions.index(last_action)
                                except Exception:
                                    selected_idx = None
                                print(
                                    "[AZ][MCTS][TRACE_B]"
                                    f" event_id={event_id}"
                                    f" generated_new={generated_new}"
                                    f" selection_source=mcts_tree"
                                    f" selected_idx={selected_idx}"
                                    f" selected_vec={_format_vec_list(last_action, full=True)}"
                                    f" state_fingerprint_online=NA"
                                    f" state_fingerprint_env={state_fp if state_fp is not None else 'NA'}"
                                    f" legal_actions_vec_hash={_hash_vec_list(node_actions)}"
                                    f" game_id={pre_ctx.get('game_id')}"
                                    f" turn={pre_ctx.get('turn')}"
                                    f" player={pre_ctx.get('player')}"
                                    f" trace_json={json.dumps({'event_id': event_id,'generated_new': int(generated_new),'selection_source': 'mcts_tree','selected_idx': selected_idx,'selected_vec': last_action,'state_fingerprint_online': 'NA','state_fingerprint_env': state_fp if state_fp is not None else 'NA','legal_actions_vec_hash': _hash_vec_list(node_actions),'game_id': pre_ctx.get('game_id'),'turn': pre_ctx.get('turn'),'player': pre_ctx.get('player')}, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)}",
                                    flush=True,
                                )
                        except Exception:
                            pass
                        try:
                            # 適用側（env.step）の lookup が重複 vec で曖昧にならないよう、選択ヒントを渡す
                            try:
                                setattr(sim_env, "_la5_selected_idx_hint", int(action_index))
                            except Exception:
                                pass
                            try:
                                setattr(sim_env, "_la5_selected_vec_hint", last_action)
                            except Exception:
                                pass
                            sim_env.step(last_action)
                        except Exception as e:
                            post_ctx = _get_env_ctx(sim_env)
                            try:
                                event_id = getattr(sim_env, "_last_step_no_match_event_id", None)
                            except Exception:
                                event_id = None
                            try:
                                print(
                                    "[AZ][MCTS][STEP_FAIL]"
                                    f" sim={int(sim_i)} depth={int(depth)}"
                                    f" game_id={pre_ctx.get('game_id')}"
                                    f" pre_turn={pre_ctx.get('turn')}"
                                    f" pre_player={pre_ctx.get('player')}"
                                    f" pre_forced={pre_ctx.get('forced_active')}"
                                    f" post_turn={post_ctx.get('turn')}"
                                    f" post_player={post_ctx.get('player')}"
                                    f" post_forced={post_ctx.get('forced_active')}"
                                    f" n_actions={len(node_actions) if isinstance(node_actions, list) else 'NA'}"
                                    f" target={_format_vec_list(last_action, full=True)}"
                                    f" node_actions={_format_vec_list(node_actions, full=True)}"
                                    f" event_id={event_id}"
                                    f" err={type(e).__name__}:{e}",
                                    flush=True,
                                )
                            except Exception:
                                pass
                            try:
                                setattr(self, "last_step_no_match_event_id", event_id)
                            except Exception:
                                pass
                            raise

                        node = best_child
                        path.append(node)

                    if sim_env.is_terminal():
                        leaf_value = float(sim_env.result())
                    else:
                        next_actions = sim_env.legal_actions()
                        if not isinstance(next_actions, list) or not next_actions:
                            raise RuntimeError("[AZ][MCTS] non-terminal but legal_actions empty (no fallback).")

                        # ★重複 fatal（leaf 側でも曖昧性を許容しない）
                        self._az_assert_no_duplicate_actions_5(next_actions, where=f"_run_mcts:leaf_legal_actions sim={int(sim_i)} depth={int(depth)}")

                        obs_leaf = sim_env.get_obs_vec()
                        try:
                            self._az_pv_callsite = "mcts:leaf"
                        except Exception:
                            pass
                        priors_leaf, v_cur_leaf = self._policy_value(obs_leaf, next_actions, cand_vecs=None)
                        try:
                            self._az_pv_callsite = None
                        except Exception:
                            pass

                        try:
                            v_root_leaf = float(sim_env.value_to_root(float(v_cur_leaf)))
                        except Exception:
                            raise RuntimeError("[AZ][MCTS] env.value_to_root failed at leaf (no fallback).")

                        leaf_value = v_root_leaf

                        p_sum = float(sum(float(x) for x in priors_leaf))
                        if not (p_sum > 0.0):
                            raise RuntimeError("[AZ][MCTS] priors_leaf sum<=0 (no fallback).")
                        priors_leaf_norm = [float(x) / p_sum for x in priors_leaf]

                        for i, (aid, p) in enumerate(zip(next_actions, priors_leaf_norm)):
                            k = (int(i), _akey(aid))
                            if k in node.children:
                                continue
                            node.children[k] = MCTSNode(parent=node, prior=float(p), state_key=None, action_from_parent=(int(i), aid))

                    if selection_aborted:
                        sim_selection_aborted = True
                    else:
                        try:
                            if isinstance(path, list) and int(len(path)) <= 1:
                                _mcts_diag_backup_pathlen1 += 1
                        except Exception:
                            pass

                        try:
                            if isinstance(path, list) and int(len(path)) <= 1:
                                raise RuntimeError(
                                    "[AZ][MCTS] backup path_len<=1 (no fallback). "
                                    f"sim={int(sim_i)} depth={int(depth)} selection_aborted={int(bool(selection_aborted))}"
                                )
                        except RuntimeError:
                            raise
                        except Exception as e:
                            raise RuntimeError(f"[AZ][MCTS] backup path_len guard failed: {type(e).__name__}:{e} (no fallback).")

                        v = float(leaf_value)
                        for nd in path:
                            nd.N += 1
                            nd.W += v
                            nd.Q = nd.W / float(max(1, nd.N))

                        sim_did_backup = True
                        sim_ok += 1
                except Exception as e:
                    sim_err += 1
                    try:
                        extra = ""
                        if mute and buf is not None:
                            try:
                                extra = str(buf.getvalue())
                                try:
                                    la5_hits = []
                                    act_hits = []
                                    for ln in str(extra).splitlines():
                                        if "[MCTS_ENV][LA5]" in ln:
                                            la5_hits.append(ln)
                                        elif "[ACTION_SERIALIZE][MCTS]" in ln:
                                            act_hits.append(ln)
                                    if la5_hits or act_hits:
                                        extra = "\n".join((la5_hits + act_hits)[:60])
                                except Exception:
                                    pass
                                if len(extra) > 2000:
                                    extra = extra[:2000] + "\n. (truncated)"
                            except Exception:
                                extra = ""
                        raise RuntimeError(
                            f"[AZ][MCTS][SIM_FAIL] sim={int(sim_i)} ok={int(sim_ok)} "
                            f"etype={type(e).__name__} err={repr(e)}"
                            + (("\n[MCTS_MUTE_BUFFER]\n" + extra) if extra else "")
                        )
                    except Exception:
                        raise
                finally:
                    if mute and buf is not None:
                        try:
                            buf_text = str(buf.getvalue())
                        except Exception:
                            buf_text = None

            if mute and buf_text:
                _echo_from_buf(buf_text)


            _emit_timeline("post", sim_i, selection_aborted=int(sim_selection_aborted), did_backup=int(sim_did_backup))

        self._last_mcts_stats = {"sims": int(sims), "ok": int(sim_ok), "err": int(sim_err)}

        visit_counts: List[float] = []
        for k in root_keys:
            child = root.children.get(k)
            n = float(child.N) if child is not None else 0.0
            visit_counts.append(n)

        total_visits = float(sum(visit_counts))
        if not (total_visits > 0.0):
            ctx_game_id = None
            ctx_turn = None
            ctx_player = None
            ctx_forced = None
            ctx_forced_len = None
            try:
                m = getattr(env, "_match", None)
                ctx_game_id = getattr(m, "game_id", None) if m is not None else None
                ctx_turn = getattr(m, "turn", None) if m is not None else None
            except Exception:
                ctx_game_id = None
                ctx_turn = None
            try:
                p = getattr(env, "_player", None)
                ctx_player = getattr(p, "name", None) if p is not None else None
            except Exception:
                ctx_player = None
            try:
                fn = getattr(env, "_forced_is_active", None)
                if callable(fn):
                    ctx_forced = bool(fn())
            except Exception:
                ctx_forced = None
            try:
                fn = getattr(env, "_get_forced_actions_raw", None)
                if callable(fn):
                    fr = fn()
                    if isinstance(fr, (list, tuple)):
                        ctx_forced_len = int(len(fr))
            except Exception:
                ctx_forced_len = None

            _sumN = int(_sum_child_N(root))
            _headN = _child_N_head(root, root_keys, _timeline_k)
            _n_pos = 0
            try:
                for x in visit_counts:
                    if float(x) > 0.0:
                        _n_pos += 1
            except Exception:
                _n_pos = -1

            try:
                import json
                print(
                    "[AZ][MCTS][TOTAL_VISITS_ZERO_DETAIL]"
                    f" game_id={ctx_game_id}"
                    f" turn={ctx_turn}"
                    f" player={ctx_player}"
                    f" forced_active={ctx_forced}"
                    f" forced_len={ctx_forced_len}"
                    f" n_actions={int(len(legal_action_ids))}"
                    f" num_simulations={int(sims)}"
                    f" sim_ok={int(sim_ok)}"
                    f" sim_err={int(sim_err)}"
                    f" root_children={int(len(root.children))}"
                    f" root_N={getattr(root, 'N', None)}"
                    f" sum_child_N={int(_sumN)}"
                    f" n_children_N_gt0={int(_n_pos)}"
                    f" child_N_head={_headN}"
                    f" diag_backup_pathlen1={int(_mcts_diag_backup_pathlen1)}"
                    f" diag_root_leaf_start={int(_mcts_diag_root_leaf_start)}"
                    f" diag_sel_mismatch={int(_mcts_diag_sel_mismatch)}"
                    f" diag_sel_mismatch_root={int(_mcts_diag_sel_mismatch_root)}"
                    f" diag_last_mismatch={json.dumps(_mcts_diag_last_mismatch, ensure_ascii=False) if _mcts_diag_last_mismatch is not None else 'NA'}",
                    flush=True,
                )
            except Exception:
                pass

            try:
                import time
                from ..debug_dump import write_debug_dump

                def _coerce_vec(v):
                    if isinstance(v, tuple):
                        v = list(v)
                    if not isinstance(v, list):
                        v = [0, 0, 0, 0, 0]
                    if len(v) < 5:
                        v = list(v) + [0] * (5 - len(v))
                    if len(v) > 5:
                        v = list(v)[:5]
                    return v

                payload = {
                    "error_type": "total_visits<=0",
                    "exception_class": "RuntimeError",
                    "error_message": "[AZ][MCTS] total_visits<=0 (no fallback).",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "run_context": {
                        "game_id": ctx_game_id,
                        "turn": ctx_turn,
                        "player": ctx_player,
                        "forced_actions_active": ctx_forced,
                    },
                    "action_context": {
                        "selected_vec": None,
                        "selected_source": "mcts",
                        "legal_actions_serialized": [
                            {"i": i, "action_type": None, "name": None, "vec": _coerce_vec(aid)}
                            for i, aid in enumerate(legal_action_ids)
                        ],
                    },
                    "mcts_context": {
                        "num_simulations": int(sims),
                        "total_visits": float(total_visits),
                        "n_actions": int(len(legal_action_ids)),
                        "any_exception_counters": {
                            "sim_ok": int(sim_ok),
                            "sim_err": int(sim_err),
                        },
                        "diag": {
                            "sum_child_N": int(_sumN),
                            "n_children_N_gt0": int(_n_pos),
                            "child_N_head": _headN,
                            "backup_pathlen1": int(_mcts_diag_backup_pathlen1),
                            "root_leaf_start": int(_mcts_diag_root_leaf_start),
                            "sel_mismatch": int(_mcts_diag_sel_mismatch),
                            "sel_mismatch_root": int(_mcts_diag_sel_mismatch_root),
                            "last_mismatch": _mcts_diag_last_mismatch,
                        },
                    },
                }

                dump_path = write_debug_dump(payload)
                if dump_path is not None:
                    try:
                        cwd = os.getcwd()
                        rel = os.path.relpath(dump_path, cwd)
                    except Exception:
                        cwd = None
                        rel = dump_path
                    print(f"[DEBUG_DUMP] wrote: {rel} cwd={cwd}", flush=True)
            except Exception:
                pass
            raise RuntimeError("[AZ][MCTS] total_visits<=0 (no fallback).")

        pi_temp = float(getattr(self, "mcts_pi_temperature", 1.0) or 1.0)
        if pi_temp <= 0.0:
            best_i = int(max(range(len(visit_counts)), key=lambda i: float(visit_counts[i])))
            pi = [0.0] * int(len(visit_counts))
            pi[best_i] = 1.0
        else:
            power = 1.0 / float(pi_temp)
            ww = [float(n) ** float(power) for n in visit_counts]
            s = float(sum(ww))
            if not (s > 0.0):
                raise RuntimeError("[AZ][MCTS] pi normalization failed (no fallback).")
            pi = [float(x) / s for x in ww]

        try:
            q_values: List[float] = []
            prior_values: List[float] = []
            for k in root_keys:
                child = root.children.get(k)
                q_values.append(float(getattr(child, "Q", 0.0)) if child is not None else 0.0)
                prior_values.append(float(getattr(child, "P", 0.0)) if child is not None else 0.0)
            self._last_mcts_detail = {
                "sims": int(sims),
                "visits": [float(x) for x in visit_counts],
                "q": q_values,
                "prior": prior_values,
            }
        except Exception:
            pass

        try:
            if os.getenv("AZ_MCTS_DEBUG", "0") == "1":
                elapsed_ms = None
                try:
                    t0 = getattr(self, "_mcts_t0_perf", None)
                    if t0 is not None:
                        elapsed_ms = int((time.perf_counter() - float(t0)) * 1000.0)
                except Exception:
                    elapsed_ms = None

                topk = 3 if len(pi) >= 3 else len(pi)
                order = sorted(range(len(pi)), key=lambda i: float(visit_counts[i]), reverse=True)[:topk]
                top_parts = []
                for i in order:
                    try:
                        a = legal_action_ids[i]
                        a_repr = repr(a)
                        if len(a_repr) > 120:
                            a_repr = a_repr[:120] + "..."
                    except Exception:
                        a_repr = "<unrepr>"
                    top_parts.append(f"#{i}:N={visit_counts[i]:.0f},p={pi[i]:.3f},a={a_repr}")

                print(
                    f"[AZ][MCTS][CONFIRM] sims={int(sims)} ok={int(sim_ok)} err=0 "
                    f"elapsed_ms={elapsed_ms} root_total_visits={float(total_visits):.1f} "
                    f"top3={' | '.join(top_parts)} stats={getattr(self, '_last_mcts_stats', None)}",
                    flush=True,
                )
        except Exception:
            pass

        if os.getenv("AZ_PI_DIAG", "1") == "1":
            try:
                _pisum = 0.0
                for _x in pi:
                    try:
                        _pisum += float(_x)
                    except Exception:
                        pass
                _vsum = 0.0
                try:
                    for _n in visit_counts:
                        _vsum += float(_n)
                except Exception:
                    _vsum = -1.0
                try:
                    _best = int(best_i)
                except Exception:
                    _best = -1
                print(f"[AZ][PI_DIAG][_RUN_MCTS][EXIT] n_actions={int(len(pi))} pi_sum={_pisum:.6f} visit_sum={_vsum:.0f} best_i={_best}", flush=True)
            except Exception:
                pass

        return [float(x) for x in pi]

    # ======================================================================
    #  以下、AlphaZero 型 MCTS 用の補助メソッド群（未使用の器。必要なら後で整理）
    # ======================================================================

    def _create_root_node(
        self,
        state_key: Optional[Any],
        legal_action_ids: List[Any],
        priors: List[float],
    ) -> MCTSNode:
        root = MCTSNode(parent=None, prior=1.0, state_key=state_key, action_from_parent=None)

        if not legal_action_ids:
            return root

        n = min(len(legal_action_ids), len(priors))
        for i in range(n):
            aid = legal_action_ids[i]
            p = float(priors[i])
            if p < 0.0:
                p = 0.0
            child = MCTSNode(parent=root, prior=p, state_key=None, action_from_parent=aid)
            root.children[aid] = child

        total_p = sum(child.P for child in root.children.values())
        if total_p > 0.0:
            inv_total = 1.0 / total_p
            for child in root.children.values():
                child.P *= inv_total

        return root

    def _select_child(self, node: MCTSNode) -> Tuple[Any, MCTSNode]:
        if not node.children:
            raise ValueError("_select_child called on a leaf node without children.")

        total_N = sum(child.N for child in node.children.values())
        if total_N <= 0:
            total_N = 1

        best_action = None
        best_child = None
        best_score = None

        for action_id, child in node.children.items():
            Q = child.Q
            U = self.c_puct * child.P * ((total_N ** 0.5) / (1.0 + child.N))
            score = Q + U

            if best_score is None or score > best_score:
                best_score = score
                best_action = action_id
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: MCTSNode, action_priors: List[Tuple[Any, float]]) -> None:
        for action_id, prior in action_priors:
            p = float(prior)
            if p < 0.0:
                p = 0.0
            if action_id in node.children:
                node.children[action_id].P = p
            else:
                node.children[action_id] = MCTSNode(
                    parent=node,
                    prior=p,
                    state_key=None,
                    action_from_parent=action_id,
                )

        total_p = sum(child.P for child in node.children.values())
        if total_p > 0.0:
            inv_total = 1.0 / total_p
            for child in node.children.values():
                child.P *= inv_total

    def _backup(self, path: List[MCTSNode], leaf_value: float) -> None:
        v = float(leaf_value)
        for node in path:
            node.N += 1
            node.W += v
            node.Q = node.W / max(1, node.N)

    def _apply_dirichlet_noise_to_root(self, root: MCTSNode) -> None:
        if not root.children:
            return

        import numpy as np

        n = len(root.children)
        if n == 0:
            return

        alpha = float(self.dirichlet_alpha)
        eps = float(self.dirichlet_eps)

        if alpha <= 0.0 or eps <= 0.0:
            return

        noise = np.random.dirichlet([alpha] * n).astype("float64")
        total_p = sum(child.P for child in root.children.values())
        if total_p <= 0.0:
            total_p = 1.0
        base_ps = [child.P / total_p for child in root.children.values()]

        for (action_id, child), base_p, eta in zip(root.children.items(), base_ps, noise):
            child.P = (1.0 - eps) * base_p + eps * float(eta)
