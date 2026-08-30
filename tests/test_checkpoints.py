"""CheckpointStore 集成测试：真实 git + 临时目录验证快照/还原全链路。

覆盖：
  快照-还原往返（含删除快照后新增的文件）
  .env 凭据不进快照、还原不触碰
  无变更时复用 HEAD（账本仍追加）
  跨会话冲突检测（第 2 层）与还原后由其它会话账本找回（第 1 层安全网）
  精确回退（第 3 层）：只撤销本会话改动、同文件冲突回滚、空快照去重
  账本持久化 / prune / truncate / clear / drop
  git 缺失降级、账本损坏容错
"""

import json

import pytest

from agent.checkpoints import CheckpointStore


@pytest.fixture()
def ws(tmp_path):
    """临时 workspace，预置一个代码文件。"""
    w = tmp_path / "ws"
    w.mkdir()
    (w / "code.py").write_text("v1\n", encoding="utf-8")
    return w


@pytest.fixture()
def ckpt_root(tmp_path):
    return tmp_path / "ckpt"


def _store(root, workspace, sid):
    return CheckpointStore(root, str(workspace), sid)


def test_snapshot_and_restore_roundtrip(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    assert store.snapshot("第一轮") is not None
    assert len(store.entries()) == 1

    # 第二轮前修改文件并打快照
    (ws / "code.py").write_text("v2\n", encoding="utf-8")
    (ws / "new.py").write_text("added later\n", encoding="utf-8")
    assert store.snapshot("第二轮") is not None
    assert len(store.entries()) == 2

    # 回退到第一轮之前：文件回 v1，且快照后新增的文件被删除
    result = store.restore(store.entries()[0])
    assert result is not None
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"
    assert not (ws / "new.py").exists()


def test_env_file_never_snapshotted(ws, ckpt_root):
    (ws / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")
    (ws / ".env").write_text("OPENAI_API_KEY=rotated\n", encoding="utf-8")
    store.snapshot("第二轮")

    store.restore(store.entries()[0])

    # .env 被内置 exclude 排除：不进快照树，还原时也不被触碰/删除
    assert (ws / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=rotated\n"


def test_no_change_snapshot_reuses_head_but_appends_ledger(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    c1 = store.snapshot("第一轮")
    c2 = store.snapshot("没有改动的第二轮")
    assert c1 is not None and c2 is not None
    assert len(store.entries()) == 2  # 账本仍按轮次追加（1:1 不变量）
    assert c1 == c2  # 无变更复用同一 commit


def test_cross_session_conflict_detection_and_recovery(ws, ckpt_root):
    """第 1+2 层防线：A 回退前警告 B 的交叉改动；B 可从自己账本找回状态。"""
    import time

    store_a = _store(ckpt_root, ws, "sessA")
    store_a.snapshot("A 第一轮")
    time.sleep(0.05)  # 保证时间戳可比较

    store_b = _store(ckpt_root, ws, "sessB")
    store_b.snapshot("B 第一轮")
    (ws / "code.py").write_text("v2 by B\n", encoding="utf-8")
    time.sleep(0.05)
    store_b.snapshot("B 第二轮")

    # A 回退到自己的第一轮之前：应检测到 sessB 在其后改过 code.py
    plan = store_a.plan_restore(store_a.entries()[0])
    assert plan is not None
    assert ("sessB", ["code.py"]) in plan.conflicts

    # A 仍然回退（模拟用户确认）：B 的改动被覆盖
    store_a.restore(store_a.entries()[0])
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"

    # 第 1 层安全网：B 从自己的账本还原，状态找回
    store_b.restore(store_b.entries()[-1])
    assert (ws / "code.py").read_text(encoding="utf-8") == "v2 by B\n"


def test_precise_restore_preserves_other_session_changes(ws, ckpt_root):
    """第 3 层核心语义：A 精确回退只撤销 A 自己的改动，B 的交叉修改原样保留。"""
    import time

    store_a = _store(ckpt_root, ws, "sessA")
    store_a.snapshot("A 第一轮")  # e_A0：code.py = v1
    # A 第一轮的工作（落在 e_A0 与下一个快照之间的 diff 里）
    (ws / "a.txt").write_text("by A\n", encoding="utf-8")
    time.sleep(0.05)

    store_b = _store(ckpt_root, ws, "sessB")
    store_b.snapshot("B 第一轮")  # e_B0：捕获 A 的工作（新 commit）
    # B 第一轮的工作
    (ws / "b.txt").write_text("by B\n", encoding="utf-8")
    time.sleep(0.05)
    store_b.snapshot("B 第二轮")  # e_B1：捕获 B 的工作

    store_a.snapshot("A 第二轮")  # e_A1：捕获 B 的工作（A 第二轮尚未干活）

    # A 回退到自己的第一轮之前：只应撤销 A 的 a.txt，保留 B 的 b.txt
    pr = store_a.restore_precise(store_a.entries()[0])
    assert pr is not None and pr.ok is True
    assert pr.files == ["a.txt"]
    assert not (ws / "a.txt").exists()
    assert (ws / "b.txt").exists()
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"


def test_precise_restore_undoes_last_turn(ws, ckpt_root):
    """回退到最后一轮之前：最后一轮的改动尚无「下一个快照」，
    由 restore 时创建的安全快照充当 diff 终点（否则最后一轮永远漏撤）。"""
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")  # e0：workspace 无该文件
    (ws / "苹果.txt").write_text("", encoding="utf-8")
    store.snapshot("第二轮")  # e1：有 苹果.txt
    (ws / "苹果.txt").rename(ws / "橘子.txt")  # 最后一轮的工作：重命名

    # /back 确认前的预览：安全快照尚未创建，最后一轮没有「下一个快照」，
    # 必须用工作区 diff 补全（含未跟踪的 橘子.txt），且不得崩溃
    preview = store.precise_files(store.entries()[1])
    assert set(preview) == {"苹果.txt", "橘子.txt"}

    # /back 2：撤销最后一轮的重命名
    pr = store.restore_precise(store.entries()[1])
    assert pr is not None and pr.ok is True
    assert set(pr.files) == {"苹果.txt", "橘子.txt"}
    assert (ws / "苹果.txt").exists()
    assert not (ws / "橘子.txt").exists()

    # 模拟命令层行为：回退后账本截断，再回退到第一轮之前 → 连创建也撤销
    store.truncate(1)
    pr1 = store.restore_precise(store.entries()[0])
    assert pr1 is not None and pr1.ok is True
    assert pr1.files == ["苹果.txt"]
    assert not (ws / "苹果.txt").exists()
    assert not (ws / "橘子.txt").exists()


def test_precise_restore_skips_shared_start(ws, ckpt_root):
    """起点被其它会话共享（两轮起点之间无任何改动）时必须跳过：
    这段区间的 diff 全部属于其它会话，撤销会误伤。"""
    import time

    store_a = _store(ckpt_root, ws, "sessA")
    store_a.snapshot("A 第一轮")  # c1
    store_a.snapshot("A 第二轮（无改动）")  # 复用 c1
    store_b = _store(ckpt_root, ws, "sessB")
    store_b.snapshot("B 第一轮")  # 仍无改动 → 也复用 c1（共享起点）
    (ws / "b.txt").write_text("by B\n", encoding="utf-8")
    time.sleep(0.05)
    store_b.snapshot("B 第二轮")  # 捕获 B 的工作

    pr = store_a.restore_precise(store_a.entries()[0])
    assert pr is not None and pr.ok is True
    assert pr.files == []
    assert (ws / "b.txt").exists()  # B 的工作未被误撤


def test_precise_restore_conflict_rolls_back(ws, ckpt_root):
    """同文件交叉时反向 apply 失败：工作区完整回滚，之后仍可降级全量回退。"""
    import time

    store_a = _store(ckpt_root, ws, "sessA")
    store_a.snapshot("A 第一轮")  # code.py = v1
    (ws / "code.py").write_text("v2 by A\n", encoding="utf-8")
    time.sleep(0.05)
    store_a.snapshot("A 第二轮")  # A 改 code.py

    store_b = _store(ckpt_root, ws, "sessB")
    store_b.snapshot("B 第一轮")  # 基于 v2 by A
    (ws / "code.py").write_text("v3 by B\n", encoding="utf-8")
    time.sleep(0.05)
    store_b.snapshot("B 第二轮")  # B 在同一文件上继续改

    # A 精确回退：A 的私有 diff（v1 -> v2 by A）无法应用到 v3 by B 上
    pr = store_a.restore_precise(store_a.entries()[0])
    assert pr is not None and pr.ok is False
    assert "code.py" in pr.files
    # 回滚完整：工作区与还原前一致
    assert (ws / "code.py").read_text(encoding="utf-8") == "v3 by B\n"

    # 降级全量回退仍可用（第 1 层安全快照保证链路完整）
    result = store_a.restore(store_a.entries()[0])
    assert result is not None
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"


def test_precise_restore_noop_and_single_session(ws, ckpt_root):
    """回退到最新快照时代码无需改动；单会话场景与全量回退结果等效。"""
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")
    (ws / "code.py").write_text("v2\n", encoding="utf-8")
    (ws / "new.py").write_text("added\n", encoding="utf-8")
    store.snapshot("第二轮")

    # 回退到最新快照：本会话其后无改动，工作区原样
    pr = store.restore_precise(store.entries()[-1])
    assert pr is not None and pr.ok is True
    assert pr.files == []
    assert (ws / "code.py").read_text(encoding="utf-8") == "v2\n"

    # 回退到第一轮：精确回退等效于全量回退（含删除新增文件）
    pr1 = store.restore_precise(store.entries()[0])
    assert pr1 is not None and pr1.ok is True
    assert set(pr1.files) == {"code.py", "new.py"}
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"
    assert not (ws / "new.py").exists()


def test_precise_restore_with_reused_head_snapshots(ws, ckpt_root):
    """复用 HEAD 的空快照条目必须去重：否则同一改动会被反向 apply 两次。"""
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")  # c1
    (ws / "code.py").write_text("v2\n", encoding="utf-8")
    c2 = store.snapshot("第二轮")
    assert c2 != store.entries()[0]["commit"]
    c3 = store.snapshot("无改动的第三轮")
    assert c3 == c2  # 复用 HEAD，账本仍追加

    pr = store.restore_precise(store.entries()[0])
    assert pr is not None and pr.ok is True
    assert (ws / "code.py").read_text(encoding="utf-8") == "v1\n"


def test_ledger_persists_across_instances(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")
    store.snapshot("第二轮")

    reopened = _store(ckpt_root, ws, "s1")
    assert [e["preview"] for e in reopened.entries()] == ["第一轮", "第二轮"]
    assert reopened.entries()[0]["commit"]


def test_prune_truncate_clear_ledger(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    for i in range(3):
        store.snapshot(f"第{i}轮")

    store.prune(1)  # 压缩丢弃第一个用户轮次
    assert len(store.entries()) == 2
    assert store.entries()[0]["preview"] == "第1轮"

    store.truncate(1)  # 回退到第 2 轮之前
    assert len(store.entries()) == 1

    store.clear_ledger()  # /clear
    assert store.entries() == []
    assert json.loads(store._ledger_path.read_text(encoding="utf-8")) == []


def test_drop_session_and_drop_all(ws, ckpt_root):
    _store(ckpt_root, ws, "s1").snapshot("第一轮")
    _store(ckpt_root, ws, "s2").snapshot("第一轮")

    store = _store(ckpt_root, ws, "s1")
    store.drop_session("s2")
    assert not (store._repo_dir / "s2.ckpt.json").exists()
    assert (store._repo_dir / "s1.ckpt.json").exists()

    store.drop_all()
    assert list(store._repo_dir.glob("*.ckpt.json")) == []
    assert store.entries() == []


def test_disabled_without_git(monkeypatch, ws, ckpt_root):
    monkeypatch.setattr("agent.checkpoints.shutil.which", lambda name: None)
    store = _store(ckpt_root, ws, "s1")
    assert store.enabled is False
    assert store.snapshot("第一轮") is None
    assert store.entries() == []
    assert store.plan_restore({"commit": "abc", "ts": 1.0}) is None
    assert store.restore({"commit": "abc"}) is None


def test_corrupt_ledger_tolerated(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    store.snapshot("第一轮")
    store._ledger_path.write_text("{broken json", encoding="utf-8")

    reopened = _store(ckpt_root, ws, "s1")
    assert reopened.entries() == []
    # 会话仍可正常打新快照（账本重建）
    assert reopened.snapshot("重新开始") is not None


def test_disabled_store_skips_ledger_writes(ws, ckpt_root):
    store = _store(ckpt_root, ws, "s1")
    store.enabled = False
    store.clear_ledger()
    assert not store._ledger_path.exists()  # 未启用时不产生账本文件
