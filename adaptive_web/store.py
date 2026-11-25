"""Independent WAL SQLite store with RLock concurrency and nested-safe transactions."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable
import uuid

from . import catalogue_bridge


class AdaptiveStore:
    """Thread-safe SQLite store compatible with MemoryService/EvidenceStream and adaptive_web tables."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._is_memory = self.path == ":memory:"
        if not self._is_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self.db = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=15.0,
            isolation_level=None,
        )
        self.db.row_factory = sqlite3.Row
        with self._lock:
            if not self._is_memory:
                self.db.execute("PRAGMA journal_mode=WAL;")
            self.db.execute("PRAGMA busy_timeout=5000;")
            self.db.execute("PRAGMA synchronous=NORMAL;")
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles(
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS interactions(
                    id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    task TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS interactions_learner ON interactions(learner, skill);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    interaction TEXT NOT NULL,
                    at REAL NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_stream ON events(learner, skill, id);
                CREATE TABLE IF NOT EXISTS requests(
                    id TEXT PRIMARY KEY,
                    result TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generated_tasks(
                    task_id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL DEFAULT '',
                    skill TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS gen_tasks_skill ON generated_tasks(skill, difficulty, task_type);
                CREATE TABLE IF NOT EXISTS learning_paths(
                    id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS paths_lookup ON learning_paths(learner, skill, created);
                CREATE TABLE IF NOT EXISTS course_materials(
                    id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS materials_lookup ON course_materials(learner, skill, created);
                CREATE TABLE IF NOT EXISTS tutor_turns(
                    id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    policy_source TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS turns_lookup ON tutor_turns(learner, skill, created);
                CREATE TABLE IF NOT EXISTS chat_messages(
                    id TEXT PRIMARY KEY,
                    learner TEXT NOT NULL,
                    skill TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    code_example TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    mastery_tier TEXT NOT NULL DEFAULT '',
                    guardrail_status TEXT NOT NULL DEFAULT '',
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chat_lookup ON chat_messages(learner, skill, created);
                """
            )
            gen_cols = {
                row["name"]
                for row in self.db.execute("PRAGMA table_info(generated_tasks)").fetchall()
            }
            if "learner" not in gen_cols:
                self.db.execute(
                    "ALTER TABLE generated_tasks ADD COLUMN learner TEXT NOT NULL DEFAULT ''"
                )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS gen_tasks_learner ON generated_tasks(learner, skill, created)"
            )

    @contextmanager
    def transaction(self):
        with self._lock:
            depth = self._tx_depth
            if depth == 0:
                self.db.execute("BEGIN")
            else:
                self.db.execute(f"SAVEPOINT _aw_sp_{depth}")
            self._tx_depth = depth + 1
            try:
                yield self
            except Exception:
                self._tx_depth = depth
                if depth == 0:
                    self.db.execute("ROLLBACK")
                else:
                    self.db.execute(f"ROLLBACK TO SAVEPOINT _aw_sp_{depth}")
                    self.db.execute(f"RELEASE SAVEPOINT _aw_sp_{depth}")
                raise
            else:
                self._tx_depth = depth
                if depth == 0:
                    self.db.execute("COMMIT")
                else:
                    self.db.execute(f"RELEASE SAVEPOINT _aw_sp_{depth}")

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # --- Profiles ---

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT id, name, created FROM profiles ORDER BY created, rowid"
            ).fetchall()
            return [{"id": r["id"], "name": r["name"], "created": float(r["created"])} for r in rows]

    def create_profile(
        self,
        name: str,
        profile_id: str | None = None,
        created: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str):
            raise ValueError("Profile name must be a string.")
        cleaned = " ".join(name.strip().split())
        if not (1 <= len(cleaned) <= 50):
            raise ValueError("Profile name must be between 1 and 50 characters.")
        pid = profile_id or uuid.uuid4().hex[:12]
        ts = float(time.time() if created is None else created)
        with self.transaction():
            self.db.execute(
                "INSERT INTO profiles(id, name, created) VALUES (?, ?, ?)",
                (pid, cleaned, ts),
            )
        return {"id": pid, "name": cleaned, "created": ts}

    def get_profile(self, learner_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                "SELECT id, name, created FROM profiles WHERE id=?",
                (learner_id,),
            ).fetchone()
            if row is None:
                return None
            return {"id": row["id"], "name": row["name"], "created": float(row["created"])}

    def require_profile(self, learner_id: str) -> dict[str, Any]:
        prof = self.get_profile(learner_id)
        if prof is None:
            raise ValueError("Choose a learner profile first.")
        return prof

    # --- Generated & v2.2 Tasks ---

    def save_generated_task(
        self,
        task: dict[str, Any],
        created: float | None = None,
        *,
        learner: str = "",
        verify_code_tests: bool = False,
    ) -> dict[str, Any]:
        catalogue_bridge.assert_valid_task(task, verify_code_tests=verify_code_tests)
        ts = float(time.time() if created is None else created)
        stored = dict(task)
        if learner:
            stored.setdefault("learner", learner)
        payload = json.dumps(stored)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO generated_tasks(task_id, learner, skill, difficulty, task_type, payload, created)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    learner=CASE WHEN excluded.learner != '' THEN excluded.learner ELSE generated_tasks.learner END,
                    skill=excluded.skill,
                    difficulty=excluded.difficulty,
                    task_type=excluded.task_type,
                    payload=excluded.payload,
                    created=excluded.created
                """,
                (
                    task["task_id"],
                    str(learner or stored.get("learner") or ""),
                    task["skill_id"],
                    task["difficulty"],
                    task["task_type"],
                    payload,
                    ts,
                ),
            )
        return json.loads(payload)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        v22 = catalogue_bridge.get_v22_task(task_id)
        if v22 is not None:
            return v22
        with self._lock:
            row = self.db.execute(
                "SELECT payload FROM generated_tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            return json.loads(row["payload"]) if row is not None else None

    def list_generated_tasks(
        self,
        skill: str | None = None,
        difficulty: str | None = None,
        task_type: str | None = None,
        *,
        learner: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT payload FROM generated_tasks WHERE 1=1"
        args: list[Any] = []
        if learner is not None:
            query += " AND learner=?"
            args.append(learner)
        if skill is not None:
            query += " AND skill=?"
            args.append(skill)
        if difficulty is not None:
            query += " AND difficulty=?"
            args.append(difficulty)
        if task_type is not None:
            query += " AND task_type=?"
            args.append(task_type)
        query += " ORDER BY created, rowid"
        with self._lock:
            rows = self.db.execute(query, args).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    # --- Interactions & Events (MemoryService / EvidenceStream compatible) ---

    def get_interaction(self, interaction_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                "SELECT payload FROM interactions WHERE id=?",
                (interaction_id,),
            ).fetchone()
            return json.loads(row["payload"]) if row is not None else None

    def get_or_init_interaction(
        self,
        interaction_id: str,
        learner: str,
        skill: str,
        task_id: str,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.transaction():
            existing = self.get_interaction(interaction_id)
            if existing is not None:
                return existing
            item = dict(defaults or {})
            item.setdefault("id", interaction_id)
            item.setdefault("learner", learner)
            item.setdefault("skill", skill)
            item.setdefault("task", task_id)
            self.save(item)
            return item

    def save(self, item: dict[str, Any]) -> None:
        payload = json.dumps(item)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO interactions(id, learner, skill, task, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    learner=excluded.learner,
                    skill=excluded.skill,
                    task=excluded.task,
                    payload=excluded.payload
                """,
                (item["id"], item["learner"], item["skill"], item["task"], payload),
            )

    save_interaction = save

    def interactions(self, learner: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT payload FROM interactions WHERE learner=? ORDER BY rowid",
                (learner,),
            ).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def get_interactions(self, learner: str, skill: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if skill is None:
                return self.interactions(learner)
            rows = self.db.execute(
                "SELECT payload FROM interactions WHERE learner=? AND skill=? ORDER BY rowid",
                (learner, skill),
            ).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def append_event(
        self,
        learner: str,
        skill: str,
        interaction_id: str,
        at: float,
        kind: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        ts = float(at)
        body = json.dumps(payload)
        with self.transaction():
            cur = self.db.execute(
                """
                INSERT INTO events(learner, skill, interaction, at, kind, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (learner, skill, interaction_id, ts, kind, body),
            )
            event_id = int(cur.lastrowid)
        result = dict(json.loads(body))
        result.update(
            id=event_id,
            at=ts,
            kind=kind,
            learner=learner,
            skill=skill,
            interaction=interaction_id,
        )
        return result

    def events(self, learner: str, skill: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT id, learner, skill, interaction, at, kind, payload FROM events WHERE learner=?"
        args: list[Any] = [learner]
        if skill is not None:
            query += " AND skill=?"
            args.append(skill)
        query += " ORDER BY id"
        with self._lock:
            rows = self.db.execute(query, args).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                ev = dict(json.loads(r["payload"]))
                ev.update(
                    id=int(r["id"]),
                    at=float(r["at"]),
                    kind=r["kind"],
                    learner=r["learner"],
                    skill=r["skill"],
                    interaction=r["interaction"],
                )
                out.append(ev)
            return out

    get_events = events

    # --- Learning Paths, Course Materials & Tutor Turns ---

    def save_learning_path(
        self,
        learner: str,
        skill: str,
        payload: dict[str, Any],
        path_id: str | None = None,
        created: float | None = None,
    ) -> dict[str, Any]:
        pid = path_id or uuid.uuid4().hex[:12]
        ts = float(time.time() if created is None else created)
        stored = dict(payload)
        stored.setdefault("id", pid)
        stored.setdefault("learner", learner)
        stored.setdefault("skill", skill)
        stored.setdefault("created", ts)
        body = json.dumps(stored)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO learning_paths(id, learner, skill, payload, created)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, created=excluded.created
                """,
                (pid, learner, skill, body, ts),
            )
        return json.loads(body)

    def get_latest_learning_path(self, learner: str, skill: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                """
                SELECT payload FROM learning_paths
                WHERE learner=? AND skill=?
                ORDER BY created DESC, rowid DESC
                LIMIT 1
                """,
                (learner, skill),
            ).fetchone()
            return json.loads(row["payload"]) if row is not None else None

    def save_course_material(
        self,
        learner: str,
        skill: str,
        payload: dict[str, Any],
        material_id: str | None = None,
        created: float | None = None,
    ) -> dict[str, Any]:
        mid = material_id or uuid.uuid4().hex[:12]
        ts = float(time.time() if created is None else created)
        stored = dict(payload)
        stored.setdefault("id", mid)
        stored.setdefault("learner", learner)
        stored.setdefault("skill", skill)
        stored.setdefault("created", ts)
        body = json.dumps(stored)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO course_materials(id, learner, skill, payload, created)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, created=excluded.created
                """,
                (mid, learner, skill, body, ts),
            )
        return json.loads(body)

    def get_latest_course_material(self, learner: str, skill: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute(
                """
                SELECT payload FROM course_materials
                WHERE learner=? AND skill=?
                ORDER BY created DESC, rowid DESC
                LIMIT 1
                """,
                (learner, skill),
            ).fetchone()
            return json.loads(row["payload"]) if row is not None else None

    def save_tutor_turn(
        self,
        learner: str,
        skill: str,
        policy_source: str,
        payload: dict[str, Any],
        turn_id: str | None = None,
        created: float | None = None,
    ) -> dict[str, Any]:
        tid = turn_id or uuid.uuid4().hex[:12]
        ts = float(time.time() if created is None else created)
        stored = dict(payload)
        stored.setdefault("id", tid)
        stored.setdefault("learner", learner)
        stored.setdefault("skill", skill)
        stored.setdefault("policy_source", policy_source)
        stored.setdefault("created", ts)
        body = json.dumps(stored)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO tutor_turns(id, learner, skill, policy_source, payload, created)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    policy_source=excluded.policy_source,
                    payload=excluded.payload,
                    created=excluded.created
                """,
                (tid, learner, skill, policy_source, body, ts),
            )
        return json.loads(body)

    def get_tutor_turns(
        self,
        learner: str,
        skill: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT payload FROM tutor_turns WHERE learner=?"
        args: list[Any] = [learner]
        if skill is not None:
            query += " AND skill=?"
            args.append(skill)
        query += " ORDER BY created, rowid"
        if limit is not None:
            query += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self.db.execute(query, args).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    # --- Doubt Tutor Chat History ---

    def save_chat_message(
        self,
        learner: str,
        skill: str,
        role: str,
        message: str,
        *,
        code_example: str = "",
        source: str = "",
        citations: list[dict[str, Any]] | None = None,
        mastery_tier: str = "",
        guardrail_status: str = "",
        msg_id: str | None = None,
        created: float | None = None,
    ) -> dict[str, Any]:
        cid = msg_id or f"chat-{uuid.uuid4().hex[:12]}"
        ts = float(time.time() if created is None else created)
        citations_list = list(citations or [])
        citations_json = json.dumps(citations_list)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO chat_messages(
                    id, learner, skill, role, message, code_example, source,
                    citations_json, mastery_tier, guardrail_status, created
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role=excluded.role,
                    message=excluded.message,
                    code_example=excluded.code_example,
                    source=excluded.source,
                    citations_json=excluded.citations_json,
                    mastery_tier=excluded.mastery_tier,
                    guardrail_status=excluded.guardrail_status,
                    created=excluded.created
                """,
                (
                    cid,
                    learner,
                    skill,
                    str(role),
                    str(message),
                    str(code_example or ""),
                    str(source or ""),
                    citations_json,
                    str(mastery_tier or ""),
                    str(guardrail_status or ""),
                    ts,
                ),
            )
        return {
            "id": cid,
            "learner": learner,
            "skill": skill,
            "role": str(role),
            "message": str(message),
            "code_example": str(code_example or ""),
            "source": str(source or ""),
            "citations": citations_list,
            "mastery_tier": str(mastery_tier or ""),
            "guardrail_status": str(guardrail_status or ""),
            "created": ts,
        }

    def get_chat_messages(
        self,
        learner: str,
        skill: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, learner, skill, role, message, code_example, source, "
            "citations_json, mastery_tier, guardrail_status, created "
            "FROM chat_messages WHERE learner=?"
        )
        args: list[Any] = [learner]
        if skill is not None:
            query += " AND skill=?"
            args.append(skill)
        query += " ORDER BY created ASC, rowid ASC"
        with self._lock:
            rows = self.db.execute(query, args).fetchall()
        items = [
            {
                "id": r["id"],
                "learner": r["learner"],
                "skill": r["skill"],
                "role": r["role"],
                "message": r["message"],
                "code_example": r["code_example"],
                "source": r["source"],
                "citations": json.loads(r["citations_json"] or "[]"),
                "mastery_tier": r["mastery_tier"],
                "guardrail_status": r["guardrail_status"],
                "created": float(r["created"]),
            }
            for r in rows
        ]
        if limit is not None and int(limit) >= 0:
            items = items[-int(limit) :] if int(limit) > 0 else []
        return items

    # --- Idempotency ---

    def _validate_request_id(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not (8 <= len(request_id) <= 100):
            raise ValueError("request_id must be a string of length 8..100.")

    def idempotent_get(self, request_id: str) -> dict[str, Any] | None:
        self._validate_request_id(request_id)
        with self._lock:
            row = self.db.execute(
                "SELECT result FROM requests WHERE id=?",
                (request_id,),
            ).fetchone()
            return json.loads(row["result"]) if row is not None else None

    def idempotent_set(self, request_id: str, result: dict[str, Any]) -> dict[str, Any]:
        self._validate_request_id(request_id)
        encoded = json.dumps(result)
        with self.transaction():
            self.db.execute(
                """
                INSERT INTO requests(id, result) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET result=excluded.result
                """,
                (request_id, encoded),
            )
        return json.loads(encoded)

    def run_idempotent(self, request_id: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        self._validate_request_id(request_id)
        with self.transaction():
            existing = self.idempotent_get(request_id)
            if existing is not None:
                return existing
            result = fn()
            self.idempotent_set(request_id, result)
            return result
