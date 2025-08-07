import sqlite3
from typing import List, Optional

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.repository.metadata_utils import MetadataUtils

class SQLHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self):
        """为每次数据库操作提供一个新的连接"""
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS result_table (
                    uid TEXT PRIMARY KEY,
                    summary_text TEXT,
                    metadata TEXT,
                    unit_table TEXT,
                    image_path TEXT,
                    bboxs_image_path TEXT,
                    masks_path TEXT,
                    masks_image_path TEXT
                )
            ''')
            conn.commit()

    def create_unit_table(self, unit_table: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {unit_table} (
                    uid TEXT PRIMARY KEY,
                    result_uid TEXT,
                    bbox TEXT,
                    source_module TEXT,
                    score REAL,
                    type TEXT,
                    text TEXT,
                    label TEXT,
                    metadata TEXT,
                    mask_image_path TEXT
                )
            ''')
            conn.commit()

    def save_result(self, result_obj: ImageParseResult):
        if not result_obj.uid:
            raise ValueError("result_obj.uid must not be empty")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            unit_table = f"unit_table_{result_obj.uid}"
            self.create_unit_table(unit_table) # 这里会创建自己的连接，但也可以在同一个with块中做

            # 保存 units
            for u in result_obj.units:
                if not u.uid:
                    raise ValueError("Unit uid must not be empty")
                u_data = {
                    "uid": u.uid,
                    "result_uid": result_obj.uid,
                    "bbox": str(MetadataUtils.preprocess_metadata(u.bbox.to_dict())),
                    "source_module": u.source_module or "",
                    "score": u.score if u.score is not None else 0.0,
                    "type": u.type or "",
                    "text": u.text or "",
                    "label": u.label or "",
                    "metadata": str(MetadataUtils.preprocess_metadata(u.metadata or {})),
                    "mask_image_path": u.storage_dict.get("mask_image_path", ""),
                }
                placeholders = ",".join(["?"] * len(u_data))
                cursor.execute(
                    f"INSERT OR REPLACE INTO {unit_table} ({','.join(u_data.keys())}) VALUES ({placeholders})",
                    list(u_data.values())
                )

            # 保存 result 自身
            r_data = {
                "uid": result_obj.uid,
                "summary_text": result_obj.summary_text or "",
                "metadata": str(MetadataUtils.preprocess_metadata(result_obj.metadata or {})),
                "unit_table": unit_table,
                "image_path": result_obj.storage_dict.get("image_path", ""),
                "bboxs_image_path": result_obj.storage_dict.get("bboxs_image_path", ""),
                "masks_path": result_obj.storage_dict.get("masks_path", ""),
                "masks_image_path": result_obj.storage_dict.get("masks_image_path", ""),
            }
            placeholders = ",".join(["?"] * len(r_data))
            cursor.execute(
                f"INSERT OR REPLACE INTO result_table ({','.join(r_data.keys())}) VALUES ({placeholders})",
                list(r_data.values())
            )
            conn.commit()

    def fetch_result(self, uid: str, as_object: bool = False):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM result_table WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            if not row:
                return None

            columns = [desc[0] for desc in cursor.description]
            result_dict = dict(zip(columns, row))
            result_dict = MetadataUtils.deserialize_metadata(result_dict)

            unit_table = result_dict.get("unit_table", "")
            units = self.fetch_units(unit_table, as_object) if unit_table else []

        if not as_object:
            return {
                "result": result_dict,
                "units": units
            }

        result = ImageParseResult(
            uid=result_dict.get("uid"),
            summary_text=result_dict.get("summary_text", ""),
            metadata=result_dict.get("metadata", {}),
            units=units,
        )
        result.storage_dict = {
            "image_path": result_dict.get("image_path", ""),
            "bboxs_image_path": result_dict.get("bboxs_image_path", ""),
            "masks_path": result_dict.get("masks_path", ""),
            "masks_image_path": result_dict.get("masks_image_path", ""),
        }
        return result

    def fetch_units(self, unit_table: str, as_object: bool = False):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {unit_table}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            result = []
            for row in rows:
                data = dict(zip(columns, row))
                data = MetadataUtils.deserialize_metadata(data)

                if not as_object:
                    result.append(data)
                    continue

                unit = ImageParseUnit(
                    uid=data.get("uid"),
                    bbox=BBox.from_dict(data.get("bbox", {})),
                    source_module=data.get("source_module", ""),
                    score=data.get("score", 0.0),
                    type=data.get("type", ""),
                    text=data.get("text", ""),
                    label=data.get("label", ""),
                    metadata=data.get("metadata", {}),
                )
                unit.storage_dict = {
                    "mask_image_path": data.get("mask_image_path", "")
                }
                result.append(unit)
        return result

    def get_all_unit_table_names(self) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'unit_table_%'")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def fuzzy_query(self, query_text: str, topk: int = 5) -> List[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            matches = set()

            # 查询 result_table.summary_text
            cursor.execute("SELECT uid FROM result_table WHERE summary_text LIKE ?", (f"%{query_text}%",))
            matches.update(uid for uid, in cursor.fetchall())

            # 查询所有 unit_table 的 text 和 label 字段
            for table_name in self.get_all_unit_table_names():
                query_sql = f"SELECT uid FROM {table_name} WHERE text LIKE ? OR label LIKE ?"
                cursor.execute(query_sql, (f"%{query_text}%", f"%{query_text}%"))
                unit_rows = cursor.fetchall()
                if unit_rows:
                    result_uid = table_name.split("unit_table_")[-1]
                    matches.add(result_uid)

        return list(matches)[:topk]

    def update_result(self, uid: str, update_fields: dict):
        if not update_fields:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()
            set_clause = []
            values = []
            for k, v in update_fields.items():
                if k == "images" or k == "embedding":
                    continue
                set_clause.append(f"{k} = ?")
                values.append(v)
            if not set_clause:
                return
            sql = f"UPDATE result_table SET {', '.join(set_clause)} WHERE uid = ?"
            values.append(uid)
            cursor.execute(sql, values)
            conn.commit()

    def delete_result(self, uid: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT unit_table FROM result_table WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            if row:
                unit_table = row[0]
                cursor.execute(f"DROP TABLE IF EXISTS {unit_table}")
            cursor.execute("DELETE FROM result_table WHERE uid = ?", (uid,))
            conn.commit()

    def list_results(self, filters: dict = None, as_object: bool = False):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT uid FROM result_table"
            values = []
            if filters:
                clauses = []
                for k, v in filters.items():
                    clauses.append(f"{k} = ?")
                    values.append(v)
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
            cursor.execute(sql, values)
            uids = [row[0] for row in cursor.fetchall()]
        return [self.fetch_result(uid, as_object) for uid in uids]

    def append_units(self, result_uid: str, new_units: list):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            unit_table = f"unit_table_{result_uid}"
            self.create_unit_table(unit_table)

            cursor.execute(f"SELECT uid FROM {unit_table}")
            existing_uids = set(row[0] for row in cursor.fetchall())
            duplicate_uids = []
            for u in new_units:
                uid = u.uid if hasattr(u, 'uid') else u.get('uid')
                if uid in existing_uids:
                    duplicate_uids.append(uid)
                    continue
                u_data = {
                    "uid": uid,
                    "result_uid": result_uid,
                    "bbox": str(MetadataUtils.preprocess_metadata(u.bbox.to_dict() if hasattr(u, 'bbox') else u['bbox'])),
                    "source_module": getattr(u, 'source_module', u.get('source_module', '')),
                    "score": getattr(u, 'score', u.get('score', 0.0)),
                    "type": getattr(u, 'type', u.get('type', '')),
                    "text": getattr(u, 'text', u.get('text', '')),
                    "label": getattr(u, 'label', u.get('label', '')),
                    "metadata": str(MetadataUtils.preprocess_metadata(getattr(u, 'metadata', u.get('metadata', {})))),
                    "mask_image_path": getattr(u, 'storage_dict', {}).get('mask_image_path', '') if hasattr(u, 'storage_dict') else u.get('storage_dict', {}).get('mask_image_path', ''),
                }
                placeholders = ",".join(["?"] * len(u_data))
                cursor.execute(
                    f"INSERT INTO {unit_table} ({','.join(u_data.keys())}) VALUES ({placeholders})",
                    list(u_data.values())
                )
            conn.commit()
        return duplicate_uids

    def replace_units(self, result_uid: str, new_units: list):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            unit_table = f"unit_table_{result_uid}"
            self.create_unit_table(unit_table)

            for u in new_units:
                uid = u.uid if hasattr(u, 'uid') else u.get('uid')
                update_fields = {}
                for field in ["label", "text", "score", "metadata"]:
                    val = getattr(u, field, u.get(field, None))
                    if val is not None:
                        if field == "metadata":
                            val = str(MetadataUtils.preprocess_metadata(val))
                        update_fields[field] = val
                if not update_fields:
                    continue
                set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
                values = list(update_fields.values())
                values.append(uid)
                cursor.execute(
                    f"UPDATE {unit_table} SET {set_clause} WHERE uid = ?",
                    values
                )
            conn.commit()

    def delete_units(self, result_uid: str, unit_uids: list):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            unit_table = f"unit_table_{result_uid}"
            self.create_unit_table(unit_table)

            for uid in unit_uids:
                cursor.execute(f"DELETE FROM {unit_table} WHERE uid = ?", (uid,))
            conn.commit()
