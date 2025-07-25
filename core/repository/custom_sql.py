import sqlite3
from typing import List, Optional

from core.entity.image_parse_data import BBox, ImageParseUnit, ImageParseResult
from core.repository.metadata_utils import MetadataUtils

class SQLHandler:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_tables()

    def _init_tables(self):
        self.cursor.execute('''
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
        self.conn.commit()

    def create_unit_table(self, unit_table: str):
        # 注意：sqlite表名参数无法用?占位符，这里需要谨慎避免SQL注入，确保unit_table合法
        self.cursor.execute(f'''
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
        self.conn.commit()

    def save_result(self, result_obj: ImageParseResult):
        if not result_obj.uid:
            raise ValueError("result_obj.uid must not be empty")

        unit_table = f"unit_table_{result_obj.uid}"
        self.create_unit_table(unit_table)

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
            self.cursor.execute(
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
        self.cursor.execute(
            f"INSERT OR REPLACE INTO result_table ({','.join(r_data.keys())}) VALUES ({placeholders})",
            list(r_data.values())
        )
        self.conn.commit()

    def fetch_result(self, uid: str, as_object: bool = False):
        self.cursor.execute("SELECT * FROM result_table WHERE uid = ?", (uid,))
        row = self.cursor.fetchone()
        if not row:
            return None

        columns = [desc[0] for desc in self.cursor.description]
        result_dict = dict(zip(columns, row))
        result_dict = MetadataUtils.deserialize_metadata(result_dict)

        # 取出 unit_table 并解析 units（也可能为空）
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
        self.cursor.execute(f"SELECT * FROM {unit_table}")
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]

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
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'unit_table_%'")
        rows = self.cursor.fetchall()
        return [row[0] for row in rows]

    def fuzzy_query(self, query_text: str, topk: int = 5) -> List[str]:
        matches = set()

        # 查询 result_table.summary_text
        self.cursor.execute("SELECT uid FROM result_table WHERE summary_text LIKE ?", (f"%{query_text}%",))
        matches.update(uid for uid, in self.cursor.fetchall())

        # 查询所有 unit_table 的 text 和 label 字段
        for table_name in self.get_all_unit_table_names():
            query_sql = f"SELECT uid FROM {table_name} WHERE text LIKE ? OR label LIKE ?"
            self.cursor.execute(query_sql, (f"%{query_text}%", f"%{query_text}%"))
            unit_rows = self.cursor.fetchall()
            if unit_rows:
                # unit_table 的名字格式是 unit_table_<result_uid>
                result_uid = table_name.split("unit_table_")[-1]
                matches.add(result_uid)

        return list(matches)[:topk]

    def update_result(self, uid: str, update_fields: dict):
        """
        根据 uid 更新 result_table 的字段。
        """
        if not update_fields:
            return
        set_clause = []
        values = []
        for k, v in update_fields.items():
            if k == "images" or k == "embedding":
                continue  # 跳过图片和embedding字段
            set_clause.append(f"{k} = ?")
            values.append(v)
        if not set_clause:
            return
        sql = f"UPDATE result_table SET {', '.join(set_clause)} WHERE uid = ?"
        values.append(uid)
        self.cursor.execute(sql, values)
        self.conn.commit()

    def delete_result(self, uid: str):
        """
        删除 result_table 及其对应的 unit_table。
        """
        # 先查 unit_table
        self.cursor.execute("SELECT unit_table FROM result_table WHERE uid = ?", (uid,))
        row = self.cursor.fetchone()
        if row:
            unit_table = row[0]
            self.cursor.execute(f"DROP TABLE IF EXISTS {unit_table}")
        self.cursor.execute("DELETE FROM result_table WHERE uid = ?", (uid,))
        self.conn.commit()

    def list_results(self, filters: dict = None, as_object: bool = False):
        """
        支持简单条件过滤，返回所有结果。
        """
        sql = "SELECT uid FROM result_table"
        values = []
        if filters:
            clauses = []
            for k, v in filters.items():
                clauses.append(f"{k} = ?")
                values.append(v)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
        self.cursor.execute(sql, values)
        uids = [row[0] for row in self.cursor.fetchall()]
        return [self.fetch_result(uid, as_object) for uid in uids]


    def close(self):
        self.conn.close()
