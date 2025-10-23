import hashlib
from enum import Enum
from typing import List, Union

from nlp_retrieval.core.models import SearchableItem
from nlp_retrieval.loaders.loader_abc import BaseLoader
from utils_database_connector.core import Database
from utils_database_connector.sqlite_db import DatabaseSqlite
from tqdm import tqdm


class SerializationStrategy(Enum):
    """Defines the different ways a database can be serialized into text items."""

    SCHEMA_LEVEL = "schema_level"
    ROW_LEVEL = "row_level"
    VALUE_LEVEL_SCHEMA_AWARE = "value_level_schema_aware"  # values with schema context
    VALUE_LEVEL = "value_level"


class DatabaseLoader(BaseLoader):
    """
    Loads data from a database connection and serializes it into `SearchableItem`s.

    Supports multiple serialization strategies to structure the database content
    for different retrieval tasks.
    """

    def __init__(
        self,
        db: Union[Database, DatabaseSqlite],
        strategy: SerializationStrategy,
        table_separator: str = " <table> ",
        column_separator: str = " <col> ",
        value_separator: str = " <val> ",
    ):
        """
        Initializes the DatabaseLoader.

        Args:
            db: An active database connection object.
            strategy: The serialization strategy to use.
            table_separator: The token to use for separating table names.
            column_separator: The token to use for separating column names.
            value_separator: The token to use for separating cell values.
        """
        self.db = db
        self.strategy = strategy
        self.table_sep = table_separator
        self.col_sep = column_separator
        self.val_sep = value_separator

    def load(self) -> List[SearchableItem]:
        """
        Executes the data loading and serialization based on the chosen strategy.

        Returns:
            A list of `SearchableItem` objects.
        """
        all_items: List[SearchableItem] = []
        schema = self.db.get_tables_and_columns()
        tables = [
            table for table in schema.get("tables", []) if table != "sqlite_sequence"
        ]

        for table_name in tqdm(tables, desc=f"Loading from DB ({self.strategy.value})"):
            columns_for_table = [
                col.split(".")[1]
                for col in schema.get("columns", [])
                if col.startswith(f"{table_name}.")
            ]
            if not columns_for_table:
                continue

            if self.strategy == SerializationStrategy.SCHEMA_LEVEL:
                items = self._load_schema_level(table_name, columns_for_table)
            elif self.strategy == SerializationStrategy.ROW_LEVEL:
                items = self._load_row_level(table_name, columns_for_table)
            elif self.strategy == SerializationStrategy.VALUE_LEVEL_SCHEMA_AWARE:
                items = self._load_value_level(
                    table_name, columns_for_table, schema_aware=True
                )
            elif self.strategy == SerializationStrategy.VALUE_LEVEL:
                items = self._load_value_level(
                    table_name, columns_for_table, schema_aware=False
                )
            else:
                items = []

            all_items.extend(items)

        return all_items

    def _load_schema_level(
        self, table_name: str, columns: List[str]
    ) -> List[SearchableItem]:
        """Serializes an entire table schema into a single item."""
        content = (
            f"{self.table_sep}{table_name}{self.col_sep}"
            f"{self.col_sep.join(columns)}"
        )
        metadata = {"table": table_name, "columns": columns}
        return [SearchableItem(item_id=table_name, content=content, metadata=metadata)]

    def _load_row_level(
        self, table_name: str, columns: List[str]
    ) -> List[SearchableItem]:
        """Serializes each row of a table into a separate item."""
        items: List[SearchableItem] = []
        df = self.db.execute(f'SELECT * FROM "{table_name}"', limit=-1)  # nosec B608

        for index, row in df.iterrows():
            row_parts = []
            row_metadata = []
            for col in columns:
                value = row.get(col)
                if value is not None:
                    value_str = str(value)
                    row_parts.append(f"{self.col_sep}{col}{self.val_sep}{value_str}")
                    row_metadata.append({"column": col, "value": value})

            if not row_parts:
                continue

            content = f"{self.table_sep}{table_name}{''.join(row_parts)}"
            item_id = f"{table_name}_row_{index}"
            metadata = {"table": table_name, "row_values": row_metadata}
            items.append(
                SearchableItem(item_id=item_id, content=content, metadata=metadata)
            )

        return items

    def _load_value_level(
        self, table_name: str, columns: List[str], schema_aware: bool
    ) -> List[SearchableItem]:
        """Serializes each unique cell value into a separate item."""
        items: List[SearchableItem] = []
        for col_name in columns:
            # Get distinct, non-null values for the column
            df = self.db.execute(
                f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL;',  # nosec B608
                limit=-1,
            )
            values = df[col_name].tolist()

            for value in values:
                value_str = str(value)
                metadata = {"table": table_name, "column": col_name, "value": value}

                # Use a hash of the value for a stable, unique ID
                value_hash = hashlib.md5(
                    value_str.encode(), usedforsecurity=False
                ).hexdigest()

                item_id = f"{table_name}.{col_name}.{value_hash}"

                if schema_aware:
                    content = (
                        f"{self.table_sep}{table_name}"
                        f"{self.col_sep}{col_name}"
                        f"{self.val_sep}{value_str}"
                    )
                else:
                    content = value_str

                items.append(
                    SearchableItem(item_id=item_id, content=content, metadata=metadata)
                )

        return items
